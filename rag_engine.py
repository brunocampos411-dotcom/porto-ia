"""
Porto IA - RAG Engine v4.0
Busca hibrida: embeddings semanticos (text-embedding-3-small) + TF-IDF por palavras-chave
Formato: chunks.json + embeddings.npy
"""
import os
import re
import json
import math
import unicodedata
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import openai

# ---- Configuracao ----
BASE_DIR = Path(__file__).parent
DOCS_PATH = BASE_DIR / "docs"
CHUNKS_PATH = BASE_DIR / "chunks.json"
EMBEDDINGS_NPY = BASE_DIR / "embeddings.npy"
EMBEDDINGS_PATH = BASE_DIR / "embeddings.json"  # legado fallback


def _load_openrouter_key() -> str:
    """Tenta obter a API key do OpenRouter de varias fontes"""
    # 1. Variavel de ambiente direta
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    # 2. Variavel OpenAI
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return key
    # 3. Arquivo .env do Hermes
    env_paths = ["/opt/data/.env", str(Path.home() / ".env")]
    for env_path in env_paths:
        try:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENROUTER_API_KEY="):
                        val = line.split("=", 1)[1].strip()
                        # Remover caracteres de controle e espacos
                        val = ''.join(c for c in val if c.isprintable()).strip()
                        if val and val != "***":
                            return val
        except Exception:
            pass
    # 4. Config do Hermes YAML
    config_paths = ["/opt/data/config.yaml"]
    for cfg_path in config_paths:
        try:
            import yaml
            cfg = yaml.safe_load(open(str(cfg_path)))
            val = cfg.get("providers", {}).get("openrouter", {}).get("api_key", "")
            if val and "..." not in val:
                return val
        except Exception:
            pass
    return ""


def get_llm_client():
    """Cliente para chat/LLM via OpenRouter"""
    key = _load_openrouter_key()
    return openai.OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1"
    )


def get_embedding_client():
    """Cliente para embeddings via OpenRouter (openai/text-embedding-3-small)"""
    key = _load_openrouter_key()
    base_url = "https://openrouter.ai/api/v1" if key.startswith("sk-or-") else "https://api.openai.com/v1"
    return openai.OpenAI(api_key=key, base_url=base_url)


# ---- Pre-processamento de sinonimos coloquiais ----
SINONIMOS = {
    r'\bbatida\b': 'colisão acidente',
    r'\bbateu\b': 'colisão acidente',
    r'\bchocou\b': 'colisão',
    r'\broubo\b': 'furto roubo',
    r'\broubaram\b': 'furto roubo',
    r'\bquebrou o vidro\b': 'danos a vidros',
    r'\bvidro quebrado\b': 'danos a vidros',
    r'\bvid[rR]os?\b': 'vidros danos a vidros',
    r'\benchente\b': 'alagamento submersão inundação',
    r'\balagou\b': 'alagamento submersão',
    r'\bchuva\b': 'chuva alagamento danos hidráulicos',
    r'\bmorreu\b': 'morte óbito falecimento',
    r'\bfaleceu\b': 'morte óbito falecimento',
    r'\bmachucou\b': 'danos corporais invalidez lesão',
    r'\bferiu\b': 'danos corporais invalidez lesão',
    r'\bpersonagem\b': 'passageiro ocupante',
    r'\bpassageiro\b': 'passageiro ocupante',
    r'\bcancelamento de voo\b': 'cancelamento voo sinistro viagem',
    r'\bvoo cancelado\b': 'cancelamento voo sinistro viagem',
    r'\bguinchou\b': 'guincho reboque',
    r'\bpane\b': 'pane seca bateria socorro mecânico',
    r'\bsinistro\b': 'sinistro acidente dano',
    r'\bfurto\b': 'furto roubo',
    r'\bincendio\b': 'incêndio',
    r'\bincendiou\b': 'incêndio',
    r'\bcarro reserva\b': 'veículo reserva locação',
    r'\bconsertar\b': 'reparo conserto oficina',
    r'\bconserto\b': 'reparo conserto oficina',
    r'\bperdeu o carro\b': 'perda total',
    r'\bperda total\b': 'perda total indenização',
    r'\bcarência\b': 'carência prazo',
    r'\bfranquia\b': 'franquia participação obrigatória',
}

# Nomes de produtos para priorizar busca
PRODUTOS_KEYWORDS = {
    'auto frota compacto': ['Auto Frota Compacto - Porto'],
    'auto frota tradicional': ['Auto Frota Tradicional - Porto'],
    'auto frota porto': ['Auto Frota Compacto - Porto', 'Auto Frota Tradicional - Porto'],
    'auto frota mitsui': ['Auto Frota - Mitsui Seguros'],
    'auto frota': ['Auto Frota Compacto - Porto', 'Auto Frota Tradicional - Porto', 'Auto Frota - Mitsui Seguros'],
    'auto compacto azul': ['Auto Compacto - Azul Seguros'],
    'auto compacto itau': ['Auto Compacto - Itau Seguros'],
    'auto compacto': ['Auto Compacto - Azul Seguros', 'Auto Compacto - Itau Seguros'],
    'protecao combinada': ['Auto Protecao Combinada - Porto Seguro'],
    'proteção combinada': ['Auto Protecao Combinada - Porto Seguro'],
    'seguro viagem': ['Viagem - Porto'],
    'viagem': ['Viagem - Porto', 'FAQ - Viagem', 'Duvidas - Viagem'],
    'residencial': ['Residencial Essencial - Porto', 'Residencial Facil - Porto',
                    'Residencial Habitual - Porto', 'Residencial Premium Private - Porto',
                    'Residencial Veraneio - Porto', 'Residencial Veraneio Premium Private - Porto'],
    'condominio': ['Condominio - Porto', 'Duvidas - Condominio'],
    'condôminio': ['Condominio - Porto', 'Duvidas - Condominio'],
    'empresarial': ['Empresarial - Porto', 'Duvidas - Empresarial'],
    'maquinas e equipamentos': ['Maquinas e Equipamentos - Porto', 'Duvidas - Maquinas e Equipamentos'],
    'máquinas': ['Maquinas e Equipamentos - Porto', 'Duvidas - Maquinas e Equipamentos'],
    'rc profissional': ['RC Profissional - Porto', 'Duvidas - RC Profissional'],
    'responsabilidade civil': ['RC Profissional - Porto', 'Duvidas - RC Profissional'],
    'vida individual': ['Vida Individual - Porto', 'Vida Individual Completo - Porto'],
    'vida mais mulher': ['Vida Mais Mulher - Porto'],
    'vida mais simples': ['Vida Mais Simples - Porto'],
    'vida on': ['Vida On - Porto'],
    'vida presente': ['Vida Presente - Porto'],
    'vida do seu jeito': ['Vida do Seu Jeito - Porto'],
    'acidentes pessoais': ['Acidentes Pessoais Individual - Porto', 'Acidentes Pessoais Individual Prazo Curto - Porto'],
    'apoio familiar': ['Apoio Familiar - Porto'],
    'moto azul': ['Moto - Azul Seguros'],
    'moto': ['Moto - Azul Seguros'],
    '24 horas itau': ['24 Horas - Itau'],
    '24 horas': ['24 Horas - Itau'],
    'celular': ['Duvidas - Celular'],
    'auto porto': ['Auto - Porto Seguro'],
    'auto azul': ['Auto - Azul Seguros'],
    'auto itau': ['Auto - Itau Seguros'],
    'auto mitsui': ['Auto - Mitsui Seguros'],
}


def expand_synonyms(query: str) -> str:
    """Expande sinonimos coloquiais para termos tecnicos das apolices"""
    expanded = query
    for pattern, replacement in SINONIMOS.items():
        expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
    return expanded


def normalize_text(text: str) -> str:
    """Remove acentos e normaliza para minusculas"""
    nfkd = unicodedata.normalize('NFD', text.lower())
    return ''.join(c for c in nfkd if unicodedata.category(c) != 'Mn')


def detect_priority_sources(query: str) -> List[str]:
    """Detecta se a pergunta menciona produto especifico e retorna fontes prioritarias"""
    q_lower = normalize_text(query)
    priority = []
    for keyword, sources in PRODUTOS_KEYWORDS.items():
        kw_norm = normalize_text(keyword)
        if kw_norm in q_lower:
            for s in sources:
                if s not in priority:
                    priority.append(s)
    return priority


# ---- TF-IDF simples para busca por palavras-chave ----
class TFIDFIndex:
    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self._build(chunks)

    def _tokenize(self, text: str) -> List[str]:
        text = normalize_text(text)
        return re.findall(r'\b[a-z\d]{2,}\b', text)

    def _build(self, chunks: List[Dict]):
        N = len(chunks)
        self.df: Dict[str, int] = {}
        self.tf_idf: List[Dict[str, float]] = []

        # TF por documento
        tfs = []
        for chunk in chunks:
            tokens = self._tokenize(chunk['text'])
            tf: Dict[str, float] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            # Normalizar por tamanho
            total = sum(tf.values())
            for k in tf:
                tf[k] = tf[k] / total if total > 0 else 0
            tfs.append(tf)
            for tok in tf:
                self.df[tok] = self.df.get(tok, 0) + 1

        # TF-IDF
        for tf in tfs:
            tfidf: Dict[str, float] = {}
            for tok, val in tf.items():
                idf = math.log((N + 1) / (self.df.get(tok, 0) + 1))
                tfidf[tok] = val * idf
            self.tf_idf.append(tfidf)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Retorna lista de (indice_chunk, score_tfidf)"""
        tokens = self._tokenize(query)
        if not tokens:
            return []
        scores = np.zeros(len(self.chunks))
        for tok in tokens:
            idf = math.log((len(self.chunks) + 1) / (self.df.get(tok, 0) + 1))
            for i, tfidf in enumerate(self.tf_idf):
                if tok in tfidf:
                    scores[i] += tfidf[tok] * idf

        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]


# ---- Embedding Index com Busca Hibrida ----
class EmbeddingIndex:
    def __init__(self):
        self.chunks: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None  # shape (N, 1536)
        self._tfidf: Optional[TFIDFIndex] = None

    def _get_tfidf(self) -> TFIDFIndex:
        if self._tfidf is None:
            self._tfidf = TFIDFIndex(self.chunks)
        return self._tfidf

    def build(self, documents: List[Dict]):
        self.chunks = documents
        print(f"Indice tem {len(documents)} chunks. Gerando embeddings...")
        client = get_embedding_client()
        emb_model = "openai/text-embedding-3-small" if 'openrouter' in str(client.base_url) else "text-embedding-3-small"
        all_embeddings = []
        batch_size = 50
        texts = [d['text'] for d in documents]
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"  Batch {i//batch_size + 1}/{math.ceil(len(texts)/batch_size)}...")
            resp = client.embeddings.create(model=emb_model, input=batch)
            for item in resp.data:
                all_embeddings.append(item.embedding)
        self.embeddings = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self.embeddings = self.embeddings / norms
        print(f"Embeddings gerados: shape={self.embeddings.shape}")

    def search_semantic(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Busca semantica por cosine similarity"""
        if self.embeddings is None or len(self.chunks) == 0:
            return []
        client = get_embedding_client()
        emb_model = "openai/text-embedding-3-small" if 'openrouter' in str(client.base_url) else "text-embedding-3-small"
        resp = client.embeddings.create(model=emb_model, input=[query])
        q_vec = np.array(resp.data[0].embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm
        scores = self.embeddings @ q_vec
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.chunks[i], float(scores[i])) for i in top_indices]

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """
        Busca hibrida: combina embeddings semanticos (60%) + TF-IDF palavras-chave (40%).
        Se pergunta mencionar produto especifico, prioriza chunks desse produto.
        """
        # Expandir sinonimos antes de buscar
        expanded_query = expand_synonyms(query)

        # Detectar fontes prioritarias
        priority_sources = detect_priority_sources(query)

        # 1. Busca semantica — pegar candidatos amplos
        fetch_k = top_k * 3
        sem_results = self.search_semantic(expanded_query, top_k=fetch_k)

        if not sem_results:
            return []

        # Normalizar scores semanticos para [0, 1]
        sem_scores = [s for _, s in sem_results]
        max_sem = max(sem_scores)
        min_sem = min(sem_scores)
        rng = max_sem - min_sem if max_sem > min_sem else 1.0

        # 2. Busca TF-IDF — apenas sobre os candidatos semanticos (mais rapido)
        tfidf = self._get_tfidf()
        kw_results_raw = tfidf.search(expanded_query, top_k=fetch_k)
        max_kw = max((s for _, s in kw_results_raw), default=1.0)
        kw_score_by_idx = {idx: score / max_kw for idx, score in kw_results_raw}

        # 3. Combinar: para cada resultado semantico, calcular score hibrido
        combined: List[Tuple[Dict, float]] = []
        for i, (chunk, sem_score) in enumerate(sem_results):
            norm_sem = (sem_score - min_sem) / rng
            # Encontrar indice no array original para o TF-IDF
            # Usamos o chunk_id para identificar
            chunk_id_str = chunk.get('chunk_id', '')
            kw_s = 0.0
            for idx in kw_score_by_idx:
                if 0 <= idx < len(self.chunks) and self.chunks[idx].get('chunk_id') == chunk_id_str:
                    kw_s = kw_score_by_idx[idx]
                    break
            hybrid_score = 0.60 * norm_sem + 0.40 * kw_s

            # Boost por fonte prioritaria
            if priority_sources and any(ps.lower() in chunk['source'].lower() for ps in priority_sources):
                hybrid_score *= 1.5

            combined.append((chunk, hybrid_score))

        # 4. Adicionar resultados TF-IDF que nao apareceram na busca semantica
        sem_ids = {c.get('chunk_id') for c, _ in sem_results}
        for idx, kw_norm in kw_score_by_idx.items():
            if 0 <= idx < len(self.chunks):
                c = self.chunks[idx]
                if c.get('chunk_id') not in sem_ids:
                    boost = 1.5 if priority_sources and any(ps.lower() in c['source'].lower() for ps in priority_sources) else 1.0
                    combined.append((c, 0.40 * kw_norm * boost))

        # 5. Ordenar e retornar top_k
        combined.sort(key=lambda x: x[1], reverse=True)
        return combined[:top_k]

    def save_split(self, chunks_path: str, npy_path: str):
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump({'chunks': self.chunks}, f, ensure_ascii=False)
        np.save(npy_path, self.embeddings)
        print(f"Salvo: {chunks_path} + {npy_path}")

    def load_split(self, chunks_path: str, npy_path: str):
        with open(chunks_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.chunks = data['chunks']
        self.embeddings = np.load(npy_path)
        print(f"Indice carregado: {len(self.chunks)} chunks, shape={self.embeddings.shape}")

    def load(self, path: str):
        """Carrega do formato legado JSON"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.chunks = data['chunks']
        self.embeddings = np.array(data['embeddings'], dtype=np.float32)
        print(f"Legado carregado: {len(self.chunks)} chunks")


# ---- System Prompt ----
SYSTEM_PROMPT = """Você é a Porto IA, assistente técnico oficial do Grupo Porto, especializado em responder dúvidas de corretores sobre produtos, coberturas, condições gerais, assistências, campanhas e diretrizes comerciais.

Responda sempre com base nos documentos fornecidos no contexto abaixo. Seja direto, claro e objetivo. Use linguagem profissional mas acessível. Quando a pergunta envolver comparação entre produtos, monte uma tabela comparativa. Quando a pergunta for sobre cobertura específica, cite a cláusula exata. Ao final de cada resposta, indique a fonte no formato: 'Fonte: [nome do documento] · [cláusula ou seção]'. Se a informação não estiver nos documentos disponíveis, responda exatamente: 'Não encontrei essa informação na base de conhecimento. Consulte seu gerente comercial ou acesse o portal Porto.' Nunca invente informações. Nunca responda com base em conhecimento geral — apenas no contexto fornecido.

═══════════════════════════════════════════════
REGRAS DE FORMATO — SIGA SEMPRE
═══════════════════════════════════════════════

1. COMPARAÇÕES entre 2 ou mais produtos/seguradoras → SEMPRE tabela markdown:
   | Critério | Porto | Azul | Itaú | Mitsui |
   Use ✅ para sim, ❌ para não, ⚠️ para parcial/limitado

2. COBERTURAS / LIMITES → lista com ícones e negrito para valores:
   • 🚗 Guincho: **300 km** (cláusula X)
   • 🔑 Chaveiro: **R$ 150,00** por acionamento

3. ASSISTÊNCIA 24H → tabela comparativa quando mencionar 2+ seguradoras

4. PROJETOS ESPECIAIS PORTO → mencione quando relevante:
   - 🕐 Projeto 15 Minutos: até 15 min entre 22h-5h (SP, Campinas, RJ, Salvador)
   - Se atrasar → 15% de desconto automático na renovação

5. BENEFÍCIOS PORTO BANK → quando perguntarem sobre vantagens Porto:
   - 💳 Cartão Porto Bank: até 15% OFF na renovação

6. NUNCA forneça telefones, 0800 ou endereços — podem estar desatualizados
7. Cite cláusula/seção quando disponível
8. Para sinistros: oriente App Porto ou portoseguro.com.br/atendimento/sinistros

═══════════════════════════════════════════════
PADRÃO DE QUALIDADE
═══════════════════════════════════════════════
✦ Pergunta simples → resposta direta em 2-4 linhas + 1 destaque em negrito
✦ Pergunta sobre 1 produto → bullets com ícones + valores em negrito
✦ Pergunta comparativa → tabela obrigatória + resumo de 2 linhas
✦ Pergunta sobre assistência 24h → tabela comparativa SEMPRE"""


# ---- RAG Query ----
def query_rag(question: str, index: EmbeddingIndex, conversation_history: List[Dict] = None) -> str:
    """Busca hibrida + resposta via LLM"""
    # Determinar se pergunta e complexa (mais contexto necessario)
    is_complex = any(w in question.lower() for w in [
        'compar', 'diferença', 'versus', ' vs ', 'todos', 'qual o melhor',
        'assistência', 'assistencia', 'sinistro', 'cobertura', 'tabela'
    ])
    top_k = 7 if is_complex else 5

    results = index.search(question, top_k=top_k)

    context_parts = []
    for chunk, score in results:
        if score > 0.1:
            context_parts.append(f"[{chunk['source']}]\n{chunk['text']}")

    if not context_parts and results:
        context_parts = [f"[{c['source']}]\n{c['text']}" for c, _ in results[:3]]

    context = "\n\n---\n\n".join(context_parts) if context_parts else "Nenhum contexto relevante encontrado."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if conversation_history:
        messages.extend(conversation_history[-6:])

    user_message = f"""Contexto dos documentos:

{context}

---

Pergunta do corretor: {question}"""

    messages.append({"role": "user", "content": user_message})

    client = get_llm_client()
    response = client.chat.completions.create(
        model="anthropic/claude-haiku-4-5",
        max_tokens=1500,
        temperature=0.2,
        messages=messages
    )

    return response.choices[0].message.content


def query_rag_stream(question: str, index: EmbeddingIndex, conversation_history: List[Dict] = None):
    """Busca hibrida + resposta em streaming (generator de chunks de texto)"""
    is_complex = any(w in question.lower() for w in [
        'compar', 'diferença', 'versus', ' vs ', 'todos', 'qual o melhor',
        'assistência', 'assistencia', 'sinistro', 'cobertura', 'tabela'
    ])
    top_k = 7 if is_complex else 5

    results = index.search(question, top_k=top_k)

    context_parts = []
    sources_used = []
    for chunk, score in results:
        if score > 0.1:
            context_parts.append(f"[{chunk['source']}]\n{chunk['text']}")
            if chunk['source'] not in sources_used:
                sources_used.append(chunk['source'])

    if not context_parts and results:
        context_parts = [f"[{c['source']}]\n{c['text']}" for c, _ in results[:3]]
        sources_used = list(set(c['source'] for c, _ in results[:3]))

    context = "\n\n---\n\n".join(context_parts) if context_parts else "Nenhum contexto relevante encontrado."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if conversation_history:
        messages.extend(conversation_history[-6:])

    user_message = f"""Contexto dos documentos:

{context}

---

Pergunta do corretor: {question}"""

    messages.append({"role": "user", "content": user_message})

    client = get_llm_client()
    stream = client.chat.completions.create(
        model="anthropic/claude-haiku-4-5",
        max_tokens=1500,
        temperature=0.2,
        messages=messages,
        stream=True
    )

    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            yield delta.content, sources_used


# ---- Singleton Index ----
_index_instance = None


def get_index() -> EmbeddingIndex:
    global _index_instance
    if _index_instance is None:
        idx = EmbeddingIndex()
        if CHUNKS_PATH.exists() and EMBEDDINGS_NPY.exists():
            idx.load_split(str(CHUNKS_PATH), str(EMBEDDINGS_NPY))
        elif EMBEDDINGS_PATH.exists():
            print("Usando formato legado embeddings.json...")
            idx.load(str(EMBEDDINGS_PATH))
        else:
            print("Nenhum indice encontrado.")
        _index_instance = idx
    return _index_instance
