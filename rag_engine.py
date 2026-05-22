"""
FOCUS AI - RAG Engine v5.0
Busca hibrida: embeddings semanticos (text-embedding-3-small) + TF-IDF por palavras-chave
Formato: chunks.json + embeddings.npy
Le PDFs recursivamente de docs/<seguradora>/<arquivo>.pdf
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
    key = os.environ.get("OPENROUTER_API_KEY", "")
    if key:
        return key
    key = os.environ.get("OPENAI_API_KEY", "")
    if key:
        return key
    env_paths = ["/opt/data/.env", str(Path.home() / ".env")]
    for env_path in env_paths:
        try:
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("OPENROUTER_API_KEY="):
                        val = line.split("=", 1)[1].strip()
                        val = ''.join(c for c in val if c.isprintable()).strip()
                        if val and val != "***":
                            return val
        except Exception:
            pass
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


# ---- Extracao de texto de PDF ----
def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extrai texto de um PDF usando pypdf"""
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        pages = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if text.strip():
                pages.append(text)
        return "\n\n".join(pages)
    except Exception as e:
        print(f"  AVISO: nao foi possivel extrair texto de {pdf_path.name}: {e}")
        return ""


# ---- Carregamento de documentos ----
def load_documents() -> List[Dict]:
    """
    Carrega todos os documentos da pasta docs/.
    Suporta PDFs em subpastas (docs/<seguradora>/<arquivo>.pdf)
    e .txt direto em docs/.
    Retorna lista de {'source': str, 'text': str}
    """
    documents = []

    if not DOCS_PATH.exists():
        print(f"Pasta docs/ nao encontrada em {DOCS_PATH}")
        return documents

    # 1. PDFs em subpastas (estrutura nova)
    for subfolder in sorted(DOCS_PATH.iterdir()):
        if subfolder.is_dir():
            seguradora = subfolder.name
            for pdf_file in sorted(subfolder.glob("*.pdf")):
                print(f"  Lendo PDF: {seguradora}/{pdf_file.name}")
                text = extract_text_from_pdf(pdf_file)
                if text.strip():
                    # Nome da fonte: "NomePDF - Seguradora" sem extensao
                    stem = pdf_file.stem  # ex: "Auto - Porto"
                    source = stem
                    documents.append({"source": source, "text": text, "seguradora": seguradora})
                else:
                    print(f"    AVISO: PDF sem texto extraivel — {pdf_file.name}")

    # 2. .txt direto em docs/ (legado — fallback)
    for txt_file in sorted(DOCS_PATH.glob("*.txt")):
        print(f"  Lendo TXT: {txt_file.name}")
        try:
            text = txt_file.read_text(encoding='utf-8', errors='replace')
            if text.strip():
                source = txt_file.stem
                documents.append({"source": source, "text": text, "seguradora": "legado"})
        except Exception as e:
            print(f"    AVISO: erro ao ler {txt_file.name}: {e}")

    print(f"\nTotal de documentos carregados: {len(documents)}")
    return documents


# ---- Chunking ----
def chunk_document(doc: Dict, chunk_size: int = 800, overlap: int = 150) -> List[Dict]:
    """Divide um documento em chunks com overlap"""
    text = doc['text']
    source = doc['source']
    seguradora = doc.get('seguradora', '')

    # Limpar texto
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' {3,}', ' ', text)
    text = text.strip()

    if not text:
        return []

    # Dividir por paragrafos primeiro
    paragraphs = re.split(r'\n\n+', text)
    chunks = []
    current = ""
    chunk_idx = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        # Se o paragrafo sozinho ja e maior que chunk_size, divide por sentencas
        if len(para) > chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sent in sentences:
                if len(current) + len(sent) > chunk_size and current:
                    chunks.append({
                        "chunk_id": f"{source}_{chunk_idx}",
                        "source": source,
                        "seguradora": seguradora,
                        "text": current.strip()
                    })
                    # Overlap: pegar ultimas palavras do chunk anterior
                    words = current.split()
                    current = " ".join(words[-30:]) + " " + sent
                    chunk_idx += 1
                else:
                    current += " " + sent if current else sent
        else:
            if len(current) + len(para) > chunk_size and current:
                chunks.append({
                    "chunk_id": f"{source}_{chunk_idx}",
                    "source": source,
                    "seguradora": seguradora,
                    "text": current.strip()
                })
                words = current.split()
                current = " ".join(words[-30:]) + "\n\n" + para
                chunk_idx += 1
            else:
                current += "\n\n" + para if current else para

    # Ultimo chunk
    if current.strip():
        chunks.append({
            "chunk_id": f"{source}_{chunk_idx}",
            "source": source,
            "seguradora": seguradora,
            "text": current.strip()
        })

    return chunks


def build_index() -> 'EmbeddingIndex':
    """Carrega documentos, chunka e gera embeddings. Salva chunks.json + embeddings.npy"""
    print("Carregando documentos...")
    docs = load_documents()

    if not docs:
        print("ERRO: nenhum documento encontrado!")
        return EmbeddingIndex()

    print("\nChunkando documentos...")
    all_chunks = []
    for doc in docs:
        chunks = chunk_document(doc)
        all_chunks.extend(chunks)
        print(f"  {doc['source']}: {len(chunks)} chunks")

    print(f"\nTotal de chunks: {len(all_chunks)}")

    idx = EmbeddingIndex()
    idx.build(all_chunks)
    idx.save_split(str(CHUNKS_PATH), str(EMBEDDINGS_NPY))
    return idx


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


def expand_synonyms(query: str) -> str:
    expanded = query
    for pattern, replacement in SINONIMOS.items():
        expanded = re.sub(pattern, replacement, expanded, flags=re.IGNORECASE)
    return expanded


def normalize_text(text: str) -> str:
    nfkd = unicodedata.normalize('NFD', text.lower())
    return ''.join(c for c in nfkd if unicodedata.category(c) != 'Mn')


def is_index_chunk(text: str) -> bool:
    dot_lines = sum(1 for line in text.splitlines() if line.strip().endswith('...') or
                    ('...' in line and line.strip()[-1].isdigit()))
    ellipsis_count = text.count('........')
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    index_pattern_lines = sum(
        1 for l in lines
        if (l.endswith(')') is False) and
           (l[-1:].isdigit() and ('...' in l or '  ' in l))
    )
    return dot_lines >= 3 or ellipsis_count >= 4 or index_pattern_lines >= 4


def detect_priority_sources(query: str) -> List[str]:
    """Detecta seguradora especifica mencionada na pergunta"""
    q_lower = normalize_text(query)
    priority = []

    # Mapa de alias -> nome de fonte (stem do PDF)
    aliases = {
        'porto': ['Auto - Porto', 'Auto Proteção Combinada - Porto', 'Auto Frota Compacto - Porto',
                  'Auto Frota Tradicional - Porto'],
        'grupo porto': ['Auto - Porto', 'Auto Proteção Combinada - Porto'],
        'allianz': ['Auto - Allianz'],
        'hdi': ['Auto Basico - HDI', 'Auto Fit - HDI', 'Auto Perfil - HDI'],
        'yelum': ['Auto Consciente - Yelum', 'Auto Exclusivo - Yelum', 'Auto Perfil - Yelum'],
        'aliro': ['Auto - Aliro'],
        'azul': ['Auto - Azul', 'Auto Compacto - Azul', 'Moto - Azul'],
        'itau': ['Auto - Itau', 'Auto Compacto - Itau', '24 Horas - Itau'],
        'mitsui': ['Auto - Mitsui', 'Auto Frota - Mitsui'],
        'mapfre': ['Auto - Mapfre'],
        'tokio': ['Auto - Tokio', 'Auto Frota - Tokio'],
        'tokio marine': ['Auto - Tokio', 'Auto Frota - Tokio'],
        'zurich': ['Auto - Zurich'],
        'bradesco': ['Auto - Bradesco'],
        'sura': ['Auto - Sura'],
        'suhai': ['Auto - Suhai'],
        'sancor': ['Auto - Sancor'],
        'banestes': ['Auto - Banestes'],
        'bvix': ['Auto - Bvix'],
        'ezze': ['Auto e Frota - Ezze'],
        'axa': ['Auto Frota - Axa'],
        'bp seguros': ['Auto Anual - BP', 'Auto Mensal - BP'],
        'pier': ['Auto - Pier'],
        'darwin': ['Auto Anual - Darwin', 'Auto Mensal - Darwin'],
        'cardif': ['Auto - Cardif'],
        'aruana': ['RCF - Aruana'],
        'too': ['Auto - Too'],
        'usebens': ['Auto - Usebens'],
        'gente': ['Auto - Gente'],
        'suica': ['Auto - Suiça'],
    }

    for alias, sources in aliases.items():
        if normalize_text(alias) in q_lower:
            for s in sources:
                if s not in priority:
                    priority.append(s)

    return priority


def detect_category(query: str) -> Optional[str]:
    q_lower = normalize_text(query)
    keywords = {
        'auto': ['seguro auto', 'seguro de auto', 'seguro carro', 'seguro veiculo',
                 'cobertura auto', 'apolice auto', 'guincho', 'carro reserva',
                 'perda total', 'colisao', 'roubo carro', 'furto carro',
                 'franquia auto', 'condutor', 'veiculo', 'automovel'],
        'residencial': ['seguro residencial', 'seguro casa', 'seguro imovel',
                        'residencial', 'duo lar'],
        'vida': ['seguro vida', 'vida individual', 'acidentes pessoais', 'morte', 'invalidez'],
        'viagem': ['seguro viagem', 'cancelamento voo', 'bagagem', 'assistencia viagem'],
        'empresarial': ['seguro empresarial', 'lucros cessantes'],
        'rc': ['rc profissional', 'responsabilidade civil'],
    }
    for cat, kws in keywords.items():
        for kw in kws:
            if normalize_text(kw) in q_lower:
                return cat
    return None


# ---- TF-IDF simples ----
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
        tfs = []
        for chunk in chunks:
            tokens = self._tokenize(chunk['text'])
            tf: Dict[str, float] = {}
            for tok in tokens:
                tf[tok] = tf.get(tok, 0) + 1
            total = sum(tf.values())
            for k in tf:
                tf[k] = tf[k] / total if total > 0 else 0
            tfs.append(tf)
            for tok in tf:
                self.df[tok] = self.df.get(tok, 0) + 1
        for tf in tfs:
            tfidf: Dict[str, float] = {}
            for tok, val in tf.items():
                idf = math.log((N + 1) / (self.df.get(tok, 0) + 1))
                tfidf[tok] = val * idf
            self.tf_idf.append(tfidf)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
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
        self.embeddings: Optional[np.ndarray] = None
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

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        expanded_query = expand_synonyms(query)
        priority_sources = detect_priority_sources(query)
        category = detect_category(query)

        # Pool de chunks
        if priority_sources:
            pool_chunks = [c for c in self.chunks if any(
                ps.lower() in c['source'].lower() for ps in priority_sources
            )]
            pool_indices = [i for i, c in enumerate(self.chunks) if any(
                ps.lower() in c['source'].lower() for ps in priority_sources
            )]
            if not pool_chunks:
                pool_chunks = self.chunks
                pool_indices = list(range(len(self.chunks)))
        elif category:
            pool_chunks = [c for c in self.chunks if normalize_text(category) in normalize_text(c.get('source', '') + ' ' + c.get('seguradora', ''))]
            pool_indices = [i for i, c in enumerate(self.chunks) if normalize_text(category) in normalize_text(c.get('source', '') + ' ' + c.get('seguradora', ''))]
            if not pool_chunks:
                pool_chunks = self.chunks
                pool_indices = list(range(len(self.chunks)))
        else:
            pool_chunks = self.chunks
            pool_indices = list(range(len(self.chunks)))

        if self.embeddings is None or len(pool_chunks) == 0:
            return []

        fetch_k = top_k * 3
        client = get_embedding_client()
        emb_model = "openai/text-embedding-3-small" if 'openrouter' in str(client.base_url) else "text-embedding-3-small"
        resp = client.embeddings.create(model=emb_model, input=[expanded_query])
        q_vec = np.array(resp.data[0].embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        pool_embs = self.embeddings[pool_indices]
        scores = pool_embs @ q_vec
        top_local = np.argsort(scores)[::-1][:fetch_k]
        sem_results = [(pool_chunks[i], float(scores[i])) for i in top_local]

        if not sem_results:
            return []

        sem_scores = [s for _, s in sem_results]
        max_sem = max(sem_scores)
        min_sem = min(sem_scores)
        rng = max_sem - min_sem if max_sem > min_sem else 1.0

        pool_tfidf = TFIDFIndex(pool_chunks)
        kw_results_raw = pool_tfidf.search(expanded_query, top_k=fetch_k)
        max_kw = max((s for _, s in kw_results_raw), default=1.0)
        kw_score_by_idx = {idx: score / max_kw for idx, score in kw_results_raw}

        combined: List[Tuple[Dict, float]] = []
        for chunk, sem_score in sem_results:
            norm_sem = (sem_score - min_sem) / rng
            chunk_id_str = chunk.get('chunk_id', '')
            kw_s = 0.0
            for idx in kw_score_by_idx:
                if 0 <= idx < len(pool_chunks) and pool_chunks[idx].get('chunk_id') == chunk_id_str:
                    kw_s = kw_score_by_idx[idx]
                    break
            hybrid_score = 0.60 * norm_sem + 0.40 * kw_s
            if priority_sources and any(ps.lower() in chunk['source'].lower() for ps in priority_sources):
                hybrid_score *= 1.5
            if is_index_chunk(chunk.get('text', '')):
                hybrid_score *= 0.4
            combined.append((chunk, hybrid_score))

        sem_ids = {c.get('chunk_id') for c, _ in sem_results}
        for idx, kw_norm in kw_score_by_idx.items():
            if 0 <= idx < len(pool_chunks):
                c = pool_chunks[idx]
                if c.get('chunk_id') not in sem_ids:
                    boost = 1.5 if priority_sources and any(ps.lower() in c['source'].lower() for ps in priority_sources) else 1.0
                    score_tfidf = 0.40 * kw_norm * boost
                    if is_index_chunk(c.get('text', '')):
                        score_tfidf *= 0.4
                    combined.append((c, score_tfidf))

        combined.sort(key=lambda x: x[1], reverse=True)
        seen_ids = set()
        deduped = []
        for chunk, score in combined:
            cid = chunk.get('chunk_id', '')
            if cid not in seen_ids:
                seen_ids.add(cid)
                deduped.append((chunk, score))
        return deduped[:top_k]

    def search_semantic(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
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
SYSTEM_PROMPT = """Você é o FOCUS AI, assistente técnico especializado em responder dúvidas de corretores de seguros sobre produtos, coberturas, condições gerais, assistências e diretrizes comerciais de múltiplas seguradoras.

Responda sempre com base nos documentos fornecidos no contexto abaixo. Seja direto, claro e objetivo. Use linguagem profissional mas acessível.

REGRAS DE RESPOSTA:
- Se o contexto contiver a informação pedida → responda com base nele, citando a fonte
- Se o contexto contiver documentos dos produtos perguntados mas não detalhar a diferença específica → sintetize o que está disponível nos docs e oriente: 'Para detalhes comerciais completos (preços, franquias, público-alvo), consulte seu gerente comercial ou o portal da seguradora.'
- Se o contexto NÃO contiver nenhum documento relevante → responda: 'Não encontrei essa informação na base de conhecimento. Consulte seu gerente comercial ou acesse o portal da seguradora.'
- NUNCA confirme nem negue cobertura de forma definitiva — sempre cite a cláusula do documento e oriente o corretor a verificar as condições gerais completas.

Quando a pergunta envolver comparação entre produtos, monte uma tabela comparativa com o que estiver disponível nos docs. Quando a pergunta for sobre cobertura específica, cite a cláusula exata. Ao final de cada resposta, indique a fonte no formato: 'Fonte: [nome do documento] · [cláusula ou seção]'. Nunca invente valores ou coberturas que não estejam explicitamente nos documentos.

═══════════════════════════════════════════════
REGRAS DE FORMATO — SIGA SEMPRE
═══════════════════════════════════════════════

1. COMPARAÇÕES entre 2 ou mais produtos/seguradoras → SEMPRE tabela markdown:
   | Critério | Seguradora A | Seguradora B |
   Use ✅ para sim, ❌ para não, ⚠️ para parcial/limitado

2. COBERTURAS / LIMITES → lista com ícones e negrito para valores:
   • 🚗 Guincho: **300 km** (cláusula X)
   • 🔑 Chaveiro: **R$ 150,00** por acionamento

3. ASSISTÊNCIA 24H → tabela comparativa quando mencionar 2+ seguradoras

4. NUNCA forneça telefones, 0800 ou endereços — podem estar desatualizados
5. Cite cláusula/seção quando disponível
6. Para sinistros: oriente o corretor a consultar o canal de atendimento da seguradora

═══════════════════════════════════════════════
PADRÃO DE QUALIDADE
═══════════════════════════════════════════════
✦ Pergunta simples → resposta direta em 2-4 linhas + 1 destaque em negrito
✦ Pergunta sobre 1 produto → bullets com ícones + valores em negrito
✦ Pergunta comparativa → tabela obrigatória + resumo de 2 linhas
✦ Pergunta sobre assistência 24h → tabela comparativa SEMPRE"""


# ---- RAG Query ----
def query_rag(question: str, index: EmbeddingIndex, conversation_history: List[Dict] = None) -> str:
    is_complex = any(w in question.lower() for w in [
        'compar', 'diferença', 'versus', ' vs ', 'todos', 'qual o melhor',
        'assistência', 'assistencia', 'sinistro', 'cobertura', 'tabela'
    ])
    top_k = 8 if is_complex else 5

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
    is_complex = any(w in question.lower() for w in [
        'compar', 'diferença', 'versus', ' vs ', 'todos', 'qual o melhor',
        'assistência', 'assistencia', 'sinistro', 'cobertura', 'tabela'
    ])
    top_k = 8 if is_complex else 5

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
