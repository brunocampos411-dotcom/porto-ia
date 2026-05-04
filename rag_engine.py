"""
Porto IA - RAG Engine v3.1
Busca semantica por embeddings (OpenAI text-embedding-3-small)
Formato de armazenamento: chunks.json + embeddings.npy (76% menor que JSON unico)
"""
import os
import re
import json
import math
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import openai

# ---- Configuracao ----
BASE_DIR = Path(__file__).parent
DOCS_PATH = BASE_DIR / "docs"
EMBEDDINGS_PATH = BASE_DIR / "embeddings.json"   # legado (fallback)
CHUNKS_PATH      = BASE_DIR / "chunks.json"       # novo formato
EMBEDDINGS_NPY   = BASE_DIR / "embeddings.npy"    # novo formato binario


def get_llm_client():
    """Cliente para chat/LLM via OpenRouter"""
    key = os.environ.get("OPENROUTER_API_KEY", "")
    return openai.OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1"
    )


def get_embedding_client():
    """Cliente para embeddings via OpenRouter (openai/text-embedding-3-small)"""
    key = os.environ.get("OPENROUTER_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    base_url = "https://openrouter.ai/api/v1" if key.startswith("sk-or-") else "https://api.openai.com/v1"
    return openai.OpenAI(api_key=key, base_url=base_url)


# ---- Chunking ----
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 80) -> List[str]:
    """Divide texto em chunks de 300-500 palavras com overlap"""
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Tentar dividir por clausulas primeiro
    clausula_pattern = re.compile(
        r'(?=►?CL[AÁ]USULA\s+\d+|►?\d+\.\s+[A-Z]{4,})',
        re.MULTILINE
    )
    sections = clausula_pattern.split(text)

    chunks = []
    for section in sections:
        section = section.strip()
        if not section or len(section) < 50:
            continue

        words = section.split()
        if len(words) <= chunk_size:
            chunks.append(section)
        else:
            paragraphs = section.split('\n\n')
            current = ""
            current_words = 0
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                para_words = len(para.split())
                if current_words + para_words <= chunk_size:
                    current += "\n\n" + para if current else para
                    current_words += para_words
                else:
                    if current:
                        chunks.append(current.strip())
                        w = current.split()
                        overlap_text = " ".join(w[-30:]) if len(w) > 30 else current
                        current = overlap_text + "\n\n" + para
                        current_words = len(current.split())
                    else:
                        chunks.append(para[:chunk_size * 6])
                        current = ""
                        current_words = 0
            if current:
                chunks.append(current.strip())

    chunks = [c for c in chunks if len(c.split()) > 30]
    return chunks


# ---- Embedding Index ----
class EmbeddingIndex:
    def __init__(self):
        self.chunks: List[Dict] = []
        self.embeddings: Optional[np.ndarray] = None  # shape (N, 1536)

    def build(self, documents: List[Dict]):
        """documents: lista de {text, source, chunk_id}"""
        self.chunks = documents
        print(f"Indice tem {len(documents)} chunks. Gerando embeddings...")

        client = get_embedding_client()
        all_embeddings = []
        batch_size = 50
        emb_model = "openai/text-embedding-3-small" if 'openrouter' in str(client.base_url) else "text-embedding-3-small"

        texts = [d['text'] for d in documents]
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            print(f"  Batch {i//batch_size + 1}/{math.ceil(len(texts)/batch_size)}...")
            resp = client.embeddings.create(
                model=emb_model,
                input=batch
            )
            for item in resp.data:
                all_embeddings.append(item.embedding)

        self.embeddings = np.array(all_embeddings, dtype=np.float32)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self.embeddings = self.embeddings / norms
        print(f"Embeddings gerados: shape={self.embeddings.shape}")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Dict, float]]:
        """Busca semantica por cosine similarity"""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        client = get_embedding_client()
        emb_model = "openai/text-embedding-3-small" if 'openrouter' in str(client.base_url) else "text-embedding-3-small"
        resp = client.embeddings.create(
            model=emb_model,
            input=[query]
        )
        q_vec = np.array(resp.data[0].embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q_vec)
        if q_norm > 0:
            q_vec = q_vec / q_norm

        scores = self.embeddings @ q_vec  # shape (N,)
        top_indices = np.argsort(scores)[::-1][:top_k]

        return [(self.chunks[i], float(scores[i])) for i in top_indices]

    def save(self, path: str):
        """Salva no formato legado JSON (para compatibilidade)"""
        data = {
            'chunks': self.chunks,
            'embeddings': self.embeddings.tolist() if self.embeddings is not None else []
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        size_mb = Path(path).stat().st_size / 1024 / 1024
        print(f"Indice salvo: {path} ({size_mb:.1f} MB)")

    def save_split(self, chunks_path: str, npy_path: str):
        """Salva no formato otimizado: chunks.json + embeddings.npy (76% menor)"""
        # Salvar chunks
        with open(chunks_path, 'w', encoding='utf-8') as f:
            json.dump({'chunks': self.chunks}, f, ensure_ascii=False)
        chunks_mb = Path(chunks_path).stat().st_size / 1024 / 1024
        print(f"chunks.json salvo: {chunks_mb:.2f} MB")

        # Salvar embeddings como numpy binario
        np.save(npy_path, self.embeddings)
        npy_mb = Path(npy_path).stat().st_size / 1024 / 1024
        print(f"embeddings.npy salvo: {npy_mb:.1f} MB")
        print(f"Total formato split: {chunks_mb + npy_mb:.1f} MB")

    def load(self, path: str):
        """Carrega do formato legado JSON"""
        print(f"Carregando embeddings de {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.chunks = data['chunks']
        self.embeddings = np.array(data['embeddings'], dtype=np.float32)
        print(f"Embeddings carregados: {len(self.chunks)} chunks, shape={self.embeddings.shape}")

    def load_split(self, chunks_path: str, npy_path: str):
        """Carrega do formato otimizado chunks.json + embeddings.npy"""
        print(f"Carregando chunks de {chunks_path}...")
        with open(chunks_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.chunks = data['chunks']
        print(f"Carregando embeddings de {npy_path}...")
        self.embeddings = np.load(npy_path)
        print(f"Embeddings carregados: {len(self.chunks)} chunks, shape={self.embeddings.shape}")


# ---- Build Index ----
def build_index():
    all_docs = []

    doc_files = {
        '24_Horas_-_Itau.txt': '24 Horas - Itau',
        'Auto_-_Azul.txt': 'Auto - Azul Seguros',
        'Auto_-_Itau.txt': 'Auto - Itau Seguros',
        'Auto_-_Mitsui.txt': 'Auto - Mitsui Seguros',
        'Auto_-_Porto.txt': 'Auto - Porto Seguro',
        'Auto_Compacto_-_Azul.txt': 'Auto Compacto - Azul Seguros',
        'Auto_Compacto_-_Itau.txt': 'Auto Compacto - Itau Seguros',
        'Auto_Frota_-_Mitsui.txt': 'Auto Frota - Mitsui Seguros',
        'Auto_Protecao_Combinada_-_Porto.txt': 'Auto Protecao Combinada - Porto Seguro',
        'Moto_-_Azul.txt': 'Moto - Azul Seguros',
        'Site_Porto_Seguro_Auto.txt': 'Site Oficial - Porto Seguro Auto',
    }

    for filename, source in doc_files.items():
        filepath = DOCS_PATH / filename
        if not filepath.exists():
            print(f"AVISO: {filepath} nao encontrado")
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = chunk_text(text, chunk_size=500, overlap=80)
        print(f"{source}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_docs.append({
                'text': chunk,
                'source': source,
                'chunk_id': f"{source}_{i}"
            })

    index = EmbeddingIndex()
    index.build(all_docs)
    # Salvar nos dois formatos
    index.save(str(EMBEDDINGS_PATH))        # legado
    index.save_split(str(CHUNKS_PATH), str(EMBEDDINGS_NPY))  # otimizado
    return index


# ---- RAG Query ----
SYSTEM_PROMPT = """Você é a IA oficial do Grupo Porto, especializada em responder dúvidas de corretores sobre produtos, coberturas e condições gerais.

Responda sempre com base nos documentos fornecidos abaixo. Ao final de cada resposta, cite o documento e a cláusula de origem no formato:
'Fonte: [nome do documento] · [cláusula ou seção]'

Se a informação não estiver nos documentos, diga claramente: 'Não encontrei essa informação nos documentos disponíveis. Consulte seu gerente comercial.'
Nunca invente informações. Seja direto, claro e profissional.

═══════════════════════════════════════════════
REGRAS DE FORMATO — SIGA SEMPRE SEM EXCEÇÃO
═══════════════════════════════════════════════

1. COMPARAÇÕES entre 2 ou mais seguradoras → SEMPRE use tabela markdown:
   | Critério | Porto | Azul | Itaú | Mitsui |
   Use ✅ para sim, ❌ para não, ⚠️ para parcial/limitado

2. COBERTURAS / LIMITES → use lista com ícones e negrito para valores:
   • 🚗 Guincho: **300 km** (cláusula X)
   • 🔑 Chaveiro: **R$ 150,00** por acionamento

3. ASSISTÊNCIA 24H → SEMPRE tabela comparativa quando mencionar 2+ seguradoras:
   | Serviço | Porto | Azul | Itaú | Mitsui |
   Inclua: guincho, pane, chaveiro, carro reserva, assistência residencial

4. PROJETOS ESPECIAIS PORTO → mencione proativamente quando relevante:
   - 🕐 Projeto 15 Minutos: atendimento em até 15 min entre 22h-5h (SP, Campinas, RJ, Salvador)
   - Se atrasar → 15% de desconto automático na renovação
   - Válido apenas para Seguro Auto Porto Individual

5. BENEFÍCIOS PORTO BANK → mencione quando perguntarem sobre vantagens da Porto:
   - 💳 Cartão Porto Bank: até 15% OFF na renovação do Seguro Auto Porto
   - IOF Zero em compras internacionais via App Porto
   - Não acumulativo, exclusivo para produto Porto (não vale para Azul/Itaú)

6. DIFERENCIAIS EXCLUSIVOS DO SEGURO AUTO PORTO (vs Azul e Itaú):
   - 🚘 Motorista da vez (profissional leva você e o carro)
   - 🅿️ Desconto em estacionamentos (até 30%)
   - 🏠 Assistência residencial COMPLETA (encanador, eletricista, chaveiro, eletrodoméstico)
   - 📱 Crédito em app de transporte (Uber/99) em caso de imprevisto
   - ⏱️ Projeto 15 Minutos

7. DIFERENCIAIS DE NÍVEL:
   - Azul: essencial (veículo apenas)
   - Itaú: intermediário (veículo + assistência casa básica)
   - Porto: completo (veículo + casa completa + benefícios exclusivos)

8. NUNCA forneça telefones, 0800 ou endereços — podem estar desatualizados
9. Cite cláusula/seção quando disponível (ex: "Cláusula 5ª", "Seção III")
10. Para sinistros: oriente a acessar o App Porto ou portoseguro.com.br/atendimento/sinistros

═══════════════════════════════════════════════
PADRÃO DE QUALIDADE DAS RESPOSTAS
═══════════════════════════════════════════════

✦ Pergunta simples → resposta direta em 2-4 linhas + 1 destaque em negrito
✦ Pergunta sobre 1 seguradora → bullets com ícones + valores em negrito
✦ Pergunta comparativa → tabela obrigatória + resumo de 2 linhas após a tabela
✦ Pergunta sobre assistência 24h → tabela comparativa SEMPRE
✦ Perguntas sobre Porto → mencione Projeto 15min e Porto Bank quando pertinente

Documentos disponíveis:
- Auto Porto Seguro (Condições Gerais + Site Oficial portoseguro.com.br/seguro-auto)
- Auto Proteção Combinada Porto Seguro
- Auto Azul Seguros + Auto Compacto Azul
- Auto Itaú Seguros + Auto Compacto Itaú + 24 Horas Itaú
- Auto Mitsui Seguros + Auto Frota Mitsui
- Moto Azul Seguros"""


def query_rag(question: str, index: EmbeddingIndex, conversation_history: List[Dict] = None) -> str:
    """Busca semantica + resposta via LLM"""

    results = index.search(question, top_k=5)

    context_parts = []
    for chunk, score in results:
        if score > 0.2:
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
        max_tokens=1000,
        temperature=0.2,
        messages=messages
    )

    return response.choices[0].message.content


# ---- Singleton Index ----
_index_instance = None


def get_index() -> EmbeddingIndex:
    global _index_instance
    if _index_instance is None:
        idx = EmbeddingIndex()
        # Preferir formato otimizado (chunks.json + embeddings.npy)
        if CHUNKS_PATH.exists() and EMBEDDINGS_NPY.exists():
            idx.load_split(str(CHUNKS_PATH), str(EMBEDDINGS_NPY))
        elif EMBEDDINGS_PATH.exists():
            print("Usando formato legado embeddings.json...")
            idx.load(str(EMBEDDINGS_PATH))
        else:
            print("Nenhum indice encontrado. Execute indexer.py primeiro.")
            idx = build_index()
        _index_instance = idx
    return _index_instance
