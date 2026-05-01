"""
Porto IA - RAG Engine
Sistema de busca semantica nos documentos da Porto Seguro
usando TF-IDF simples sem dependencias externas
"""
import os
import re
import math
import json
from pathlib import Path
from typing import List, Dict, Tuple
import openai

# ---- Configuracao ----
BASE_DIR = Path(__file__).parent
DOCS_PATH = BASE_DIR / "docs"
INDEX_PATH = BASE_DIR / "index.json"


def get_llm_client():
    key = os.environ.get("OPENROUTER_API_KEY", "")
    return openai.OpenAI(
        api_key=key,
        base_url="https://openrouter.ai/api/v1"
    )


# ---- Chunking por clausula ----
def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> List[str]:
    """Divide o texto em chunks, respeitando clausulas quando possivel"""
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

        if len(section) <= chunk_size:
            chunks.append(section)
        else:
            # Dividir secao grande em sub-chunks com overlap
            paragraphs = section.split('\n\n')
            current = ""
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                if len(current) + len(para) <= chunk_size:
                    current += "\n\n" + para if current else para
                else:
                    if current:
                        chunks.append(current.strip())
                        # overlap
                        words = current.split()
                        overlap_text = " ".join(words[-40:]) if len(words) > 40 else current
                        current = overlap_text + "\n\n" + para
                    else:
                        chunks.append(para[:chunk_size])
                        current = para[chunk_size - overlap:]
            if current:
                chunks.append(current.strip())

    chunks = [c for c in chunks if len(c) > 80]
    return chunks


# ---- TF-IDF Index ----
class TFIDFIndex:
    def __init__(self):
        self.chunks: List[Dict] = []
        self.vocab: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.tfidf_matrix: List[Dict[str, float]] = []

    def tokenize(self, text: str) -> List[str]:
        text = text.lower()
        tokens = re.findall(r'\b[a-záéíóúâêîôûãõçàüñ\w]{2,}\b', text)
        stopwords = {
            'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para',
            'com', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as',
            'dos', 'como', 'mas', 'ao', 'ele', 'das', 'seu', 'sua',
            'ou', 'quando', 'muito', 'nos', 'ja', 'eu', 'também',
            'pelo', 'pela', 'ate', 'isso', 'ela', 'entre', 'depois',
            'sem', 'mesmo', 'aos', 'seus', 'quem', 'nas', 'me', 'esse',
            'eles', 'essa', 'num', 'nem', 'suas', 'meu', 'minha', 'te',
            'essa', 'nao', 'nesta', 'deste', 'estava', 'este', 'havia',
            'ser', 'ter', 'pode', 'foi', 'sao', 'esta', 'tambem',
        }
        return [t for t in tokens if t not in stopwords]

    def build(self, documents: List[Dict]):
        print(f"Construindo indice com {len(documents)} chunks...")
        self.chunks = documents
        N = len(documents)

        tokenized = [self.tokenize(doc['text']) for doc in documents]

        all_tokens = set()
        for tokens in tokenized:
            all_tokens.update(tokens)
        self.vocab = {t: i for i, t in enumerate(sorted(all_tokens))}

        doc_freq = {}
        for tokens in tokenized:
            for t in set(tokens):
                doc_freq[t] = doc_freq.get(t, 0) + 1

        self.idf = {}
        for term, df in doc_freq.items():
            self.idf[term] = math.log((N + 1) / (df + 1)) + 1

        self.tfidf_matrix = []
        for tokens in tokenized:
            tf = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1
            tfidf = {}
            for t, count in tf.items():
                tf_val = count / len(tokens) if tokens else 0
                idf_val = self.idf.get(t, 1.0)
                tfidf[t] = tf_val * idf_val
            self.tfidf_matrix.append(tfidf)

        print(f"Indice construido: {len(self.vocab)} termos unicos")

    def search(self, query: str, top_k: int = 6) -> List[Tuple[Dict, float]]:
        query_tokens = self.tokenize(query)

        # Busca por numero de clausula se mencionado
        clausula_match = re.search(r'cl[aá]usula\s+(\d+\w*)', query.lower())
        if clausula_match:
            num = clausula_match.group(1).upper()
            exact = []
            for i, chunk in enumerate(self.chunks):
                if re.search(rf'CL[AÁ]USULA\s+{num}\b', chunk['text'], re.IGNORECASE):
                    exact.append((chunk, 1.0))
            if exact:
                return exact[:top_k]

        query_tf = {}
        for t in query_tokens:
            query_tf[t] = query_tf.get(t, 0) + 1

        query_tfidf = {}
        for t, count in query_tf.items():
            tf_val = count / len(query_tokens) if query_tokens else 0
            idf_val = self.idf.get(t, 1.0)
            query_tfidf[t] = tf_val * idf_val

        scores = []
        query_norm = math.sqrt(sum(v**2 for v in query_tfidf.values()))

        for i, doc_tfidf in enumerate(self.tfidf_matrix):
            dot = sum(query_tfidf.get(t, 0) * doc_tfidf.get(t, 0) for t in query_tfidf)
            doc_norm = math.sqrt(sum(v**2 for v in doc_tfidf.values()))
            if query_norm > 0 and doc_norm > 0:
                score = dot / (query_norm * doc_norm)
            else:
                score = 0.0
            scores.append((self.chunks[i], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def save(self, path: str):
        data = {
            'chunks': self.chunks,
            'idf': self.idf,
            'tfidf_matrix': self.tfidf_matrix
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False)
        print(f"Indice salvo em {path}")

    def load(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.chunks = data['chunks']
        self.idf = data['idf']
        self.tfidf_matrix = data['tfidf_matrix']
        print(f"Indice carregado: {len(self.chunks)} chunks")


# ---- Build Index ----
def build_index():
    all_docs = []

    doc_files = {
        'Auto Porto - Condicoes Gerais': str(DOCS_PATH / 'auto_porto_full.txt'),
        'Auto Protecao Combinada Porto': str(DOCS_PATH / 'auto_protecao_combinada_full.txt'),
    }

    for source, filepath in doc_files.items():
        if not Path(filepath).exists():
            print(f"AVISO: {filepath} nao encontrado")
            continue

        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()

        chunks = chunk_text(text, chunk_size=1200, overlap=200)
        print(f"{source}: {len(chunks)} chunks")

        for i, chunk in enumerate(chunks):
            all_docs.append({
                'text': chunk,
                'source': source,
                'chunk_id': f"{source}_{i}"
            })

    index = TFIDFIndex()
    index.build(all_docs)
    index.save(str(INDEX_PATH))
    return index


# ---- RAG Query ----
SYSTEM_PROMPT = """Você é a Porto IA, assistente inteligente da Insurian para corretores de seguros que vendem produtos Porto Seguro.

Seu objetivo é responder dúvidas técnicas sobre coberturas, cláusulas, limites e condições do Seguro Auto Porto Seguro com base nas Condições Gerais e Manual do Segurado.

REGRAS:
1. Responda SEMPRE em português brasileiro, de forma clara e objetiva
2. Use as informações do contexto fornecido para fundamentar suas respostas
3. Se a informação estiver no contexto, seja preciso com valores, limites e condições
4. Se não houver informação suficiente no contexto, diga claramente que não encontrou essa informação nos documentos
5. Cite a cláusula ou seção relevante quando disponível
6. Seja cordial e profissional
7. Para perguntas sobre coberturas específicas, sempre mencione os limites e exclusões relevantes
8. NUNCA invente informações — se não souber, diga que não encontrou nos documentos
9. NUNCA forneça números de telefone, 0800, WhatsApp ou canais de atendimento — esses dados ficam desatualizados e você pode passar informação errada. Se perguntarem sobre contato, oriente o corretor a acessar o site www.portoseguro.com.br
10. Foque em responder sobre COBERTURAS, CLÁUSULAS, LIMITES, FRANQUIAS, EXCLUSÕES e PROCEDIMENTOS — esse é seu foco como assistente técnico

Você tem acesso ao Manual do Segurado e às Condições Gerais do Seguro Auto Porto Seguro."""


def query_rag(question: str, index: TFIDFIndex, conversation_history: List[Dict] = None) -> str:
    results = index.search(question, top_k=6)

    context_parts = []
    for chunk, score in results:
        if score > 0.02:
            context_parts.append(f"[{chunk['source']}]\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts) if context_parts else "Nenhum contexto relevante encontrado."

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    if conversation_history:
        messages.extend(conversation_history[-6:])

    user_message = f"""Contexto dos documentos Porto Seguro:

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


def get_index() -> TFIDFIndex:
    global _index_instance
    if _index_instance is None:
        # Sempre reconstruir para garantir chunks atualizados
        idx = build_index()
        _index_instance = idx
    return _index_instance
