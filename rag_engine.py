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
        import unicodedata
        # Normalizar acentos: "Itaú" -> "itau", "assistências" -> "assistencias"
        text = ''.join(
            c for c in unicodedata.normalize('NFD', text.lower())
            if unicodedata.category(c) != 'Mn'
        )
        tokens = re.findall(r'\b[a-z\w]{2,}\b', text)
        stopwords = {
            'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para',
            'com', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as',
            'dos', 'como', 'mas', 'ao', 'ele', 'das', 'seu', 'sua',
            'ou', 'quando', 'muito', 'nos', 'ja', 'eu', 'tambem',
            'pelo', 'pela', 'ate', 'isso', 'ela', 'entre', 'depois',
            'sem', 'mesmo', 'aos', 'seus', 'quem', 'nas', 'me', 'esse',
            'eles', 'essa', 'num', 'nem', 'suas', 'meu', 'minha', 'te',
            'nao', 'nesta', 'deste', 'estava', 'este', 'havia',
            'ser', 'ter', 'pode', 'foi', 'sao', 'esta',
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

    def search_in_source(self, query: str, source_key: str, top_k: int = 3) -> List[Tuple[Dict, float]]:
        """Busca apenas nos chunks de uma seguradora especifica, com reranking por topico"""
        import unicodedata

        def norm(t):
            return ''.join(c for c in unicodedata.normalize('NFD', t.lower())
                           if unicodedata.category(c) != 'Mn')

        src_key_norm = norm(source_key)
        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        query_tf = {}
        for t in query_tokens:
            query_tf[t] = query_tf.get(t, 0) + 1
        query_tfidf = {t: (c/len(query_tokens))*self.idf.get(t, 1.0) for t, c in query_tf.items()}
        query_norm_val = math.sqrt(sum(v**2 for v in query_tfidf.values()))

        scores = []
        for i, doc_tfidf in enumerate(self.tfidf_matrix):
            if src_key_norm not in norm(self.chunks[i]['source']):
                continue
            dot = sum(query_tfidf.get(t, 0)*doc_tfidf.get(t, 0) for t in query_tfidf)
            dnorm = math.sqrt(sum(v**2 for v in doc_tfidf.values()))
            score = dot/(query_norm_val*dnorm) if query_norm_val > 0 and dnorm > 0 else 0.0
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

    # Mapeamento de arquivo -> nome legivel da fonte
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
SYSTEM_PROMPT = """Você é a Porto IA, assistente técnico especializado para corretores de seguros da Insurian.
Você tem acesso às Condições Gerais, Manuais do Segurado e informações do site oficial da Porto Seguro, Azul Seguros, Itaú Seguros e Mitsui Seguros.

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
10. Se não tiver a informação no contexto: diga claramente "Não encontrei essa informação no contexto disponível"
11. Para sinistros: oriente a acessar o App Porto ou portoseguro.com.br/atendimento/sinistros

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


def query_rag(question: str, index: TFIDFIndex, conversation_history: List[Dict] = None) -> str:
    import unicodedata

    def norm(t):
        return ''.join(c for c in unicodedata.normalize('NFD', t.lower()) if unicodedata.category(c) != 'Mn')

    # Detectar quais seguradoras foram mencionadas na pergunta
    q_norm = norm(question)
    seguradora_keys = {
        'porto': ['porto seguro', 'auto protecao combinada', 'porto'],
        'azul': ['azul'],
        'itau': ['itau'],
        'mitsui': ['mitsui'],
    }
    mencoes = [key for key in seguradora_keys if key in q_norm]

    # Detectar se eh pergunta comparativa geral (sem citar seguradoras especificas)
    palavras_comparativas = ['compare', 'comparar', 'comparativo', 'diferenca', 'diferente',
                              'todas', 'cada', 'por seguradora', 'entre as seguradoras',
                              'todas as seguradoras', 'qual seguradora', 'quais seguradoras']
    eh_comparativa_geral = any(p in q_norm for p in palavras_comparativas)

    # Se pergunta mencionar "todas" as seguradoras implicitamente, buscar em todas
    if eh_comparativa_geral and len(mencoes) == 0:
        mencoes = list(seguradora_keys.keys())

    seen_ids = set()
    context_parts = []

    if len(mencoes) >= 2:
        # Para perguntas comparativas: busca focada por seguradora usando search_in_source
        # Remove nomes das seguradoras da query para focar no topico
        topic_query = question
        for key in seguradora_keys:
            for variant in [key, key.capitalize(), key.upper()]:
                topic_query = topic_query.replace(variant, '')
        topic_query = ' '.join(topic_query.split())  # normalizar espacos

        for key in mencoes:
            results = index.search_in_source(topic_query, key, top_k=3)
            for chunk, score in results:
                cid = (chunk['source'], chunk['chunk_id'])
                if cid not in seen_ids:
                    context_parts.append(f"[{chunk['source']}]\n{chunk['text']}")
                    seen_ids.add(cid)
    else:
        results = index.search(question, top_k=6)
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
