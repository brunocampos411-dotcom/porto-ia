"""
Porto IA - Backend FastAPI v4.0
API do chatbot para corretores — com streaming SSE
"""
import os
import time
import secrets
import datetime
import json
import asyncio
from typing import List, Optional, AsyncGenerator
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel

from rag_engine import get_index, query_rag, query_rag_stream, EmbeddingIndex

BASE_DIR = Path(__file__).parent

app = FastAPI(title="Porto IA", description="Assistente inteligente Porto Seguro para corretores")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

static_dir = BASE_DIR / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

_index: Optional[EmbeddingIndex] = None

# ---- Credenciais de acesso ----
APP_USERNAME = os.environ.get("APP_USERNAME", "corretor")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "porto2024")

# ---- Tokens de sessao em memoria (validos por 8h) ----
_sessions: dict = {}
SESSION_TTL_HOURS = 8


def create_session_token() -> str:
    token = secrets.token_urlsafe(32)
    expiry = time.time() + (SESSION_TTL_HOURS * 3600)
    _sessions[token] = expiry
    return token


def validate_token(token: str) -> bool:
    if not token or token not in _sessions:
        return False
    if time.time() > _sessions[token]:
        del _sessions[token]
        return False
    return True


def cleanup_sessions():
    now = time.time()
    expired = [t for t, exp in _sessions.items() if now > exp]
    for t in expired:
        del _sessions[t]


def require_auth(request: Request):
    token = request.headers.get("X-Session-Token") or request.cookies.get("session_token")
    if not validate_token(token):
        raise HTTPException(status_code=401, detail="Nao autorizado. Faca login novamente.")


# ---- Contador semanal (persistente em counter.json) ----
COUNTER_FILE = BASE_DIR / "counter.json"
_counter_data = {"count": 0, "week": 0, "year": 0}


def _load_counter():
    """Carrega contador do arquivo JSON ao iniciar."""
    global _counter_data
    try:
        if COUNTER_FILE.exists():
            data = json.loads(COUNTER_FILE.read_text(encoding='utf-8'))
            week, year = get_current_week()
            if data.get("week") == week and data.get("year") == year:
                _counter_data = data
                print(f"Contador semanal carregado: {data['count']} interacoes (semana {week}/{year})")
            else:
                print(f"Nova semana — contador resetado.")
    except Exception as e:
        print(f"Aviso: nao foi possivel carregar counter.json: {e}")


def _save_counter():
    """Salva contador no arquivo JSON."""
    try:
        COUNTER_FILE.write_text(json.dumps(_counter_data, ensure_ascii=False), encoding='utf-8')
    except Exception as e:
        print(f"Aviso: nao foi possivel salvar counter.json: {e}")


def get_current_week():
    today = datetime.date.today()
    return today.isocalendar()[1], today.year


def increment_counter() -> int:
    global _counter_data
    week, year = get_current_week()
    if _counter_data["week"] != week or _counter_data["year"] != year:
        _counter_data = {"count": 0, "week": week, "year": year}
    _counter_data["count"] += 1
    _save_counter()
    return _counter_data["count"]


def get_counter_value() -> int:
    global _counter_data
    week, year = get_current_week()
    if _counter_data["week"] != week or _counter_data["year"] != year:
        return 0
    return _counter_data["count"]


def get_cached_index():
    global _index
    if _index is None:
        _load_counter()
        print("Carregando indice...")
        _index = get_index()
    return _index


def extract_pdf_text(base64_data: str) -> str:
    """Extrai texto de PDF enviado em base64."""
    import base64, tempfile, sys
    sys.path.insert(0, '/usr/local/lib/python3.13/dist-packages')
    try:
        import pdfplumber
        raw = base64.b64decode(base64_data)
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name
        text_parts = []
        with pdfplumber.open(tmp_path) as pdf:
            for page in pdf.pages:
                t = page.extract_text()
                if t:
                    text_parts.append(t)
        import os
        os.unlink(tmp_path)
        return '\n\n'.join(text_parts)[:12000]  # limita 12k chars
    except Exception as e:
        return f"[Erro ao extrair texto do PDF: {e}]"


def build_message_with_attachment(question: str, attachment, context: str) -> list:
    """Monta a lista de messages para a API com suporte a imagem (multimodal) ou texto de doc."""
    from rag_engine import SYSTEM_PROMPT

    if attachment.type == 'image':
        # Multimodal — imagem em base64
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": (
                    f"Contexto dos documentos indexados:\n{context}\n\n"
                    f"O corretor enviou uma imagem ({attachment.name}) e perguntou:\n{question}\n\n"
                    "Descreva o conteúdo da imagem e responda a pergunta. "
                    "Lembre-se: não afirme se algo é ou não coberto — siga a regra jurídica."
                )},
                {"type": "image_url", "image_url": {
                    "url": f"data:{attachment.mime_type};base64,{attachment.data}"
                }}
            ]}
        ]
    else:
        # PDF / DOCX — texto extraído
        doc_text = extract_pdf_text(attachment.data) if attachment.ext == 'pdf' else \
                   f"[Arquivo DOCX: extração de texto não implementada — peça ao usuário para copiar o texto.]"
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": (
                f"Contexto dos documentos indexados:\n{context}\n\n"
                f"O corretor anexou o arquivo '{attachment.name}' com o seguinte conteúdo:\n"
                f"---\n{doc_text}\n---\n\n"
                f"Pergunta do corretor: {question}\n\n"
                "Responda com base no conteúdo do arquivo. "
                "Não afirme se algo é ou não coberto — siga a regra jurídica."
            )}
        ]


# ---- Models ----
class Message(BaseModel):
    role: str
    content: str


class Attachment(BaseModel):
    type: str          # 'image' | 'doc'
    name: str
    data: str          # base64
    mime_type: str
    ext: Optional[str] = None


class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []
    attachment: Optional[Attachment] = None


class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []
    response_time: float = 0.0
    weekly_count: int = 0


class LoginRequest(BaseModel):
    username: str
    password: str


# ---- Endpoints publicos ----
@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = BASE_DIR / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Porto IA</h1><p>Interface em construcao...</p>")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "Porto IA", "version": "4.0.0"}


@app.post("/api/login")
async def login(req: LoginRequest):
    cleanup_sessions()
    if req.username.strip() == APP_USERNAME and req.password == APP_PASSWORD:
        token = create_session_token()
        return {"success": True, "token": token, "expires_in_hours": SESSION_TTL_HOURS}
    raise HTTPException(status_code=401, detail="Usuario ou senha incorretos")


@app.post("/api/logout")
async def logout(request: Request):
    token = request.headers.get("X-Session-Token") or request.cookies.get("session_token")
    if token and token in _sessions:
        del _sessions[token]
    return {"success": True}


# ---- Endpoints protegidos ----
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, _auth=Depends(require_auth)):
    """Chat normal (resposta completa de uma vez) — com suporte a anexos PDF e imagem."""
    import traceback
    start = time.time()
    try:
        idx = get_cached_index()
        history = [{"role": msg.role, "content": msg.content} for msg in request.history]

        results = idx.search(request.message, top_k=5)
        sources = list(set([r[0]['source'] for r in results if r[1] > 0.1]))
        if not sources and results:
            sources = list(set([r[0]['source'] for r in results[:3]]))

        if request.attachment:
            # Com anexo: monta contexto RAG + processa o arquivo
            context_parts = [f"[{r[0]['source']}]\n{r[0]['text']}" for r in results if r[1] > 0.05]
            context = "\n\n---\n\n".join(context_parts[:4]) if context_parts else "Nenhum contexto adicional nos documentos indexados."

            from rag_engine import get_llm_client, SYSTEM_PROMPT
            client = get_llm_client()

            messages = build_message_with_attachment(request.message, request.attachment, context)
            # Injeta historico antes da mensagem do usuario
            if history and len(messages) >= 2:
                messages = [messages[0]] + history[-6:] + [messages[-1]]

            model = "anthropic/claude-haiku-4-5" if request.attachment.type == 'doc' else "anthropic/claude-haiku-4-5"
            resp = client.chat.completions.create(
                model=model,
                max_tokens=1200,
                temperature=0.2,
                messages=messages
            )
            answer = resp.choices[0].message.content
            if request.attachment.type == 'image':
                sources = [f"Imagem: {request.attachment.name}"]
            else:
                sources = [f"Arquivo: {request.attachment.name}"] + sources
        else:
            # Sem anexo: fluxo normal
            answer = query_rag(request.message, idx, history)

        weekly_count = increment_counter()
        elapsed = time.time() - start

        return ChatResponse(
            answer=answer,
            sources=sources,
            response_time=round(elapsed, 2),
            weekly_count=weekly_count
        )
    except Exception as e:
        tb = traceback.format_exc()
        print(f"ERRO /api/chat: {e}\n{tb}")
        raise HTTPException(status_code=500, detail=f"{type(e).__name__}: {str(e)}")


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest, _auth=Depends(require_auth)):
    """
    Chat com streaming SSE — resposta palavra por palavra.
    Formato dos eventos:
      data: {"type": "chunk", "content": "texto"}
      data: {"type": "sources", "sources": [...], "weekly_count": N}
      data: {"type": "done"}
    """
    async def generate() -> AsyncGenerator[str, None]:
        try:
            idx = get_cached_index()
            history = [{"role": msg.role, "content": msg.content} for msg in request.history]

            sources_sent = False
            for text_chunk, sources in query_rag_stream(request.message, idx, history):
                event = json.dumps({"type": "chunk", "content": text_chunk}, ensure_ascii=False)
                yield f"data: {event}\n\n"

                if not sources_sent and sources:
                    count = increment_counter()
                    meta = json.dumps({
                        "type": "sources",
                        "sources": sources,
                        "weekly_count": count
                    }, ensure_ascii=False)
                    yield f"data: {meta}\n\n"
                    sources_sent = True

                await asyncio.sleep(0)  # Cede controle para o event loop

            yield "data: {\"type\": \"done\"}\n\n"

        except Exception as e:
            err = json.dumps({"type": "error", "message": str(e)}, ensure_ascii=False)
            yield f"data: {err}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/api/counter")
async def get_counter(_auth=Depends(require_auth)):
    week, year = get_current_week()
    return {"weekly_count": get_counter_value(), "week": week, "year": year}


@app.get("/api/suggest")
async def suggest_questions(_auth=Depends(require_auth)):
    return {
        "suggestions": [
            "Quais os produtos disponíveis no Grupo Porto?",
            "Compare a assistência 24h da Porto, Itaú, Azul e Mitsui",
            "O seguro viagem cobre cancelamento de voo?",
            "Qual a diferença entre Auto Compacto e Auto Frota Compacto?",
            "Como funciona o Projeto 15 Minutos da Porto?",
            "O que cobre o seguro residencial da Porto?",
            "Quais são as exclusões para perda total?",
            "Como funciona o seguro de vida Porto?",
            "O que é RC Profissional e quem precisa?",
            "Quais as vantagens do Cartão Porto Bank?",
        ]
    }


@app.get("/api/stats")
async def stats(_auth=Depends(require_auth)):
    idx = get_cached_index()
    return {
        "total_chunks": len(idx.chunks),
        "embedding_dim": idx.embeddings.shape[1] if idx.embeddings is not None else 0,
        "sources": sorted(list(set([c['source'] for c in idx.chunks]))),
        "status": "operacional",
        "engine": "hibrido-embeddings-tfidf"
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    print("Iniciando Porto IA v4.0 (Busca Hibrida + Streaming)...")
    print("Carregando base de conhecimento...")
    _load_counter()
    _index = get_index()
    print(f"Base carregada: {len(_index.chunks)} chunks")
    print(f"\nServidor: http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
