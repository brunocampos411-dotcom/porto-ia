"""
FOCUS AI - Backend FastAPI v2.0
API do chatbot para corretores de seguros — com streaming SSE
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

# ---- Tracking de metricas ----
_metrics = {
    "total_interacoes": 0,
    "interacoes_por_hora": {},       # "YYYY-MM-DD HH" -> count
    "interacoes_por_dia": {},        # "YYYY-MM-DD" -> count
    "top_perguntas": [],             # lista de {pergunta, count, ultima_vez}
    "top_sources": {},               # source -> count de uso
    "response_times": [],            # ultimos 100 response times
    "sessoes_ativas": 0,
    "inicio": datetime.datetime.now().isoformat(),
}
_perguntas_map: dict = {}           # pergunta normalizada -> {original, count, ultima_vez}


def _track_interaction(pergunta: str, sources: list, response_time: float):
    global _metrics
    now = datetime.datetime.now()
    hora_key = now.strftime("%Y-%m-%d %H")
    dia_key = now.strftime("%Y-%m-%d")

    _metrics["total_interacoes"] += 1
    _metrics["interacoes_por_hora"][hora_key] = _metrics["interacoes_por_hora"].get(hora_key, 0) + 1
    _metrics["interacoes_por_dia"][dia_key] = _metrics["interacoes_por_dia"].get(dia_key, 0) + 1

    # Tracking de response time (guarda ultimos 100)
    _metrics["response_times"].append(round(response_time, 2))
    if len(_metrics["response_times"]) > 100:
        _metrics["response_times"] = _metrics["response_times"][-100:]

    # Tracking de fontes usadas
    for s in sources:
        _metrics["top_sources"][s] = _metrics["top_sources"].get(s, 0) + 1

    # Tracking de perguntas (normaliza para agrupar similares)
    key = pergunta.strip().lower()[:100]
    if key in _perguntas_map:
        _perguntas_map[key]["count"] += 1
        _perguntas_map[key]["ultima_vez"] = now.isoformat()
    else:
        _perguntas_map[key] = {
            "pergunta": pergunta[:200],
            "count": 1,
            "ultima_vez": now.isoformat()
        }
    # Atualiza top_perguntas (top 20)
    sorted_q = sorted(_perguntas_map.values(), key=lambda x: x["count"], reverse=True)
    _metrics["top_perguntas"] = sorted_q[:20]

app = FastAPI(title="FOCUS AI", description="IA para Corretores de Seguros - FOCUS AI")

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


# ---- Contador semanal ----
_counter_data = {"count": 0, "week": 0, "year": 0}


def get_current_week():
    today = datetime.date.today()
    return today.isocalendar()[1], today.year


def increment_counter() -> int:
    global _counter_data
    week, year = get_current_week()
    if _counter_data["week"] != week or _counter_data["year"] != year:
        _counter_data = {"count": 0, "week": week, "year": year}
    _counter_data["count"] += 1
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
        print("Carregando indice...")
        _index = get_index()
    return _index


# ---- Models ----
class Message(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[Message] = []


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
    return HTMLResponse(content="<h1>FOCUS AI</h1><p>IA para Corretores de Seguros</p>")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "FOCUS AI", "version": "2.0.0"}


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
    """Chat normal (resposta completa de uma vez)"""
    import traceback
    start = time.time()
    try:
        idx = get_cached_index()
        history = [{"role": msg.role, "content": msg.content} for msg in request.history]

        results = idx.search(request.message, top_k=5)
        sources = list(set([r[0]['source'] for r in results if r[1] > 0.1]))
        if not sources and results:
            sources = list(set([r[0]['source'] for r in results[:3]]))

        answer = query_rag(request.message, idx, history)
        weekly_count = increment_counter()
        elapsed = time.time() - start
        _track_interaction(request.message, sources, elapsed)

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
                    _track_interaction(request.message, sources, 0)
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
            "Quais coberturas estão incluídas no seguro auto?",
            "Qual a diferença entre cobertura básica e ampla?",
            "O seguro viagem cobre cancelamento de voo?",
            "Quais são as exclusões para perda total?",
            "Como funciona o seguro de vida?",
            "O que cobre o seguro residencial?",
            "O que é RC Profissional e quem precisa?",
            "Como funciona a assistência 24h para guincho?",
            "O seguro cobre quando o condutor não é o proprietário?",
            "Quais produtos cobrem equipamentos eletrônicos?",
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


@app.get("/api/dashboard")
async def dashboard_data(_auth=Depends(require_auth)):
    """Retorna todos os dados de metricas para o dashboard."""
    idx = get_cached_index()
    sources_list = sorted(list(set([c['source'] for c in idx.chunks])))

    # Calcula tempo medio de resposta
    rts = _metrics["response_times"]
    avg_response = round(sum(rts) / len(rts), 2) if rts else 0

    # Ultimos 7 dias
    hoje = datetime.date.today()
    ultimos_7_dias = []
    for i in range(6, -1, -1):
        d = (hoje - datetime.timedelta(days=i)).strftime("%Y-%m-%d")
        ultimos_7_dias.append({
            "data": d,
            "label": (hoje - datetime.timedelta(days=i)).strftime("%d/%m"),
            "count": _metrics["interacoes_por_dia"].get(d, 0)
        })

    # Top fontes consultadas
    top_fontes = sorted(
        [{"source": k, "count": v} for k, v in _metrics["top_sources"].items()],
        key=lambda x: x["count"], reverse=True
    )[:10]

    # Sessoes ativas (tokens validos agora)
    now_ts = time.time()
    sessoes_ativas = sum(1 for exp in _sessions.values() if now_ts < exp)

    return {
        "resumo": {
            "total_interacoes": _metrics["total_interacoes"],
            "interacoes_semana": get_counter_value(),
            "avg_response_time": avg_response,
            "sessoes_ativas": sessoes_ativas,
            "total_documentos": len(sources_list),
            "total_chunks": len(idx.chunks),
            "engine": "Hibrido (Embeddings + TF-IDF)",
            "status": "Operacional",
            "online_desde": _metrics["inicio"],
        },
        "grafico_7_dias": ultimos_7_dias,
        "top_perguntas": _metrics["top_perguntas"][:15],
        "top_fontes": top_fontes,
        "documentos": sources_list,
    }


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    html_path = BASE_DIR / "static" / "dashboard.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Dashboard em construcao</h1>")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    print("Iniciando FOCUS AI v2.0 (Busca Hibrida + Streaming)...")
    print("Carregando base de conhecimento...")
    _index = get_index()
    print(f"Base carregada: {len(_index.chunks)} chunks")
    print(f"\nServidor: http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
