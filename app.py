"""
Porto IA - Backend FastAPI
API do chatbot da Porto Seguro para corretores
"""
import os
import time
import secrets
import datetime
import json
from typing import List, Optional
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from rag_engine import get_index, query_rag, EmbeddingIndex

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

# ---- Credenciais de acesso (via variaveis de ambiente) ----
# No Render: defina APP_USERNAME e APP_PASSWORD no painel de Environment
APP_USERNAME = os.environ.get("APP_USERNAME", "corretor")
APP_PASSWORD = os.environ.get("APP_PASSWORD", "porto2024")

# ---- Tokens de sessao em memoria ----
# Cada token e valido por 8 horas
_sessions: dict = {}  # token -> expiry_timestamp

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
    """Remove sessoes expiradas"""
    now = time.time()
    expired = [t for t, exp in _sessions.items() if now > exp]
    for t in expired:
        del _sessions[t]

def require_auth(request: Request):
    """Dependencia FastAPI que verifica autenticacao"""
    token = request.headers.get("X-Session-Token") or request.cookies.get("session_token")
    if not validate_token(token):
        raise HTTPException(status_code=401, detail="Nao autorizado. Faca login novamente.")

# ---- Contador semanal (em memoria, reseta ao reiniciar no Render free tier) ----
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
    return HTMLResponse(content="<h1>Porto IA</h1><p>Interface em construcao...</p>")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "Porto IA", "version": "2.1.0"}


@app.post("/api/login")
async def login(req: LoginRequest):
    """Autentica o usuario e retorna um token de sessao"""
    cleanup_sessions()
    if req.username.strip() == APP_USERNAME and req.password == APP_PASSWORD:
        token = create_session_token()
        return {"success": True, "token": token, "expires_in_hours": SESSION_TTL_HOURS}
    raise HTTPException(status_code=401, detail="Usuario ou senha incorretos")


@app.post("/api/logout")
async def logout(request: Request):
    """Invalida o token de sessao"""
    token = request.headers.get("X-Session-Token") or request.cookies.get("session_token")
    if token and token in _sessions:
        del _sessions[token]
    return {"success": True}


# ---- Endpoints protegidos ----
@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, _auth=Depends(require_auth)):
    start = time.time()

    try:
        idx = get_cached_index()

        history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.history
        ]

        # Buscar fontes (top 5 chunks semanticos)
        results = idx.search(request.message, top_k=5)
        sources = list(set([
            r[0]['source']
            for r in results
            if r[1] > 0.2
        ]))
        if not sources and results:
            sources = list(set([r[0]['source'] for r in results[:3]]))

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
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/counter")
async def get_counter(_auth=Depends(require_auth)):
    """Retorna contador de interacoes da semana"""
    week, year = get_current_week()
    return {
        "weekly_count": get_counter_value(),
        "week": week,
        "year": year
    }


@app.get("/api/suggest")
async def suggest_questions(_auth=Depends(require_auth)):
    """Sugestoes de perguntas frequentes"""
    return {
        "suggestions": [
            "Compare a assistência 24h da Porto, Itaú, Azul e Mitsui",
            "Quais são os diferenciais exclusivos do Seguro Auto Porto?",
            "Como funciona o Projeto 15 Minutos da Porto?",
            "Qual é o limite do guincho em cada seguradora?",
            "O que cobre a assistência 24 horas em caso de pane?",
            "Quais as vantagens do Cartão Porto Bank?",
            "Como funciona o carro reserva em caso de sinistro?",
            "Quais são as exclusões para perda total?",
            "O que é a cláusula 87 - Reparo Rápido e Supermartelinho?",
            "Compare coberturas de danos a terceiros entre Porto e Itaú",
        ]
    }


@app.get("/api/stats")
async def stats(_auth=Depends(require_auth)):
    idx = get_cached_index()
    return {
        "total_chunks": len(idx.chunks),
        "embedding_dim": idx.embeddings.shape[1] if idx.embeddings is not None else 0,
        "sources": list(set([c['source'] for c in idx.chunks])),
        "status": "operacional",
        "engine": "embeddings-semanticos"
    }


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8001))
    print("Iniciando Porto IA v3.0 (Embeddings Semanticos)...")
    print("Carregando base de conhecimento...")
    _index = get_index()
    print(f"Base carregada: {len(_index.chunks)} chunks")
    print(f"\nServidor: http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
