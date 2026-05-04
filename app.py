"""
Porto IA - Backend FastAPI
API do chatbot da Porto Seguro para corretores
"""
import time
import datetime
import json
from typing import List, Optional
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from rag_engine import get_index, query_rag, TFIDFIndex

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

_index: Optional[TFIDFIndex] = None

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


# ---- Endpoints ----
@app.get("/", response_class=HTMLResponse)
async def root():
    html_path = BASE_DIR / "static" / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding='utf-8'))
    return HTMLResponse(content="<h1>Porto IA</h1><p>Interface em construcao...</p>")


@app.get("/health")
async def health():
    return {"status": "ok", "service": "Porto IA", "version": "2.0.0"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    start = time.time()

    try:
        idx = get_cached_index()

        history = [
            {"role": msg.role, "content": msg.content}
            for msg in request.history
        ]

        results = idx.search(request.message, top_k=5)
        sources = list(set([
            r[0]['source']
            for r in results
            if r[1] > 0.01
        ]))

        answer = query_rag(request.message, idx, history)

        # Incrementar contador semanal
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
async def get_counter():
    """Retorna contador de interacoes da semana"""
    week, year = get_current_week()
    return {
        "weekly_count": get_counter_value(),
        "week": week,
        "year": year
    }


@app.get("/api/suggest")
async def suggest_questions():
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
async def stats():
    idx = get_cached_index()
    return {
        "total_chunks": len(idx.chunks),
        "total_terms": len(idx.idf),
        "sources": list(set([c['source'] for c in idx.chunks])),
        "status": "operacional"
    }


if __name__ == "__main__":
    port = int(__import__("os").environ.get("PORT", 8001))
    print("Iniciando Porto IA v2...")
    print("Carregando base de conhecimento...")
    _index = get_index()
    print(f"Base carregada: {len(_index.chunks)} chunks")
    print(f"\nServidor: http://0.0.0.0:{port}")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
