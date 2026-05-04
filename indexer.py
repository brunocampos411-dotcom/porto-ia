"""
Porto IA - Script de Indexacao (roda UMA VEZ)
Gera embeddings de todos os documentos e salva em embeddings.json

Como usar:
  OPENAI_API_KEY=sk-... python indexer.py

Custo estimado: ~$0.05-0.10 para todos os documentos
Tempo: 2-5 minutos
"""
import os
import sys

# Garantir que está rodando da pasta correta
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

print("=" * 50)
print("Porto IA - Gerador de Embeddings")
print("=" * 50)

if not os.environ.get("OPENAI_API_KEY"):
    print("\nERRO: defina OPENAI_API_KEY antes de rodar")
    print("Exemplo: OPENAI_API_KEY=sk-... python indexer.py")
    sys.exit(1)

from rag_engine import build_index

print("\nIniciando indexacao dos documentos...")
print("Isso pode levar alguns minutos...\n")

index = build_index()

print(f"\n✅ Concluido! {len(index.chunks)} chunks indexados.")
print("Arquivo gerado: embeddings.json")
print("\nAgora faca o deploy normalmente — o embeddings.json sera carregado automaticamente.")
