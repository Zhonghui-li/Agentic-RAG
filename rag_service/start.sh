#!/bin/bash
set -e

VECTORSTORE_DIR="/app/agent_integration/vectorstore-hotpot/hotpotqa_faiss_v3"

# Download FAISS from Cloud Storage if not already present
if [ ! -d "$VECTORSTORE_DIR" ]; then
    echo "[start] Downloading FAISS vectorstore from Cloud Storage..."
    mkdir -p /app/agent_integration/vectorstore-hotpot
    gsutil -m cp -r gs://proslm-vectorstore/hotpotqa_faiss_v3 \
        /app/agent_integration/vectorstore-hotpot/
    echo "[start] FAISS download complete."
else
    echo "[start] FAISS vectorstore already present, skipping download."
fi

echo "[start] Starting RAG service..."
exec python /app/rag_service/main.py
