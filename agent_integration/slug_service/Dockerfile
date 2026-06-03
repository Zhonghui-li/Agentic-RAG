# UCSC Slug Advisor — live demo for Cloud Run.
# Reranker kept (quality = evaluated baseline); CrossEncoder model pre-baked so
# cold starts don't download it. Build context must be agent_integration/.
FROM python:3.11-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    RERANK=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1 \
    TRANSFORMERS_VERBOSITY=error

# CPU-only torch first (avoids the ~2GB CUDA build), then the rest.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY slug_service/requirements.txt ./slug_service/requirements.txt
RUN pip install --no-cache-dir -r slug_service/requirements.txt

# Pre-bake the reranker model into the image (no runtime download / cold-start stall).
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')"

# Code + data + prebuilt FAISS vectorstore (vectorstore is gitignored, baked here).
COPY agents/ ./agents/
COPY data-ucsc/ ./data-ucsc/
COPY vectorstore-ucsc/ ./vectorstore-ucsc/
COPY slug_service/ ./slug_service/

# Cloud Run injects $PORT (default 8080).
ENV PORT=8080
CMD uvicorn slug_service.app:app --host 0.0.0.0 --port ${PORT}
