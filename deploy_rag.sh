#!/bin/bash
gcloud run deploy rag-service \
  --image us-central1-docker.pkg.dev/proslm/proslm-repo/rag-service:v1 \
  --region us-central1 \
  --platform managed \
  --memory 4Gi \
  --cpu 2 \
  --timeout 300 \
  --set-secrets OPENAI_API_KEY=OPENAI_API_KEY:latest \
  --set-env-vars VECTORSTORE_PATH=/app/agent_integration/vectorstore-hotpot/hotpotqa_faiss_v3,GEN_LLM_MODEL=gpt-4o-mini,EMB_MODEL=text-embedding-3-large \
  --allow-unauthenticated
