#!/bin/bash
gcloud run deploy backend-service \
  --image us-central1-docker.pkg.dev/proslm/proslm-repo/backend-service:v1 \
  --region us-central1 \
  --platform managed \
  --memory 512Mi \
  --cpu 1 \
  --timeout 300 \
  --set-secrets OPENAI_API_KEY=OPENAI_API_KEY:latest,MONGO_URI=MONGO_URI:latest \
  --set-env-vars RAG_SERVICE_URL=https://rag-service-759005971862.us-central1.run.app \
  --allow-unauthenticated
