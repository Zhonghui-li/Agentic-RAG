#!/bin/bash
gcloud run deploy frontend-service \
  --image us-central1-docker.pkg.dev/proslm/proslm-repo/frontend-service:v1 \
  --region us-central1 \
  --platform managed \
  --memory 512Mi \
  --cpu 1 \
  --timeout 60 \
  --port 3000 \
  --allow-unauthenticated
