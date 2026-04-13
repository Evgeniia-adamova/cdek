#!/usr/bin/env bash
# Start FastAPI server (lightweight — no ML models loaded here)
# Run on the API VM (or same VM during development)
set -e
cd "$(dirname "$0")"
source .env 2>/dev/null || true

echo "Starting FastAPI server on port 80..."
uvicorn app.api:app --host 0.0.0.0 --port 80 --workers 2
