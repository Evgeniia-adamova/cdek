#!/usr/bin/env bash
# Start FastAPI server (lightweight — no ML models loaded here)
# Run on the API VM (or same VM during development)
set -e
cd "$(dirname "$0")"
source .env 2>/dev/null || true

echo "Starting FastAPI server on port 8000..."
uvicorn app.api:app --host 127.0.0.1 --port 8000 --workers 1
