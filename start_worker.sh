#!/usr/bin/env bash
# Start Celery worker (heavy ML — run on the powerful VM)
# Requires: Redis running (apt install redis-server && redis-server --daemonize yes)
# Requires: all pip deps installed (pip install -r requirements.txt)
set -e
cd "$(dirname "$0")"
source .env 2>/dev/null || true

echo "Starting Celery worker..."
echo "  Broker: ${REDIS_URL:-redis://localhost:6379/0}"
echo "  Concurrency: 1 (pipeline is CPU/memory intensive)"

celery -A app.worker worker \
    --loglevel=info \
    --concurrency=1 \
    --queues=celery \
    --hostname=cdek-worker@%h
