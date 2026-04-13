"""
Celery worker for CDEK pipeline.

Start worker:
    celery -A app.worker worker --loglevel=info --concurrency=1

Requires Redis (set REDIS_URL in .env).
"""

import os
import sys
import traceback
from pathlib import Path

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from celery import Celery

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery("cdek_pipeline", broker=REDIS_URL, backend=REDIS_URL)

celery_app.conf.update(
    task_track_started=True,
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    worker_prefetch_multiplier=1,       # one task at a time (heavy pipeline)
    task_acks_late=True,                # ack only after task completes (safe retry)
    result_expires=86400,               # keep results 24h
)

# Redis key pattern: "task_id:{video_id}" → Celery task UUID
TASK_KEY_PREFIX = "cdek:task_id:"


def _store_task_id(video_id: str, task_id: str):
    """Save Celery task_id keyed by video_id in Redis backend."""
    try:
        celery_app.backend.client.set(
            TASK_KEY_PREFIX + video_id, task_id, ex=86400
        )
    except Exception:
        pass  # non-critical


def get_task_id(video_id: str) -> str | None:
    """Retrieve the Celery task_id for a given video_id."""
    try:
        val = celery_app.backend.client.get(TASK_KEY_PREFIX + video_id)
        return val.decode() if isinstance(val, bytes) else val
    except Exception:
        return None


# ─── Step labels exposed to the API ───────────────────────────────────────────

STEPS = [
    (5,  "Запуск пайплайна..."),
    (15, "Извлечение кадров..."),
    (30, "Детекция лиц..."),
    (45, "Распознавание эмоций..."),
    (55, "Извлечение аудио..."),
    (65, "Транскрипция речи (SpeechKit)..."),
    (78, "Анализ диалога и тональности..."),
    (88, "Объединённый анализ эмоций..."),
    (95, "Оценка KPI..."),
]


@celery_app.task(bind=True, name="cdek.process_video")
def process_video(self, video_path: str):
    """Run the full CDEK analysis pipeline for a single video."""
    video_id = Path(video_path).stem

    def progress(pct: int, step: str):
        self.update_state(state="PROGRESS", meta={"pct": pct, "step": step})

    progress(5, "Запуск пайплайна...")

    original_argv = sys.argv
    sys.argv = ["run_full_pipeline", video_path]
    try:
        from app.run_full_pipeline import main as run_pipeline
        run_pipeline()
        return {"status": "completed", "video_id": video_id}
    except Exception as exc:
        traceback.print_exc()
        # Mark session as failed in DB
        try:
            from app.backend.database.schema import get_connection
            from app.backend.database.repository import PipelineRepository
            conn = get_connection()
            repo = PipelineRepository(conn)
            session = repo.get_session_by_video_id(video_id)
            if session:
                repo.update_session_status(session["session_id"], "failed")
            conn.close()
        except Exception:
            pass
        raise
    finally:
        sys.argv = original_argv
