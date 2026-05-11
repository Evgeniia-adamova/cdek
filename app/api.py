"""
CDEK Quality Control — FastAPI Backend
Serves data from PostgreSQL to the frontend dashboard.
"""

import os
import json
import traceback
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import boto3
from botocore.client import Config
import requests as http_requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.backend.database.schema import get_connection
from app.backend.database.repository import PipelineRepository
from app.worker import celery_app, process_video, get_task_id, _store_task_id

app = FastAPI(title="CDEK Quality Control API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_repo() -> PipelineRepository:
    conn = get_connection()
    return PipelineRepository(conn)


# ─────────────────────────────────────────────
# S3 STORAGE
# ─────────────────────────────────────────────

def _s3_client():
    return boto3.client(
        "s3",
        endpoint_url="https://storage.yandexcloud.net",
        aws_access_key_id=os.environ.get("YANDEX_S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("YANDEX_S3_SECRET_ACCESS_KEY"),
        config=Config(signature_version="s3v4"),
        region_name="ru-central1",
    )


def _upload_to_s3_background(local_path: str, filename: str):
    """Upload a file to S3 in a background thread (non-blocking)."""
    def _do():
        try:
            bucket = os.environ.get("YANDEX_S3_BUCKET")
            _s3_client().upload_file(local_path, bucket, f"uploads/{filename}")
        except Exception:
            traceback.print_exc()
    threading.Thread(target=_do, daemon=True).start()


# ─────────────────────────────────────────────
# SESSIONS / ANALYSES
# ─────────────────────────────────────────────

@app.get("/api/analyses")
def list_analyses():
    """List all processed sessions with scores — feeds the dashboard cards."""
    repo = get_repo()
    rows = repo.get_session_summary()
    results = []
    for r in rows:
        processed = r.get("processed_at") or r.get("created_at")
        results.append({
            "session_id": r["session_id"],
            "video_id": r.get("video_id", ""),
            "manager_name": r.get("operator_name") or "Неизвестно",
            "processed_at": processed.isoformat() if processed else None,
            "score_percentage": r.get("score_percentage", 0),
            "traffic_light": r.get("traffic_light", ""),
            "status": r.get("pipeline_status", ""),
            "duration_sec": r.get("duration_sec", 0),
            "contract_number": r.get("contract_number", ""),
        })
    return results


@app.get("/api/analyses/{session_id}")
def get_analysis(session_id: int):
    """Full analysis detail for a single session."""
    repo = get_repo()

    summary = repo.get_session_summary(session_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Session not found")
    s = summary[0]

    kpi_detail = repo.get_kpi_detail(session_id)
    transcript = repo.get_transcript(session_id)
    emotions = repo.get_emotion_timeline(session_id)

    # Group KPI items by category
    categories = {}
    for item in kpi_detail:
        cat = item.get("category_name", "Other")
        if cat not in categories:
            categories[cat] = {"items": [], "total_score": 0, "max_score": 0}
        categories[cat]["items"].append({
            "key": item.get("key", ""),
            "name": item.get("criterion_name", ""),
            "score": item.get("score", 0),
            "max_score": item.get("max_score", 0),
            "status": item.get("status", ""),
            "details": item.get("details", ""),
        })
        categories[cat]["total_score"] += item.get("score", 0)
        categories[cat]["max_score"] += item.get("max_score", 0)

    for cat_data in categories.values():
        mx = cat_data["max_score"]
        cat_data["percentage"] = round(cat_data["total_score"] / mx * 100) if mx else 0

    return {
        "session_id": session_id,
        "video_id": s.get("video_id", ""),
        "manager": s.get("operator_name") or "—",
        "score_percentage": s.get("score_percentage", 0),
        "overall_score": s.get("overall_score", 0),
        "max_possible_score": s.get("max_possible_score", 0),
        "traffic_light": s.get("traffic_light", ""),
        "overall_status": s.get("overall_status", ""),
        "duration_sec": s.get("duration_sec", 0),
        "categories": categories,
        "transcript": [
            {
                "start": t["start_sec"],
                "end": t["end_sec"],
                "text": t["text"],
                "speaker": t.get("speaker_label", ""),
                "role": t.get("role", ""),
                "sentiment": t.get("sentiment_label", "neutral"),
            }
            for t in transcript
        ],
        "emotion_timeline": [
            {
                "interval_id": e.get("interval_id"),
                "person": e.get("person_label", ""),
                "video_emotion": e.get("video_emotion", ""),
                "audio_sentiment": e.get("audio_sentiment", ""),
                "combined_emotion": e.get("combined_emotion", ""),
                "combined_valence": e.get("combined_valence", 0),
            }
            for e in emotions
        ],
    }


# ─────────────────────────────────────────────
# DASHBOARD STATS
# ─────────────────────────────────────────────

@app.get("/api/dashboard")
def dashboard_stats():
    """Aggregated stats for the dashboard header cards."""
    repo = get_repo()
    rows = repo.get_session_summary()
    if not rows:
        return {"total_analyses": 0, "avg_score": 0, "min_score": 0, "max_score": 0,
                "manager_count": 0, "person_count": 0}
    scores = [r.get("score_percentage") or 0 for r in rows]
    managers = set(r.get("operator_name") for r in rows if r.get("operator_name"))
    return {
        "total_analyses": len(rows),
        "avg_score": round(sum(scores) / len(scores), 2) if scores else 0,
        "min_score": min(scores) if scores else 0,
        "max_score": max(scores) if scores else 0,
        "manager_count": len(managers),
        "person_count": len(rows),  # approximate
    }


# ─────────────────────────────────────────────
# MANAGERS
# ─────────────────────────────────────────────

@app.get("/api/managers")
def list_managers():
    """Manager-level statistics for the analytics page."""
    repo = get_repo()
    rows = repo.get_session_summary()
    mgr_stats: dict = {}
    for r in rows:
        name = r.get("operator_name") or "Неизвестно"
        if name not in mgr_stats:
            mgr_stats[name] = {"name": name, "scores": [], "call_count": 0}
        mgr_stats[name]["scores"].append(r.get("score_percentage") or 0)
        mgr_stats[name]["call_count"] += 1

    result = []
    for m in mgr_stats.values():
        s = [x for x in m["scores"] if x is not None]
        result.append({
            "name": m["name"],
            "call_count": m["call_count"],
            "avg_score": round(sum(s) / len(s), 2) if s else None,
            "min_score": min(s) if s else None,
            "max_score": max(s) if s else None,
            "excellent_count": sum(1 for x in s if x >= 90),
        })
    return result


# ─────────────────────────────────────────────
# SCORE DYNAMICS
# ─────────────────────────────────────────────

@app.get("/api/score-dynamics")
def score_dynamics():
    """Daily average scores for the trend chart."""
    repo = get_repo()
    rows = repo.get_session_summary()
    daily: dict = {}
    for r in rows:
        dt = r.get("created_at")
        if not dt:
            continue
        day = dt.strftime("%Y-%m-%d")
        if day not in daily:
            daily[day] = []
        daily[day].append(r.get("score_percentage") or 0)

    result = []
    for day in sorted(daily.keys()):
        scores = daily[day]
        result.append({
            "day": day,
            "avg_score": round(sum(scores) / len(scores), 2),
            "count": len(scores),
        })
    return result


# ─────────────────────────────────────────────
# RECOMMENDATIONS
# ─────────────────────────────────────────────

@app.get("/api/recommendations")
def recommendations():
    """Auto-generated recommendations based on analysis data."""
    repo = get_repo()
    rows = repo.get_session_summary()

    # Manager stats
    mgr_stats: dict = {}
    for r in rows:
        name = r.get("operator_name") or "Неизвестно"
        if name not in mgr_stats:
            mgr_stats[name] = {"name": name, "scores": []}
        mgr_stats[name]["scores"].append(r.get("score_percentage") or 0)

    managers = []
    for m in mgr_stats.values():
        s = [x for x in m["scores"] if x is not None]
        managers.append({
            "name": m["name"],
            "avg_score": round(sum(s) / len(s), 2) if s else None,
            "excellent_count": sum(1 for x in s if x >= 90),
        })

    # Category stats across all sessions
    cat_stats: dict = {}
    for r in rows:
        sid = r.get("session_id")
        if not sid:
            continue
        try:
            kpi = repo.get_kpi_detail(sid)
            for item in kpi:
                cat = item.get("category_name", "Other")
                if cat not in cat_stats:
                    cat_stats[cat] = {"name": cat, "scores": [], "max_scores": []}
                cat_stats[cat]["scores"].append(item.get("score", 0))
                cat_stats[cat]["max_scores"].append(item.get("max_score", 0))
        except Exception:
            pass

    categories = []
    for c in cat_stats.values():
        total = sum(c["scores"])
        max_total = sum(c["max_scores"])
        categories.append({
            "name": c["name"],
            "avg_percentage": round(total / max_total * 100) if max_total else 0,
        })

    # Date range
    dates = [r.get("created_at") for r in rows if r.get("created_at")]
    date_range = {}
    if dates:
        date_range = {
            "first_date": min(dates).isoformat(),
            "last_date": max(dates).isoformat(),
            "total": len(rows),
        }

    # DB recommendations
    db_recs = repo._fetchall(
        "SELECT * FROM recommendations ORDER BY priority DESC, created_at DESC LIMIT 20")

    return {
        "managers": managers,
        "categories": categories,
        "date_range": date_range,
        "recommendations": [
            {"category": r.get("category", ""), "text": r.get("text", ""),
             "manager_name": ""}
            for r in db_recs
        ],
    }


# ─────────────────────────────────────────────
# EMOTION SUMMARY
# ─────────────────────────────────────────────

@app.get("/api/emotion-summary")
def emotion_summary():
    """Aggregated emotion distribution across all sessions for the dashboard."""
    repo = get_repo()

    # Combined emotions (video + audio fused)
    combined_rows = repo._fetchall(
        """SELECT combined_emotion, COUNT(*) AS cnt
           FROM combined_emotions
           WHERE combined_emotion IS NOT NULL AND combined_emotion <> ''
           GROUP BY combined_emotion
           ORDER BY cnt DESC"""
    )
    combined_total = sum(r["cnt"] for r in combined_rows)

    # Dominant video emotions across all interval_emotions
    video_rows = repo._fetchall(
        """SELECT dominant_emotion, COUNT(*) AS cnt
           FROM interval_emotions
           WHERE dominant_emotion IS NOT NULL AND dominant_emotion <> ''
           GROUP BY dominant_emotion
           ORDER BY cnt DESC"""
    )
    video_total = sum(r["cnt"] for r in video_rows)

    # Audio sentiment from transcript_segments
    audio_rows = repo._fetchall(
        """SELECT sentiment_label, COUNT(*) AS cnt
           FROM transcript_segments
           WHERE sentiment_label IS NOT NULL AND sentiment_label <> ''
           GROUP BY sentiment_label
           ORDER BY cnt DESC"""
    )
    audio_total = sum(r["cnt"] for r in audio_rows)

    # Avg combined valence per session (linked to manager name)
    manager_emotion = repo._fetchall(
        """SELECT o.name AS manager_name,
                  ROUND(AVG(ce.combined_valence)::numeric, 3) AS avg_valence,
                  COUNT(DISTINCT ce.session_id) AS sessions
           FROM combined_emotions ce
           JOIN session_operators so ON so.session_id = ce.session_id
           JOIN operators o ON o.operator_id = so.operator_id
           GROUP BY o.name
           ORDER BY avg_valence DESC"""
    )

    def to_pct(rows, total):
        return [
            {"label": r["dominant_emotion"] if "dominant_emotion" in r else (r.get("combined_emotion") or r.get("sentiment_label", "")),
             "count": r["cnt"],
             "pct": round(r["cnt"] / total * 100, 1) if total else 0}
            for r in rows
        ]

    combined_list = [
        {"label": r["combined_emotion"], "count": r["cnt"],
         "pct": round(r["cnt"] / combined_total * 100, 1) if combined_total else 0}
        for r in combined_rows
    ]
    video_list = [
        {"label": r["dominant_emotion"], "count": r["cnt"],
         "pct": round(r["cnt"] / video_total * 100, 1) if video_total else 0}
        for r in video_rows
    ]
    audio_list = [
        {"label": r["sentiment_label"], "count": r["cnt"],
         "pct": round(r["cnt"] / audio_total * 100, 1) if audio_total else 0}
        for r in audio_rows
    ]

    return {
        "combined": combined_list,
        "video": video_list,
        "audio": audio_list,
        "manager_valence": [
            {"name": r["manager_name"], "avg_valence": float(r["avg_valence"]) if r["avg_valence"] is not None else 0,
             "sessions": r["sessions"]}
            for r in manager_emotion
        ],
        "has_data": combined_total > 0 or video_total > 0,
    }


# ─────────────────────────────────────────────
# OPERATORS (Company Structure CRUD)
# ─────────────────────────────────────────────

class OperatorCreate(BaseModel):
    name: str
    department: Optional[str] = None
    position: Optional[str] = None
    employee_id: Optional[str] = None


@app.get("/api/operators")
def list_operators():
    """List all operators in the company structure."""
    repo = get_repo()
    rows = repo._fetchall(
        """SELECT o.operator_id, o.name, o.employee_id, o.department, o.created_at,
                  COUNT(so.session_id) AS call_count,
                  ROUND(AVG(vs.score_percentage)::numeric, 2) AS avg_score
           FROM operators o
           LEFT JOIN session_operators so ON so.operator_id = o.operator_id
           LEFT JOIN v_session_summary vs ON vs.session_id = so.session_id
           GROUP BY o.operator_id
           ORDER BY o.name""")
    return [
        {
            "id": r["operator_id"],
            "name": r["name"],
            "employee_id": r.get("employee_id"),
            "department": r.get("department", "Отдел поддержки"),
            "position": "Менеджер",
            "call_count": r.get("call_count", 0),
            "avg_score": float(r["avg_score"]) if r.get("avg_score") else None,
        }
        for r in rows
    ]


@app.get("/api/operators/{operator_id}")
def get_operator(operator_id: int):
    """Get a single operator with their call history."""
    repo = get_repo()
    op = repo._fetchone(
        "SELECT * FROM operators WHERE operator_id = %s", (operator_id,))
    if not op:
        raise HTTPException(status_code=404, detail="Operator not found")

    calls = repo._fetchall(
        """SELECT vs.*
           FROM session_operators so
           JOIN v_session_summary vs ON vs.session_id = so.session_id
           WHERE so.operator_id = %s
           ORDER BY vs.created_at DESC""",
        (operator_id,))

    return {
        "id": op["operator_id"],
        "name": op["name"],
        "employee_id": op.get("employee_id"),
        "department": op.get("department"),
        "calls": [
            {
                "session_id": c["session_id"],
                "video_id": c.get("video_id", ""),
                "score_percentage": c.get("score_percentage", 0),
                "traffic_light": c.get("traffic_light", ""),
                "contract_number": c.get("contract_number", ""),
                "created_at": c.get("created_at").isoformat() if c.get("created_at") else "",
            }
            for c in calls
        ],
    }


@app.post("/api/operators")
def create_operator(op: OperatorCreate):
    """Add a new operator to company structure."""
    repo = get_repo()
    op_id = repo.upsert_operator(op.name, op.employee_id)
    if op.department:
        repo._execute(
            "UPDATE operators SET department = %s WHERE operator_id = %s",
            (op.department, op_id))
        repo._commit()
    return {"id": op_id, "name": op.name, "department": op.department}


@app.put("/api/operators/{operator_id}")
def update_operator(operator_id: int, op: OperatorCreate):
    """Update an operator."""
    repo = get_repo()
    existing = repo._fetchone(
        "SELECT * FROM operators WHERE operator_id = %s", (operator_id,))
    if not existing:
        raise HTTPException(status_code=404, detail="Operator not found")
    repo._execute(
        "UPDATE operators SET name = %s, department = %s, employee_id = %s WHERE operator_id = %s",
        (op.name, op.department, op.employee_id, operator_id))
    repo._commit()
    return {"id": operator_id, "name": op.name, "department": op.department}


@app.delete("/api/operators/{operator_id}")
def delete_operator(operator_id: int):
    """Remove an operator."""
    repo = get_repo()
    repo._execute("DELETE FROM operators WHERE operator_id = %s", (operator_id,))
    repo._commit()
    return {"deleted": operator_id}


@app.delete("/api/analyses/{session_id}")
def delete_analysis(session_id: int):
    """Remove a session and its KPI results."""
    repo = get_repo()
    repo._execute("DELETE FROM kpi_results WHERE session_id = %s", (session_id,))
    repo._execute("DELETE FROM sessions WHERE session_id = %s", (session_id,))
    repo._commit()
    return {"deleted": session_id}


@app.post("/api/analyses/{session_id}/rerun")
def rerun_analysis(session_id: int):
    """Re-run the AI evaluation for an existing session."""
    repo = get_repo()
    session = repo._fetchone("SELECT * FROM sessions WHERE session_id = %s", (session_id,))
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    repo._execute(
        "UPDATE sessions SET pipeline_status = %s WHERE session_id = %s",
        ("pending", session_id),
    )
    repo._execute("DELETE FROM kpi_results WHERE session_id = %s", (session_id,))
    repo._commit()
    return {"session_id": session_id, "status": "pending", "video_id": session.get("video_id")}


# ─────────────────────────────────────────────
# PIPELINE STATUS
# ─────────────────────────────────────────────

@app.get("/api/pipeline-status/{video_id}")
def pipeline_status(video_id: str):
    """Check pipeline processing status for a video."""
    repo = get_repo()
    session = repo.get_session_by_video_id(video_id)
    if not session:
        for vid in [video_id, video_id.rsplit(".", 1)[0]]:
            session = repo.get_session_by_video_id(vid)
            if session:
                break
    if not session:
        return {"status": "not_found", "video_id": video_id}

    db_status = session.get("pipeline_status", "unknown")

    # Enrich with current step label from Celery task state
    step = None
    pct = None
    task_id = get_task_id(video_id)
    if task_id and db_status == "running":
        try:
            result = celery_app.AsyncResult(task_id)
            if result.state == "PROGRESS" and isinstance(result.info, dict):
                step = result.info.get("step")
                pct = result.info.get("pct")
            elif result.state == "STARTED":
                step = "Запуск пайплайна..."
                pct = 5
        except Exception:
            pass

    return {
        "status": db_status,
        "session_id": session.get("session_id"),
        "video_id": video_id,
        "step": step,
        "pct": pct,
    }


# ─────────────────────────────────────────────
# KPI STATS (legacy)
# ─────────────────────────────────────────────

@app.get("/api/stats/failed-criteria")
def failed_criteria(limit: int = 10):
    """Most frequently failed KPI criteria across all sessions."""
    repo = get_repo()
    return repo.get_most_failed_criteria(limit)


@app.get("/api/stats/score-distribution")
def score_distribution():
    """Score distribution for the chart on dashboard."""
    repo = get_repo()
    rows = repo.get_session_summary()
    buckets = {"0-20": 0, "21-40": 0, "41-60": 0, "61-80": 0, "81-100": 0}
    for r in rows:
        pct = r.get("score_percentage", 0)
        if pct <= 20:
            buckets["0-20"] += 1
        elif pct <= 40:
            buckets["21-40"] += 1
        elif pct <= 60:
            buckets["41-60"] += 1
        elif pct <= 80:
            buckets["61-80"] += 1
        else:
            buckets["81-100"] += 1
    return buckets


# ─────────────────────────────────────────────
# UPLOAD & PROCESS
# ─────────────────────────────────────────────

def _run_pipeline(video_path: str):
    """Run the full pipeline in background."""
    video_id = Path(video_path).stem
    try:
        import sys, json
        from app.run_full_pipeline import main as run_pipeline
        # main() reads video path from sys.argv
        original_argv = sys.argv
        sys.argv = ["run_full_pipeline", video_path]
        try:
            run_pipeline()
        finally:
            sys.argv = original_argv

        # After pipeline completes, read KPI JSON and update DB
        try:
            repo = get_repo()
            session = repo.get_session_by_video_id(video_id)
            if session:
                sid = session["session_id"]
                kpi_path = Path(f"data/{video_id}/kpi/kpi_evaluation.json")
                if kpi_path.exists():
                    kpi_data = json.loads(kpi_path.read_text(encoding="utf-8"))
                    score = kpi_data.get("overall_score", 0)
                    max_score = kpi_data.get("max_possible_score", 100)
                    pct = kpi_data.get("score_percentage", 0)
                    status = kpi_data.get("overall_status", "")
                    light = kpi_data.get("traffic_light", "")
                    # Update session with KPI results
                    repo._execute(
                        """UPDATE sessions SET pipeline_status = 'completed'
                           WHERE session_id = %s""", (sid,))
                    # Insert or update kpi_evaluations
                    repo._execute(
                        """INSERT INTO kpi_evaluations
                           (session_id, overall_score, max_possible_score,
                            score_percentage, overall_status, traffic_light)
                           VALUES (%s, %s, %s, %s, %s, %s)
                           ON CONFLICT (session_id) DO UPDATE SET
                            overall_score = EXCLUDED.overall_score,
                            max_possible_score = EXCLUDED.max_possible_score,
                            score_percentage = EXCLUDED.score_percentage,
                            overall_status = EXCLUDED.overall_status,
                            traffic_light = EXCLUDED.traffic_light""",
                        (sid, score, max_score, pct, status, light))
                    repo._commit()
                else:
                    repo.update_session_status(sid, "completed")
        except Exception:
            traceback.print_exc()
    except Exception:
        traceback.print_exc()
        # Mark as failed
        try:
            repo = get_repo()
            session = repo.get_session_by_video_id(video_id)
            if session:
                repo.update_session_status(session["session_id"], "failed")
        except Exception:
            pass


def _run_pipeline_with_contract(video_path: str, contract_number: str, operator_id: str):
    """Run the full pipeline and save contract number to the session."""
    try:
        from app.run_full_pipeline import main as run_pipeline
        run_pipeline(video_path)
    except Exception:
        traceback.print_exc()

    # Save contract_number to the session
    if contract_number:
        try:
            repo = get_repo()
            video_id = Path(video_path).stem
            repo._execute(
                "UPDATE sessions SET contract_number = %s WHERE video_id = %s",
                (contract_number, video_id),
            )
            repo._commit()
        except Exception:
            traceback.print_exc()


@app.post("/api/upload")
async def upload_video(
    file: UploadFile = File(...),
    contract_number: str = Form(""),
    operator_id: str = Form(""),
):
    """Upload a video file and dispatch pipeline task to Celery worker."""
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Archive to S3 in background — local copy is cleaned up by worker after processing
    _upload_to_s3_background(str(file_path), file.filename)

    video_id = Path(file.filename).stem

    # Create or update session in DB with 'running' status so polling works
    try:
        repo = get_repo()
        existing = repo.get_session_by_video_id(video_id)
        if existing:
            repo.update_session_status(existing["session_id"], "running")
            if contract_number:
                repo._execute(
                    "UPDATE sessions SET contract_number = %s WHERE video_id = %s",
                    (contract_number, video_id))
                repo._commit()
        else:
            repo._execute(
                """INSERT INTO sessions (video_id, video_path, pipeline_status,
                   fps, frame_count, width, height, duration_sec, contract_number)
                   VALUES (%s, %s, 'running', 0, 0, 0, 0, 0, %s)
                   RETURNING session_id""",
                (video_id, str(file_path), contract_number or None))
            repo._commit()
    except Exception:
        traceback.print_exc()

    # Dispatch to Celery worker
    task = process_video.delay(str(file_path))
    _store_task_id(video_id, task.id)

    return {
        "status": "processing",
        "video_id": video_id,
        "filename": file.filename,
        "task_id": task.id,
    }


# ─────────────────────────────────────────────
# YANDEX GPT CHAT
# ─────────────────────────────────────────────

YANDEX_GPT_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

SYSTEM_PROMPT = (
    "Ты — AI-ассистент системы контроля качества звонков СДЭК (CDEK Quality Control). "
    "Ты помогаешь менеджерам и супервайзерам пользоваться системой, анализировать результаты оценки звонков, "
    "объяснять критерии KPI и давать рекомендации по улучшению качества обслуживания. "
    "Отвечай кратко и по делу на русском языке.\n\n"
    "## Разделы системы и навигация (меню слева):\n\n"
    "### КОНТРОЛЬ КАЧЕСТВА\n"
    "1. **Загрузка встреч** — загрузка видеозаписей звонков для анализа. "
    "Нужно выбрать файл (mp4/webm/avi/mov/mkv), указать менеджера из выпадающего списка и нажать «Загрузить и начать анализ». "
    "Система автоматически: извлечёт аудио → распознает речь (Whisper) → разделит на реплики → оценит по KPI через AI. "
    "Прогресс обработки отображается в реальном времени с шагами пайплайна.\n"
    "2. **Аналитика** — главный дашборд. Показывает: общее количество проанализированных звонков, средний балл, "
    "распределение оценок (зелёный/жёлтый/красный), тренд оценок по дням, таблицу всех анализов с возможностью "
    "открыть детальный отчёт по каждому звонку. Есть экспорт в Excel, CSV и PDF.\n\n"
    "### ДЛЯ РУКОВОДИТЕЛЯ\n"
    "3. **Аналитика менеджеров** — сводная статистика по каждому менеджеру: средний балл, количество звонков, "
    "динамика по датам, сравнение менеджеров между собой. Можно фильтровать по отделу и периоду. Экспорт доступен.\n"
    "4. **Рекомендации** — автоматически сгенерированные рекомендации на основе анализа звонков. "
    "Разделены по приоритету (высокий/средний/низкий) и категориям. Показывают конкретные проблемы и шаги для улучшения.\n\n"
    "### УПРАВЛЕНИЕ\n"
    "5. **Структура компании** — справочник менеджеров/операторов. Можно добавлять, редактировать и удалять сотрудников. "
    "У каждого сотрудника: ФИО, отдел (Отдел поддержки / Отдел договоров), должность (Менеджер / Руководитель). "
    "По клику на менеджера открывается карточка с историей звонков и персональной статистикой.\n"
    "6. **Конструктор чек-листов** — настройка критериев KPI для оценки звонков. "
    "4 вкладки: Основные (список критериев с весами), Тюнинг (настройка порогов), Поля (какие поля извлекать из разговора), "
    "Светофор (настройка цветовых зон: зелёный/жёлтый/красный).\n"
    "7. **Настройки** — конфигурация системы: ключи API (YandexGPT, Whisper), пороги оценок, уведомления.\n\n"
    "### ИНСТРУМЕНТЫ\n"
    "8. **AI-ассистент** — это ты! Чат для вопросов по работе системы.\n"
    "9. **Журнал обработок** — таблица со всеми загруженными и обработанными файлами: имя файла, дата, менеджер, статус, оценка.\n\n"
    "## Как работает оценка звонков:\n"
    "- Видео загружается → из него извлекается аудиодорожка (FFmpeg)\n"
    "- Аудио отправляется на распознавание речи (Whisper)\n"
    "- Транскрипт разделяется на реплики менеджера и клиента\n"
    "- AI (YandexGPT) оценивает разговор по каждому критерию KPI из чек-листа\n"
    "- Каждый критерий получает оценку «Да» или «Нет» с цитатой-доказательством из разговора\n"
    "- Итоговая оценка = сумма баллов выполненных критериев / максимально возможный балл × 100%\n"
    "- Светофор: зелёный (≥80%), жёлтый (50-79%), красный (<50%)\n\n"
    "## Частые вопросы:\n"
    "- Чтобы загрузить звонок: раздел «Загрузка встреч» → выбрать файл → выбрать менеджера → нажать кнопку загрузки\n"
    "- Чтобы посмотреть результат: раздел «Аналитика» → найти звонок в таблице → нажать «Отчёт»\n"
    "- Чтобы добавить менеджера: раздел «Структура компании» → кнопка «Добавить менеджера»\n"
    "- Чтобы изменить критерии оценки: раздел «Конструктор чек-листов»\n"
    "- Чтобы сравнить менеджеров: раздел «Аналитика менеджеров»\n"
    "- Чтобы скачать отчёт: на страницах «Аналитика» или «Аналитика менеджеров» нажать кнопку экспорта (Excel/CSV/PDF)\n"
)


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    text: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    session_id: Optional[int] = None


@app.post("/api/chat")
def chat(req: ChatRequest):
    """Send messages to Yandex GPT and return the response."""
    api_key = os.environ.get("YANDEX_API_KEY")
    folder_id = os.environ.get("YANDEX_FOLDER_ID")
    if not api_key or not folder_id:
        raise HTTPException(status_code=500, detail="Yandex GPT credentials not configured")

    # Build context: if session_id provided, include KPI summary
    system_text = SYSTEM_PROMPT
    if req.session_id:
        try:
            repo = get_repo()
            summary = repo.get_session_summary(req.session_id)
            if summary:
                s = summary[0]
                kpi_detail = repo.get_kpi_detail(req.session_id)
                failed = [item for item in kpi_detail if item.get("status") == "Нет"]
                passed = [item for item in kpi_detail if item.get("status") == "Да"]
                system_text += (
                    f"\n\nКонтекст текущего анализа (session {req.session_id}):\n"
                    f"- Видео: {s.get('video_id', '?')}\n"
                    f"- Оценка: {s.get('score_percentage', 0)}% "
                    f"({s.get('overall_score', 0)}/{s.get('max_possible_score', 0)})\n"
                    f"- Статус: {s.get('traffic_light', '?')}\n"
                    f"- Выполнено: {len(passed)} критериев\n"
                    f"- Не выполнено: {len(failed)} критериев: "
                    + ", ".join(item.get("criterion_name", item.get("key", "")) for item in failed[:10])
                )
        except Exception:
            pass

    # Build Yandex GPT messages
    yandex_messages = [{"role": "system", "text": system_text}]
    for msg in req.messages:
        yandex_messages.append({"role": msg.role, "text": msg.text})

    body = {
        "modelUri": f"gpt://{folder_id}/yandexgpt/latest",
        "completionOptions": {
            "stream": False,
            "temperature": 0.3,
            "maxTokens": 1000,
        },
        "messages": yandex_messages,
    }

    resp = http_requests.post(
        YANDEX_GPT_URL,
        headers={
            "Authorization": f"Api-Key {api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=30,
    )

    if resp.status_code != 200:
        raise HTTPException(
            status_code=resp.status_code,
            detail=f"Yandex GPT error: {resp.text}",
        )

    data = resp.json()
    answer = data.get("result", {}).get("alternatives", [{}])[0].get("message", {}).get("text", "")
    return {"reply": answer}


# ─────────────────────────────────────────────
# HEALTH
# ─────────────────────────────────────────────

@app.get("/api/health")
def health():
    """Health check — also verifies DB connection."""
    try:
        repo = get_repo()
        repo._fetchone("SELECT 1 AS ok")
        return {"status": "ok", "database": "connected"}
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "error", "database": str(e)},
        )


# ─────────────────────────────────────────────
# FRONTEND (serve index.html)
# ─────────────────────────────────────────────

FRONTEND_DIR = Path(__file__).resolve().parent / "frontend"


@app.get("/")
def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")
