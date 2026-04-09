"""
CDEK Quality Control — FastAPI Backend
Serves data from PostgreSQL to the frontend dashboard.
"""

import os
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import requests as http_requests
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.backend.database.schema import get_connection
from app.backend.database.repository import PipelineRepository

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
# SESSIONS / ANALYSES
# ─────────────────────────────────────────────

@app.get("/api/analyses")
def list_analyses():
    """List all processed sessions with scores — feeds the dashboard cards."""
    repo = get_repo()
    rows = repo.get_session_summary()
    results = []
    for r in rows:
        results.append({
            "session_id": r["session_id"],
            "video_id": r.get("video_id", ""),
            "manager": r.get("operator_name") or "—",
            "date": r.get("created_at", datetime.now()).strftime("%d.%m.%Y") if r.get("created_at") else "",
            "time": r.get("created_at", datetime.now()).strftime("%H:%M") if r.get("created_at") else "",
            "score": r.get("score_percentage", 0),
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
        return {"status": "not_found", "video_id": video_id}
    return {
        "status": session.get("pipeline_status", "unknown"),
        "session_id": session.get("session_id"),
        "video_id": video_id,
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
    try:
        from app.run_full_pipeline import main as run_pipeline
        run_pipeline(video_path)
    except Exception:
        traceback.print_exc()


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
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    contract_number: str = Form(""),
    operator_id: str = Form(""),
):
    """Upload a video file and start pipeline processing."""
    upload_dir = Path("data/uploads")
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = upload_dir / file.filename
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Save contract_number to session after pipeline creates it
    background_tasks.add_task(
        _run_pipeline_with_contract, str(file_path), contract_number, operator_id
    )

    return {"status": "processing", "filename": file.filename, "path": str(file_path)}


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
