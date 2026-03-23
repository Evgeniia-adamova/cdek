"""
REST API server for the CDEK Call Analysis Dashboard.
Connects frontend to Yandex Managed PostgreSQL.

Usage:
    python -m app.api          # default port 5000
    python -m app.api --port 8080
"""

import os
import sys
import threading
import subprocess
from pathlib import Path
from datetime import datetime, date
from decimal import Decimal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from app.backend.database.schema import get_connection
from app.backend.database.repository import PipelineRepository


app = Flask(__name__, static_folder=None)
CORS(app)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _get_repo():
    conn = get_connection()
    return PipelineRepository(conn)


def _serialize(obj):
    """Make query results JSON-serializable."""
    if isinstance(obj, dict):
        return {k: _serialize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize(v) for v in obj]
    if isinstance(obj, Decimal):
        return float(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    return obj


# ─────────────────────────────────────────────
# Frontend static files
# ─────────────────────────────────────────────

FRONTEND_DIR = PROJECT_ROOT / "app" / "frontend"


@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/<path:path>")
def static_files(path):
    return send_from_directory(str(FRONTEND_DIR), path)


# ─────────────────────────────────────────────
# API: Dashboard stats
# ─────────────────────────────────────────────

@app.route("/api/dashboard")
def api_dashboard():
    """Aggregate stats for the main dashboard."""
    repo = _get_repo()
    try:
        summaries = repo.get_session_summary()

        total = len(summaries)
        scores = [s["score_percentage"] for s in summaries if s.get("score_percentage") is not None]
        avg_score = round(sum(scores) / len(scores), 2) if scores else 0
        min_score = round(min(scores), 2) if scores else 0
        max_score = round(max(scores), 2) if scores else 0

        # Manager count
        managers = repo._fetchall("""
            SELECT DISTINCT o.name
            FROM operators o
            JOIN session_operators so ON so.operator_id = o.operator_id
        """)
        manager_count = len(managers)

        # Total unique persons (from persons table)
        persons = repo._fetchall("SELECT COUNT(DISTINCT person_label) AS cnt FROM persons")
        person_count = persons[0]["cnt"] if persons else 0

        return jsonify(_serialize({
            "total_analyses": total,
            "avg_score": avg_score,
            "min_score": min_score,
            "max_score": max_score,
            "manager_count": manager_count,
            "person_count": person_count,
        }))
    finally:
        repo.conn.close()


# ─────────────────────────────────────────────
# API: Analyses list (for cards)
# ─────────────────────────────────────────────

@app.route("/api/analyses")
def api_analyses():
    """List all analyses with scores, managers, dates — for the card grid."""
    repo = _get_repo()
    try:
        rows = repo._fetchall("""
            SELECT
                s.session_id,
                s.video_id,
                s.processed_at,
                s.duration_sec,
                s.pipeline_status,
                ke.overall_score,
                ke.score_percentage,
                ke.overall_status,
                ke.traffic_light,
                ke.checklist_name,
                o.name AS manager_name
            FROM sessions s
            LEFT JOIN kpi_evaluations ke ON ke.session_id = s.session_id
            LEFT JOIN session_operators so ON so.session_id = s.session_id
            LEFT JOIN operators o ON o.operator_id = so.operator_id
            ORDER BY s.processed_at DESC
        """)
        return jsonify(_serialize(rows))
    finally:
        repo.conn.close()


# ─────────────────────────────────────────────
# API: Manager statistics
# ─────────────────────────────────────────────

@app.route("/api/managers")
def api_managers():
    """Per-manager aggregated stats for the ranking table."""
    repo = _get_repo()
    try:
        rows = repo._fetchall("""
            SELECT
                o.name,
                COUNT(DISTINCT s.session_id) AS call_count,
                ROUND(AVG(ke.score_percentage)::numeric, 1) AS avg_score,
                MIN(ke.score_percentage) AS min_score,
                MAX(ke.score_percentage) AS max_score
            FROM operators o
            JOIN session_operators so ON so.operator_id = o.operator_id
            JOIN sessions s ON s.session_id = so.session_id
            LEFT JOIN kpi_evaluations ke ON ke.session_id = s.session_id
            GROUP BY o.operator_id, o.name
            ORDER BY avg_score DESC NULLS LAST
        """)
        return jsonify(_serialize(rows))
    finally:
        repo.conn.close()


# ─────────────────────────────────────────────
# API: Session detail (KPI + transcript)
# ─────────────────────────────────────────────

@app.route("/api/sessions/<int:session_id>")
def api_session_detail(session_id):
    """Full detail for a single session."""
    repo = _get_repo()
    try:
        summary = repo.get_session_summary(session_id)
        kpi = repo.get_kpi_detail(session_id)
        transcript = repo.get_transcript(session_id)

        return jsonify(_serialize({
            "summary": summary[0] if summary else None,
            "kpi_items": kpi,
            "transcript": transcript,
        }))
    finally:
        repo.conn.close()


# ─────────────────────────────────────────────
# API: Score dynamics (for line chart)
# ─────────────────────────────────────────────

@app.route("/api/score-dynamics")
def api_score_dynamics():
    """Daily average scores for the trend chart."""
    repo = _get_repo()
    try:
        rows = repo._fetchall("""
            SELECT
                DATE(s.processed_at) AS day,
                ROUND(AVG(ke.score_percentage)::numeric, 1) AS avg_score,
                COUNT(*) AS count
            FROM sessions s
            JOIN kpi_evaluations ke ON ke.session_id = s.session_id
            GROUP BY DATE(s.processed_at)
            ORDER BY day
        """)
        return jsonify(_serialize(rows))
    finally:
        repo.conn.close()


# ─────────────────────────────────────────────
# API: Recommendations
# ─────────────────────────────────────────────

@app.route("/api/recommendations")
def api_recommendations():
    """Full recommendations page data: managers, category scores, failed criteria, DB recommendations."""
    repo = _get_repo()
    try:
        # Per-manager stats
        managers = repo._fetchall("""
            SELECT
                o.name,
                COUNT(DISTINCT s.session_id) AS call_count,
                ROUND(AVG(ke.score_percentage)::numeric, 1) AS avg_score,
                MIN(ke.score_percentage) AS min_score,
                MAX(ke.score_percentage) AS max_score,
                SUM(CASE WHEN ke.score_percentage >= 90 THEN 1 ELSE 0 END) AS excellent_count,
                SUM(CASE WHEN ke.score_percentage < 65 THEN 1 ELSE 0 END) AS low_count
            FROM operators o
            JOIN session_operators so ON so.operator_id = o.operator_id
            JOIN sessions s ON s.session_id = so.session_id
            LEFT JOIN kpi_evaluations ke ON ke.session_id = s.session_id
            GROUP BY o.operator_id, o.name
            ORDER BY avg_score DESC NULLS LAST
        """)

        # Category scores aggregated across all evaluations
        categories = repo._fetchall("""
            SELECT
                kc.name,
                ROUND(AVG(cs.percentage)::numeric, 1) AS avg_percentage,
                kc.display_order
            FROM kpi_category_scores cs
            JOIN kpi_categories kc ON kc.category_id = cs.category_id
            GROUP BY kc.category_id, kc.name, kc.display_order
            ORDER BY avg_percentage ASC
        """)

        # Most failed criteria
        failed = repo.get_most_failed_criteria(limit=10)

        # Session date range
        date_range = repo._fetchone("""
            SELECT
                MIN(s.processed_at) AS first_date,
                MAX(s.processed_at) AS last_date,
                COUNT(*) AS total
            FROM sessions s
            JOIN kpi_evaluations ke ON ke.session_id = s.session_id
        """)

        # Recommendations from DB (if any)
        db_recs = repo._fetchall("""
            SELECT r.category, r.priority, r.text, o.name AS manager_name
            FROM recommendations r
            JOIN sessions s ON s.session_id = r.session_id
            LEFT JOIN session_operators so ON so.session_id = s.session_id
            LEFT JOIN operators o ON o.operator_id = so.operator_id
            ORDER BY r.priority DESC, r.created_at DESC
            LIMIT 20
        """)

        return jsonify(_serialize({
            "managers": managers,
            "categories": categories,
            "failed_criteria": failed,
            "date_range": date_range,
            "recommendations": db_recs,
        }))
    finally:
        repo.conn.close()


# ─────────────────────────────────────────────
# API: Operators CRUD (Company Structure)
# ─────────────────────────────────────────────

@app.route("/api/operators")
def api_operators_list():
    """List all operators with call stats."""
    repo = _get_repo()
    try:
        rows = repo._fetchall("""
            SELECT
                o.operator_id,
                o.name,
                o.employee_id AS position,
                o.department,
                o.created_at,
                COUNT(DISTINCT so.session_id) AS call_count,
                ROUND(AVG(ke.score_percentage)::numeric, 1) AS avg_score
            FROM operators o
            LEFT JOIN session_operators so ON so.operator_id = o.operator_id
            LEFT JOIN kpi_evaluations ke ON ke.session_id = so.session_id
            GROUP BY o.operator_id, o.name, o.employee_id, o.department, o.created_at
            ORDER BY o.name
        """)
        return jsonify(_serialize(rows))
    finally:
        repo.conn.close()


@app.route("/api/operators", methods=["POST"])
def api_operators_create():
    """Add a new operator/manager."""
    data = request.get_json()
    if not data or not data.get("name", "").strip():
        return jsonify({"error": "Имя обязательно"}), 400

    repo = _get_repo()
    try:
        op_id = repo.upsert_operator(
            name=data["name"].strip(),
            employee_id=data.get("position"),
            department=data.get("department"),
        )
        return jsonify({"operator_id": op_id}), 201
    finally:
        repo.conn.close()


@app.route("/api/operators/<int:operator_id>")
def api_operator_detail(operator_id):
    """Detailed info about a single operator with their call history."""
    repo = _get_repo()
    try:
        operator = repo._fetchone(
            "SELECT operator_id, name, employee_id AS position, department, created_at FROM operators WHERE operator_id = %s",
            (operator_id,),
        )
        if not operator:
            return jsonify({"error": "Менеджер не найден"}), 404

        # Call history with scores
        calls = repo._fetchall("""
            SELECT
                s.session_id,
                s.video_id,
                s.processed_at,
                s.duration_sec,
                s.pipeline_status,
                ke.overall_score,
                ke.score_percentage,
                ke.overall_status,
                ke.traffic_light,
                ke.summary
            FROM session_operators so
            JOIN sessions s ON s.session_id = so.session_id
            LEFT JOIN kpi_evaluations ke ON ke.session_id = s.session_id
            WHERE so.operator_id = %s
            ORDER BY s.processed_at DESC
        """, (operator_id,))

        # Stats
        scores = [c["score_percentage"] for c in calls if c.get("score_percentage") is not None]
        stats = {
            "call_count": len(calls),
            "avg_score": round(sum(scores) / len(scores), 1) if scores else None,
            "min_score": round(min(scores), 1) if scores else None,
            "max_score": round(max(scores), 1) if scores else None,
        }

        # Worst criteria for this operator
        failed_criteria = repo._fetchall("""
            SELECT kr.name, kc.name AS category, COUNT(*) AS fail_count,
                   ROUND(AVG(ki.score)::numeric, 1) AS avg_score, kr.max_score
            FROM session_operators so
            JOIN kpi_evaluations ke ON ke.session_id = so.session_id
            JOIN kpi_evaluation_items ki ON ki.evaluation_id = ke.evaluation_id
            JOIN kpi_criteria kr ON kr.criterion_id = ki.criterion_id
            JOIN kpi_categories kc ON kc.category_id = kr.category_id
            WHERE so.operator_id = %s AND ki.status = 'Нет'
            GROUP BY kr.criterion_id, kr.name, kc.name, kr.max_score
            ORDER BY fail_count DESC
            LIMIT 8
        """, (operator_id,))

        # Recommendations from DB for this operator's sessions
        recs = repo._fetchall("""
            SELECT DISTINCT r.category, r.text, r.priority
            FROM recommendations r
            JOIN session_operators so ON so.session_id = r.session_id
            WHERE so.operator_id = %s
            ORDER BY r.priority DESC
            LIMIT 10
        """, (operator_id,))

        # Category scores for this operator
        cat_scores = repo._fetchall("""
            SELECT kc.name, ROUND(AVG(cs.percentage)::numeric, 1) AS avg_percentage
            FROM session_operators so
            JOIN kpi_evaluations ke ON ke.session_id = so.session_id
            JOIN kpi_category_scores cs ON cs.evaluation_id = ke.evaluation_id
            JOIN kpi_categories kc ON kc.category_id = cs.category_id
            WHERE so.operator_id = %s
            GROUP BY kc.category_id, kc.name
            ORDER BY avg_percentage ASC
        """, (operator_id,))

        return jsonify(_serialize({
            "operator": operator,
            "calls": calls,
            "stats": stats,
            "failed_criteria": failed_criteria,
            "recommendations": recs,
            "category_scores": cat_scores,
        }))
    finally:
        repo.conn.close()


@app.route("/api/operators/<int:operator_id>", methods=["PUT"])
def api_operators_update(operator_id):
    """Update operator details."""
    data = request.get_json()
    if not data or not data.get("name", "").strip():
        return jsonify({"error": "Имя обязательно"}), 400

    repo = _get_repo()
    try:
        repo._execute(
            "UPDATE operators SET name = %s, department = %s, employee_id = %s WHERE operator_id = %s",
            (data["name"].strip(), data.get("department"), data.get("position"), operator_id),
        )
        repo._commit()
        return jsonify({"ok": True})
    finally:
        repo.conn.close()


@app.route("/api/operators/<int:operator_id>", methods=["DELETE"])
def api_operators_delete(operator_id):
    """Delete an operator (only if no linked sessions)."""
    repo = _get_repo()
    try:
        linked = repo._fetchone(
            "SELECT COUNT(*) AS cnt FROM session_operators WHERE operator_id = %s",
            (operator_id,),
        )
        if linked and linked["cnt"] > 0:
            return jsonify({"error": f"Нельзя удалить — у менеджера есть {linked['cnt']} привязанных звонков"}), 409

        repo._execute("DELETE FROM operators WHERE operator_id = %s", (operator_id,))
        repo._commit()
        return jsonify({"ok": True})
    finally:
        repo.conn.close()


# ─────────────────────────────────────────────
# API: Upload & Run Pipeline
# ─────────────────────────────────────────────

UPLOAD_DIR = PROJECT_ROOT / "data" / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Track pipeline status per video_id
pipeline_status = {}


def _run_pipeline_background(video_path: str, video_id: str, operator_id: int = None):
    """Run the full pipeline in background and link to operator."""
    pipeline_status[video_id] = {"status": "running", "progress": 0}
    try:
        # Run pipeline as subprocess
        result = subprocess.run(
            [sys.executable, "-m", "app.backend.run_full_pipeline", video_path],
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=600,
        )
        if result.returncode != 0:
            pipeline_status[video_id] = {"status": "failed", "error": result.stderr[-500:] if result.stderr else "Unknown error"}
            return

        # Link operator to session
        if operator_id:
            try:
                repo = _get_repo()
                session = repo.get_session_by_video_id(video_id)
                if session:
                    repo.link_session_operator(session["session_id"], operator_id)
                repo.conn.close()
            except Exception as e:
                print(f"Warning: could not link operator: {e}")

        pipeline_status[video_id] = {"status": "completed"}
    except subprocess.TimeoutExpired:
        pipeline_status[video_id] = {"status": "failed", "error": "Превышено время ожидания (10 мин)"}
    except Exception as e:
        pipeline_status[video_id] = {"status": "failed", "error": str(e)}


@app.route("/api/upload", methods=["POST"])
def api_upload():
    """Upload a media file and start pipeline analysis."""
    if "file" not in request.files:
        return jsonify({"error": "Файл не прикреплён"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Пустое имя файла"}), 400

    operator_id = request.form.get("operator_id", type=int)

    # Save file
    safe_name = file.filename.replace(" ", "_")
    video_path = str(UPLOAD_DIR / safe_name)
    file.save(video_path)

    video_id = Path(safe_name).stem

    # Start pipeline in background thread
    thread = threading.Thread(
        target=_run_pipeline_background,
        args=(video_path, video_id, operator_id),
        daemon=True,
    )
    thread.start()

    return jsonify({"video_id": video_id, "status": "started"}), 202


@app.route("/api/pipeline-status/<video_id>")
def api_pipeline_status(video_id):
    """Check pipeline status for a video."""
    status = pipeline_status.get(video_id, {"status": "unknown"})
    return jsonify(status)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    print(f"Starting API server on http://{args.host}:{args.port}")
    print(f"Frontend: http://localhost:{args.port}/")
    app.run(host=args.host, port=args.port, debug=True)
