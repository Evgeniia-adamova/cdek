"""
Database schema and connection for Call Center Video Analysis Pipeline.
PostgreSQL (Yandex Managed).

Usage:
    from app.backend.database.schema import get_connection, init_db

    conn = get_connection()
    init_db(conn)

Standalone:
    python -m app.backend.database.schema [--init] [--check]

Environment variables:
    PG_HOST, PG_PORT (default 6432), PG_DATABASE, PG_USER, PG_PASSWORD,
    PG_SSLMODE (default verify-full), PG_SSLROOTCERT (default certs/root.crt)
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import psycopg2
from app.backend.calls_kpi_evaluation.kpi_analyzer import CHECKLIST_CRITERIA


# ═══════════════════════════════════════════════════════════════
# Connection
# ═══════════════════════════════════════════════════════════════

def get_connection():
    """Connect to Yandex Managed PostgreSQL. Requires PG_* env vars."""
    conn = psycopg2.connect(
        host=os.environ["PG_HOST"],
        port=int(os.environ.get("PG_PORT", "6432")),
        dbname=os.environ["PG_DATABASE"],
        user=os.environ["PG_USER"],
        password=os.environ["PG_PASSWORD"],
        sslmode=os.environ.get("PG_SSLMODE", "verify-full"),
        sslrootcert=os.environ.get(
            "PG_SSLROOTCERT",
            str(PROJECT_ROOT / "certs" / "root.crt"),
        ),
        options="-c search_path=public",
        # Keep the connection alive during long pipeline runs (multi-hour videos).
        # Without keepalives, Yandex Managed PG drops idle connections after ~10min.
        keepalives=1,
        keepalives_idle=60,
        keepalives_interval=10,
        keepalives_count=5,
    )
    conn.autocommit = False
    return conn


# ═══════════════════════════════════════════════════════════════
# PostgreSQL DDL
# ═══════════════════════════════════════════════════════════════

TABLES = [
    """
    CREATE TABLE IF NOT EXISTS sessions (
        session_id          SERIAL PRIMARY KEY,
        video_id            TEXT    NOT NULL UNIQUE,
        video_path          TEXT,
        fps                 DOUBLE PRECISION NOT NULL,
        frame_count         INTEGER NOT NULL,
        width               INTEGER NOT NULL,
        height              INTEGER NOT NULL,
        duration_sec        DOUBLE PRECISION NOT NULL,

        interval_sec        DOUBLE PRECISION NOT NULL DEFAULT 10.0,
        frames_per_interval INTEGER NOT NULL DEFAULT 6,
        max_frames          INTEGER NOT NULL DEFAULT 0,
        expected_people     INTEGER NOT NULL DEFAULT 2,
        match_threshold     DOUBLE PRECISION NOT NULL DEFAULT 0.55,
        detector            TEXT    NOT NULL DEFAULT 'yunet_2023mar',
        emotion_model       TEXT    NOT NULL DEFAULT 'onnx_ferplus',

        contract_number     TEXT,

        processed_at        TIMESTAMP NOT NULL DEFAULT NOW(),
        pipeline_status     TEXT NOT NULL DEFAULT 'pending'
            CHECK (pipeline_status IN ('pending', 'running', 'completed', 'failed'))
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS persons (
        person_id        SERIAL PRIMARY KEY,
        session_id       INTEGER NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
        person_label     TEXT    NOT NULL,
        total_samples    INTEGER DEFAULT 0,
        dominant_emotion TEXT,
        dominant_share   DOUBLE PRECISION,
        UNIQUE(session_id, person_label)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS frames (
        frame_id      SERIAL PRIMARY KEY,
        session_id    INTEGER NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
        sample_index  INTEGER NOT NULL,
        frame_index   INTEGER NOT NULL,
        timestamp_sec DOUBLE PRECISION NOT NULL,
        timestamp_fmt TEXT,
        image_path    TEXT,
        interval_id   INTEGER,
        face_count    INTEGER NOT NULL DEFAULT 0,
        UNIQUE(session_id, sample_index)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS face_detections (
        detection_id  SERIAL PRIMARY KEY,
        frame_id      INTEGER NOT NULL REFERENCES frames(frame_id) ON DELETE CASCADE,
        person_id     INTEGER NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
        face_idx      INTEGER NOT NULL DEFAULT 0,

        bbox_x        INTEGER NOT NULL,
        bbox_y        INTEGER NOT NULL,
        bbox_w        INTEGER NOT NULL,
        bbox_h        INTEGER NOT NULL,
        center_x      DOUBLE PRECISION NOT NULL,
        center_y      DOUBLE PRECISION NOT NULL,
        area          INTEGER NOT NULL,

        confidence    DOUBLE PRECISION NOT NULL,
        match_score   DOUBLE PRECISION NOT NULL DEFAULT 0.0,
        crop_path     TEXT
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS emotion_predictions (
        prediction_id SERIAL PRIMARY KEY,
        detection_id  INTEGER NOT NULL REFERENCES face_detections(detection_id) ON DELETE CASCADE,
        emotion       TEXT    NOT NULL,
        confidence    DOUBLE PRECISION NOT NULL,
        probabilities JSONB
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS interval_emotions (
        id               SERIAL PRIMARY KEY,
        session_id       INTEGER NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
        person_id        INTEGER NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
        interval_id      INTEGER NOT NULL,
        start_ts         TEXT,
        frame_count      INTEGER NOT NULL,
        total_detections INTEGER NOT NULL,

        emotion_counts   JSONB   NOT NULL,
        dominant_emotion TEXT    NOT NULL,
        dominant_share   DOUBLE PRECISION NOT NULL,
        avg_confidence   DOUBLE PRECISION NOT NULL,
        valence_score    DOUBLE PRECISION NOT NULL,

        UNIQUE(session_id, person_id, interval_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS speakers (
        speaker_id    SERIAL PRIMARY KEY,
        session_id    INTEGER NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
        speaker_label TEXT    NOT NULL,
        role          TEXT    NOT NULL,
        UNIQUE(session_id, speaker_label)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS transcript_segments (
        segment_id      SERIAL PRIMARY KEY,
        session_id      INTEGER NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
        speaker_id      INTEGER NOT NULL REFERENCES speakers(speaker_id) ON DELETE CASCADE,
        start_sec       DOUBLE PRECISION NOT NULL,
        end_sec         DOUBLE PRECISION NOT NULL,
        text            TEXT    NOT NULL,
        sentiment_label TEXT    NOT NULL DEFAULT 'neutral'
            CHECK (sentiment_label IN ('positive', 'neutral', 'negative')),
        sentiment_score DOUBLE PRECISION NOT NULL DEFAULT 0.0,
        interval_id     INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS combined_emotions (
        id                  SERIAL PRIMARY KEY,
        session_id          INTEGER NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
        person_id           INTEGER NOT NULL REFERENCES persons(person_id) ON DELETE CASCADE,
        interval_id         INTEGER NOT NULL,

        video_emotion       TEXT    NOT NULL,
        video_valence       DOUBLE PRECISION NOT NULL,
        video_confidence    DOUBLE PRECISION NOT NULL,

        audio_sentiment     TEXT    NOT NULL,
        audio_valence       DOUBLE PRECISION NOT NULL,
        audio_confidence    DOUBLE PRECISION NOT NULL,

        combined_emotion    TEXT    NOT NULL
            CHECK (combined_emotion IN ('very_positive', 'positive', 'neutral', 'negative', 'very_negative')),
        combined_valence    DOUBLE PRECISION NOT NULL,
        combined_confidence DOUBLE PRECISION NOT NULL,

        video_weight        DOUBLE PRECISION NOT NULL DEFAULT 0.6,
        audio_weight        DOUBLE PRECISION NOT NULL DEFAULT 0.4,

        UNIQUE(session_id, person_id, interval_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS kpi_categories (
        category_id   SERIAL PRIMARY KEY,
        name          TEXT    NOT NULL UNIQUE,
        display_order INTEGER NOT NULL DEFAULT 0,
        max_score     INTEGER NOT NULL
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS kpi_criteria (
        criterion_id  SERIAL PRIMARY KEY,
        key           TEXT    NOT NULL UNIQUE,
        name          TEXT    NOT NULL,
        category_id   INTEGER NOT NULL REFERENCES kpi_categories(category_id),
        max_score     INTEGER NOT NULL,
        description   TEXT,
        display_order INTEGER NOT NULL DEFAULT 0
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS kpi_evaluations (
        evaluation_id      SERIAL PRIMARY KEY,
        session_id         INTEGER NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
        checklist_name     TEXT    NOT NULL DEFAULT 'Пример для СДЭК',
        evaluated_at       TIMESTAMP NOT NULL DEFAULT NOW(),

        overall_score      INTEGER NOT NULL,
        max_possible_score INTEGER NOT NULL DEFAULT 100,
        score_percentage   DOUBLE PRECISION NOT NULL,
        overall_status     TEXT    NOT NULL,
        traffic_light      TEXT    NOT NULL
            CHECK (traffic_light IN ('red', 'orange', 'yellow', 'green')),

        summary          TEXT,
        detailed_summary TEXT,
        next_step        TEXT,
        positive_aspects TEXT,
        negative_aspects TEXT,

        UNIQUE(session_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS kpi_evaluation_items (
        item_id       SERIAL PRIMARY KEY,
        evaluation_id INTEGER NOT NULL REFERENCES kpi_evaluations(evaluation_id) ON DELETE CASCADE,
        criterion_id  INTEGER NOT NULL REFERENCES kpi_criteria(criterion_id),
        score         INTEGER NOT NULL,
        status        TEXT    NOT NULL CHECK (status IN ('Да', 'Нет')),
        details       TEXT,
        UNIQUE(evaluation_id, criterion_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS kpi_category_scores (
        id            SERIAL PRIMARY KEY,
        evaluation_id INTEGER NOT NULL REFERENCES kpi_evaluations(evaluation_id) ON DELETE CASCADE,
        category_id   INTEGER NOT NULL REFERENCES kpi_categories(category_id),
        total_score   INTEGER NOT NULL,
        max_score     INTEGER NOT NULL,
        percentage    DOUBLE PRECISION NOT NULL,
        UNIQUE(evaluation_id, category_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS operators (
        operator_id SERIAL PRIMARY KEY,
        name        TEXT NOT NULL,
        employee_id TEXT UNIQUE,
        department  TEXT,
        created_at  TIMESTAMP NOT NULL DEFAULT NOW()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS session_operators (
        session_id  INTEGER NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
        operator_id INTEGER NOT NULL REFERENCES operators(operator_id),
        role        TEXT    NOT NULL DEFAULT 'company_representative',
        PRIMARY KEY (session_id, operator_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS recommendations (
        recommendation_id SERIAL PRIMARY KEY,
        session_id        INTEGER NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
        category          TEXT,
        priority          INTEGER NOT NULL DEFAULT 0,
        text              TEXT    NOT NULL,
        created_at        TIMESTAMP NOT NULL DEFAULT NOW()
    )
    """,
]

INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_sessions_video_id        ON sessions(video_id)",
    "CREATE INDEX IF NOT EXISTS idx_sessions_processed_at     ON sessions(processed_at)",
    "CREATE INDEX IF NOT EXISTS idx_persons_session           ON persons(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_frames_session_interval   ON frames(session_id, interval_id)",
    "CREATE INDEX IF NOT EXISTS idx_frames_session_ts         ON frames(session_id, timestamp_sec)",
    "CREATE INDEX IF NOT EXISTS idx_detections_frame          ON face_detections(frame_id)",
    "CREATE INDEX IF NOT EXISTS idx_detections_person         ON face_detections(person_id)",
    "CREATE INDEX IF NOT EXISTS idx_emotion_pred_detection    ON emotion_predictions(detection_id)",
    "CREATE INDEX IF NOT EXISTS idx_interval_emo_session      ON interval_emotions(session_id, interval_id)",
    "CREATE INDEX IF NOT EXISTS idx_interval_emo_person       ON interval_emotions(person_id, interval_id)",
    "CREATE INDEX IF NOT EXISTS idx_speakers_session          ON speakers(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_transcript_session        ON transcript_segments(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_transcript_speaker        ON transcript_segments(speaker_id)",
    "CREATE INDEX IF NOT EXISTS idx_transcript_interval       ON transcript_segments(session_id, interval_id)",
    "CREATE INDEX IF NOT EXISTS idx_transcript_time           ON transcript_segments(session_id, start_sec)",
    "CREATE INDEX IF NOT EXISTS idx_combined_session          ON combined_emotions(session_id, interval_id)",
    "CREATE INDEX IF NOT EXISTS idx_combined_person           ON combined_emotions(person_id, interval_id)",
    "CREATE INDEX IF NOT EXISTS idx_combined_emotion          ON combined_emotions(session_id, combined_emotion)",
    "CREATE INDEX IF NOT EXISTS idx_kpi_criteria_category     ON kpi_criteria(category_id)",
    "CREATE INDEX IF NOT EXISTS idx_kpi_eval_session          ON kpi_evaluations(session_id)",
    "CREATE INDEX IF NOT EXISTS idx_kpi_eval_score            ON kpi_evaluations(score_percentage)",
    "CREATE INDEX IF NOT EXISTS idx_kpi_eval_traffic          ON kpi_evaluations(traffic_light)",
    "CREATE INDEX IF NOT EXISTS idx_kpi_items_eval            ON kpi_evaluation_items(evaluation_id)",
    "CREATE INDEX IF NOT EXISTS idx_kpi_items_criterion       ON kpi_evaluation_items(criterion_id)",
    "CREATE INDEX IF NOT EXISTS idx_kpi_cat_scores_eval       ON kpi_category_scores(evaluation_id)",
    "CREATE INDEX IF NOT EXISTS idx_recommendations_session   ON recommendations(session_id)",
]

VIEWS = [
    "DROP VIEW IF EXISTS v_session_summary",
    """
    CREATE VIEW v_session_summary AS
    SELECT
        s.session_id, s.video_id, s.duration_sec, s.processed_at, s.pipeline_status, s.contract_number,
        ke.overall_score, ke.max_possible_score, ke.score_percentage, ke.traffic_light, ke.overall_status,
        o.name AS operator_name, o.operator_id,
        (SELECT COUNT(*) FROM persons p WHERE p.session_id = s.session_id) AS person_count,
        (SELECT COUNT(*) FROM transcript_segments ts WHERE ts.session_id = s.session_id) AS segment_count
    FROM sessions s
    LEFT JOIN kpi_evaluations ke ON ke.session_id = s.session_id
    LEFT JOIN session_operators so ON so.session_id = s.session_id AND so.role = 'manager'
    LEFT JOIN operators o ON o.operator_id = so.operator_id
    """,
    """
    CREATE OR REPLACE VIEW v_emotion_timeline AS
    SELECT
        ie.session_id, s.video_id, p.person_label, ie.interval_id, ie.start_ts,
        ie.dominant_emotion, ie.valence_score, ie.avg_confidence,
        ce.combined_emotion, ce.combined_valence, ce.audio_sentiment
    FROM interval_emotions ie
    JOIN sessions s ON s.session_id = ie.session_id
    JOIN persons p ON p.person_id = ie.person_id
    LEFT JOIN combined_emotions ce
        ON ce.session_id = ie.session_id
        AND ce.person_id = ie.person_id
        AND ce.interval_id = ie.interval_id
    """,
    """
    CREATE OR REPLACE VIEW v_kpi_detail AS
    SELECT
        ke.session_id, s.video_id,
        kc.name AS category_name, kr.key AS criterion_key, kr.name AS criterion_name,
        ki.score, kr.max_score, ki.status, ki.details,
        ke.score_percentage AS session_score_pct, ke.traffic_light
    FROM kpi_evaluation_items ki
    JOIN kpi_evaluations ke ON ke.evaluation_id = ki.evaluation_id
    JOIN kpi_criteria kr ON kr.criterion_id = ki.criterion_id
    JOIN kpi_categories kc ON kc.category_id = kr.category_id
    JOIN sessions s ON s.session_id = ke.session_id
    ORDER BY kc.display_order, kr.display_order
    """,
]


# ═══════════════════════════════════════════════════════════════
# Seed Data
# ═══════════════════════════════════════════════════════════════

_CATEGORY_ORDER = [
    "Установление контакта",
    "Управление встречей",
    "Сбор информации",
    "Демонстрация услуг",
    "Финализация",
]


def _build_category_seeds():
    cat_scores = {}
    for c in CHECKLIST_CRITERIA:
        cat_scores[c["category"]] = cat_scores.get(c["category"], 0) + c["max_score"]
    return [
        (name, idx + 1, cat_scores.get(name, 0))
        for idx, name in enumerate(_CATEGORY_ORDER)
    ]


def _build_criteria_seeds(cat_name_to_id: dict):
    cat_order_counter = {}
    seeds = []
    for c in CHECKLIST_CRITERIA:
        cat_id = cat_name_to_id[c["category"]]
        cat_order_counter[cat_id] = cat_order_counter.get(cat_id, 0) + 1
        seeds.append((
            c["key"], c["name"], cat_id, c["max_score"],
            c.get("description", ""), cat_order_counter[cat_id],
        ))
    return seeds


# ═══════════════════════════════════════════════════════════════
# Init
# ═══════════════════════════════════════════════════════════════

def init_db(conn) -> None:
    """
    Initialize database: create tables, indexes, views, seed KPI data.
    Works with psycopg2 connection (PostgreSQL).
    """
    cur = conn.cursor()

    for ddl in TABLES:
        cur.execute(ddl)

    # Migrations: add columns that may not exist in older databases
    _migrations = [
        "ALTER TABLE sessions ADD COLUMN IF NOT EXISTS contract_number TEXT",
    ]
    for m in _migrations:
        try:
            cur.execute(m)
        except Exception:
            conn.rollback()

    for ddl in INDEXES:
        cur.execute(ddl)
    for ddl in VIEWS:
        cur.execute(ddl)

    # Seed KPI categories + criteria (skip if already seeded)
    cur.execute("SELECT COUNT(*) FROM kpi_categories")
    if cur.fetchone()[0] == 0:
        for name, order, max_score in _build_category_seeds():
            cur.execute(
                "INSERT INTO kpi_categories (name, display_order, max_score) VALUES (%s, %s, %s)",
                (name, order, max_score),
            )
        cur.execute("SELECT category_id, name FROM kpi_categories")
        cat_name_to_id = {row[1]: row[0] for row in cur.fetchall()}

        for key, name, cat_id, max_score, desc, order in _build_criteria_seeds(cat_name_to_id):
            cur.execute(
                "INSERT INTO kpi_criteria (key, name, category_id, max_score, description, display_order) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (key, name, cat_id, max_score, desc, order),
            )

    conn.commit()


# ═══════════════════════════════════════════════════════════════
# Standalone CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Initialize pipeline database.")
    parser.add_argument("--init", action="store_true", help="Create tables and seed data")
    parser.add_argument("--check", action="store_true", help="Show table counts")
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
    except ImportError:
        pass

    conn = get_connection()
    print(f"Connected: {os.environ['PG_HOST']}:{os.environ.get('PG_PORT', '6432')}")

    if args.init:
        print("Initializing schema...")
        init_db(conn)
        print("Schema initialized.")

    if args.check or args.init:
        cur = conn.cursor()
        cur.execute("""
            SELECT table_name FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """)
        tables = [r[0] for r in cur.fetchall()]
        print(f"\nTables ({len(tables)}): {', '.join(tables)}")
        for table in tables:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            cnt = cur.fetchone()[0]
            if cnt > 0:
                print(f"  {table}: {cnt} rows")

    conn.close()
    print("Done.")
