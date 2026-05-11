"""
Repository module — insert/query functions for the pipeline database (PostgreSQL).

Usage:
    from app.backend.database.schema import get_connection
    from app.backend.database.repository import PipelineRepository

    conn = get_connection()
    repo = PipelineRepository(conn)
"""

import json
from typing import Any, Dict, List, Optional


class PipelineRepository:
    """Data access layer for PostgreSQL (Yandex Managed)."""

    def __init__(self, conn):
        self.conn = conn

    def _execute(self, sql: str, params=None):
        cur = self.conn.cursor()
        cur.execute(sql, params)
        return cur

    def _fetchone(self, sql: str, params=None) -> Optional[dict]:
        cur = self._execute(sql, params)
        row = cur.fetchone()
        if row is None:
            return None
        cols = [desc[0] for desc in cur.description]
        return dict(zip(cols, row))

    def _fetchall(self, sql: str, params=None) -> List[dict]:
        cur = self._execute(sql, params)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        return [dict(zip(cols, row)) for row in rows]

    def _commit(self):
        self.conn.commit()

    # ─────────────────────────────────────────────
    # SESSIONS
    # ─────────────────────────────────────────────

    def create_session(
        self,
        video_id: str,
        fps: float,
        frame_count: int,
        width: int,
        height: int,
        duration_sec: float,
        video_path: str = None,
        interval_sec: float = 10.0,
        frames_per_interval: int = 6,
        max_frames: int = 0,
        expected_people: int = 2,
        match_threshold: float = 0.55,
        detector: str = "yunet_2023mar",
        emotion_model: str = "onnx_ferplus",
    ) -> int:
        cur = self._execute(
            """INSERT INTO sessions
               (video_id, video_path, fps, frame_count, width, height, duration_sec,
                interval_sec, frames_per_interval, max_frames, expected_people,
                match_threshold, detector, emotion_model, pipeline_status)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, 'running')
               RETURNING session_id""",
            (video_id, video_path, fps, frame_count, width, height, duration_sec,
             interval_sec, frames_per_interval, max_frames, expected_people,
             match_threshold, detector, emotion_model),
        )
        session_id = cur.fetchone()[0]
        self._commit()
        return session_id

    def update_session_status(self, session_id: int, status: str) -> None:
        self._execute(
            "UPDATE sessions SET pipeline_status = %s WHERE session_id = %s",
            (status, session_id),
        )
        self._commit()

    def get_session_by_video_id(self, video_id: str) -> Optional[dict]:
        return self._fetchone("SELECT * FROM sessions WHERE video_id = %s", (video_id,))

    # ─────────────────────────────────────────────
    # PERSONS
    # ─────────────────────────────────────────────

    def insert_persons(self, session_id: int, person_labels: List[str]) -> Dict[str, int]:
        label_to_id = {}
        for label in person_labels:
            cur = self._execute(
                """INSERT INTO persons (session_id, person_label)
                   VALUES (%s, %s)
                   ON CONFLICT (session_id, person_label) DO NOTHING
                   RETURNING person_id""",
                (session_id, label),
            )
            row = cur.fetchone()
            if row:
                label_to_id[label] = row[0]
            else:
                r = self._fetchone(
                    "SELECT person_id FROM persons WHERE session_id = %s AND person_label = %s",
                    (session_id, label),
                )
                label_to_id[label] = r["person_id"]
        self._commit()
        return label_to_id

    def update_person_stats(
        self, person_id: int, total_samples: int,
        dominant_emotion: str = None, dominant_share: float = None,
    ) -> None:
        self._execute(
            "UPDATE persons SET total_samples = %s, dominant_emotion = %s, dominant_share = %s "
            "WHERE person_id = %s",
            (total_samples, dominant_emotion, dominant_share, person_id),
        )
        self._commit()

    def get_person_ids(self, session_id: int) -> Dict[str, int]:
        rows = self._fetchall(
            "SELECT person_label, person_id FROM persons WHERE session_id = %s",
            (session_id,),
        )
        return {r["person_label"]: r["person_id"] for r in rows}

    # ─────────────────────────────────────────────
    # FRAMES
    # ─────────────────────────────────────────────

    def insert_frames(self, session_id: int, frames: List[dict]) -> Dict[int, int]:
        sample_to_id = {}
        for f in frames:
            cur = self._execute(
                """INSERT INTO frames
                   (session_id, sample_index, frame_index, timestamp_sec,
                    timestamp_fmt, image_path, interval_id, face_count)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING frame_id""",
                (session_id, f["sample_index"], f["frame_index"],
                 f["timestamp_sec"], f.get("timestamp"), f.get("image_path"),
                 f.get("interval_id"), f.get("face_count", 0)),
            )
            sample_to_id[f["sample_index"]] = cur.fetchone()[0]
        self._commit()
        return sample_to_id

    # ─────────────────────────────────────────────
    # FACE DETECTIONS
    # ─────────────────────────────────────────────

    def insert_detections(
        self, frame_id: int, person_label_to_id: Dict[str, int], detections: List[dict],
    ) -> List[int]:
        ids = []
        for d in detections:
            person_id = person_label_to_id.get(d.get("person_id"))
            if person_id is None:
                continue
            bbox = d.get("bbox", {})
            center = d.get("center", {})
            cur = self._execute(
                """INSERT INTO face_detections
                   (frame_id, person_id, face_idx,
                    bbox_x, bbox_y, bbox_w, bbox_h,
                    center_x, center_y, area,
                    confidence, match_score, crop_path)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   RETURNING detection_id""",
                (frame_id, person_id, d.get("face_idx", 0),
                 bbox.get("x", 0), bbox.get("y", 0),
                 bbox.get("w", 0), bbox.get("h", 0),
                 center.get("x", 0.0), center.get("y", 0.0),
                 d.get("area", 0),
                 d.get("confidence", 0.0), d.get("match_score", 0.0),
                 d.get("crop_path")),
            )
            ids.append(cur.fetchone()[0])
        self._commit()
        return ids

    # ─────────────────────────────────────────────
    # EMOTION PREDICTIONS
    # ─────────────────────────────────────────────

    def insert_emotion_prediction(
        self, detection_id: int, emotion: str, confidence: float,
        probabilities: dict = None,
    ) -> int:
        prob_val = json.dumps(probabilities) if probabilities else None
        cur = self._execute(
            """INSERT INTO emotion_predictions
               (detection_id, emotion, confidence, probabilities)
               VALUES (%s, %s, %s, %s)
               RETURNING prediction_id""",
            (detection_id, emotion, confidence, prob_val),
        )
        pred_id = cur.fetchone()[0]
        self._commit()
        return pred_id

    # ─────────────────────────────────────────────
    # INTERVAL EMOTIONS
    # ─────────────────────────────────────────────

    def insert_interval_emotions(
        self, session_id: int, person_id: int, intervals: List[dict],
    ) -> None:
        for iv in intervals:
            emotion_counts = json.dumps(iv["emotion_counts"])
            self._execute(
                """INSERT INTO interval_emotions
                   (session_id, person_id, interval_id, start_ts, frame_count,
                    total_detections, emotion_counts, dominant_emotion,
                    dominant_share, avg_confidence, valence_score)
                   VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s)
                   ON CONFLICT (session_id, person_id, interval_id)
                   DO UPDATE SET start_ts = EXCLUDED.start_ts,
                       frame_count = EXCLUDED.frame_count,
                       total_detections = EXCLUDED.total_detections,
                       emotion_counts = EXCLUDED.emotion_counts,
                       dominant_emotion = EXCLUDED.dominant_emotion,
                       dominant_share = EXCLUDED.dominant_share,
                       avg_confidence = EXCLUDED.avg_confidence,
                       valence_score = EXCLUDED.valence_score""",
                (session_id, person_id, iv["interval_id"], iv.get("start_ts"),
                 iv["frame_count"], iv["total_detections"],
                 emotion_counts,
                 iv["dominant_emotion"], iv["dominant_share"],
                 iv["avg_confidence"], iv["valence_score"]),
            )
        self._commit()

    # ─────────────────────────────────────────────
    # SPEAKERS
    # ─────────────────────────────────────────────

    def insert_speakers(
        self, session_id: int, speaker_roles: Dict[str, str],
    ) -> Dict[str, int]:
        label_to_id = {}
        for label, role in speaker_roles.items():
            cur = self._execute(
                """INSERT INTO speakers (session_id, speaker_label, role)
                   VALUES (%s, %s, %s)
                   ON CONFLICT (session_id, speaker_label) DO NOTHING
                   RETURNING speaker_id""",
                (session_id, label, role),
            )
            row = cur.fetchone()
            if row:
                label_to_id[label] = row[0]
            else:
                r = self._fetchone(
                    "SELECT speaker_id FROM speakers WHERE session_id = %s AND speaker_label = %s",
                    (session_id, label),
                )
                label_to_id[label] = r["speaker_id"]
        self._commit()
        return label_to_id

    # ─────────────────────────────────────────────
    # TRANSCRIPT SEGMENTS
    # ─────────────────────────────────────────────

    def insert_transcript_segments(
        self, session_id: int, speaker_label_to_id: Dict[str, int],
        segments: List[dict],
    ) -> None:
        for seg in segments:
            sp_id = speaker_label_to_id.get(seg.get("speaker_id", ""))
            if sp_id is None:
                continue
            sentiment = seg.get("sentiment", {})
            interval_id = int(seg["start"] / 10.0)
            self._execute(
                """INSERT INTO transcript_segments
                   (session_id, speaker_id, start_sec, end_sec, text,
                    sentiment_label, sentiment_score, interval_id)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
                (session_id, sp_id, seg["start"], seg["end"], seg["text"],
                 sentiment.get("label", "neutral"),
                 sentiment.get("score", 0.0),
                 interval_id),
            )
        self._commit()

    # ─────────────────────────────────────────────
    # COMBINED EMOTIONS
    # ─────────────────────────────────────────────

    def insert_combined_emotions(
        self, session_id: int, person_id: int, intervals: List[dict],
    ) -> None:
        for iv in intervals:
            self._execute(
                """INSERT INTO combined_emotions
                   (session_id, person_id, interval_id,
                    video_emotion, video_valence, video_confidence,
                    audio_sentiment, audio_valence, audio_confidence,
                    combined_emotion, combined_valence, combined_confidence)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT (session_id, person_id, interval_id)
                   DO UPDATE SET
                       video_emotion = EXCLUDED.video_emotion,
                       video_valence = EXCLUDED.video_valence,
                       video_confidence = EXCLUDED.video_confidence,
                       audio_sentiment = EXCLUDED.audio_sentiment,
                       audio_valence = EXCLUDED.audio_valence,
                       audio_confidence = EXCLUDED.audio_confidence,
                       combined_emotion = EXCLUDED.combined_emotion,
                       combined_valence = EXCLUDED.combined_valence,
                       combined_confidence = EXCLUDED.combined_confidence""",
                (session_id, person_id, iv["interval_id"],
                 iv["video_emotion"], iv["video_valence"], iv["video_confidence"],
                 iv["audio_sentiment"], iv["audio_valence"], iv["audio_confidence"],
                 iv["combined_emotion"], iv["combined_valence"], iv["combined_confidence"]),
            )
        self._commit()

    # ─────────────────────────────────────────────
    # KPI EVALUATIONS
    # ─────────────────────────────────────────────

    def insert_kpi_evaluation(self, session_id: int, kpi_result: dict) -> int:
        text_fields = kpi_result.get("text_fields", {})
        cur = self._execute(
            """INSERT INTO kpi_evaluations
               (session_id, checklist_name, overall_score, max_possible_score,
                score_percentage, overall_status, traffic_light,
                summary, detailed_summary, next_step,
                positive_aspects, negative_aspects)
               VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (session_id) DO UPDATE SET
                   overall_score = EXCLUDED.overall_score,
                   max_possible_score = EXCLUDED.max_possible_score,
                   score_percentage = EXCLUDED.score_percentage,
                   overall_status = EXCLUDED.overall_status,
                   traffic_light = EXCLUDED.traffic_light,
                   summary = EXCLUDED.summary,
                   detailed_summary = EXCLUDED.detailed_summary,
                   next_step = EXCLUDED.next_step,
                   positive_aspects = EXCLUDED.positive_aspects,
                   negative_aspects = EXCLUDED.negative_aspects,
                   evaluated_at = NOW()
               RETURNING evaluation_id""",
            (session_id,
             kpi_result.get("checklist_name", "Пример для СДЭК"),
             kpi_result["overall_score"],
             kpi_result["max_possible_score"],
             kpi_result["score_percentage"],
             kpi_result["overall_status"],
             kpi_result["traffic_light"],
             text_fields.get("summary"),
             text_fields.get("detailed_summary"),
             text_fields.get("next_step"),
             text_fields.get("positive_aspects"),
             text_fields.get("negative_aspects")),
        )
        evaluation_id = cur.fetchone()[0]

        # Per-criterion items
        criteria_rows = self._fetchall("SELECT criterion_id, key FROM kpi_criteria")
        key_to_crit_id = {r["key"]: r["criterion_id"] for r in criteria_rows}

        for item in kpi_result.get("all_items", []):
            crit_id = key_to_crit_id.get(item["key"])
            if crit_id is None:
                continue
            self._execute(
                """INSERT INTO kpi_evaluation_items
                   (evaluation_id, criterion_id, score, status, details)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (evaluation_id, criterion_id)
                   DO UPDATE SET score = EXCLUDED.score,
                       status = EXCLUDED.status, details = EXCLUDED.details""",
                (evaluation_id, crit_id, item["score"], item["status"], item.get("details")),
            )

        # Category scores
        cat_rows = self._fetchall("SELECT category_id, name FROM kpi_categories")
        cat_name_to_id = {r["name"]: r["category_id"] for r in cat_rows}

        for cat_name, cat_data in kpi_result.get("categories", {}).items():
            cat_id = cat_name_to_id.get(cat_name)
            if cat_id is None:
                continue
            self._execute(
                """INSERT INTO kpi_category_scores
                   (evaluation_id, category_id, total_score, max_score, percentage)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (evaluation_id, category_id)
                   DO UPDATE SET total_score = EXCLUDED.total_score,
                       max_score = EXCLUDED.max_score, percentage = EXCLUDED.percentage""",
                (evaluation_id, cat_id, cat_data["total_score"],
                 cat_data["max_score"], cat_data["percentage"]),
            )

        self._commit()
        return evaluation_id

    # ─────────────────────────────────────────────
    # RECOMMENDATIONS
    # ─────────────────────────────────────────────

    def insert_recommendations(self, session_id: int, recommendations: List[dict]) -> None:
        for rec in recommendations:
            self._execute(
                """INSERT INTO recommendations (session_id, category, priority, text)
                   VALUES (%s, %s, %s, %s)""",
                (session_id, rec.get("category"), rec.get("priority", 0), rec["text"]),
            )
        self._commit()

    # ─────────────────────────────────────────────
    # OPERATORS
    # ─────────────────────────────────────────────

    def upsert_operator(self, name: str, employee_id: str = None, department: str = None) -> int:
        if employee_id:
            row = self._fetchone(
                "SELECT operator_id FROM operators WHERE employee_id = %s",
                (employee_id,),
            )
            if row:
                return row["operator_id"]
        cur = self._execute(
            "INSERT INTO operators (name, employee_id, department) VALUES (%s, %s, %s) RETURNING operator_id",
            (name, employee_id, department),
        )
        op_id = cur.fetchone()[0]
        self._commit()
        return op_id

    def link_session_operator(self, session_id: int, operator_id: int,
                               role: str = "company_representative") -> None:
        self._execute(
            """INSERT INTO session_operators (session_id, operator_id, role)
               VALUES (%s, %s, %s) ON CONFLICT DO NOTHING""",
            (session_id, operator_id, role),
        )
        self._commit()

    # ─────────────────────────────────────────────
    # QUERY HELPERS
    # ─────────────────────────────────────────────

    def get_session_summary(self, session_id: int = None) -> List[dict]:
        if session_id:
            return self._fetchall(
                "SELECT * FROM v_session_summary WHERE session_id = %s", (session_id,))
        return self._fetchall("SELECT * FROM v_session_summary")

    def get_emotion_timeline(self, session_id: int, person_label: str = None) -> List[dict]:
        if person_label:
            return self._fetchall(
                "SELECT * FROM v_emotion_timeline WHERE session_id = %s AND person_label = %s ORDER BY interval_id",
                (session_id, person_label))
        return self._fetchall(
            "SELECT * FROM v_emotion_timeline WHERE session_id = %s ORDER BY interval_id",
            (session_id,))

    def get_kpi_detail(self, session_id: int) -> List[dict]:
        return self._fetchall(
            "SELECT * FROM v_kpi_detail WHERE session_id = %s", (session_id,))

    def get_sessions_by_score(self, max_pct: float = None, min_pct: float = None) -> List[dict]:
        query = "SELECT * FROM v_session_summary WHERE 1=1"
        params = []
        if max_pct is not None:
            query += " AND score_percentage <= %s"
            params.append(max_pct)
        if min_pct is not None:
            query += " AND score_percentage >= %s"
            params.append(min_pct)
        query += " ORDER BY score_percentage"
        return self._fetchall(query, tuple(params))

    def get_most_failed_criteria(self, limit: int = 10) -> List[dict]:
        return self._fetchall(
            """SELECT kr.key, kr.name, kc.name AS category, COUNT(*) AS fail_count
               FROM kpi_evaluation_items ki
               JOIN kpi_criteria kr ON kr.criterion_id = ki.criterion_id
               JOIN kpi_categories kc ON kc.category_id = kr.category_id
               WHERE ki.status = 'Нет'
               GROUP BY ki.criterion_id, kr.key, kr.name, kc.name
               ORDER BY fail_count DESC
               LIMIT %s""",
            (limit,))

    def get_transcript(self, session_id: int) -> List[dict]:
        return self._fetchall(
            """SELECT ts.start_sec, ts.end_sec, ts.text,
                      sp.speaker_label, sp.role,
                      ts.sentiment_label, ts.sentiment_score
               FROM transcript_segments ts
               JOIN speakers sp ON sp.speaker_id = ts.speaker_id
               WHERE ts.session_id = %s
               ORDER BY ts.start_sec""",
            (session_id,))
