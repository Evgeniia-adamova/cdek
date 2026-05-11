#!/usr/bin/env python
# coding: utf-8
"""
Run full pipeline from video: dense frame extraction -> face detection ->
emotion recognition -> extract audio -> text extraction (Whisper) ->
text analysis (diarization + sentiment) -> combined emotion -> KPI evaluation.

All outputs are isolated per video under data/<video_id>/ and logs/<video_id>/,
so multiple videos can be processed concurrently without conflicts.

Usage:
  python -m app.run_full_pipeline [path/to/video.webm] [--clean]

Options:
  --clean   Remove old face crop folders before running.
"""
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

from app.backend.database.schema import get_connection, init_db
from app.backend.database.repository import PipelineRepository

LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR = PROJECT_ROOT / "data"


def _setup_dirs(video_id: str):
    """Create per-video output directory structure."""
    data_root = DATA_DIR / video_id
    logs_root = LOGS_DIR / video_id
    dirs = {
        "face_detection": str(data_root / "face_detection"),
        "emotion": str(data_root / "emotion"),
        "text": str(data_root / "text"),
        "sentiment": str(data_root / "sentiment"),
        "combined_emotion": str(data_root / "combined_emotion"),
        "kpi": str(data_root / "kpi"),
        "frames": str(logs_root / "extracted_frames"),
    }
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    return dirs


def _clean_old_crops(frames_dir: str):
    """Remove old face crop folders (with retry for OneDrive/Windows locks)."""
    import time
    crops_path = os.path.join(frames_dir, "faces_by_person")
    if not os.path.isdir(crops_path):
        return
    for attempt in range(3):
        try:
            shutil.rmtree(crops_path, ignore_errors=False)
            print(f"Removed {crops_path}")
            return
        except PermissionError:
            if attempt < 2:
                print(f"Retry {attempt+1}/3: folder locked, waiting...")
                time.sleep(2)
    # Last resort: ignore errors
    shutil.rmtree(crops_path, ignore_errors=True)
    print(f"Removed {crops_path} (some files may remain)")


def main():
    # Handle --clean flag
    args = [a for a in sys.argv[1:] if a != "--clean"]
    do_clean = "--clean" in sys.argv

    # Resolve video path
    if args:
        video_path = Path(args[0])
        if not video_path.is_file():
            print(f"Error: video not found: {video_path}", file=sys.stderr)
            return 1
        video_path = str(video_path.resolve())
    else:
        local_video = DATA_DIR / "video_from_bucket.webm"
        if local_video.is_file():
            video_path = str(local_video)
            print("Using local video:", video_path)
        else:
            try:
                from app.backend.text_extraction import get_video_path_from_bucket
                print("=== DOWNLOADING VIDEO FROM BUCKET ===")
                video_path = get_video_path_from_bucket(local_dir=str(DATA_DIR))
            except Exception as e:
                print(f"Error: no video path given and bucket failed: {e}", file=sys.stderr)
                print("Usage: python -m app.run_full_pipeline [path/to/video.webm]", file=sys.stderr)
                return 1

    video_id = Path(video_path).stem
    dirs = _setup_dirs(video_id)

    if do_clean:
        _clean_old_crops(dirs["frames"])

    # --- Initialize database ---
    db_conn = get_connection()
    init_db(db_conn)
    repo = PipelineRepository(db_conn)

    # --- Dense frame extraction + face detection ---
    from app.backend.face_detection import run_step2, run_step3, run_step4
    from app.backend.emotion_recognition import run_emotion_baseline_onnx, run_step6_emotion_report

    print("\n=== DENSE FRAME EXTRACTION + FACE DETECTION ===")
    step4_out = run_step4(
        video_path=video_path,
        output_dir=dirs["frames"],
        interval_sec=10.0,
        frames_per_interval=6,
        max_frames=None,
        manifest_path=os.path.join(dirs["face_detection"], "frame_manifest.json"),
    )
    run_step2(
        manifest_path=step4_out["manifest_path"],
        output_path=os.path.join(dirs["face_detection"], "face_detection.json"),
        interval_sec=10.0,
        save_face_crops=True,
        crops_dir=os.path.join(dirs["frames"], "faces_by_person"),
        expected_people=2,
        analysis_people=2,
        match_threshold=0.55,
    )
    run_step3(
        input_path=os.path.join(dirs["face_detection"], "face_detection.json"),
        output_path=os.path.join(dirs["face_detection"], "face_detection_extended.json"),
        interval_sec=10.0,
    )

    # --- DB: save face detection data ---
    session_id = None
    label_to_id = {}
    try:
        fd_path = os.path.join(dirs["face_detection"], "face_detection.json")
        with open(fd_path, encoding="utf-8") as f:
            fd_data = json.load(f)
        meta = fd_data["video_metadata"]
        config = fd_data.get("step2_config", {})

        # Collect person labels
        all_person_labels = set()
        for frame in fd_data["frames"]:
            for det in frame.get("detections", []):
                pid = det.get("person_id")
                if pid:
                    all_person_labels.add(pid)

        # Reuse session created by upload handler (same video_id) instead of
        # inserting a duplicate, which would fail the UNIQUE constraint and leave
        # session_id=None, causing all DB writes (including KPI) to be skipped.
        existing = repo.get_session_by_video_id(video_id)
        if existing:
            session_id = existing["session_id"]
            repo._execute(
                """UPDATE sessions SET fps=%s, frame_count=%s, width=%s, height=%s,
                   duration_sec=%s, video_path=%s, pipeline_status='running'
                   WHERE session_id=%s""",
                (meta["fps"], meta["frame_count"], meta["width"], meta["height"],
                 meta["duration_sec"], meta.get("video_path"), session_id),
            )
            repo._commit()
        else:
            session_id = repo.create_session(
                video_id=video_id, video_path=meta.get("video_path"),
                fps=meta["fps"], frame_count=meta["frame_count"],
                width=meta["width"], height=meta["height"],
                duration_sec=meta["duration_sec"],
                interval_sec=config.get("interval_sec", 10.0),
                match_threshold=config.get("match_threshold", 0.55),
                detector=config.get("detector", "yunet_2023mar"),
            )
        label_to_id = repo.insert_persons(session_id, sorted(all_person_labels))

        frames_data = [{
            "sample_index": fr["sample_index"], "frame_index": fr["frame_index"],
            "timestamp_sec": fr["timestamp_sec"], "timestamp": fr.get("timestamp"),
            "image_path": fr.get("image_path"), "interval_id": fr.get("interval_id"),
            "face_count": fr.get("face_count", 0),
        } for fr in fd_data["frames"]]
        sample_to_frame_id = repo.insert_frames(session_id, frames_data)

        for frame in fd_data["frames"]:
            frame_id = sample_to_frame_id[frame["sample_index"]]
            if frame.get("detections"):
                repo.insert_detections(frame_id, label_to_id, frame["detections"])

        print(f"  DB: session={session_id}, persons={list(label_to_id.keys())}, "
              f"frames={len(sample_to_frame_id)}")
    except Exception as e:
        print(f"  DB warning (face detection): {e}", file=sys.stderr)

    # --- Emotion recognition ---
    print("\n=== EMOTION RECOGNITION ===")
    run_emotion_baseline_onnx(
        analysis_path=os.path.join(dirs["face_detection"], "face_detection.json"),
        output_path=os.path.join(dirs["emotion"], "emotion_baseline.json"),
        max_per_person=300,
    )
    run_step6_emotion_report(
        analysis_path=os.path.join(dirs["face_detection"], "face_detection_extended.json"),
        emotion_path=os.path.join(dirs["emotion"], "emotion_baseline.json"),
        output_path=os.path.join(dirs["emotion"], "emotion_report.json"),
    )

    # --- DB: save emotion report ---
    if session_id:
        try:
            emo_path = os.path.join(dirs["emotion"], "emotion_report.json")
            with open(emo_path, encoding="utf-8") as f:
                emo_data = json.load(f)
            emo_count = 0
            for interval in emo_data.get("interval_emotion_report", []):
                for plabel, pe in interval.get("person_emotions", {}).items():
                    pid = label_to_id.get(plabel)
                    if pid is None:
                        continue
                    repo.insert_interval_emotions(session_id, pid, [{
                        "interval_id": interval["interval_id"],
                        "start_ts": interval.get("start_ts"),
                        "frame_count": interval.get("frame_count", 0),
                        "total_detections": pe["total_detections"],
                        "emotion_counts": pe["emotion_counts"],
                        "dominant_emotion": pe["dominant_emotion"],
                        "dominant_share": pe["dominant_share"],
                        "avg_confidence": pe["avg_confidence"],
                        "valence_score": pe["valence_score"],
                    }])
                    emo_count += 1
            print(f"  DB: interval_emotions={emo_count}")
        except Exception as e:
            print(f"  DB warning (emotion): {e}", file=sys.stderr)

    # --- Extract audio from video ---
    ogg_path = Path(dirs["text"]) / "audio.ogg"
    print("\n=== EXTRACTING AUDIO ===")
    from app.backend.text_extraction import extract_audio_to_ogg
    if not extract_audio_to_ogg(video_path, ogg_path):
        print("Warning: ffmpeg failed or not installed. Skipping text extraction/analysis.", file=sys.stderr)
        print("Pipeline (frames + emotion) completed. Install ffmpeg and re-run for text.", file=sys.stderr)
        if session_id:
            repo.update_session_status(session_id, "completed")
        db_conn.close()
        return 0
    print("Audio:", ogg_path)

    # --- Text extraction (SpeechKit) ---
    print("\n=== TEXT EXTRACTION (SpeechKit) ===")
    from app.backend.text_extraction import transcribe
    transcribe(
        ogg_path,
        output_path=Path(dirs["text"]) / "transcript.txt",
        language="ru-RU",
        verbose=True,
        save_timestamps=True,
    )
    segments_path = Path(dirs["text"]) / "audio_segments.json"
    if not segments_path.is_file():
        print("Error: segments file not created.", file=sys.stderr)
        if session_id:
            repo.update_session_status(session_id, "failed")
        db_conn.close()
        return 1

    # --- Text analysis (diarization + sentiment) ---
    print("\n=== TEXT ANALYSIS (diarization + sentiment) ===")
    from app.backend.text_analysis import run_diarize_sentiment
    result = run_diarize_sentiment(
        ogg_path,
        segments_path,
        output_json_path=Path(dirs["sentiment"]) / "diarized_sentiment.json",
        output_txt_path=Path(dirs["sentiment"]) / "diarized_sentiment.txt",
    )
    print("Output JSON:", result["output_json"])
    print("Output TXT:", result["output_txt"])

    # --- DB: save transcript + speakers ---
    if session_id:
        try:
            with open(result["output_json"], encoding="utf-8") as f:
                sent_data = json.load(f)
            speaker_roles = sent_data.get("speaker_roles", {})
            speaker_label_to_id = repo.insert_speakers(session_id, speaker_roles)
            repo.insert_transcript_segments(
                session_id, speaker_label_to_id, sent_data.get("segments", []),
            )
            print(f"  DB: speakers={list(speaker_roles.keys())}, "
                  f"segments={len(sent_data.get('segments', []))}")
        except Exception as e:
            print(f"  DB warning (transcript): {e}", file=sys.stderr)

    # --- Combined video+audio emotion analysis ---
    print("\n=== COMBINED VIDEO+AUDIO EMOTION ANALYSIS ===")
    combined_output_path = None
    try:
        from app.backend.combined_emotion_analysis import run_combined_emotion_analysis
        emotion_report_path = os.path.join(dirs["emotion"], "emotion_report.json")
        audio_sentiment_path = result["output_json"]
        combined_output_path = os.path.join(dirs["combined_emotion"], "combined_emotion.json")

        if os.path.isfile(emotion_report_path) and os.path.isfile(audio_sentiment_path):
            combined_result = run_combined_emotion_analysis(
                emotion_report_path=emotion_report_path,
                audio_sentiment_path=audio_sentiment_path,
                output_path=combined_output_path,
                video_weight=0.6,
                audio_weight=0.4,
            )
            print("Combined emotion analysis:", combined_output_path)

            # --- DB: save combined emotions ---
            if session_id:
                try:
                    comb_count = 0
                    for interval in combined_result.get("combined_intervals", []):
                        for plabel, pe in interval.get("person_emotions", {}).items():
                            pid = label_to_id.get(plabel)
                            if pid is None or "combined_emotion" not in pe:
                                continue
                            repo.insert_combined_emotions(session_id, pid, [{
                                "interval_id": interval["interval_id"],
                                "video_emotion": pe.get("video_emotion", ""),
                                "video_valence": pe.get("video_valence", 0.0),
                                "video_confidence": pe.get("video_confidence", 0.0),
                                "audio_sentiment": pe.get("audio_sentiment", "neutral"),
                                "audio_valence": pe.get("audio_valence", 0.0),
                                "audio_confidence": pe.get("audio_confidence", 0.0),
                                "combined_emotion": pe["combined_emotion"],
                                "combined_valence": pe["combined_valence"],
                                "combined_confidence": pe["combined_confidence"],
                            }])
                            comb_count += 1
                    print(f"  DB: combined_emotions={comb_count}")
                except Exception as e:
                    print(f"  DB warning (combined): {e}", file=sys.stderr)
        else:
            print("Warning: Missing emotion report or audio sentiment file. Skipping combined analysis.")
    except Exception as e:
        print(f"Warning: Combined emotion analysis failed: {e}", file=sys.stderr)

    # --- KPI Evaluation ---
    print("\n=== KPI EVALUATION ===")
    kpi_output_path = None
    try:
        from app.backend.calls_kpi_evaluation import run_kpi_evaluation, print_kpi_report

        transcript_path = Path(dirs["text"]) / "transcript.txt"
        if transcript_path.is_file():
            transcript_text = transcript_path.read_text(encoding="utf-8")
        else:
            transcript_text = ""

        if transcript_text.strip():
            kpi_output_path = os.path.join(dirs["kpi"], "kpi_evaluation.json")
            combined_for_kpi = None
            if 'combined_result' in locals():
                combined_for_kpi = combined_result

            # Загружаем сегменты с таймстемпами для evidence в KPI
            kpi_segments = None
            segments_file = Path(dirs["text"]) / "audio_segments.json"
            if segments_file.is_file():
                try:
                    with open(segments_file, encoding="utf-8") as sf:
                        seg_data = json.load(sf)
                    kpi_segments = seg_data.get("segments", seg_data) if isinstance(seg_data, dict) else seg_data
                except Exception as e:
                    print(f"  Warning: could not load segments for KPI evidence: {e}", file=sys.stderr)

            kpi_result = run_kpi_evaluation(
                transcript=transcript_text,
                output_path=kpi_output_path,
                video_id=video_id,
                combined_emotion_result=combined_for_kpi,
                segments=kpi_segments,
            )
            print_kpi_report(kpi_result)
            print("KPI evaluation saved:", kpi_output_path)

            # --- DB: save KPI evaluation ---
            if session_id:
                try:
                    eval_id = repo.insert_kpi_evaluation(session_id, kpi_result)
                    print(f"  DB: kpi_evaluation={eval_id}, "
                          f"score={kpi_result['overall_score']}/{kpi_result['max_possible_score']}")
                except Exception as e:
                    print(f"  DB warning (KPI): {e}", file=sys.stderr)
        else:
            print("Warning: Empty transcript, skipping KPI evaluation.")
    except Exception as e:
        print(f"Warning: KPI evaluation failed: {e}", file=sys.stderr)

    # --- Finalize DB session ---
    if session_id:
        repo.update_session_status(session_id, "completed")
    db_conn.close()

    print("\n=== FULL PIPELINE COMPLETE ===")
    print(f"  Video ID: {video_id}")
    print(f"  Data:     data/{video_id}/")
    print(f"  Logs:     logs/{video_id}/")
    print(f"  Database: PostgreSQL ({os.environ.get('PG_HOST', 'unknown')})")
    print(f"  Outputs:")
    print(f"    Face detection:    {dirs['face_detection']}")
    print(f"    Emotion:           {dirs['emotion']}")
    print(f"    Text:              {dirs['text']}")
    print(f"    Sentiment:         {dirs['sentiment']}")
    print(f"    Combined emotion:  {combined_output_path or 'N/A'}")
    print(f"    KPI:               {kpi_output_path or 'N/A'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
