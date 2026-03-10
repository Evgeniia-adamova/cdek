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
                from app.backend.speech_processor import get_video_path_from_bucket
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

    # --- Dense frame extraction + face detection ---
    from app.backend.face_detection import run_step2, run_step3, run_step4
    from app.backend.emotion_recognition import run_emotion_baseline_onnx, run_step6_emotion_report

    print("\n=== DENSE FRAME EXTRACTION + FACE DETECTION ===")
    step4_out = run_step4(
        video_path=video_path,
        output_dir=dirs["frames"],
        interval_sec=10.0,
        frames_per_interval=6,
        max_frames=1200,
        manifest_path=os.path.join(dirs["face_detection"], "frame_manifest.json"),
    )
    run_step2(
        manifest_path=step4_out["manifest_path"],
        output_path=os.path.join(dirs["face_detection"], "face_detection.json"),
        interval_sec=10.0,
        save_face_crops=True,
        crops_dir=os.path.join(dirs["frames"], "faces_by_person"),
        expected_people=2,
    )
    run_step3(
        input_path=os.path.join(dirs["face_detection"], "face_detection.json"),
        output_path=os.path.join(dirs["face_detection"], "face_detection_extended.json"),
        interval_sec=10.0,
    )

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

    # --- Extract audio from video ---
    ogg_path = Path(dirs["text"]) / "audio.ogg"
    print("\n=== EXTRACTING AUDIO ===")
    from app.backend.speech_processor import _extract_audio_to_ogg
    if not _extract_audio_to_ogg(video_path, ogg_path):
        print("Warning: ffmpeg failed or not installed. Skipping text extraction/analysis.", file=sys.stderr)
        print("Pipeline (frames + emotion) completed. Install ffmpeg and re-run for text.", file=sys.stderr)
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

            kpi_result = run_kpi_evaluation(
                transcript=transcript_text,
                output_path=kpi_output_path,
                video_id=video_id,
                combined_emotion_result=combined_for_kpi,
            )
            print_kpi_report(kpi_result)
            print("KPI evaluation saved:", kpi_output_path)
        else:
            print("Warning: Empty transcript, skipping KPI evaluation.")
    except Exception as e:
        print(f"Warning: KPI evaluation failed: {e}", file=sys.stderr)

    print("\n=== FULL PIPELINE COMPLETE ===")
    print(f"  Video ID: {video_id}")
    print(f"  Data:     data/{video_id}/")
    print(f"  Logs:     logs/{video_id}/")
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
