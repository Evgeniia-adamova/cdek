#!/usr/bin/env python
# coding: utf-8
"""
Run full pipeline from video: extract frames → face detection → emotion recognition
→ extract audio → text extraction (Whisper) → text analysis (diarization + sentiment).

Video: with no argument uses data/video_from_bucket.webm if present, else downloads from
Yandex bucket (requires .env with YANDEX_S3_*). With one argument: uses that video path.
Extracted face crops (faces_by_person) are stored locally under logs/; canonical storage
for faces is Yandex Cloud (upload/sync to bucket is separate).

  python -m app.run_full_pipeline [path/to/video.webm]
"""
import os
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
FRAMES_V2 = str(LOGS_DIR / "extracted_frames_v2")
FRAMES_V3 = str(LOGS_DIR / "extracted_frames_v3_dense")
DATA_V2 = str(DATA_DIR / "v2")
DATA_V3 = str(DATA_DIR / "v3_dense")


def main():
    # Resolve video path
    if len(sys.argv) >= 2:
        video_path = Path(sys.argv[1])
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
                from app.speech_processor import get_video_path_from_bucket
                print("=== DOWNLOADING VIDEO FROM BUCKET ===")
                video_path = get_video_path_from_bucket(local_dir=str(DATA_DIR))
            except Exception as e:
                print(f"Error: no video path given and bucket failed: {e}", file=sys.stderr)
                print("Usage: python -m app.run_full_pipeline [path/to/video.webm]", file=sys.stderr)
                return 1

    os.makedirs(DATA_V2, exist_ok=True)
    os.makedirs(DATA_V3, exist_ok=True)

    # --- Face detection + emotion ---
    from app.backend.face_detection import run_step1, run_step2, run_step3, run_step4
    from app.backend.emotion_recognition import run_emotion_baseline_onnx, run_step6_emotion_report

    print("\n=== FRAME EXTRACTION + FACE DETECTION (v2) ===")
    step1_out = run_step1(
        video_path=video_path,
        interval_sec=0.5,
        max_frames=200,
        output_dir=FRAMES_V2,
        manifest_path=os.path.join(DATA_V2, "frame_manifest_step1.json"),
    )
    run_step2(
        manifest_path=step1_out["manifest_path"],
        output_path=os.path.join(DATA_V2, "analysis_step2_consolidated.json"),
        interval_sec=10.0,
        save_face_crops=True,
        crops_dir=os.path.join(FRAMES_V2, "faces_by_person"),
        expected_people=2,
    )
    run_step3(
        input_path=os.path.join(DATA_V2, "analysis_step2_consolidated.json"),
        output_path=os.path.join(DATA_V2, "analysis_step3_extended.json"),
        interval_sec=10.0,
    )

    print("\n=== DENSE SAMPLING + FACE DETECTION (v3) ===")
    step4_out = run_step4(
        video_path=video_path,
        output_dir=FRAMES_V3,
        interval_sec=10.0,
        frames_per_interval=6,
        max_frames=1200,
        manifest_path=os.path.join(DATA_V3, "frame_manifest_step4_dense.json"),
    )
    run_step2(
        manifest_path=step4_out["manifest_path"],
        output_path=os.path.join(DATA_V3, "analysis_step2_dense_consolidated.json"),
        interval_sec=10.0,
        save_face_crops=True,
        crops_dir=os.path.join(FRAMES_V3, "faces_by_person"),
        expected_people=3,
        max_tracked_people=3,
        analysis_people=2,
    )
    run_step3(
        input_path=os.path.join(DATA_V3, "analysis_step2_dense_consolidated.json"),
        output_path=os.path.join(DATA_V3, "analysis_step3_extended.json"),
        interval_sec=10.0,
    )

    print("\n=== EMOTION RECOGNITION ===")
    run_emotion_baseline_onnx(
        analysis_path=os.path.join(DATA_V3, "analysis_step2_dense_consolidated.json"),
        output_path=os.path.join(DATA_V3, "emotion_baseline_onnx_ferplus.json"),
        max_per_person=300,
    )
    run_step6_emotion_report(
        analysis_path=os.path.join(DATA_V3, "analysis_step3_extended.json"),
        emotion_path=os.path.join(DATA_V3, "emotion_baseline_onnx_ferplus.json"),
        output_path=os.path.join(DATA_V3, "analysis_step6_emotion_report.json"),
    )

    # --- Extract audio from video ---
    video_stem = Path(video_path).stem
    ogg_path = DATA_DIR / f"{video_stem}_audio.ogg"
    print("\n=== EXTRACTING AUDIO ===")
    from app.speech_processor import _extract_audio_to_ogg
    if not _extract_audio_to_ogg(video_path, ogg_path):
        print("Warning: ffmpeg failed or not installed. Skipping text extraction/analysis.", file=sys.stderr)
        print("Pipeline (frames + emotion) completed. Install ffmpeg and re-run for text.", file=sys.stderr)
        return 0
    print("Audio:", ogg_path)

    # --- Text extraction (Whisper) ---
    print("\n=== TEXT EXTRACTION (Whisper) ===")
    from app.backend.text_extraction import transcribe
    transcribe(
        ogg_path,
        model_name="base",
        language="ru",
        output_path=ogg_path.with_suffix(".txt"),
        verbose=True,
        save_timestamps=True,
    )
    segments_path = ogg_path.parent / f"{ogg_path.stem}_segments.json"
    if not segments_path.is_file():
        print("Error: segments file not created.", file=sys.stderr)
        return 1

    # --- Text analysis (diarization + sentiment) ---
    print("\n=== TEXT ANALYSIS (diarization + sentiment) ===")
    from app.backend.text_analysis import run_diarize_sentiment
    result = run_diarize_sentiment(ogg_path, segments_path)
    print("Output JSON:", result["output_json"])
    print("Output TXT:", result["output_txt"])

    # --- Combined video+audio emotion analysis ---
    print("\n=== COMBINED VIDEO+AUDIO EMOTION ANALYSIS ===")
    try:
        from app.backend.combined_emotion_analysis import run_combined_emotion_analysis
        emotion_report_path = os.path.join(DATA_V3, "analysis_step6_emotion_report.json")
        audio_sentiment_path = result["output_json"]
        combined_output_path = os.path.join(DATA_V3, "combined_emotion_analysis.json")
        
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

    # --- KPI Evaluation (СДЭК checklist, 100 points) ---
    print("\n=== KPI EVALUATION (СДЭК CHECKLIST) ===")
    kpi_output_path = None
    try:
        from app.backend.calls_kpi_evaluation import run_kpi_evaluation, print_kpi_report

        transcript_path = ogg_path.with_suffix(".txt")
        if transcript_path.is_file():
            transcript_text = transcript_path.read_text(encoding="utf-8")
        else:
            transcript_text = ""

        if transcript_text.strip():
            kpi_output_path = os.path.join(DATA_V3, "kpi_evaluation.json")
            combined_for_kpi = None
            if 'combined_result' in locals():
                combined_for_kpi = combined_result

            kpi_result = run_kpi_evaluation(
                transcript=transcript_text,
                output_path=kpi_output_path,
                video_id=video_stem,
                combined_emotion_result=combined_for_kpi,
            )
            print_kpi_report(kpi_result)
            print("KPI evaluation saved:", kpi_output_path)
        else:
            print("Warning: Empty transcript, skipping KPI evaluation.")
    except Exception as e:
        print(f"Warning: KPI evaluation failed: {e}", file=sys.stderr)

    print("\n=== FULL PIPELINE COMPLETE ===")
    print("  Frames + faces: logs/extracted_frames_v2, logs/extracted_frames_v3_dense")
    print("  Analysis: data/v2, data/v3_dense")
    print("  Transcript: ", ogg_path.with_suffix(".txt"))
    print("  Diarized sentiment: ", result["output_json"])
    print("  Combined emotion analysis: ", combined_output_path if 'combined_output_path' in locals() else "N/A")
    print("  KPI evaluation: ", kpi_output_path if kpi_output_path else "N/A")
    return 0


if __name__ == "__main__":
    sys.exit(main())
