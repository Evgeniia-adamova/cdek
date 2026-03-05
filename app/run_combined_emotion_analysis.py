#!/usr/bin/env python
# coding: utf-8
"""
Run combined video+audio emotion analysis on existing emotion and sentiment data.

Usage:
  python -m app.run_combined_emotion_analysis
    [emotion_report_path] [audio_sentiment_path] [output_path] [video_weight] [audio_weight]

Examples:
  # Use defaults (auto-detects paths)
  python -m app.run_combined_emotion_analysis

  # Specify custom paths
  python -m app.run_combined_emotion_analysis \\
    data/v3_dense/analysis_step6_emotion_report.json \\
    data/video_from_bucket_audio_diarized_sentiment.json \\
    data/v3_dense/combined_emotion_analysis.json \\
    0.6 0.4

Weights:
  video_weight + audio_weight should ideally sum to 1.0
  - video_weight=0.6, audio_weight=0.4 (default): facial emotions weighted more
  - video_weight=0.5, audio_weight=0.5: equal weight
  - video_weight=0.4, audio_weight=0.6: audio sentiment weighted more
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

DATA_DIR = PROJECT_ROOT / "data"
DATA_V3 = str(DATA_DIR / "v3_dense")


def main():
    # Parse arguments
    if len(sys.argv) >= 2:
        emotion_report_path = sys.argv[1]
    else:
        emotion_report_path = os.path.join(DATA_V3, "analysis_step6_emotion_report.json")
    
    if len(sys.argv) >= 3:
        audio_sentiment_path = sys.argv[2]
    else:
        # Try to find audio sentiment file
        candidates = [
            DATA_DIR / "video_from_bucket_audio_diarized_sentiment.json",
            DATA_DIR / "video_from_bucket_audio_diarized_sentiment.json",
        ]
        audio_sentiment_path = None
        for candidate in candidates:
            if candidate.is_file():
                audio_sentiment_path = str(candidate)
                break
        
        if audio_sentiment_path is None:
            print("Error: audio sentiment file not found. Please provide path as argument.", file=sys.stderr)
            print(f"Searched in: {candidates}", file=sys.stderr)
            return 1
    
    if len(sys.argv) >= 4:
        output_path = sys.argv[3]
    else:
        output_path = os.path.join(DATA_V3, "combined_emotion_analysis.json")
    
    video_weight = float(sys.argv[4]) if len(sys.argv) >= 5 else 0.6
    audio_weight = float(sys.argv[5]) if len(sys.argv) >= 6 else 0.4
    
    # Validate inputs
    if not os.path.isfile(emotion_report_path):
        print(f"Error: emotion report not found: {emotion_report_path}", file=sys.stderr)
        return 1
    
    if not os.path.isfile(audio_sentiment_path):
        print(f"Error: audio sentiment file not found: {audio_sentiment_path}", file=sys.stderr)
        return 1
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    
    # Run combined analysis
    from app.backend.combined_emotion_analysis import run_combined_emotion_analysis
    
    try:
        result = run_combined_emotion_analysis(
            emotion_report_path=emotion_report_path,
            audio_sentiment_path=audio_sentiment_path,
            output_path=output_path,
            video_weight=video_weight,
            audio_weight=audio_weight,
        )
        
        print("\n=== SUMMARY ===")
        print(f"Emotion report: {emotion_report_path}")
        print(f"Audio sentiment: {audio_sentiment_path}")
        print(f"Video weight: {video_weight}, Audio weight: {audio_weight}")
        print(f"Output: {output_path}")
        return 0
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
