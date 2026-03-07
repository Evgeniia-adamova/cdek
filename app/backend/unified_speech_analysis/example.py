#!/usr/bin/env python
# coding: utf-8

"""
Example: Combined SpeechKit + Whisper/Torch + HuggingFace analysis.

This script demonstrates how to use the unified speech analysis module
combining:
- Yandex SpeechKit for speech recognition
- Torch-based pyannote for speaker diarization  
- HuggingFace transformers for sentiment analysis

Usage (from project root):
    python -m app.backend.unified_speech_analysis.example [path/to/audio.ogg]

Or with segments JSON:
    python -m app.backend.unified_speech_analysis.example [path/to/audio.ogg] [path/to/segments.json]

Defaults to: data/video_from_bucket_audio.ogg
"""

import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass


def main():
    # Input paths
    audio_path = Path(sys.argv[1]) if len(sys.argv) > 1 else PROJECT_ROOT / "data" / "video_from_bucket_audio.ogg"
    segments_json_path = Path(sys.argv[2]) if len(sys.argv) > 2 else audio_path.parent / f"{audio_path.stem}_segments.json"
    
    if not audio_path.is_file():
        print(f"Error: audio file not found: {audio_path}")
        sys.exit(1)
    
    print(f"Audio: {audio_path}")
    
    # Step 1: Extract text with Whisper if segments don't exist
    if not segments_json_path.is_file():
        print(f"\n=== STEP 1: Whisper Text Extraction ===")
        from app.backend.text_extraction import transcribe
        
        try:
            transcribe(
                audio_path,
                model_name="base",
                language="ru",
                output_path=audio_path.with_suffix(".txt"),
                verbose=True,
                save_timestamps=True,
            )
            print(f"Segments: {segments_json_path}")
        except Exception as e:
            print(f"Error during text extraction: {e}")
            sys.exit(1)
    
    if not segments_json_path.is_file():
        print(f"Error: segments file not created")
        sys.exit(1)
    
    # Load segments
    segments_data = json.loads(segments_json_path.read_text(encoding="utf-8"))
    segments = segments_data.get("segments", [])
    print(f"Loaded {len(segments)} segments")
    
    # Step 2: Unified analysis (diarization + sentiment)
    print(f"\n=== STEP 2: Unified Speech Analysis ===")
    print("Components: SpeechKit + Torch (pyannote) + HuggingFace")
    
    from app.backend.unified_speech_analysis import run_unified_analysis
    
    try:
        role_1 = os.environ.get("SPEAKER_1_ROLE", "client")
        role_2 = os.environ.get("SPEAKER_2_ROLE", "company_representative")
        
        result = run_unified_analysis(
            audio_path,
            segments,
            role_1=role_1,
            role_2=role_2,
        )
        
        print(f"✓ Analysis complete")
        print(f"  JSON output: {result['output_json']}")
        print(f"  TXT output: {result['output_txt']}")
        print(f"  Speakers: {result['num_speakers']}")
        print(f"  Segments analyzed: {result['num_segments']}")
        
        # Show sample
        print(f"\n=== Sample Results (first 3 segments) ===")
        for r in result["segments"][:3]:
            text_preview = r["text"][:50].replace("\n", " ")
            print(f"[{r['start']:.1f}s-{r['end']:.1f}s] {r['role']}")
            print(f"  Text: {text_preview}...")
            print(f"  Speaker: {r['speaker_id']}")
            print(f"  Sentiment: {r['sentiment']['label']} ({r['sentiment']['score']:.2f})")
            print()
        
        return 0
    except Exception as e:
        print(f"Error during unified analysis: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
