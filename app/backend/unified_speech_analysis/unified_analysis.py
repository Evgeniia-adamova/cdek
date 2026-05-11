"""
Unified speech analysis: SpeechKit STT + HuggingFace sentiment.
Speakers are assigned by alternating segment order.
"""
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


class UnifiedSpeechAnalyzer:
    """Unified analyzer combining SpeechKit and HuggingFace models."""

    def __init__(self,
                 api_key: Optional[str] = None,
                 folder_id: Optional[str] = None,
                 s3_client=None,
                 bucket: Optional[str] = None,
                 logger=None):
        import logging
        self.logger = logger or logging.getLogger(__name__)
        self.api_key = api_key or os.environ.get("YANDEX_API_KEY")
        self.folder_id = folder_id or os.environ.get("YANDEX_FOLDER_ID")
        self.bucket = bucket or os.environ.get("YANDEX_S3_BUCKET")
        self.s3_client = s3_client

        from app.backend.text_extraction import SpeechProcessor
        if not self.s3_client:
            from app.backend.text_extraction import create_yandex_s3_client_from_env
            self.s3_client = create_yandex_s3_client_from_env()

        self.speech_processor = SpeechProcessor(
            folder_id=self.folder_id,
            api_key=self.api_key,
            s3_client=self.s3_client,
            bucket=self.bucket,
            logger=self.logger
        )
        self.logger.info("UnifiedSpeechAnalyzer initialized")

    def get_sentiment_torch(self, text: str) -> Dict[str, Any]:
        """Get sentiment using HuggingFace + Torch. Returns dict with label and score."""
        if not text.strip():
            return {"label": "neutral", "score": 0.0}
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            self.logger.error("transformers not installed")
            return {"label": "unknown", "score": 0.0, "error": "transformers not installed"}
        try:
            model_name = "blanchefort/rubert-base-cased-sentiment-rusentiment"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
            if torch.cuda.is_available():
                model = model.cuda()
            id2label = {int(k): v for k, v in model.config.id2label.items()}
            inputs = tokenizer(text[:512], return_tensors="pt", truncation=True,
                               padding=True, max_length=512)
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            with torch.no_grad():
                logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            idx = int(probs.argmax())
            label = id2label.get(idx, "unknown").lower()
            return {"label": label, "score": float(probs[idx])}
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return {"label": "unknown", "score": 0.0, "error": str(e)}

    def transcribe_with_speechkit(
        self,
        file_bytes: bytes,
        file_extension: str = ".ogg",
        language_code: str = "ru-RU",
    ) -> Dict[str, Any]:
        """Use SpeechKit for speech-to-text."""
        return self.speech_processor.speech_to_text(
            file_bytes,
            file_extension=file_extension,
            language_code=language_code,
        )

    def analyze_segments(
        self,
        audio_path: Path,
        segments: List[Dict[str, Any]],
        role_1: str = "client",
        role_2: str = "company_representative",
        use_alternating_fallback: bool = True,
    ) -> Dict[str, Any]:
        """
        Assign speakers by alternating segment order and compute sentiment.
        """
        speaker_to_role = {"SPEAKER_00": role_1, "SPEAKER_01": role_2}
        self.logger.info("Using alternating speaker assignment")

        results = []
        for idx, seg in enumerate(segments):
            start, end = seg["start"], seg["end"]
            text = seg.get("text", "").strip()
            speaker_id = "SPEAKER_00" if (idx % 2 == 0) else "SPEAKER_01"
            role = role_1 if (idx % 2 == 0) else role_2
            sentiment = self.get_sentiment_torch(text) if text else {"label": "neutral", "score": 0.0}
            results.append({
                "start": start, "end": end, "text": text,
                "speaker_id": speaker_id, "role": role,
                "sentiment": sentiment,
            })

        meta = {
            "speaker_roles": speaker_to_role,
            "diarization_fallback_used": True,
            "note": "Speakers assigned by alternating segment order.",
            "num_segments": len(results),
            "num_speakers": len(speaker_to_role),
        }
        return {"segments": results, "metadata": meta}


def run_unified_analysis(
    audio_path: Path,
    segments: List[Dict[str, Any]],
    output_json_path: Optional[Path] = None,
    output_txt_path: Optional[Path] = None,
    role_1: str = "client",
    role_2: str = "company_representative",
) -> Dict[str, Any]:
    """Run full unified analysis on segments. Returns dict with segments, metadata, output paths."""
    analyzer = UnifiedSpeechAnalyzer()
    result = analyzer.analyze_segments(audio_path, segments, role_1=role_1, role_2=role_2)

    base = audio_path.parent / audio_path.stem
    out_json = Path(output_json_path or (base.parent / f"{base.name}_unified_analysis.json"))
    out_txt = Path(output_txt_path or (base.parent / f"{base.name}_unified_analysis.txt"))

    out_json.write_text(
        json.dumps({"segments": result["segments"], **result["metadata"]},
                   ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    lines = []
    for r in result["segments"]:
        lines.append(f"[{r['start']:.1f}s - {r['end']:.1f}s] {r['role']}: {r['text']}")
        lines.append(f"  sentiment: {r['sentiment'].get('label', '?')} ({r['sentiment'].get('score', 0):.2f})")
    out_txt.write_text("\n".join(lines), encoding="utf-8")

    return {
        "segments": result["segments"],
        **result["metadata"],
        "output_json": str(out_json),
        "output_txt": str(out_txt),
    }


def main() -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    script_dir = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = script_dir / "data"
    default_audio = data_dir / "video_from_bucket_audio.ogg"

    if len(sys.argv) < 2:
        print("Usage: python -m app.backend.unified_speech_analysis <segments_json> [audio_path]",
              file=sys.stderr)
        return 1

    segments_path = Path(sys.argv[1])
    audio_path = Path(sys.argv[2]) if len(sys.argv) > 2 else default_audio

    if not segments_path.is_file():
        print(f"Error: segments JSON not found: {segments_path}", file=sys.stderr)
        return 1
    if not audio_path.is_file():
        print(f"Error: audio not found: {audio_path}", file=sys.stderr)
        return 1

    segments_data = json.loads(segments_path.read_text(encoding="utf-8"))
    segments = segments_data.get("segments", [])

    role_1 = os.environ.get("SPEAKER_1_ROLE", "client")
    role_2 = os.environ.get("SPEAKER_2_ROLE", "company_representative")

    print(f"Analyzing {len(segments)} segments...")
    result = run_unified_analysis(audio_path, segments, role_1=role_1, role_2=role_2)
    print(f"JSON: {result['output_json']}")
    print(f"TXT: {result['output_txt']}")
    print(f"Summary: {len(segments)} segments, {result['num_speakers']} speakers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
