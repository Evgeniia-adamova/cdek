"""
Unified speech analysis: SpeechKit STT + Torch-based diarization + HuggingFace sentiment.
"""
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


class UnifiedSpeechAnalyzer:
    """Unified analyzer combining SpeechKit, Torch, and HuggingFace models."""
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 folder_id: Optional[str] = None,
                 s3_client=None,
                 bucket: Optional[str] = None,
                 huggingface_token: Optional[str] = None,
                 logger=None):
        """
        Initialize unified analyzer.
        
        Args:
            api_key: Yandex SpeechKit API key
            folder_id: Yandex folder ID
            s3_client: Boto3 S3 client
            bucket: S3 bucket name
            huggingface_token: HuggingFace token for pyannote access
            logger: Logger instance
        """
        import logging
        self.logger = logger or logging.getLogger(__name__)
        self.api_key = api_key or os.environ.get("YANDEX_API_KEY")
        self.folder_id = folder_id or os.environ.get("YANDEX_FOLDER_ID")
        self.bucket = bucket or os.environ.get("YANDEX_S3_BUCKET")
        self.s3_client = s3_client
        self.huggingface_token = huggingface_token or os.environ.get("HUGGINGFACE_TOKEN")
        
        # Initialize SpeechKit processor for STT
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
    
    def get_diarization_torch(
        self,
        audio_path: Path,
        num_speakers: int = 2,
    ) -> List[Tuple[float, float, str]]:
        """
        Get speaker diarization using pyannote (Torch-based).
        
        Returns list of (start, end, speaker_id) tuples.
        """
        try:
            from pyannote.audio import Pipeline
        except ImportError:
            self.logger.warning("pyannote.audio not installed, skipping diarization")
            return []
        
        if not self.huggingface_token:
            self.logger.warning("HUGGINGFACE_TOKEN not set, skipping diarization")
            return []
        
        try:
            pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-community-1",
                use_auth_token=self.huggingface_token,
            )
            diar = pipeline(str(audio_path), num_speakers=num_speakers, min_duration_off=0.0)
            result = [
                (segment.start, segment.end, speaker) 
                for segment, _track, speaker in diar.itertracks(yield_label=True)
            ]
            self.logger.info(f"Diarization complete: {len(result)} segments")
            return result
        except Exception as e:
            self.logger.error(f"Diarization failed: {e}")
            return []
    
    def get_sentiment_torch(self, text: str) -> Dict[str, Any]:
        """
        Get sentiment using Hugging Face + Torch.
        
        Returns dict with label and score.
        """
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
            
            # Use GPU if available
            if torch.cuda.is_available():
                model = model.cuda()
            
            id2label = {int(k): v for k, v in model.config.id2label.items()}
            inputs = tokenizer(
                text[:512], 
                return_tensors="pt", 
                truncation=True, 
                padding=True, 
                max_length=512
            )
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                logits = model(**inputs).logits
            
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
            idx = int(probs.argmax())
            label = id2label.get(idx, "unknown").lower()
            score = float(probs[idx])
            
            return {"label": label, "score": score}
        except Exception as e:
            self.logger.error(f"Sentiment analysis failed: {e}")
            return {"label": "unknown", "score": 0.0, "error": str(e)}
    
    def transcribe_with_speechkit(
        self,
        file_bytes: bytes,
        file_extension: str = ".ogg",
        language_code: str = "ru-RU",
    ) -> Dict[str, Any]:
        """
        Use SpeechKit for speech-to-text.
        
        Returns dict with text and metadata.
        """
        result = self.speech_processor.speech_to_text(
            file_bytes,
            file_extension=file_extension,
            language_code=language_code,
        )
        return result
    
    def assign_speaker_to_segment(
        self,
        segment_start: float,
        segment_end: float,
        diarization: List[Tuple[float, float, str]],
    ) -> Optional[str]:
        """Assign speaker by maximum overlap with diarization segments."""
        if not diarization:
            return None
        
        best_speaker = None
        best_overlap = 0.0
        seg_mid = (segment_start + segment_end) / 2
        
        for d_start, d_end, speaker in diarization:
            overlap_start = max(segment_start, d_start)
            overlap_end = min(segment_end, d_end)
            if overlap_end > overlap_start:
                overlap = overlap_end - overlap_start
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = speaker
        
        if best_speaker is not None:
            return best_speaker
        
        for d_start, d_end, speaker in diarization:
            if d_start <= seg_mid <= d_end:
                return speaker
        
        return diarization[0][2] if diarization else None
    
    def analyze_segments(
        self,
        audio_path: Path,
        segments: List[Dict[str, Any]],
        role_1: str = "client",
        role_2: str = "company_representative",
        use_alternating_fallback: bool = True,
    ) -> Dict[str, Any]:
        """
        Analyze segments: assign speakers, compute sentiment.
        
        Args:
            audio_path: Path to audio file
            segments: List of segment dicts with start, end, text
            role_1: First speaker role
            role_2: Second speaker role
            use_alternating_fallback: Use alternating assignment if diarization fails
        
        Returns:
            Dict with analyzed segments and metadata
        """
        diarization = self.get_diarization_torch(audio_path, num_speakers=2)
        
        speaker_to_role = {}
        if diarization:
            speakers = sorted(set(s[2] for s in diarization))
            for i, sp in enumerate(speakers):
                speaker_to_role[sp] = role_1 if i == 0 else role_2
            self.logger.info(f"Found {len(speakers)} speakers: {speakers}")
        else:
            if use_alternating_fallback:
                speaker_to_role = {"SPEAKER_00": role_1, "SPEAKER_01": role_2}
                self.logger.info("Using alternating speaker assignment")
        
        results = []
        for idx, seg in enumerate(segments):
            start, end = seg["start"], seg["end"]
            text = seg.get("text", "").strip()
            
            speaker_id = self.assign_speaker_to_segment(start, end, diarization) if diarization else None
            
            if speaker_id is not None:
                role = speaker_to_role.get(speaker_id, "speaker_unknown")
            elif use_alternating_fallback and not diarization:
                role = role_1 if (idx % 2 == 0) else role_2
                speaker_id = "SPEAKER_00" if (idx % 2 == 0) else "SPEAKER_01"
            else:
                role = "speaker_unknown"
            
            sentiment = self.get_sentiment_torch(text) if text else {"label": "neutral", "score": 0.0}
            
            results.append({
                "start": start,
                "end": end,
                "text": text,
                "speaker_id": speaker_id,
                "role": role,
                "sentiment": sentiment,
            })
        
        used_fallback = not diarization and use_alternating_fallback
        meta = {
            "speaker_roles": speaker_to_role,
            "diarization_fallback_used": used_fallback,
            "num_segments": len(results),
            "num_speakers": len(speaker_to_role),
        }
        
        if used_fallback:
            meta["note"] = "Speakers assigned by alternating segment order."
        
        return {
            "segments": results,
            "metadata": meta,
        }


def run_unified_analysis(
    audio_path: Path,
    segments: List[Dict[str, Any]],
    output_json_path: Optional[Path] = None,
    output_txt_path: Optional[Path] = None,
    role_1: str = "client",
    role_2: str = "company_representative",
) -> Dict[str, Any]:
    """
    Run full unified analysis on segments.
    
    Returns dict with segments, metadata, and output paths.
    """
    analyzer = UnifiedSpeechAnalyzer()
    
    result = analyzer.analyze_segments(
        audio_path,
        segments,
        role_1=role_1,
        role_2=role_2,
    )
    
    # Write outputs
    base = audio_path.parent / audio_path.stem
    out_json = output_json_path or (base.parent / f"{base.name}_unified_analysis.json")
    out_txt = output_txt_path or (base.parent / f"{base.name}_unified_analysis.txt")
    
    out_json = Path(out_json)
    out_txt = Path(out_txt)
    
    # Save JSON
    out_json.write_text(
        json.dumps(
            {
                "segments": result["segments"],
                **result["metadata"],
            },
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )
    
    # Save TXT
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
    """CLI entry point."""
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    
    script_dir = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = script_dir / "data"
    default_audio = data_dir / "video_from_bucket_audio.ogg"
    
    if len(sys.argv) < 2:
        print("Usage: python -m app.backend.unified_speech_analysis <segments_json> [audio_path]", file=sys.stderr)
        return 1
    
    segments_path = Path(sys.argv[1])
    audio_path = Path(sys.argv[2]) if len(sys.argv) > 2 else default_audio
    
    if not segments_path.is_file():
        print(f"Error: segments JSON not found: {segments_path}", file=sys.stderr)
        return 1
    
    if not audio_path.is_file():
        print(f"Error: audio not found: {audio_path}", file=sys.stderr)
        return 1
    
    # Load segments
    segments_data = json.loads(segments_path.read_text(encoding="utf-8"))
    segments = segments_data.get("segments", [])
    
    role_1 = os.environ.get("SPEAKER_1_ROLE", "client")
    role_2 = os.environ.get("SPEAKER_2_ROLE", "company_representative")
    
    print(f"Analyzing {len(segments)} segments...")
    result = run_unified_analysis(
        audio_path,
        segments,
        role_1=role_1,
        role_2=role_2,
    )
    
    print("\n=== Unified Analysis Complete ===")
    print(f"JSON: {result['output_json']}")
    print(f"TXT: {result['output_txt']}")
    print(f"\nSummary: {len(segments)} segments, {result['num_speakers']} speakers")
    
    for r in result["segments"][:5]:
        print(f"  [{r['start']:.1f}s] {r['role']}: {r['text'][:50]}... | {r['sentiment'].get('label', '?')}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
