"""
Text analysis module: speaker diarization + sentiment on timestamped transcript.
"""
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_SPEAKER_1_ROLE = "client"
DEFAULT_SPEAKER_2_ROLE = "company_representative"


def load_segments(segments_path: Path) -> List[Dict[str, Any]]:
    """Load Whisper segments JSON."""
    data = json.loads(segments_path.read_text(encoding="utf-8"))
    return data.get("segments", [])


def diarize_audio(
    audio_path: Path,
    num_speakers: int = 2,
    token: Optional[str] = None,
    _log_reason: Optional[list] = None,
) -> List[Tuple[float, float, str]]:
    """
    Run pyannote speaker diarization. Returns list of (start, end, speaker_id).
    """
    reason = []
    try:
        from pyannote.audio import Pipeline
    except ImportError:
        reason.append("pyannote.audio not installed (pip install pyannote.audio)")
        if _log_reason is not None:
            _log_reason.extend(reason)
        return []
    token = token or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        reason.append("HUGGINGFACE_TOKEN not set in .env")
        if _log_reason is not None:
            _log_reason.extend(reason)
        return []
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1",
            use_auth_token=token,
        )
        diar = pipeline(str(audio_path), num_speakers=num_speakers, min_duration_off=0.0)
        result = [(segment.start, segment.end, speaker) for segment, _track, speaker in diar.itertracks(yield_label=True)]
        return result
    except Exception as e:
        reason.append(f"pyannote failed: {e}")
        if _log_reason is not None:
            _log_reason.extend(reason)
        return []


def assign_speaker_to_segment(
    segment_start: float,
    segment_end: float,
    diarization: List[Tuple[float, float, str]],
) -> Optional[str]:
    """Assign a speaker to a segment by maximum overlap with diarization segments."""
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


def run_sentiment_ru(text: str) -> Dict[str, Any]:
    """Run Russian sentiment on a segment. Returns label and score."""
    if not text.strip():
        return {"label": "neutral", "score": 0.0}
    try:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except ImportError:
        return {"label": "unknown", "score": 0.0, "error": "transformers/torch not installed"}
    model_name = "blanchefort/rubert-base-cased-sentiment-rusentiment"
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
    except Exception as e:
        return {"label": "unknown", "score": 0.0, "error": str(e)}
    labels = ["negative", "neutral", "positive"]
    inputs = tokenizer(text[:512], return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).numpy()[0]
    idx = int(probs.argmax())
    return {"label": labels[idx], "score": float(probs[idx])}


def run_diarize_sentiment(
    audio_path: Path,
    segments_path: Path,
    output_json_path: Optional[Path] = None,
    output_txt_path: Optional[Path] = None,
    role_1: str = DEFAULT_SPEAKER_1_ROLE,
    role_2: str = DEFAULT_SPEAKER_2_ROLE,
    use_alternating_fallback: bool = True,
) -> Dict[str, Any]:
    """
    Run diarization + sentiment on segments. Returns result dict with segments and meta.
    """
    segments = load_segments(segments_path)
    if not segments:
        raise ValueError("No segments in JSON.")

    diarization_reason: List[str] = []
    diarization = diarize_audio(audio_path, num_speakers=2, token=os.environ.get("HUGGINGFACE_TOKEN"), _log_reason=diarization_reason)
    speaker_to_role = {}
    if diarization:
        speakers = sorted(set(s[2] for s in diarization))
        for i, sp in enumerate(speakers):
            speaker_to_role[sp] = role_1 if i == 0 else role_2
    else:
        if use_alternating_fallback:
            speaker_to_role = {"SPEAKER_00": role_1, "SPEAKER_01": role_2}

    results = []
    for idx, seg in enumerate(segments):
        start, end = seg["start"], seg["end"]
        text = seg.get("text", "").strip()
        speaker_id = assign_speaker_to_segment(start, end, diarization) if diarization else None
        if speaker_id is not None:
            role = speaker_to_role.get(speaker_id, "speaker_unknown")
        elif use_alternating_fallback and not diarization:
            role = role_1 if (idx % 2 == 0) else role_2
            speaker_id = "SPEAKER_00" if (idx % 2 == 0) else "SPEAKER_01"
        else:
            role = "speaker_unknown"
        sentiment = run_sentiment_ru(text) if text else {"label": "neutral", "score": 0.0}
        results.append({
            "start": start, "end": end, "text": text,
            "speaker_id": speaker_id, "role": role,
            "sentiment": sentiment,
        })

    used_fallback = not diarization and use_alternating_fallback
    meta = {"speaker_roles": speaker_to_role, "diarization_fallback_used": used_fallback}
    if used_fallback:
        meta["note"] = "Speakers assigned by alternating segment order."

    base = audio_path.parent / audio_path.stem
    out_json = output_json_path or (base.parent / f"{base.name}_diarized_sentiment.json")
    out_txt = output_txt_path or (base.parent / f"{base.name}_diarized_sentiment.txt")
    out_json.write_text(json.dumps({"segments": results, **meta}, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = []
    for r in results:
        lines.append(f"[{r['start']:.1f}s - {r['end']:.1f}s] {r['role']}: {r['text']}")
        lines.append(f"  sentiment: {r['sentiment'].get('label', '?')} ({r['sentiment'].get('score', 0):.2f})")
    out_txt.write_text("\n".join(lines), encoding="utf-8")

    return {"segments": results, **meta, "output_json": str(out_json), "output_txt": str(out_txt)}


def main() -> int:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    script_dir = Path(__file__).resolve().parent.parent
    data_dir = script_dir / "data"
    default_audio = data_dir / "video_from_bucket_audio.ogg"
    default_segments = data_dir / "video_from_bucket_audio_segments.json"

    audio_path = Path(sys.argv[1]) if len(sys.argv) >= 2 else default_audio
    segments_path = Path(sys.argv[2]) if len(sys.argv) >= 3 else (audio_path.parent / f"{audio_path.stem}_segments.json")
    if not default_segments.exists() and segments_path == default_segments:
        segments_path = data_dir / "video_from_bucket_audio_segments.json"

    if not audio_path.is_file():
        print(f"Error: audio not found: {audio_path}", file=sys.stderr)
        return 1
    if not segments_path.is_file():
        print(f"Error: segments JSON not found: {segments_path}. Run text_extraction first.", file=sys.stderr)
        return 1

    role_1 = os.environ.get("SPEAKER_1_ROLE", DEFAULT_SPEAKER_1_ROLE)
    role_2 = os.environ.get("SPEAKER_2_ROLE", DEFAULT_SPEAKER_2_ROLE)
    use_fallback = os.environ.get("USE_ALTERNATING_SPEAKERS_FALLBACK", "1").strip().lower() in ("1", "true", "yes")

    print("Running diarization...")
    result = run_diarize_sentiment(
        audio_path, segments_path,
        role_1=role_1, role_2=role_2,
        use_alternating_fallback=use_fallback,
    )
    print("\n=== Output ===")
    print("JSON:", result["output_json"])
    print("TXT:", result["output_txt"])
    for r in result["segments"][:5]:
        print(f"  {r['role']}: {r['text'][:60]}... | {r['sentiment'].get('label', '?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
