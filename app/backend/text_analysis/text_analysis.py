"""
Text analysis module: speaker diarization + sentiment on timestamped transcript.
Uses pyannote.audio when HUGGINGFACE_TOKEN is set, falls back to alternating order.
"""
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_SPEAKER_1_ROLE = "client"
DEFAULT_SPEAKER_2_ROLE = "company_representative"


def load_segments(segments_path: Path) -> List[Dict[str, Any]]:
    data = json.loads(segments_path.read_text(encoding="utf-8"))
    return data.get("segments", [])


_sentiment_model_cache: Dict[str, Any] = {}


def _get_sentiment_model():
    if _sentiment_model_cache:
        return _sentiment_model_cache["tokenizer"], _sentiment_model_cache["model"], _sentiment_model_cache["id2label"]
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model_name = "blanchefort/rubert-base-cased-sentiment-rusentiment"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    _sentiment_model_cache["tokenizer"] = tokenizer
    _sentiment_model_cache["model"] = model
    _sentiment_model_cache["id2label"] = id2label
    return tokenizer, model, id2label


def run_sentiment_ru(text: str) -> Dict[str, Any]:
    if not text.strip():
        return {"label": "neutral", "score": 0.0}
    try:
        import torch
        tokenizer, model, id2label = _get_sentiment_model()
    except ImportError:
        return {"label": "unknown", "score": 0.0, "error": "transformers/torch not installed"}
    except Exception as e:
        return {"label": "unknown", "score": 0.0, "error": str(e)}
    inputs = tokenizer(text[:512], return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.softmax(logits, dim=1).numpy()[0]
    idx = int(probs.argmax())
    return {"label": id2label.get(idx, "unknown").lower(), "score": float(probs[idx])}


def _load_audio_waveform(audio_path: Path) -> Dict[str, Any]:
    """Load audio as {waveform, sample_rate} without torchcodec/FFmpeg system DLLs.

    Tries soundfile (WAV/FLAC/OGG Vorbis), then imageio-ffmpeg conversion (OGG Opus etc.).
    """
    import numpy as np
    import torch

    # soundfile: works for WAV, FLAC, OGG Vorbis (not Opus)
    try:
        import soundfile as sf
        data, sr = sf.read(str(audio_path), dtype="float32", always_2d=True)
        return {"waveform": torch.from_numpy(data.T), "sample_rate": sr}
    except Exception:
        pass

    # imageio-ffmpeg: bundles FFmpeg binaries, handles OGG Opus and everything else
    import subprocess
    import tempfile
    import os
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        raise RuntimeError(
            "Cannot decode audio. Run: pip install imageio-ffmpeg"
        )
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            [ffmpeg_exe, "-i", str(audio_path), "-ar", "16000", "-ac", "1", "-y", tmp_path],
            capture_output=True, check=True,
        )
        import soundfile as sf
        data, sr = sf.read(tmp_path, dtype="float32", always_2d=True)
        return {"waveform": torch.from_numpy(data.T), "sample_rate": sr}
    finally:
        os.unlink(tmp_path)


def _patch_compat() -> None:
    """Patch torchaudio 2.x and huggingface_hub 1.x API breaks vs pyannote 3.x."""
    import collections
    import functools
    import warnings
    # suppress torchcodec/FFmpeg load noise — torchcodec is not needed (we pass waveforms directly)
    warnings.filterwarnings("ignore", message="torchcodec is not installed correctly")

    # --- torchaudio 2.x removed APIs ---
    import torchaudio
    if not hasattr(torchaudio, "AudioMetaData"):
        torchaudio.AudioMetaData = collections.namedtuple(
            "AudioMetaData",
            ["sample_rate", "num_frames", "num_channels", "bits_per_sample", "encoding"],
        )
    if not hasattr(torchaudio, "list_audio_backends"):
        torchaudio.list_audio_backends = lambda: ["soundfile"]
    if not hasattr(torchaudio, "get_audio_backend"):
        torchaudio.get_audio_backend = lambda: "soundfile"
    if not hasattr(torchaudio, "set_audio_backend"):
        torchaudio.set_audio_backend = lambda backend: None

    # --- torch 2.6+ / lightning 2.x pass weights_only=True, breaking pyannote 3.x ---
    # Force weights_only=False on every torch.load call (pyannote checkpoints are trusted).
    import torch
    import torch.serialization
    if not getattr(torch.load, "_compat_patched", False):
        _orig_load = torch.serialization.load
        def _load_compat(*args, **kwargs):
            kwargs["weights_only"] = False  # force, override any explicit True
            return _orig_load(*args, **kwargs)
        _load_compat._compat_patched = True
        torch.serialization.load = _load_compat
        torch.load = _load_compat
        # Also patch lightning's internal loader if present
        try:
            import lightning.fabric.utilities.cloud_io as _lf
            if hasattr(_lf, "_load"):
                _lf._load = lambda path, map_location=None, **kwargs: _load_compat(path, map_location=map_location)
        except Exception:
            pass

    # --- speechbrain registers DeprecatedModuleRedirect proxies that fail on access ---
    # Import speechbrain first so its __init__ registers those proxies, then replace
    # every non-real-module entry with a proper empty stub.
    import sys
    import types
    try:
        import speechbrain as _sb  # noqa: F401 — triggers proxy registration
        for _key in list(sys.modules.keys()):
            if "speechbrain" in _key:
                _mod = sys.modules[_key]
                # DeprecatedModuleRedirect (and any other proxy) has type name != 'module'
                if _mod is not None and type(_mod).__name__ != "module":
                    sys.modules[_key] = types.ModuleType(_key)
        # Pre-stub integration submodules that may be imported later
        for _stub_name in (
            "speechbrain.integrations",
            "speechbrain.integrations.k2_fsa",
            "speechbrain.integrations.nlp",
            "speechbrain.integrations.huggingface",
            "k2",
        ):
            sys.modules.setdefault(_stub_name, types.ModuleType(_stub_name))
    except Exception:
        pass

    # --- huggingface_hub 1.x removed use_auth_token ---
    # pyannote 3.x calls hf_hub_download/snapshot_download(use_auth_token=...)
    # HF_TOKEN env var handles auth; just drop the deprecated kwarg.
    import huggingface_hub

    def _drop_use_auth_token(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            kwargs.pop("use_auth_token", None)
            return fn(*args, **kwargs)
        wrapper._compat_patched = True
        return wrapper

    for _name in ("hf_hub_download", "snapshot_download", "cached_download"):
        _orig = getattr(huggingface_hub, _name, None)
        if _orig and not getattr(_orig, "_compat_patched", False):
            setattr(huggingface_hub, _name, _drop_use_auth_token(_orig))


def _diarize_with_pyannote(audio_path: Path, hf_token: str) -> List[Dict[str, Any]]:
    """Return [{start, end, speaker}, ...] using pyannote/speaker-diarization-3.1."""
    _patch_compat()

    from pyannote.audio import Pipeline

    # Try v4 API (token=), fall back to v3 (use_auth_token=)
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
        )
    except TypeError:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token,
        )

    # Load audio without torchcodec/FFmpeg, pass as waveform dict to pyannote
    audio_input = _load_audio_waveform(audio_path)
    diarization = pipeline(audio_input, num_speakers=2)
    # pyannote 3.3+ returns DiarizeOutput; older versions return Annotation directly
    annotation = diarization.speaker_diarization if hasattr(diarization, "speaker_diarization") else diarization
    return [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]


def _label_segments_by_overlap(
    whisper_segments: List[Dict[str, Any]],
    diar_segments: List[Dict[str, Any]],
    role_1: str,
    role_2: str,
) -> List[Dict[str, Any]]:
    """Assign pyannote speaker IDs to Whisper segments by maximum time overlap.

    Role assignment by total speaking time:
    - Most speech → role_2 (company_representative: manager explains and instructs)
    - Less speech  → role_1 (client: asks questions, listens)
    """
    # Total speaking duration per speaker
    speaker_duration: Dict[str, float] = {}
    for d in diar_segments:
        dur = d["end"] - d["start"]
        speaker_duration[d["speaker"]] = speaker_duration.get(d["speaker"], 0.0) + dur

    # Sort descending: index 0 = most speech = company_representative
    sorted_speakers = sorted(speaker_duration, key=lambda s: -speaker_duration[s])

    def role_for(speaker: Optional[str]) -> str:
        if speaker is None or speaker not in speaker_duration:
            return role_1
        return role_2 if sorted_speakers.index(speaker) == 0 else role_1

    result = []
    for seg in whisper_segments:
        s, e = seg["start"], seg["end"]
        overlaps: Dict[str, float] = {}
        for d in diar_segments:
            ov = min(e, d["end"]) - max(s, d["start"])
            if ov > 0:
                overlaps[d["speaker"]] = overlaps.get(d["speaker"], 0.0) + ov
        speaker = max(overlaps, key=overlaps.get) if overlaps else None
        result.append({**seg, "speaker_id": speaker or "SPEAKER_UNKNOWN", "role": role_for(speaker)})
    return result


def run_diarize_sentiment(
    audio_path: Path,
    segments_path: Path,
    output_json_path: Optional[Path] = None,
    output_txt_path: Optional[Path] = None,
    role_1: str = DEFAULT_SPEAKER_1_ROLE,
    role_2: str = DEFAULT_SPEAKER_2_ROLE,
    use_alternating_fallback: bool = True,
) -> Dict[str, Any]:
    segments = load_segments(segments_path)
    if not segments:
        raise ValueError("No segments in JSON.")

    hf_token = os.environ.get("HUGGINGFACE_TOKEN", "").strip()
    # HF Hub reads HF_TOKEN globally; sync it so the sentiment model also authenticates
    if hf_token and not os.environ.get("HF_TOKEN"):
        os.environ["HF_TOKEN"] = hf_token
    diarization_fallback_used = True
    note = "Speakers assigned by alternating segment order."
    labeled: Optional[List[Dict[str, Any]]] = None

    if hf_token:
        try:
            diar_segments = _diarize_with_pyannote(audio_path, hf_token)
            labeled = _label_segments_by_overlap(segments, diar_segments, role_1, role_2)
            diarization_fallback_used = False
            note = "Speakers assigned by pyannote/speaker-diarization-3.1."
        except Exception as exc:
            print(f"[text_analysis] pyannote diarization failed ({exc}), using alternating fallback.", file=sys.stderr)

    if labeled is None:
        labeled = []
        for idx, seg in enumerate(segments):
            speaker_id = "SPEAKER_00" if (idx % 2 == 0) else "SPEAKER_01"
            role = role_1 if (idx % 2 == 0) else role_2
            labeled.append({**seg, "speaker_id": speaker_id, "role": role})

    results = []
    for seg in labeled:
        text = seg.get("text", "").strip()
        sentiment = run_sentiment_ru(text) if text else {"label": "neutral", "score": 0.0}
        results.append({
            "start": seg["start"], "end": seg["end"], "text": text,
            "speaker_id": seg["speaker_id"], "role": seg["role"],
            "sentiment": sentiment,
        })

    speaker_roles = {r["speaker_id"]: r["role"] for r in results}
    meta = {
        "speaker_roles": speaker_roles,
        "diarization_fallback_used": diarization_fallback_used,
        "note": note,
    }

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
    script_dir = Path(__file__).resolve().parent.parent.parent.parent
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

    result = run_diarize_sentiment(audio_path, segments_path, role_1=role_1, role_2=role_2)
    print("JSON:", result["output_json"])
    print("TXT:", result["output_txt"])
    print("Diarization fallback used:", result["diarization_fallback_used"])
    for r in result["segments"][:5]:
        print(f"  {r['role']}: {r['text'][:60]}... | {r['sentiment'].get('label', '?')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
