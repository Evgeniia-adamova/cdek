"""
Emotion recognition module: ONNX FER+ inference, report merge, and visualization.
Merged from pipeline steps 5 and 6.
"""
import copy
import json
import os
import time
import urllib.request
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

# Optional matplotlib for visual
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


DEFAULT_MODEL_URL = (
    "https://github.com/onnx/models/raw/main/"
    "validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
)

EMOTION_LABELS = [
    "neutral", "happiness", "surprise", "sadness",
    "anger", "disgust", "fear", "contempt",
]

EMOTION_VALENCE = {
    "happiness": 1.0, "surprise": 0.4, "neutral": 0.0,
    "sadness": -0.7, "anger": -1.0, "disgust": -0.9,
    "fear": -0.8, "contempt": -0.6,
}


def _default_model_path() -> str:
    return os.path.join(os.path.dirname(__file__), "models", "emotion-ferplus-8.onnx")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_ferplus_onnx(
    model_path: Optional[str] = None,
    model_url: str = DEFAULT_MODEL_URL,
) -> str:
    if model_path is None:
        model_path = _default_model_path()
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path):
        return model_path
    print(f"Downloading ONNX emotion model: {model_url}")
    urllib.request.urlretrieve(model_url, model_path)
    return model_path


def collect_samples(
    analysis_path: str,
    max_per_person: int = 300,
    analysis_person_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """
    Collect face crop paths from step2 analysis for emotion inference.
    If analysis_person_ids is set (e.g. [P001, P002]), only those persons are included.
    Note: face crops are stored on Yandex Cloud; crop_path in JSON may be local
    or a key/URL — ensure files are available (e.g. downloaded from bucket) before inference.
    """
    data = load_json(analysis_path)
    frames = data.get("frames", [])
    allowed_ids = set(analysis_person_ids) if analysis_person_ids else None
    samples = []
    person_counter = defaultdict(int)
    for fr in frames:
        for det in fr.get("detections", []):
            pid = det.get("person_id")
            crop_path = det.get("crop_path")
            if not pid or not crop_path or not os.path.exists(crop_path):
                continue
            if allowed_ids is not None and pid not in allowed_ids:
                continue
            if person_counter[pid] >= max_per_person:
                continue
            person_counter[pid] += 1
            samples.append({
                "person_id": pid,
                "sample_index": fr.get("sample_index"),
                "timestamp_sec": fr.get("timestamp_sec"),
                "timestamp": fr.get("timestamp"),
                "interval_id": fr.get("interval_id"),
                "crop_path": crop_path,
            })
    return samples


def preprocess_ferplus(face_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    x = gray.astype(np.float32)
    x = np.expand_dims(np.expand_dims(x, axis=0), axis=0)
    return x


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / np.sum(e)


def aggregate_predictions(
    predictions: List[Dict[str, Any]],
    analysis_path: str,
    model_name: str,
    elapsed_sec: float,
    max_per_person: int,
) -> Dict[str, Any]:
    by_person = {}
    by_interval = defaultdict(lambda: defaultdict(Counter))
    for p in predictions:
        pid = p["person_id"]
        interval_id = str(p.get("interval_id", -1))
        emo = p["emotion_label"]
        if pid not in by_person:
            by_person[pid] = {"person_id": pid, "total_samples": 0, "emotion_counts": Counter()}
        by_person[pid]["total_samples"] += 1
        by_person[pid]["emotion_counts"][emo] += 1
        by_interval[pid][interval_id][emo] += 1
    by_person_out = {}
    for pid, stats in sorted(by_person.items()):
        total = max(1, stats["total_samples"])
        counts = dict(stats["emotion_counts"])
        dominant = max(counts, key=counts.get) if counts else "unknown"
        by_person_out[pid] = {
            "person_id": pid,
            "total_samples": stats["total_samples"],
            "emotion_counts": counts,
            "dominant_emotion": dominant,
            "dominant_share": round(counts.get(dominant, 0) / total, 4),
        }
    by_interval_out = {
        pid: {iid: dict(counter) for iid, counter in sorted(intervals.items(), key=lambda x: int(x[0]))}
        for pid, intervals in by_interval.items()
    }
    unknown_count = sum(1 for p in predictions if p.get("emotion_label") == "unknown")
    return {
        "model": model_name,
        "analysis_path": analysis_path,
        "total_predictions": len(predictions),
        "unknown_share": round(unknown_count / max(1, len(predictions)), 4),
        "elapsed_sec": round(elapsed_sec, 3),
        "throughput_samples_per_sec": round(len(predictions) / max(1e-6, elapsed_sec), 3),
        "max_per_person": max_per_person,
        "by_person": by_person_out,
        "by_person_interval": by_interval_out,
        "predictions": predictions,
    }


def run_emotion_baseline_onnx(
    analysis_path: str = "extracted_frames_v3_dense/analysis_step2_dense_consolidated.json",
    output_path: str = "extracted_frames_v3_dense/emotion_baseline_onnx_ferplus.json",
    model_path: Optional[str] = None,
    max_per_person: int = 300,
    analysis_person_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    import onnxruntime as ort
    model_path = ensure_ferplus_onnx(model_path=model_path)
    if analysis_person_ids is None:
        data = load_json(analysis_path)
        analysis_person_ids = data.get("analysis_person_ids")
    samples = collect_samples(
        analysis_path=analysis_path,
        max_per_person=max_per_person,
        analysis_person_ids=analysis_person_ids,
    )
    if not samples:
        raise RuntimeError("No face crop files for emotion inference.")
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    started = time.time()
    predictions = []
    for s in samples:
        img = cv2.imread(s["crop_path"])
        if img is None:
            continue
        x = preprocess_ferplus(img)
        logits = sess.run(None, {input_name: x})[0][0]
        probs = softmax(logits)
        idx = int(np.argmax(probs))
        label = EMOTION_LABELS[idx] if 0 <= idx < len(EMOTION_LABELS) else "unknown"
        confidence = float(probs[idx]) if 0 <= idx < len(probs) else 0.0
        predictions.append({
            **s,
            "emotion_label": label,
            "emotion_confidence": round(confidence, 6),
            "model": "onnx_ferplus",
        })
    elapsed = time.time() - started
    result = aggregate_predictions(
        predictions=predictions,
        analysis_path=analysis_path,
        model_name="onnx_ferplus",
        elapsed_sec=elapsed,
        max_per_person=max_per_person,
    )
    save_json(result, output_path)
    print("=== EMOTION (ONNX FER+) ===")
    print(f"Model path: {model_path}")
    print(f"Samples inferred: {result['total_predictions']}")
    print(f"Saved: {output_path}")
    return result


# --- Step 6: Merge emotions into frames and build report ---

def _build_prediction_indexes(
    predictions: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, int], Dict[str, Any]]]:
    by_crop = {}
    by_person_sample = {}
    for pred in predictions:
        crop_path = pred.get("crop_path")
        if crop_path:
            by_crop[str(crop_path)] = pred
        pid = str(pred.get("person_id", ""))
        sample_index = int(pred.get("sample_index", -1))
        if pid and sample_index >= 0:
            by_person_sample[(pid, sample_index)] = pred
    return by_crop, by_person_sample


def merge_emotions_into_frames(
    analysis_data: Dict[str, Any],
    emotion_data: Dict[str, Any],
) -> Dict[str, Any]:
    out = copy.deepcopy(analysis_data)
    frames = out.get("frames", [])
    predictions = emotion_data.get("predictions", [])
    pred_by_crop, pred_by_person_sample = _build_prediction_indexes(predictions)
    attached = 0
    for fr in frames:
        sample_index = int(fr.get("sample_index", -1))
        for det in fr.get("detections", []):
            pid = str(det.get("person_id", ""))
            pred = pred_by_crop.get(str(det.get("crop_path", ""))) if det.get("crop_path") else None
            if pred is None and pid and sample_index >= 0:
                pred = pred_by_person_sample.get((pid, sample_index))
            if pred is not None:
                det["emotion"] = {
                    "label": pred.get("emotion_label", "unknown"),
                    "confidence": pred.get("emotion_confidence"),
                    "model": emotion_data.get("model", "unknown"),
                }
                attached += 1
            else:
                det.setdefault("emotion", {"label": "unknown", "confidence": None, "model": emotion_data.get("model", "unknown")})
    out["emotion_merge_info"] = {
        "total_predictions": len(predictions),
        "attached_to_detections": attached,
        "model": emotion_data.get("model", "unknown"),
    }
    return out


def _person_interval_stats(dets: List[Dict[str, Any]]) -> Dict[str, Any]:
    counts = Counter()
    conf_sum, conf_n, valence_sum = 0.0, 0, 0.0
    for d in dets:
        emo = d.get("emotion", {})
        label = str(emo.get("label", "unknown"))
        counts[label] += 1
        conf = emo.get("confidence")
        if conf is not None:
            conf_sum += float(conf)
            conf_n += 1
        valence_sum += EMOTION_VALENCE.get(label, 0.0)
    total = sum(counts.values())
    dominant = counts.most_common(1)[0][0] if total > 0 else "unknown"
    dominant_share = (counts[dominant] / total) if total > 0 else 0.0
    avg_conf = conf_sum / conf_n if conf_n > 0 else None
    valence = valence_sum / total if total > 0 else 0.0
    return {
        "total_detections": total,
        "emotion_counts": dict(counts),
        "dominant_emotion": dominant,
        "dominant_share": round(dominant_share, 4),
        "avg_confidence": round(avg_conf, 4) if avg_conf is not None else None,
        "valence_score": round(valence, 4),
    }


def build_interval_emotion_report(
    frames: List[Dict[str, Any]],
    allowed_person_ids: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    allowed = set(allowed_person_ids) if allowed_person_ids else None
    buckets = {}
    for fr in frames:
        interval_id = int(fr.get("interval_id", -1))
        if interval_id < 0:
            continue
        if interval_id not in buckets:
            buckets[interval_id] = {"interval_id": interval_id, "start_ts": fr.get("timestamp"), "frame_count": 0, "people": defaultdict(list)}
        b = buckets[interval_id]
        b["frame_count"] += 1
        for det in fr.get("detections", []):
            pid = str(det.get("person_id", ""))
            if not pid or (allowed is not None and pid not in allowed):
                continue
            b["people"][pid].append(det)
    report = []
    for interval_id in sorted(buckets.keys()):
        b = buckets[interval_id]
        person_stats = {pid: _person_interval_stats(dets) for pid, dets in b["people"].items()}
        report.append({
            "interval_id": interval_id,
            "start_ts": b["start_ts"],
            "frame_count": b["frame_count"],
            "persons": sorted(person_stats.keys()),
            "person_emotions": person_stats,
        })
    return report


def compare_interval_emotions(interval_report: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    comparisons = []
    for i in range(1, len(interval_report)):
        prev_it, curr_it = interval_report[i - 1], interval_report[i]
        shared = sorted(set(prev_it["persons"]) & set(curr_it["persons"]))
        changes = []
        for pid in shared:
            prev_stats = prev_it["person_emotions"].get(pid, {})
            curr_stats = curr_it["person_emotions"].get(pid, {})
            changes.append({
                "person_id": pid,
                "dominant_changed": prev_stats.get("dominant_emotion") != curr_stats.get("dominant_emotion"),
                "prev_dominant": prev_stats.get("dominant_emotion"),
                "curr_dominant": curr_stats.get("dominant_emotion"),
                "valence_delta": round(float(curr_stats.get("valence_score", 0.0)) - float(prev_stats.get("valence_score", 0.0)), 4),
                "dominant_share_delta": round(float(curr_stats.get("dominant_share", 0.0)) - float(prev_stats.get("dominant_share", 0.0)), 4),
            })
        comparisons.append({
            "from_interval_id": prev_it["interval_id"],
            "to_interval_id": curr_it["interval_id"],
            "from_ts": prev_it.get("start_ts"),
            "to_ts": curr_it.get("start_ts"),
            "shared_person_changes": changes,
        })
    return comparisons


def build_person_emotion_timeline(interval_report: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_person = defaultdict(list)
    for row in interval_report:
        interval_id = row["interval_id"]
        start_ts = row.get("start_ts")
        for pid, stats in row["person_emotions"].items():
            per_person[pid].append({
                "interval_id": interval_id,
                "start_ts": start_ts,
                "dominant_emotion": stats.get("dominant_emotion"),
                "dominant_share": stats.get("dominant_share"),
                "valence_score": stats.get("valence_score"),
            })
    summary = {}
    for pid, timeline in sorted(per_person.items()):
        dominant_counter = Counter([x["dominant_emotion"] for x in timeline if x["dominant_emotion"]])
        most_common = dominant_counter.most_common(1)[0][0] if dominant_counter else "unknown"
        avg_valence = sum(float(x.get("valence_score", 0.0)) for x in timeline) / max(1, len(timeline))
        summary[pid] = {
            "person_id": pid,
            "intervals": timeline,
            "most_common_interval_emotion": most_common,
            "avg_valence_score": round(avg_valence, 4),
        }
    return summary


def run_step6_emotion_report(
    analysis_path: str = "extracted_frames_v3_dense/analysis_step3_extended.json",
    emotion_path: str = "extracted_frames_v3_dense/emotion_baseline_onnx_ferplus.json",
    output_path: str = "extracted_frames_v3_dense/analysis_step6_emotion_report.json",
) -> Dict[str, Any]:
    analysis_data = load_json(analysis_path)
    emotion_data = load_json(emotion_path)
    merged = merge_emotions_into_frames(analysis_data, emotion_data)
    frames = merged.get("frames", [])
    analysis_person_ids = analysis_data.get("analysis_person_ids")
    interval_report = build_interval_emotion_report(frames, allowed_person_ids=analysis_person_ids)
    interval_comparison = compare_interval_emotions(interval_report)
    person_timeline = build_person_emotion_timeline(interval_report)
    result = {
        "meta": {"analysis_path": analysis_path, "emotion_path": emotion_path, "emotion_model": emotion_data.get("model"), "analysis_person_ids": analysis_person_ids},
        "emotion_merge_info": merged.get("emotion_merge_info", {}),
        "interval_emotion_report": interval_report,
        "interval_emotion_comparison": interval_comparison,
        "person_emotion_timeline": person_timeline,
    }
    save_json(result, output_path)
    print("=== EMOTION REPORT ===")
    print(f"Intervals: {len(interval_report)}")
    print(f"Saved: {output_path}")
    return result


def plot_emotion_report(
    report_path: str,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
) -> str:
    if not _HAS_MATPLOTLIB:
        raise ImportError("matplotlib is required for emotion visual. pip install matplotlib")
    data = load_json(report_path)
    interval_report = data.get("interval_emotion_report", [])
    person_timeline = data.get("person_emotion_timeline", {})
    if not interval_report and not person_timeline:
        raise ValueError(f"No interval_emotion_report or person_emotion_timeline in {report_path}")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[1.2, 1])
    intervals = sorted(r["interval_id"] for r in interval_report)
    if intervals:
        for pid, info in sorted(person_timeline.items()):
            timeline = info.get("intervals", [])
            if not timeline:
                continue
            by_i = {t["interval_id"]: t for t in timeline}
            ys = [by_i.get(i, {}).get("valence_score", 0.0) for i in intervals]
            ax1.plot(intervals, ys, marker="o", markersize=4, label=f"Person {pid}", alpha=0.9)
        ax1.set_xlabel("Interval")
        ax1.set_ylabel("Valence score")
        ax1.set_title("Emotion valence over time (per person)")
        ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax1.legend(loc="best", fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-1.1, 1.1)
    emotion_counts = Counter()
    for row in interval_report:
        for pid, stats in row.get("person_emotions", {}).items():
            for emotion, count in stats.get("emotion_counts", {}).items():
                emotion_counts[emotion] += count
    if emotion_counts:
        labels = list(emotion_counts.keys())
        counts = [emotion_counts[e] for e in labels]
        colors = plt.cm.RdYlGn([(c - min(counts)) / max((max(counts) - min(counts)) or 1, 1) * 0.5 + 0.25 for c in counts])
        bars = ax2.bar(labels, counts, color=colors, edgecolor="gray", linewidth=0.5)
        ax2.set_xlabel("Emotion")
        ax2.set_ylabel("Count")
        ax2.set_title("Overall emotion distribution")
        ax2.tick_params(axis="x", rotation=45)
        for bar, c in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(c), ha="center", va="bottom", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No emotion counts in report", ha="center", va="center", transform=ax2.transAxes)
    plt.tight_layout()
    if output_path is None:
        output_path = f"{os.path.splitext(report_path)[0]}_visual.png"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def run_emotion_visual(
    report_path: str = "extracted_frames_v3_dense/analysis_step6_emotion_report.json",
    output_path: Optional[str] = None,
) -> str:
    out = plot_emotion_report(report_path, output_path=output_path)
    print(f"Emotion visual saved: {out}")
    return out


if __name__ == "__main__":
    run_emotion_baseline_onnx()
