import json
import os
import time
import urllib.request
from collections import Counter, defaultdict
from typing import Any, Dict, List

import cv2
import numpy as np
import onnxruntime as ort


DEFAULT_MODEL_URL = (
    "https://github.com/onnx/models/raw/main/"
    "validated/vision/body_analysis/emotion_ferplus/model/emotion-ferplus-8.onnx"
)

# ONNX Model Zoo FER+ label order.
EMOTION_LABELS = [
    "neutral",
    "happiness",
    "surprise",
    "sadness",
    "anger",
    "disgust",
    "fear",
    "contempt",
]


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_ferplus_onnx(
    model_path: str = "models/emotion-ferplus-8.onnx",
    model_url: str = DEFAULT_MODEL_URL,
) -> str:
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if os.path.exists(model_path):
        return model_path
    print(f"Downloading ONNX emotion model: {model_url}")
    urllib.request.urlretrieve(model_url, model_path)
    return model_path


def collect_samples(
    analysis_path: str,
    max_per_person: int = 300,
) -> List[Dict[str, Any]]:
    data = load_json(analysis_path)
    frames = data.get("frames", [])

    samples: List[Dict[str, Any]] = []
    person_counter: Dict[str, int] = defaultdict(int)

    for fr in frames:
        for det in fr.get("detections", []):
            pid = det.get("person_id")
            crop_path = det.get("crop_path")
            if not pid or not crop_path:
                continue
            if not os.path.exists(crop_path):
                continue
            if person_counter[pid] >= max_per_person:
                continue
            samples.append(
                {
                    "person_id": pid,
                    "sample_index": fr.get("sample_index"),
                    "timestamp_sec": fr.get("timestamp_sec"),
                    "timestamp": fr.get("timestamp"),
                    "interval_id": fr.get("interval_id"),
                    "crop_path": crop_path,
                }
            )
            person_counter[pid] += 1
    return samples


def preprocess_ferplus(face_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    x = gray.astype(np.float32)
    x = np.expand_dims(np.expand_dims(x, axis=0), axis=0)  # [1,1,64,64]
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
    by_person: Dict[str, Dict[str, Any]] = {}
    by_interval: Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

    for p in predictions:
        pid = p["person_id"]
        interval_id = str(p.get("interval_id", -1))
        emo = p["emotion_label"]

        if pid not in by_person:
            by_person[pid] = {
                "person_id": pid,
                "total_samples": 0,
                "emotion_counts": Counter(),
            }
        by_person[pid]["total_samples"] += 1
        by_person[pid]["emotion_counts"][emo] += 1
        by_interval[pid][interval_id][emo] += 1

    by_person_out: Dict[str, Dict[str, Any]] = {}
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

    by_interval_out: Dict[str, Dict[str, Dict[str, int]]] = {}
    for pid, intervals in by_interval.items():
        by_interval_out[pid] = {
            iid: dict(counter)
            for iid, counter in sorted(intervals.items(), key=lambda x: int(x[0]))
        }

    unknown_count = sum(1 for p in predictions if p.get("emotion_label") == "unknown")
    throughput = len(predictions) / max(1e-6, elapsed_sec)

    return {
        "model": model_name,
        "analysis_path": analysis_path,
        "total_predictions": len(predictions),
        "unknown_share": round(unknown_count / max(1, len(predictions)), 4),
        "elapsed_sec": round(elapsed_sec, 3),
        "throughput_samples_per_sec": round(throughput, 3),
        "max_per_person": max_per_person,
        "by_person": by_person_out,
        "by_person_interval": by_interval_out,
        "predictions": predictions,
    }


def run_emotion_baseline_onnx(
    analysis_path: str = "extracted_frames_v3_dense/analysis_step2_dense_consolidated.json",
    output_path: str = "extracted_frames_v3_dense/emotion_baseline_onnx_ferplus.json",
    model_path: str = "models/emotion-ferplus-8.onnx",
    max_per_person: int = 300,
) -> Dict[str, Any]:
    model_path = ensure_ferplus_onnx(model_path=model_path)
    samples = collect_samples(analysis_path=analysis_path, max_per_person=max_per_person)
    if not samples:
        raise RuntimeError("No face crop files for emotion inference.")

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    started = time.time()
    predictions: List[Dict[str, Any]] = []

    for s in samples:
        img = cv2.imread(s["crop_path"])
        if img is None:
            continue
        x = preprocess_ferplus(img)
        logits = sess.run(None, {input_name: x})[0][0]  # [8]
        probs = softmax(logits)
        idx = int(np.argmax(probs))
        label = EMOTION_LABELS[idx] if 0 <= idx < len(EMOTION_LABELS) else "unknown"
        confidence = float(probs[idx]) if 0 <= idx < len(probs) else 0.0

        predictions.append(
            {
                **s,
                "emotion_label": label,
                "emotion_confidence": round(confidence, 6),
                "model": "onnx_ferplus",
            }
        )

    elapsed = time.time() - started
    result = aggregate_predictions(
        predictions=predictions,
        analysis_path=analysis_path,
        model_name="onnx_ferplus",
        elapsed_sec=elapsed,
        max_per_person=max_per_person,
    )
    save_json(result, output_path)

    print("=== STEP 5C: EMOTION BASELINE (ONNX FER+) ===")
    print(f"Model path: {model_path}")
    print(f"Samples inferred: {result['total_predictions']}")
    print(f"Unknown share: {result['unknown_share']:.3f}")
    print(f"Throughput: {result['throughput_samples_per_sec']:.2f} samples/sec")
    print(f"Persons: {list(result['by_person'].keys())}")
    print(f"Saved: {output_path}")
    return result


if __name__ == "__main__":
    run_emotion_baseline_onnx()
