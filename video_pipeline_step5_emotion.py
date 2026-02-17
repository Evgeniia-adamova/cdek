import json
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional

import cv2


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _collect_samples(
    analysis_path: str,
    max_per_person: int = 300,
) -> List[Dict[str, Any]]:
    data = load_json(analysis_path)
    frames = data.get("frames", [])

    samples: List[Dict[str, Any]] = []
    per_person_counter: Dict[str, int] = defaultdict(int)

    for fr in frames:
        for det in fr.get("detections", []):
            pid = det.get("person_id")
            crop_path = det.get("crop_path")
            if not pid or not crop_path:
                continue
            if per_person_counter[pid] >= max_per_person:
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
            per_person_counter[pid] += 1
    return samples


def _build_fer_model():
    import importlib

    try:
        fer_module = importlib.import_module("fer")
    except Exception as exc:
        return None, {
            "ok": False,
            "reason": f"import fer failed: {exc}",
        }

    constructors = []
    try:
        constructors.append(("fer.FER", getattr(fer_module, "FER")))
    except Exception:
        pass

    try:
        fer_submodule = importlib.import_module("fer.fer")
        constructors.append(("fer.fer.FER", getattr(fer_submodule, "FER")))
    except Exception:
        pass

    for ctor_name, ctor in constructors:
        try:
            model = ctor(mtcnn=False)
            return model, {
                "ok": True,
                "constructor": ctor_name,
                "module_path": getattr(fer_module, "__file__", None),
            }
        except Exception:
            continue

    return None, {
        "ok": False,
        "reason": "FER class not found in installed fer package",
        "module_path": getattr(fer_module, "__file__", None),
        "available_attrs_head": sorted(dir(fer_module))[:40],
    }


def run_emotion_baseline_fer(
    analysis_path: str = "extracted_frames_v2/analysis_step3_extended.json",
    output_path: str = "extracted_frames_v2/emotion_baseline_fer.json",
    max_per_person: int = 300,
) -> Dict[str, Any]:
    model, model_info = _build_fer_model()
    if model is None:
        raise RuntimeError(
            "Не удалось инициализировать FER.\n"
            f"DIAGNOSTICS: {model_info}\n"
            "Попробуйте в ноутбуке:\n"
            "  1) !pip uninstall -y fer\n"
            "  2) !pip install fer\n"
            "  3) перезапустите kernel"
        )

    samples = _collect_samples(analysis_path, max_per_person=max_per_person)
    if not samples:
        raise RuntimeError("Нет face crop файлов для инференса эмоций.")

    started = time.time()
    predictions: List[Dict[str, Any]] = []

    for s in samples:
        image = cv2.imread(s["crop_path"])
        if image is None:
            continue

        # FER на crop без дополнительного детектора.
        try:
            emotions = model.detect_emotions(image)
        except Exception:
            emotions = []

        if emotions and isinstance(emotions, list):
            best_face = emotions[0]
            emo_probs = best_face.get("emotions", {})
            if emo_probs:
                label = max(emo_probs, key=emo_probs.get)
                confidence = float(emo_probs[label])
            else:
                label = "unknown"
                confidence = 0.0
        else:
            label = "unknown"
            confidence = 0.0

        predictions.append(
            {
                **s,
                "emotion_label": label,
                "emotion_confidence": round(confidence, 6),
                "model": "fer",
            }
        )

    elapsed = max(1e-6, time.time() - started)
    throughput = len(predictions) / elapsed

    by_person: Dict[str, Dict[str, Any]] = {}
    person_interval_stats: Dict[str, Dict[str, Counter]] = defaultdict(
        lambda: defaultdict(Counter)
    )

    for p in predictions:
        pid = p["person_id"]
        interval_id = str(p.get("interval_id", -1))
        label = p["emotion_label"]

        if pid not in by_person:
            by_person[pid] = {
                "person_id": pid,
                "total_samples": 0,
                "emotion_counts": Counter(),
            }
        by_person[pid]["total_samples"] += 1
        by_person[pid]["emotion_counts"][label] += 1
        person_interval_stats[pid][interval_id][label] += 1

    by_person_out: Dict[str, Dict[str, Any]] = {}
    for pid, stats in sorted(by_person.items()):
        total = max(1, stats["total_samples"])
        counts = dict(stats["emotion_counts"])
        dominant_label = max(counts, key=counts.get) if counts else "unknown"
        dominant_share = counts.get(dominant_label, 0) / total
        by_person_out[pid] = {
            "person_id": pid,
            "total_samples": stats["total_samples"],
            "emotion_counts": counts,
            "dominant_emotion": dominant_label,
            "dominant_share": round(dominant_share, 4),
        }

    by_interval_out: Dict[str, Dict[str, Dict[str, int]]] = {}
    for pid, intervals in person_interval_stats.items():
        by_interval_out[pid] = {
            interval_id: dict(counter)
            for interval_id, counter in sorted(intervals.items(), key=lambda x: int(x[0]))
        }

    result = {
        "model": "fer",
        "model_info": model_info,
        "analysis_path": analysis_path,
        "total_predictions": len(predictions),
        "elapsed_sec": round(elapsed, 3),
        "throughput_samples_per_sec": round(throughput, 3),
        "max_per_person": max_per_person,
        "by_person": by_person_out,
        "by_person_interval": by_interval_out,
        "predictions": predictions,
    }
    save_json(result, output_path)

    print("=== STEP 5: EMOTION BASELINE (FER) ===")
    print(f"Samples inferred: {len(predictions)}")
    print(f"Throughput: {throughput:.2f} samples/sec")
    print(f"Persons: {list(by_person_out.keys())}")
    print(f"Saved: {output_path}")

    return result


def print_emotion_model_setup_help() -> None:
    print("=== EMOTION MODEL SETUP HELP ===")
    print("1) Fast baseline (recommended):")
    print("   !pip install fer")
    print("2) Better robustness for moving faces:")
    print("   !pip install hsemotion")
    print("3) Next action:")
    print("   - Run FER baseline on consolidated crops")
    print("   - Compare with HSEmotion on same sample set")


if __name__ == "__main__":
    print_emotion_model_setup_help()
