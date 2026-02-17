import copy
import json
from collections import Counter, defaultdict
from typing import Any, Dict, List, Tuple


EMOTION_VALENCE = {
    "happiness": 1.0,
    "surprise": 0.4,
    "neutral": 0.0,
    "sadness": -0.7,
    "anger": -1.0,
    "disgust": -0.9,
    "fear": -0.8,
    "contempt": -0.6,
}


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _build_prediction_indexes(
    predictions: List[Dict[str, Any]],
) -> Tuple[Dict[str, Dict[str, Any]], Dict[Tuple[str, int], Dict[str, Any]]]:
    by_crop: Dict[str, Dict[str, Any]] = {}
    by_person_sample: Dict[Tuple[str, int], Dict[str, Any]] = {}

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
            pred = None

            crop_path = det.get("crop_path")
            if crop_path:
                pred = pred_by_crop.get(str(crop_path))

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
                det.setdefault(
                    "emotion",
                    {"label": "unknown", "confidence": None, "model": emotion_data.get("model", "unknown")},
                )

    out["emotion_merge_info"] = {
        "total_predictions": len(predictions),
        "attached_to_detections": attached,
        "model": emotion_data.get("model", "unknown"),
    }
    return out


def _person_interval_stats(
    dets: List[Dict[str, Any]],
) -> Dict[str, Any]:
    counts = Counter()
    conf_sum = 0.0
    conf_n = 0
    valence_sum = 0.0

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
) -> List[Dict[str, Any]]:
    buckets: Dict[int, Dict[str, Any]] = {}

    for fr in frames:
        interval_id = int(fr.get("interval_id", -1))
        if interval_id < 0:
            continue

        if interval_id not in buckets:
            buckets[interval_id] = {
                "interval_id": interval_id,
                "start_ts": fr.get("timestamp"),
                "frame_count": 0,
                "people": defaultdict(list),
            }

        bucket = buckets[interval_id]
        bucket["frame_count"] += 1

        for det in fr.get("detections", []):
            pid = str(det.get("person_id", ""))
            if not pid:
                continue
            bucket["people"][pid].append(det)

    report: List[Dict[str, Any]] = []
    for interval_id in sorted(buckets.keys()):
        bucket = buckets[interval_id]
        person_stats: Dict[str, Any] = {}

        for pid, dets in bucket["people"].items():
            person_stats[pid] = _person_interval_stats(dets)

        report.append(
            {
                "interval_id": interval_id,
                "start_ts": bucket["start_ts"],
                "frame_count": bucket["frame_count"],
                "persons": sorted(person_stats.keys()),
                "person_emotions": person_stats,
            }
        )

    return report


def compare_interval_emotions(
    interval_report: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    comparisons: List[Dict[str, Any]] = []
    for i in range(1, len(interval_report)):
        prev_it = interval_report[i - 1]
        curr_it = interval_report[i]
        prev_persons = set(prev_it["persons"])
        curr_persons = set(curr_it["persons"])
        shared = sorted(prev_persons & curr_persons)

        changes = []
        for pid in shared:
            prev_stats = prev_it["person_emotions"].get(pid, {})
            curr_stats = curr_it["person_emotions"].get(pid, {})
            changes.append(
                {
                    "person_id": pid,
                    "dominant_changed": prev_stats.get("dominant_emotion")
                    != curr_stats.get("dominant_emotion"),
                    "prev_dominant": prev_stats.get("dominant_emotion"),
                    "curr_dominant": curr_stats.get("dominant_emotion"),
                    "valence_delta": round(
                        float(curr_stats.get("valence_score", 0.0))
                        - float(prev_stats.get("valence_score", 0.0)),
                        4,
                    ),
                    "dominant_share_delta": round(
                        float(curr_stats.get("dominant_share", 0.0))
                        - float(prev_stats.get("dominant_share", 0.0)),
                        4,
                    ),
                }
            )

        comparisons.append(
            {
                "from_interval_id": prev_it["interval_id"],
                "to_interval_id": curr_it["interval_id"],
                "from_ts": prev_it.get("start_ts"),
                "to_ts": curr_it.get("start_ts"),
                "shared_person_changes": changes,
            }
        )

    return comparisons


def build_person_emotion_timeline(
    interval_report: List[Dict[str, Any]],
) -> Dict[str, Any]:
    per_person: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in interval_report:
        interval_id = row["interval_id"]
        start_ts = row.get("start_ts")
        for pid, stats in row["person_emotions"].items():
            per_person[pid].append(
                {
                    "interval_id": interval_id,
                    "start_ts": start_ts,
                    "dominant_emotion": stats.get("dominant_emotion"),
                    "dominant_share": stats.get("dominant_share"),
                    "valence_score": stats.get("valence_score"),
                }
            )

    summary: Dict[str, Any] = {}
    for pid, timeline in sorted(per_person.items()):
        dominant_counter = Counter([x["dominant_emotion"] for x in timeline if x["dominant_emotion"]])
        most_common = dominant_counter.most_common(1)[0][0] if dominant_counter else "unknown"
        avg_valence = (
            sum(float(x.get("valence_score", 0.0)) for x in timeline) / max(1, len(timeline))
        )
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

    interval_report = build_interval_emotion_report(frames)
    interval_comparison = compare_interval_emotions(interval_report)
    person_timeline = build_person_emotion_timeline(interval_report)

    result = {
        "meta": {
            "analysis_path": analysis_path,
            "emotion_path": emotion_path,
            "emotion_model": emotion_data.get("model"),
        },
        "emotion_merge_info": merged.get("emotion_merge_info", {}),
        "interval_emotion_report": interval_report,
        "interval_emotion_comparison": interval_comparison,
        "person_emotion_timeline": person_timeline,
    }
    save_json(result, output_path)

    print("=== STEP 6: INTERVAL EMOTION REPORT ===")
    print(f"Intervals: {len(interval_report)}")
    print(f"People: {list(person_timeline.keys())}")
    print(f"Merge info: {result['emotion_merge_info']}")
    for pid, s in person_timeline.items():
        print(
            f"- {pid}: most_common_interval_emotion={s['most_common_interval_emotion']}, "
            f"avg_valence={s['avg_valence_score']}"
        )
    print(f"Saved: {output_path}")

    return result


if __name__ == "__main__":
    run_step6_emotion_report(
        analysis_path="extracted_frames_v3_dense/analysis_step2_dense_consolidated.json",
        emotion_path="extracted_frames_v3_dense/emotion_baseline_onnx_ferplus.json",
    )
