"""
Visual representation of emotion recognition: valence over time and emotion distribution.
Reads the step6 emotion report JSON and saves a figure (PNG).
"""
import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def plot_emotion_report(
    report_path: str,
    output_path: Optional[str] = None,
    figsize: Tuple[float, float] = (12, 8),
) -> str:
    """
    Load step6 emotion report and create a figure with:
    - Valence over time (one line per person)
    - Overall emotion distribution (bar chart)
    Returns the path where the figure was saved.
    """
    data = load_json(report_path)
    interval_report: List[Dict[str, Any]] = data.get("interval_emotion_report", [])
    person_timeline: Dict[str, Any] = data.get("person_emotion_timeline", {})

    if not interval_report and not person_timeline:
        raise ValueError(f"No interval_emotion_report or person_emotion_timeline in {report_path}")

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, height_ratios=[1.2, 1])

    # --- Valence over time (per person) ---
    intervals = sorted(r["interval_id"] for r in interval_report)
    if intervals:
        for pid, info in sorted(person_timeline.items()):
            timeline = info.get("intervals", [])
            if not timeline:
                continue
            by_i = {t["interval_id"]: t for t in timeline}
            xs = intervals
            ys = [by_i.get(i, {}).get("valence_score", 0.0) for i in xs]
            ax1.plot(xs, ys, marker="o", markersize=4, label=f"Person {pid}", alpha=0.9)

        ax1.set_xlabel("Interval")
        ax1.set_ylabel("Valence score")
        ax1.set_title("Emotion valence over time (per person)")
        ax1.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax1.legend(loc="best", fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(-1.1, 1.1)

    # --- Overall emotion distribution ---
    emotion_counts: Counter = Counter()
    for row in interval_report:
        for pid, stats in row.get("person_emotions", {}).items():
            for emotion, count in stats.get("emotion_counts", {}).items():
                emotion_counts[emotion] += count

    if emotion_counts:
        labels = list(emotion_counts.keys())
        counts = [emotion_counts[e] for e in labels]
        colors = plt.cm.RdYlGn(
            [(c - min(counts)) / max((max(counts) - min(counts)) or 1, 1) * 0.5 + 0.25 for c in counts]
        )
        bars = ax2.bar(labels, counts, color=colors, edgecolor="gray", linewidth=0.5)
        ax2.set_xlabel("Emotion")
        ax2.set_ylabel("Count")
        ax2.set_title("Overall emotion distribution (all intervals, all persons)")
        ax2.tick_params(axis="x", rotation=45)
        for bar, c in zip(bars, counts):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(c), ha="center", va="bottom", fontsize=8)
    else:
        ax2.text(0.5, 0.5, "No emotion counts in report", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Overall emotion distribution")

    plt.tight_layout()

    if output_path is None:
        base = os.path.splitext(report_path)[0]
        output_path = f"{base}_visual.png"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    return output_path


def run_emotion_visual(
    report_path: str = "extracted_frames_v3_dense/analysis_step6_emotion_report.json",
    output_path: Optional[str] = None,
) -> str:
    """Load step6 report and save emotion visualization. Returns path to saved figure."""
    out = plot_emotion_report(report_path, output_path=output_path)
    print(f"Emotion visual saved: {out}")
    return out


if __name__ == "__main__":
    run_emotion_visual(
        report_path="extracted_frames_v3_dense/analysis_step6_emotion_report.json",
    )
