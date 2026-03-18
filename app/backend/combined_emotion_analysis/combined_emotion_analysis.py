"""
Combined video+audio emotion analysis module.
Merges video facial emotions with audio sentiment for comprehensive emotion report.
"""
import json
import os
import sys
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _find_overlapping_segments(
    interval_start: float,
    interval_end: float,
    segments: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Find audio segments that overlap with video interval [interval_start, interval_end].
    Returns list of overlapping segments with adjusted start/end times.
    """
    overlapping = []
    for seg in segments:
        seg_start = float(seg.get("start", 0))
        seg_end = float(seg.get("end", 0))
        
        # Check if segments overlap
        if seg_start < interval_end and seg_end > interval_start:
            overlap_start = max(seg_start, interval_start)
            overlap_end = min(seg_end, interval_end)
            
            seg_copy = seg.copy()
            seg_copy["overlap_start"] = overlap_start
            seg_copy["overlap_end"] = overlap_end
            seg_copy["overlap_duration"] = overlap_end - overlap_start
            overlapping.append(seg_copy)
    
    return overlapping


def _aggregate_audio_sentiment(segments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate sentiment from audio segments.
    Returns overall sentiment and per-speaker sentiment.
    """
    if not segments:
        return {
            "sentiment_label": "unknown",
            "sentiment_score": 0.0,
            "total_segments": 0,
            "by_speaker": {},
        }
    
    sentiment_scores = []
    by_speaker = defaultdict(list)
    
    for seg in segments:
        sentiment = seg.get("sentiment", {})
        label = sentiment.get("label", "unknown")
        score = float(sentiment.get("score", 0.0))
        
        sentiment_scores.append(score)
        speaker_id = seg.get("speaker_id", "unknown")
        by_speaker[speaker_id].append({
            "label": label,
            "score": score,
            "text": seg.get("text", ""),
        })
    
    # Determine dominant sentiment
    avg_score = sum(sentiment_scores) / len(sentiment_scores) if sentiment_scores else 0.0
    
    # Map average score to sentiment label
    if avg_score >= 0.7:
        dominant_label = "positive"
    elif avg_score >= 0.4:
        dominant_label = "neutral"
    else:
        dominant_label = "negative"
    
    # Aggregate per-speaker
    by_speaker_agg = {}
    for speaker_id, scores_list in by_speaker.items():
        avg = sum(s["score"] for s in scores_list) / len(scores_list) if scores_list else 0.0
        if avg >= 0.7:
            label = "positive"
        elif avg >= 0.4:
            label = "neutral"
        else:
            label = "negative"
        
        by_speaker_agg[speaker_id] = {
            "sentiment_label": label,
            "sentiment_score": round(avg, 4),
            "segment_count": len(scores_list),
            "segments": scores_list,
        }
    
    return {
        "sentiment_label": dominant_label,
        "sentiment_score": round(avg_score, 4),
        "total_segments": len(segments),
        "by_speaker": by_speaker_agg,
    }


def _compute_combined_emotion_score(
    video_emotion: Dict[str, Any],
    audio_sentiment: Dict[str, Any],
    video_weight: float = 0.6,
    audio_weight: float = 0.4,
) -> Dict[str, Any]:
    """
    Combine video emotion with audio sentiment.
    Returns: combined dominance, confidence, and detailed breakdown.
    """
    # Map emotion/sentiment to valence score [-1, 1]
    emotion_valence_map = {
        "happiness": 1.0,
        "surprise": 0.5,
        "neutral": 0.0,
        "sadness": -0.7,
        "anger": -1.0,
        "disgust": -0.9,
        "fear": -0.8,
        "contempt": -0.6,
    }
    
    sentiment_valence_map = {
        "positive": 0.7,
        "neutral": 0.0,
        "negative": -0.7,
    }
    
    # Get video valence
    video_emo = video_emotion.get("dominant_emotion", "unknown")
    video_valence = emotion_valence_map.get(video_emo, 0.0)
    avg_conf = video_emotion.get("avg_confidence")
    video_confidence = float(avg_conf) if avg_conf is not None else 0.5
    
    # Get audio valence
    audio_sent = audio_sentiment.get("sentiment_label", "unknown")
    audio_valence = sentiment_valence_map.get(audio_sent, 0.0)
    sent_score = audio_sentiment.get("sentiment_score")
    audio_confidence = abs(float(sent_score) - 0.5) * 2 if sent_score is not None else 0.5  # Convert to confidence [0, 1]
    
    # Weighted combination
    combined_valence = (
        video_valence * video_weight * video_confidence +
        audio_valence * audio_weight * audio_confidence
    ) / (video_weight * video_confidence + audio_weight * audio_confidence) if (video_weight * video_confidence + audio_weight * audio_confidence) > 0 else 0.0
    
    # Map combined valence back to emotion
    if combined_valence >= 0.8:
        combined_emotion = "very_positive"
    elif combined_valence >= 0.4:
        combined_emotion = "positive"
    elif combined_valence >= -0.2:
        combined_emotion = "neutral"
    elif combined_valence >= -0.6:
        combined_emotion = "negative"
    else:
        combined_emotion = "very_negative"
    
    combined_confidence = (video_confidence * video_weight + audio_confidence * audio_weight)
    
    return {
        "combined_emotion": combined_emotion,
        "combined_valence": round(combined_valence, 4),
        "combined_confidence": round(combined_confidence, 4),
        "video_emotion": video_emo,
        "video_valence": round(video_valence, 4),
        "video_confidence": round(video_confidence, 4),
        "audio_sentiment": audio_sent,
        "audio_valence": round(audio_valence, 4),
        "audio_confidence": round(audio_confidence, 4),
    }


def run_combined_emotion_analysis(
    emotion_report_path: str,
    audio_sentiment_path: str,
    output_path: str,
    video_weight: float = 0.6,
    audio_weight: float = 0.4,
) -> Dict[str, Any]:
    """
    Combine video emotion and audio sentiment analyses.
    
    Args:
        emotion_report_path: Path to emotion report JSON (from step 6)
        audio_sentiment_path: Path to audio diarized sentiment JSON
        output_path: Path to save combined analysis
        video_weight: Weight for video emotion (default 0.6)
        audio_weight: Weight for audio sentiment (default 0.4)
    
    Returns:
        Combined analysis result dictionary
    """
    # Load inputs
    emotion_data = load_json(emotion_report_path)
    audio_data = load_json(audio_sentiment_path)
    
    emotion_intervals = emotion_data.get("interval_emotion_report", [])
    audio_segments = audio_data.get("segments", [])
    
    # Process each interval
    combined_intervals = []
    
    for interval in emotion_intervals:
        interval_id = int(interval.get("interval_id", -1))
        start_ts = interval.get("start_ts", "00:00:00.000")
        
        # Parse timestamp to seconds (format: HH:MM:SS.mmm)
        try:
            parts = start_ts.split(":")
            hours = int(parts[0]) if len(parts) > 0 else 0
            minutes = int(parts[1]) if len(parts) > 1 else 0
            seconds_ms = parts[2] if len(parts) > 2 else "0"
            seconds_parts = seconds_ms.split(".")
            seconds = int(seconds_parts[0]) if seconds_parts else 0
            millis = int(seconds_parts[1]) if len(seconds_parts) > 1 else 0
            
            interval_start_sec = hours * 3600 + minutes * 60 + seconds + millis / 1000.0
        except (ValueError, IndexError):
            interval_start_sec = 0.0
        
        # Estimate interval end (assuming 10-second intervals)
        interval_end_sec = interval_start_sec + 10.0
        
        # Find overlapping audio segments
        overlapping_segments = _find_overlapping_segments(
            interval_start_sec,
            interval_end_sec,
            audio_segments,
        )
        
        # Aggregate audio sentiment
        audio_sentiment = _aggregate_audio_sentiment(overlapping_segments)
        
        # Combine with person emotions
        person_emotions = interval.get("person_emotions", {})
        combined_persons = {}
        
        for person_id, person_emo in person_emotions.items():
            combined = _compute_combined_emotion_score(
                person_emo,
                audio_sentiment,
                video_weight=video_weight,
                audio_weight=audio_weight,
            )
            
            combined_persons[person_id] = {
                **person_emo,
                **combined,
                "audio_sentiment_info": audio_sentiment,
                "overlapping_segments_count": len(overlapping_segments),
            }
        
        combined_interval = {
            **interval,
            "person_emotions": combined_persons,
            "audio_sentiment": audio_sentiment,
            "overlapping_audio_segments": len(overlapping_segments),
        }
        
        combined_intervals.append(combined_interval)
    
    # Overall summary
    all_combined_emotions = []
    all_combined_valences = []
    for interval in combined_intervals:
        for person_id, data in interval.get("person_emotions", {}).items():
            all_combined_emotions.append(data.get("combined_emotion", "unknown"))
            valence = data.get("combined_valence", 0.0)
            if valence is not None:
                all_combined_valences.append(valence)
    
    dominant_combined = Counter(all_combined_emotions).most_common(1)[0][0] if all_combined_emotions else "unknown"
    avg_combined_valence = sum(all_combined_valences) / len(all_combined_valences) if all_combined_valences else 0.0
    
    result = {
        "meta": {
            "emotion_report_path": emotion_report_path,
            "audio_sentiment_path": audio_sentiment_path,
            "video_weight": video_weight,
            "audio_weight": audio_weight,
        },
        "summary": {
            "total_intervals": len(combined_intervals),
            "dominant_combined_emotion": dominant_combined,
            "average_combined_valence": round(avg_combined_valence, 4),
            "emotion_distribution": dict(Counter(all_combined_emotions)),
        },
        "combined_intervals": combined_intervals,
    }
    
    save_json(result, output_path)
    
    print("=== COMBINED VIDEO+AUDIO EMOTION ANALYSIS ===")
    print(f"Total intervals analyzed: {result['summary']['total_intervals']}")
    print(f"Dominant combined emotion: {result['summary']['dominant_combined_emotion']}")
    print(f"Average combined valence: {result['summary']['average_combined_valence']}")
    print(f"Saved: {output_path}")
    
    return result


if __name__ == "__main__":
    # Example usage
    if len(sys.argv) < 3:
        print("Usage: python -m app.backend.combined_emotion_analysis <emotion_report> <audio_sentiment> [output_path]")
        sys.exit(1)
    
    emotion_report = sys.argv[1]
    audio_sentiment = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) > 3 else "combined_emotion_analysis.json"
    
    run_combined_emotion_analysis(emotion_report, audio_sentiment, output)
