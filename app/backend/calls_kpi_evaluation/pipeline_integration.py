"""
KPI Analysis Pipeline Integration — СДЭК Чеклист
Интеграция анализа KPI (100 баллов) в основной pipeline.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime


def _get_analyze_call():
    """Import analyze_call with fallback for different import contexts."""
    try:
        from app.backend.calls_kpi_evaluation.kpi_analyzer import analyze_call
        return analyze_call
    except ImportError:
        from calls_kpi_evaluation.kpi_analyzer import analyze_call
        return analyze_call


def run_kpi_evaluation(
    transcript: str,
    output_path: Optional[str] = None,
    video_id: Optional[str] = None,
    combined_emotion_result: Optional[Dict] = None,
) -> Dict:
    """
    Запускает KPI-оценку транскрипции по СДЭК чеклисту (100 баллов).

    Это основная функция для интеграции в pipeline.
    Вызывается после text_extraction/text_analysis.

    Args:
        transcript: Текст транскрипции диалога
        output_path: Путь для сохранения JSON-результата (опционально)
        video_id: ID видео/звонка (опционально)
        combined_emotion_result: Результат combined_emotion_analysis (опционально)

    Returns:
        Словарь с результатами оценки
    """
    analyze_call = _get_analyze_call()

    # Анализируем по чеклисту
    raw_results = analyze_call(transcript)

    # Формируем результат pipeline
    result = {
        "video_id": video_id or "unknown",
        "timestamp": datetime.now().isoformat(),
        "checklist_name": raw_results.get("checklist_name", "Пример для СДЭК"),
        "overall_score": raw_results["overall_score"],
        "max_possible_score": raw_results["max_possible_score"],
        "score_percentage": raw_results["score_percentage"],
        "overall_status": raw_results["overall_status"],
        "traffic_light": raw_results["traffic_light"],
        "categories": raw_results["categories"],
        "all_items": raw_results["all_items"],
        "text_fields": raw_results.get("text_fields", {}),
    }

    # Добавляем emotion если есть
    if combined_emotion_result:
        result["combined_emotion"] = combined_emotion_result

    # Сохраняем в файл если указан путь
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    return result


def run_kpi_from_file(
    transcript_path: str,
    output_path: Optional[str] = None,
) -> Dict:
    """
    Запускает KPI-оценку из файла транскрипции.

    Args:
        transcript_path: Путь к .txt файлу с транскрипцией
        output_path: Путь для сохранения результата

    Returns:
        Словарь с результатами оценки
    """
    p = Path(transcript_path)
    if not p.is_file():
        raise FileNotFoundError(f"Transcript file not found: {p}")

    transcript = p.read_text(encoding="utf-8")
    video_id = p.stem

    if not output_path:
        output_path = str(p.parent / f"{p.stem}_kpi_evaluation.json")

    return run_kpi_evaluation(
        transcript=transcript,
        output_path=output_path,
        video_id=video_id,
    )


def print_kpi_report(result: Dict) -> None:
    """Печатает читаемый отчёт KPI в консоль."""
    print("\n" + "=" * 70)
    print(f"  ОЦЕНКА ПО ЧЕКЛИСТУ СДЭК: {result['score_percentage']}% "
          f"({result['overall_score']}/{result['max_possible_score']})")
    print(f"  Статус: {result['overall_status']} [{result['traffic_light'].upper()}]")
    print("=" * 70)

    for cat_name, cat_data in result.get("categories", {}).items():
        pct = cat_data["percentage"]
        print(f"\n  [{cat_name}] — {cat_data['total_score']}/{cat_data['max_score']} ({pct}%)")
        for item in cat_data.get("items", []):
            marker = "+" if item["status"] == "Да" else "-"
            print(f"    {marker} {item['name']}: {item['status']} ({item['score']}/{item['max_score']})")

    text_fields = result.get("text_fields", {})
    if text_fields.get("next_step"):
        print(f"\n  Следующий шаг: {text_fields['next_step'][:200]}")
    print("=" * 70)


if __name__ == "__main__":
    # CLI: python -m app.backend.calls_kpi_evaluation.pipeline_integration [transcript.txt]
    if len(sys.argv) > 1:
        path = sys.argv[1]
    else:
        path = "data/video_from_bucket_audio.txt"

    try:
        result = run_kpi_from_file(path)
        print_kpi_report(result)
        if result.get("output_path"):
            print(f"\nResults saved to: {result.get('output_path')}")
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
