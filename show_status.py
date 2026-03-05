#!/usr/bin/env python
# coding: utf-8
"""Final status report for combined emotion analysis feature."""
import os
import json

print("=" * 70)
print("✅ СОВМЕСТНЫЙ АНАЛИЗ ВИДЕО+АУДИО - ФИНАЛЬНЫЙ ОТЧЕТ")
print("=" * 70)
print()

# Check files
files_to_check = [
    ("Модуль анализа", "app/backend/combined_emotion_analysis/combined_emotion_analysis.py"),
    ("Скрипт запуска", "app/run_combined_emotion_analysis.py"),
    ("Документация модуля", "app/backend/combined_emotion_analysis/README.md"),
    ("Выходные данные", "data/v3_dense/combined_emotion_analysis.json"),
]

print("📁 ПРОВЕРКА ФАЙЛОВ:")
for name, path in files_to_check:
    if os.path.exists(path):
        size = os.path.getsize(path)
        print(f"   ✅ {name}: {size:,} байт")
    else:
        print(f"   ❌ {name}: НЕ НАЙДЕН")

print()
print("📚 ДОКУМЕНТАЦИЯ:")
docs = [
    ("QUICK_START.md", "Быстрый старт (3 способа запуска)"),
    ("FEATURE_SUMMARY.md", "Полный отчет о реализации"),
    ("EXAMPLES_COMBINED_EMOTION.md", "8 практических примеров"),
    ("ARCHITECTURE.md", "Архитектура с ASCII диаграммами"),
    ("INDEX.md", "Индекс всей документации"),
]

for doc, desc in docs:
    if os.path.exists(doc):
        print(f"   ✅ {doc}: {desc}")
    else:
        print(f"   ❌ {doc}")

print()
print("📊 РЕЗУЛЬТАТЫ АНАЛИЗА:")
if os.path.exists("data/v3_dense/combined_emotion_analysis.json"):
    with open("data/v3_dense/combined_emotion_analysis.json") as f:
        result = json.load(f)
        summary = result["summary"]
        print(f"   Интервалов проанализировано: {summary['total_intervals']}")
        print(f"   Доминирующая эмоция: {summary['dominant_combined_emotion']}")
        print(f"   Средняя валентность: {summary['average_combined_valence']}")
        print(f"   Распределение эмоций: {summary['emotion_distribution']}")

print()
print("🚀 ИСПОЛЬЗОВАНИЕ:")
print()
print("   Способ 1 - Весь пайплайн:")
print("   $ python -m app.run_full_pipeline data/my_video.webm")
print()
print("   Способ 2 - Только совместный анализ:")
print("   $ python -m app.run_combined_emotion_analysis")
print()
print("   Способ 3 - В Python коде:")
print("   >>> from app.backend.combined_emotion_analysis import run_combined_emotion_analysis")
print("   >>> result = run_combined_emotion_analysis(...)")
print()
print("=" * 70)
print("✨ ГОТОВО К ИСПОЛЬЗОВАНИЮ!")
print("=" * 70)
print()
print("📖 Начните с: QUICK_START.md или INDEX.md")
