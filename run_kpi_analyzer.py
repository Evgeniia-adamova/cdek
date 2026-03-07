#!/usr/bin/env python3
"""
KPI Analysis Quick Start Script
Скрипт для быстрого анализа звонков по KPI критериям
"""

import sys
import json
from pathlib import Path

# Убеждаемся, что мы в правильной директории
sys.path.insert(0, str(Path(__file__).parent))

from app.backend.calls_kpi_evaluation import analyze_call


def main():
    """Главная функция"""
    
    print("\n" + "="*80)
    print("🎯 KPI ANALYZER - Анализ качества звонков")
    print("="*80 + "\n")
    
    # Меню выбора
    print("Выберите вариант:")
    print("1. Вставить текст вручную")
    print("2. Загрузить из файла")
    print("3. Запустить тесты")
    print()
    
    choice = input("Ваш выбор (1-3): ").strip()
    
    transcript = None
    
    if choice == "1":
        print("\nВставьте текст транскрипции (введите 'END' на новой строке чтобы завершить):")
        print("-" * 80)
        lines = []
        while True:
            line = input()
            if line == "END":
                break
            lines.append(line)
        transcript = "\n".join(lines)
        
    elif choice == "2":
        file_path = input("\nВведите путь к файлу: ").strip()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.json'):
                    data = json.load(f)
                    transcript = data.get('text') or data.get('transcript') or str(data)
                else:
                    transcript = f.read()
        except FileNotFoundError:
            print(f"❌ Файл не найден: {file_path}")
            return
        except Exception as e:
            print(f"❌ Ошибка при чтении файла: {e}")
            return
            
    elif choice == "3":
        print("\n📝 Запускаем встроенные тесты...\n")
        import subprocess
        result = subprocess.run(
            [sys.executable, "app/backend/calls_kpi_evaluation/test_kpi.py"],
            cwd=Path(__file__).parent
        )
        return
    else:
        print("❌ Неверный выбор")
        return
    
    if not transcript:
        print("❌ Текст не введен")
        return
    
    # Анализируем
    print("\n⏳ Анализируем звонок...\n")
    
    results = analyze_call(transcript)
    
    # Выводим результаты
    print("="*80)
    print("📊 РЕЗУЛЬТАТЫ АНАЛИЗА")
    print("="*80)
    
    print(f"\n🎯 ИТОГОВАЯ ОЦЕНКА")
    print(f"   Баллы: {results['overall_score']}/{results['max_possible_score']}")
    print(f"   Процент: {results['score_percentage']}%")
    print(f"   Статус: {results['overall_status']} {results['rating']}")
    
    print(f"\n📋 ОЦЕНКА ПО КАТЕГОРИЯМ")
    print()
    
    total_weight = 0
    weighted_score = 0
    
    weights = {
        'Установление контакта': 0.15,
        'Управление встречей': 0.20,
        'Сбор информации': 0.25,
        'Демонстрация услуг': 0.30,
        'Финализация': 0.10
    }
    
    for category, data in results['categories'].items():
        percentage = data['percentage']
        weight = weights.get(category, 0)
        
        print(f"   {category}")
        print(f"   {'─' * 70}")
        print(f"   Баллы: {data['total_score']}/{data['max_score']} ({percentage}%)")
        print(f"   Вес: {int(weight*100)}%")
        
        # Показываем невыполненные критерии
        failed_items = [item for item in data['items'] if item['status'] == 'no']
        if failed_items:
            print(f"   ❌ Не выполнено ({len(failed_items)}):")
            for item in failed_items[:5]:  # Показываем первые 5
                print(f"      - {item['name']}")
            if len(failed_items) > 5:
                print(f"      ... и еще {len(failed_items) - 5}")
        
        print()
        
        # Добавляем к взвешенной оценке
        total_weight += weight
        if data['max_score'] > 0:
            weighted_score += (data['total_score'] / data['max_score']) * weight
    
    print("="*80)
    
    # Рекомендации
    if results['score_percentage'] < 50:
        print("\n⚠️  РЕКОМЕНДАЦИИ ДЛЯ УЛУЧШЕНИЯ:")
        print("   • Обязательно приветствуйте клиента в начале звонка")
        print("   • Представляйтесь по имени")
        print("   • Объясняйте цель встречи")
        print("   • Расспрашивайте об объёмах и типе товаров")
        print("   • Подробно рассказывайте об услугах")
        print("   • Подводите итоги в конце звонка")
        
    elif results['score_percentage'] < 75:
        print("\n💡 РЕКОМЕНДАЦИИ ДЛЯ УЛУЧШЕНИЯ:")
        print("   • Добавьте специальные услуги (фулфилмент, реверс)")
        print("   • Объясняйте подробнее процессы оплаты")
        print("   • Берите обратную связь от клиента")
        print("   • Отправляйте дополнительные материалы")
        
    else:
        print("\n✅ ОТЛИЧНАЯ РАБОТА!")
        print("   • Звонок соответствует большинству KPI критериев")
        print("   • Продолжайте в том же духе!")
    
    print("\n" + "="*80)
    
    # Опцион на сохранение
    save = input("\nСохранить результаты в JSON? (y/n): ").strip().lower()
    if save == 'y':
        output_file = Path("kpi_analysis_result.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✓ Результаты сохранены в {output_file}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Программа прервана пользователем")
    except Exception as e:
        print(f"\n❌ Ошибка: {e}")
        import traceback
        traceback.print_exc()
