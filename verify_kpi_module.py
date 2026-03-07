#!/usr/bin/env python3
"""
Verification Script - Проверка что все компоненты работают
"""

import sys
import json
from pathlib import Path

def check_files():
    """Проверяет что все файлы созданы"""
    print("\n" + "="*80)
    print("1️⃣  ПРОВЕРКА ФАЙЛОВ")
    print("="*80)
    
    required_files = [
        "app/backend/calls_kpi_evaluation/__init__.py",
        "app/backend/calls_kpi_evaluation/kpi_analyzer.py",
        "app/backend/calls_kpi_evaluation/pipeline_integration.py",
        "app/backend/calls_kpi_evaluation/example_usage.py",
        "app/backend/calls_kpi_evaluation/test_kpi.py",
        "app/backend/calls_kpi_evaluation/README.md",
        "app/backend/calls_kpi_evaluation/QUICKSTART.md",
        "KPI_ANALYSIS_FINAL_DOCUMENTATION.md",
        "run_kpi_analyzer.py",
    ]
    
    all_exist = True
    for file in required_files:
        path = Path(file)
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {file}")
        all_exist = all_exist and exists
    
    return all_exist


def check_imports():
    """Проверяет что модули импортируются"""
    print("\n" + "="*80)
    print("2️⃣  ПРОВЕРКА ИМПОРТОВ")
    print("="*80)
    
    try:
        from app.backend.calls_kpi_evaluation import analyze_call
        print("  ✓ from app.backend.calls_kpi_evaluation import analyze_call")
    except ImportError as e:
        print(f"  ✗ Ошибка импорта: {e}")
        return False
    
    try:
        from app.backend.calls_kpi_evaluation.pipeline_integration import add_kpi_to_pipeline_step4
        print("  ✓ from app.backend.calls_kpi_evaluation.pipeline_integration import add_kpi_to_pipeline_step4")
    except ImportError as e:
        print(f"  ✗ Ошибка импорта: {e}")
        return False
    
    try:
        from app.backend.calls_kpi_evaluation.example_usage import analyze_from_json
        print("  ✓ from app.backend.calls_kpi_evaluation.example_usage import analyze_from_json")
    except ImportError as e:
        print(f"  ✗ Ошибка импорта: {e}")
        return False
    
    return True


def check_functionality():
    """Проверяет что функциональность работает"""
    print("\n" + "="*80)
    print("3️⃣  ПРОВЕРКА ФУНКЦИОНАЛЬНОСТИ")
    print("="*80)
    
    try:
        from app.backend.calls_kpi_evaluation import analyze_call
        
        test_transcript = """
        Менеджер: Здравствуйте, меня зовут Иван.
        Клиент: Привет.
        Менеджер: Скажите, как вас зовут?
        Клиент: Петр.
        Менеджер: Спасибо, Петр. Сегодня мы обсудим доставку.
        Менеджер: Сколько отправок в месяц планируете?
        Клиент: Примерно 100.
        Менеджер: Какие товары?
        Клиент: Электронику.
        Менеджер: Я отправлю вам информацию.
        """
        
        results = analyze_call(test_transcript)
        
        # Проверяем структуру результатов
        required_keys = [
            'overall_score',
            'max_possible_score',
            'score_percentage',
            'overall_status',
            'rating',
            'categories',
            'all_items'
        ]
        
        for key in required_keys:
            if key in results:
                print(f"  ✓ {key}: {type(results[key]).__name__}")
            else:
                print(f"  ✗ {key}: отсутствует")
                return False
        
        # Выводим результат анализа
        print(f"\n  📊 Результат анализа теста:")
        print(f"     Баллы: {results['overall_score']}/{results['max_possible_score']}")
        print(f"     Процент: {results['score_percentage']}%")
        print(f"     Статус: {results['overall_status']} {results['rating']}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Ошибка при тестировании: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_output_dir():
    """Проверяет что директория для результатов создается"""
    print("\n" + "="*80)
    print("4️⃣  ПРОВЕРКА ДИРЕКТОРИИ РЕЗУЛЬТАТОВ")
    print("="*80)
    
    output_dir = Path("data/kpi_analysis")
    
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Директория {output_dir} создана/существует")
        
        # Проверяем что мы можем писать в неё
        test_file = output_dir / ".test_write"
        test_file.write_text("test")
        test_file.unlink()
        print(f"  ✓ Возможна запись в директорию")
        
        return True
    except Exception as e:
        print(f"  ✗ Ошибка: {e}")
        return False


def check_test_results():
    """Проверяет что тесты работают"""
    print("\n" + "="*80)
    print("5️⃣  ПРОВЕРКА ТЕСТОВЫХ РЕЗУЛЬТАТОВ")
    print("="*80)
    
    test_file = Path("app/backend/calls_kpi_evaluation/test_results.json")
    
    if test_file.exists():
        try:
            with open(test_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            print(f"  ✓ Файл тестовых результатов существует")
            
            if 'test_cases' in data:
                print(f"  ✓ Найдено тестовых случаев: {len(data['test_cases'])}")
                
                for test in data['test_cases']:
                    print(f"    • {test['name']}: {test['percentage']}% - {test['status']}")
                
                return True
            else:
                print(f"  ✗ Структура файла неправильная")
                return False
        except json.JSONDecodeError:
            print(f"  ✗ Ошибка парсинга JSON")
            return False
    else:
        print(f"  ⚠ Файл тестовых результатов не найден (это нормально если тесты не запускались)")
        return True


def main():
    """Главная функция"""
    
    print("\n" + "╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "  ✅ ПРОВЕРКА МОДУЛЯ KPI АНАЛИЗА ЗВОНКОВ".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    checks = [
        ("Файлы", check_files),
        ("Импорты", check_imports),
        ("Функциональность", check_functionality),
        ("Директория результатов", check_output_dir),
        ("Тестовые результаты", check_test_results),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ Ошибка в {name}: {e}")
            results.append((name, False))
    
    # Итоговый отчет
    print("\n" + "="*80)
    print("📋 ИТОГОВЫЙ ОТЧЕТ")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓" if result else "✗"
        print(f"  {status} {name}")
    
    print(f"\n  Пройдено: {passed}/{total}")
    
    if passed == total:
        print("\n" + "="*80)
        print("✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ УСПЕШНО!")
        print("="*80)
        print("\n📖 Дальше можно:")
        print("  1. Запустить интерактивный анализатор: python run_kpi_analyzer.py")
        print("  2. Запустить более подробные тесты: python app/backend/calls_kpi_evaluation/test_kpi.py")
        print("  3. Интегрировать в pipeline: app/run_full_pipeline.py")
        print("  4. Проверить документацию: KPI_ANALYSIS_FINAL_DOCUMENTATION.md")
        return 0
    else:
        print("\n" + "="*80)
        print("⚠️  НЕКОТОРЫЕ ПРОВЕРКИ НЕ ПРОШЛИ")
        print("="*80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
