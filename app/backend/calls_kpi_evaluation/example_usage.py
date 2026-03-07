"""
Example: KPI Analysis Integration
Пример использования KPI анализатора в pipeline
"""

import json
import sys
from pathlib import Path

# Добавляем путь к backend модулям
sys.path.insert(0, str(Path(__file__).parent.parent))

from calls_kpi_evaluation import analyze_call


def analyze_call_from_transcript(transcript_path: str, output_path: str = None) -> dict:
    """
    Анализирует транскрипцию звонка и сохраняет результаты
    
    Args:
        transcript_path: Путь к файлу с транскрипцией
        output_path: Путь для сохранения результатов (опционально)
        
    Returns:
        Результаты анализа
    """
    
    # Читаем транскрипцию
    with open(transcript_path, 'r', encoding='utf-8') as f:
        if transcript_path.endswith('.json'):
            data = json.load(f)
            # Извлекаем текст из JSON (предполагаем, что есть поле 'text' или 'transcript')
            transcript = data.get('text') or data.get('transcript') or json.dumps(data)
        else:
            transcript = f.read()
    
    # Анализируем
    results = analyze_call(transcript)
    
    # Сохраняем результаты если указан путь
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"✓ Результаты сохранены в {output_path}")
    
    return results


def print_analysis_report(results: dict) -> None:
    """
    Красивый вывод результатов анализа
    """
    
    print("\n" + "="*80)
    print(f"ОТЧЕТ ПО KPI АНАЛИЗУ ЗВОНКА")
    print("="*80)
    
    print(f"\n📊 ОБЩАЯ ОЦЕНКА")
    print(f"   Баллы: {results['overall_score']}/{results['max_possible_score']}")
    print(f"   Процент: {results['score_percentage']}%")
    print(f"   Статус: {results['overall_status']} {results['rating']}")
    
    print(f"\n📋 ОЦЕНКА ПО КАТЕГОРИЯМ\n")
    
    for category, cat_data in results['categories'].items():
        print(f"   {category}")
        print(f"   {'─' * 70}")
        print(f"   Баллы: {cat_data['total_score']}/{cat_data['max_score']} ({cat_data['percentage']}%)")
        
        for item in cat_data['items']:
            status_icon = "✓" if item['status'] == "yes" else "✗"
            points = f"+{item['score']}" if item['score'] > 0 else "0"
            print(f"      {status_icon} {item['name']}: {points} баллов")
            if item['details']:
                print(f"         {item['details']}")
        print()
    
    print("="*80)


def analyze_from_json(json_file_path: str) -> dict:
    """
    Анализирует JSON файл с результатами text_extraction
    """
    
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Извлекаем текст - пробуем разные возможные поля
    transcript = None
    
    if isinstance(data, dict):
        # Проверяем обычные поля для транскрипции
        for field in ['text', 'transcript', 'extracted_text', 'content']:
            if field in data and data[field]:
                transcript = data[field]
                break
        
        # Если это результат от combined_emotion_analysis
        if 'transcription' in data:
            transcript = data['transcription']
        
        # Если это массив текстов
        if not transcript and 'texts' in data:
            transcript = ' '.join(data['texts'])
    
    if not transcript:
        raise ValueError("Не удалось найти транскрипцию в JSON файле")
    
    return analyze_call(transcript)


# Пример использования
if __name__ == "__main__":
    # Пример 1: Анализ из файла с транскрипцией
    example_transcript = """
    Менеджер: Здравствуйте, это консультант СДЭК, меня зовут Александр.
    Клиент: Привет, я слушаю.
    Менеджер: Скажите, как к вам обращаться?
    Клиент: Иван.
    Менеджер: Спасибо, Иван. Сегодня мы поговорим о том, как вы сможете использовать наши услуги доставки.
    Менеджер: Я вам расскажу о возможностях личного кабинета, о тарифах и дополнительных услугах.
    Менеджер: Скажите, а какие товары вы планируете отправлять?
    Клиент: Одежду, в основном.
    Менеджер: Понимаю. А сколько отправок примерно в месяц вы планируете?
    Клиент: Примерно 50-100 в месяц.
    Менеджер: Спасибо за информацию, Иван. Позвольте рассказать о страховке. 
    Менеджер: Объявленная стоимость - это ваша страховка. Если груз потеряется или повредится, мы компенсируем именно эту сумму.
    Менеджер: Также есть опция наложенного платежа - когда получатель оплачивает товар при получении.
    Менеджер: За перевод средств взимается комиссия банка, но получатель её оплачивает, поэтому вы получите всю сумму полностью.
    Менеджер: По итогам встречи, мы обсудили ваше количество отправок, типы товаров и способы оплаты.
    Менеджер: У вас остались какие-нибудь вопросы?
    Клиент: Нет, спасибо, всё понятно.
    Менеджер: Спасибо за внимание! Я отправлю вам ссылки на личный кабинет и инструкции по почте.
    """
    
    results = analyze_call(example_transcript)
    print_analysis_report(results)
    
    # Сохраняем в JSON
    output_file = "call_kpi_analysis_result.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n✓ Результаты сохранены в {output_file}")
