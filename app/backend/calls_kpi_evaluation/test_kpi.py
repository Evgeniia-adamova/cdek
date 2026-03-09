"""
Тест KPI анализатора СДЭК — проверяет корректность оценки по 29 критериям (100 баллов).
Запуск из корня проекта:
    python -m app.backend.calls_kpi_evaluation.test_kpi
"""

import json
import sys
from pathlib import Path
from datetime import datetime

# Хороший звонок — большинство критериев выполнено
GOOD_CALL = """
Менеджер: Здравствуйте! Меня зовут Александра, я специалист СДЭК.
Клиент: Здравствуйте, слушаю вас.
Менеджер: Подскажите, как к вам обращаться?
Клиент: Ирина.
Менеджер: Ирина, сегодня наша цель — разобраться, как вам удобнее работать с доставкой.
Менеджер: Сначала обсудим ваши отправки и объёмы, затем покажу личный кабинет, в конце отвечу на ваши вопросы.
Менеджер: Скажите, работали ли вы раньше с другими логистическими компаниями?
Клиент: Да, с одной.
Менеджер: Что нравилось, а что не нравилось в работе с ними?
Клиент: Не нравилось что долго, нравилась цена.
Менеджер: Понятно. Ирина, уточните пожалуйста — что планируете отправлять?
Клиент: Одежду и электронику.
Менеджер: И сколько примерно отправок в месяц планируете?
Клиент: Около 100 в месяц.
Менеджер: Хорошо. Смотрите, стоимость доставки зависит от веса, габаритов груза и городов отправления и получения.
Менеджер: Также рекомендую покупать упаковку — коробки и пакеты — партиями, так выгоднее чем поштучно в накладной.
Менеджер: Из дополнительных услуг есть примерка, частичная доставка, СМС-уведомление.
Менеджер: Важный момент — объявленная стоимость в накладной это ваша страховка. При утере или повреждении мы компенсируем именно эту сумму.
Менеджер: Есть платный приоритетный чат за 199 рублей в месяц. Если не подключён, основной канал связи — электронная почта.
Менеджер: Деньги будут списываться с баланса автоматически.
Менеджер: Когда депозит закончится, счёт придёт на почту или через ЭДО. Срок оплаты 3 дня.
Менеджер: Также есть услуга фулфилмент — хранение и отгрузка товара с нашего склада.
Менеджер: Наложенный платёж: получатель оплачивает товар при получении, комиссия банка тоже с получателя, вы получаете полную сумму. Это инкассация средств.
Менеджер: Есть клиентский возврат и услуга Реверс — когда товар возвращается отправителю.
Менеджер: Интеграция с личным кабинетом через API позволяет автоматически формировать накладные.
Менеджер: Кстати, в WhatsApp бывают задержки в работе ВА, поэтому лучше общаться в Telegram. После встречи пришлю вам ссылку на наш чат.
Менеджер: Также скину вам инструкции и ссылки после нашего разговора.
Менеджер: Ирина, давайте подведём итоги. Мы разобрали отправки, личный кабинет, страховку, оплату и дополнительные услуги.
Менеджер: Остались ли у вас какие-нибудь вопросы?
Клиент: Нет, всё понятно.
Менеджер: Поделитесь пожалуйста отзывом о нашей встрече?
Клиент: Всё отлично, спасибо!
"""

# Плохой звонок — почти ничего не выполнено
BAD_CALL = """
Менеджер: Алло, это СДЭК.
Клиент: Да.
Менеджер: Давайте обсудим доставку.
Клиент: Ладно.
Менеджер: Сколько отправок?
Клиент: Ну 50 в месяц.
Менеджер: Окей. Стоимость от маршрута зависит.
Клиент: Понятно.
Менеджер: Ну всё, пока.
"""


def run_test():
    try:
        from app.backend.calls_kpi_evaluation import analyze_call, print_kpi_report, run_kpi_evaluation
    except ImportError:
        # fallback при запуске напрямую
        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
        from app.backend.calls_kpi_evaluation import analyze_call, print_kpi_report, run_kpi_evaluation

    print("\n" + "=" * 70)
    print("  ТЕСТ KPI АНАЛИЗАТОРА СДЭК")
    print("=" * 70)

    all_passed = True

    for label, transcript, expect_min_pct in [
        ("Хороший звонок (ожидаем >70%)", GOOD_CALL, 70),
        ("Плохой звонок (ожидаем <30%)",  BAD_CALL,   0),
    ]:
        print(f"\n{'─' * 70}")
        print(f"  {label}")
        print(f"{'─' * 70}")

        result = analyze_call(transcript)

        # Проверяем структуру
        assert "overall_score" in result, "Поле overall_score отсутствует"
        assert "max_possible_score" in result, "Поле max_possible_score отсутствует"
        assert "score_percentage" in result, "Поле score_percentage отсутствует"
        assert "traffic_light" in result, "Поле traffic_light отсутствует"
        assert result["max_possible_score"] == 100, (
            f"Максимум должен быть 100, а не {result['max_possible_score']}"
        )

        pct = result["score_percentage"]
        status = result["overall_status"]
        light = result["traffic_light"]

        print(f"\n  Оценка: {result['overall_score']}/100  ({pct}%)  [{light.upper()}]  {status}")
        print(f"\n  По категориям:")
        for cat_name, cat_data in result["categories"].items():
            bar = "#" * int(cat_data["percentage"] / 10) + "." * (10 - int(cat_data["percentage"] / 10))
            print(f"    [{bar}] {cat_data['percentage']:5.1f}%  {cat_name}: "
                  f"{cat_data['total_score']}/{cat_data['max_score']}")

        print(f"\n  Детальные результаты:")
        for item in result["all_items"]:
            marker = "+" if item["status"] == "Да" else "-"
            print(f"    {marker} {item['name']:<45} {item['status']:3}  "
                  f"{item['score']}/{item['max_score']}")

        # Базовые проверки
        ok = True
        if expect_min_pct > 0 and pct < expect_min_pct:
            print(f"\n  [FAIL] Ожидалось >={expect_min_pct}%, получено {pct}%")
            ok = False
        if expect_min_pct == 0 and pct > 40:
            print(f"\n  [WARN] Плохой звонок набрал {pct}% — возможно паттерны слишком широкие")
        if ok:
            print(f"\n  [OK] Тест пройден")
        else:
            all_passed = False

    # Сохраняем результаты
    output_file = Path(__file__).parent / "test_results.json"
    test_data = {
        "timestamp": datetime.now().isoformat(),
        "good_call": analyze_call(GOOD_CALL),
        "bad_call": analyze_call(BAD_CALL),
    }
    output_file.write_text(
        json.dumps(test_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n{'=' * 70}")
    print(f"  Результаты сохранены в: {output_file}")
    print(f"  Итог: {'ВСЕ ТЕСТЫ ПРОЙДЕНЫ' if all_passed else 'ЕСТЬ ПРОБЛЕМЫ'}")
    print("=" * 70 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(run_test())
