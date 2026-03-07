"""
Call KPI Analyzer - Анализ звонков по установленным критериям
Проверяет соответствие диалога KPI критериям и выдает оценку
"""

import re
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict


@dataclass
class KPIResult:
    """Результат проверки одного KPI"""
    key: str
    name: str
    category: str
    score: int
    max_score: int
    status: str  # "yes", "no", "partial"
    details: str = ""


class CallKPIAnalyzer:
    """Анализатор звонков по KPI критериям"""
    
    def __init__(self):
        self.results: List[KPIResult] = []
        self.total_score = 0
        self.max_total_score = 0
        
    def analyze(self, transcript: str) -> Dict:
        """
        Анализирует транскрипцию диалога по всем KPI
        
        Args:
            transcript: Текст транскрипции диалога
            
        Returns:
            Словарь с результатами анализа и итоговой оценкой
        """
        self.results = []
        self.total_score = 0
        self.max_total_score = 0
        
        # Нормализуем текст
        text = transcript.lower().strip()
        
        # ЭТАП 1: УСТАНОВЛЕНИЕ КОНТАКТА (вес: 15%)
        self._check_greeting(text)
        self._check_manager_intro(text)
        self._check_ask_client_name(text)
        self._check_call_by_name(text)
        
        # ЭТАП 2: УПРАВЛЕНИЕ ВСТРЕЧЕЙ (вес: 20%)
        self._check_meeting_purpose(text)
        self._check_meeting_plan(text)
        self._check_summary(text)
        self._check_ask_questions(text)
        
        # ЭТАП 3: СБОР ИНФОРМАЦИИ (вес: 25%)
        self._check_volumes(text)
        self._check_cargo_type(text)
        self._check_competitor_experience(text)
        self._check_preferences(text)
        
        # ЭТАП 4: ДЕМОНСТРАЦИЯ УСЛУГ (вес: 30%)
        self._check_cost_parameters(text)
        self._check_bulk_packaging(text)
        self._check_additional_services(text)
        self._check_insurance(text)
        self._check_cod_and_collection(text)
        self._check_paid_channels(text)
        self._check_free_channels(text)
        self._check_payment_process(text)
        self._check_deposit_end(text)
        self._check_special_services(text)
        self._check_integration(text)
        self._check_company_interaction(text)
        
        # ЭТАП 5: ФИНАЛИЗАЦИЯ (вес: 10%)
        self._check_telegram_warning(text)
        self._check_materials_sending(text)
        self._check_feedback_request(text)
        
        # Расчет итоговой оценки
        return self._calculate_final_score()
    
    # ===== ЭТАП 1: УСТАНОВЛЕНИЕ КОНТАКТА =====
    
    def _check_greeting(self, text: str) -> None:
        """Проверка приветствия клиента"""
        greeting_patterns = [
            r'\bздравствуй',
            r'\bдобрый\s+день',
            r'\bприветству',
            r'\bпривет\b',
            r'\bрада\s+вас\s+видеть'
        ]
        
        found = any(re.search(pattern, text) for pattern in greeting_patterns)
        status = "yes" if found else "no"
        score = 2 if found else 0
        
        self.results.append(KPIResult(
            key="greeting",
            name="Приветствие клиента",
            category="Установление контакта",
            score=score,
            max_score=2,
            status=status,
            details="Найдено приветствие в начале разговора" if found else "Приветствие не найдено"
        ))
        self.max_total_score += 2
        self.total_score += score
    
    def _check_manager_intro(self, text: str) -> None:
        """Проверка представления менеджера"""
        # Ищем фразы типа "меня зовут", "мое имя", "я менеджер"
        patterns = [
            r'меня\s+зовут\s+[а-яА-Я]+',
            r'мое\s+имя\s+[а-яА-Я]+',
            r'я\s+(?:менеджер|специалист|консультант)\s+[а-яА-Я]+',
            r'(?:представляюсь|представляюсь)\s+(?:я\s+)?[а-яА-Я]+'
        ]
        
        found = any(re.search(pattern, text) for pattern in patterns)
        status = "yes" if found else "no"
        score = 2 if found else 0
        
        self.results.append(KPIResult(
            key="manager_intro",
            name="Представление менеджера",
            category="Установление контакта",
            score=score,
            max_score=2,
            status=status,
            details="Менеджер представился" if found else "Менеджер не представился"
        ))
        self.max_total_score += 2
        self.total_score += score
    
    def _check_ask_client_name(self, text: str) -> None:
        """Проверка уточнения имени клиента"""
        patterns = [
            r'как\s+к\s+вам\s+обращаться',
            r'как\s+ваше\s+имя',
            r'назовите\s+(?:ваше\s+)?имя',
            r'правильно\s+произношу\s+(?:ваше\s+)?имя',
            r'как\s+я\s+могу\s+к\s+вам\s+обращаться',
            r'подскажите\s+имя'
        ]
        
        found = any(re.search(pattern, text) for pattern in patterns)
        status = "yes" if found else "no"
        score = 2 if found else 0
        
        self.results.append(KPIResult(
            key="ask_client_name",
            name="Уточнение имени клиента",
            category="Установление контакта",
            score=score,
            max_score=2,
            status=status,
            details="Менеджер уточнил имя клиента" if found else "Имя клиента не уточнено"
        ))
        self.max_total_score += 2
        self.total_score += score
    
    def _check_call_by_name(self, text: str) -> None:
        """Проверка обращения по имени клиента"""
        # Это сложная проверка - ищем множество обращений по имени
        # Ищем паттерны типа "имя," или "уважаемый имя"
        name_patterns = [
            r'[а-яА-Я]+,?\s+[а-яА-Я]+',
            r'уважаемый',
        ]
        matches = []
        for pattern in name_patterns:
            matches.extend(re.findall(pattern, text))
        
        # Простая эвристика: если несколько обращений - значит выполнено
        found = len(matches) >= 2
        status = "yes" if found else "no"
        score = 2 if found else 0
        
        self.results.append(KPIResult(
            key="call_by_name",
            name="Обращение по имени",
            category="Установление контакта",
            score=score,
            max_score=2,
            status=status,
            details=f"Найдено {len(matches)} обращений по имени" if found else "Обращения по имени отсутствуют"
        ))
        self.max_total_score += 2
        self.total_score += score
    
    # ===== ЭТАП 2: УПРАВЛЕНИЕ ВСТРЕЧЕЙ =====
    
    def _check_meeting_purpose(self, text: str) -> None:
        """Проверка объяснения цели встречи"""
        patterns = [
            r'цель\s+(?:этого\s+)?разговора',
            r'сегодня\s+(?:расскажу|поговорим|обсудим)',
            r'давайте\s+разберемся',
            r'вы\s+сможете\s+(?:узнать|разобраться|понять)',
            r'по\s+итогам\s+(?:вы\s+)?(?:поймете|получите)'
        ]
        
        found = any(re.search(pattern, text) for pattern in patterns)
        status = "yes" if found else "no"
        score = 2 if found else 0
        
        self.results.append(KPIResult(
            key="meeting_purpose",
            name="Объяснение цели встречи",
            category="Управление встречей",
            score=score,
            max_score=2,
            status=status,
            details="Цель встречи объяснена" if found else "Цель встречи не объяснена"
        ))
        self.max_total_score += 2
        self.total_score += score
    
    def _check_meeting_plan(self, text: str) -> None:
        """Проверка План встречи"""
        patterns = [
            r'сначала\s+(?:обсудим|разберемся)',
            r'затем\s+(?:покажу|расскажу)',
            r'потом\s+(?:ответим|обсудим)',
            r'в\s+конце\s+(?:ответ|вопрос)',
            r'по\s+плану',
            r'последовательность',
            r'сначала.*затем.*потом'
        ]
        
        found = any(re.search(pattern, text) for pattern in patterns)
        status = "yes" if found else "no"
        score = 2 if found else 0
        
        self.results.append(KPIResult(
            key="meeting_plan",
            name="План встречи",
            category="Управление встречей",
            score=score,
            max_score=2,
            status=status,
            details="План встречи озвучен" if found else "План встречи не озвучен"
        ))
        self.max_total_score += 2
        self.total_score += score
    
    def _check_summary(self, text: str) -> None:
        """Проверка подведения итогов"""
        patterns = [
            r'подвод[яи]\s+итог',
            r'(?:итак|значит|получается),',
            r'(?:мы\s+)?разобрали',
            r'обсудили\s+(?:с\s+вами\s+)?',
            r'закрепим\s+(?:пройденное|информацию)',
            r'в\s+итоге'
        ]
        
        found = any(re.search(pattern, text) for pattern in patterns)
        status = "yes" if found else "no"
        score = 5 if found else 0
        
        self.results.append(KPIResult(
            key="summary",
            name="Подведение итогов",
            category="Управление встречей",
            score=score,
            max_score=5,
            status=status,
            details="Итоги подведены" if found else "Итоги не подведены"
        ))
        self.max_total_score += 5
        self.total_score += score
    
    def _check_ask_questions(self, text: str) -> None:
        """Проверка приглашения к вопросам"""
        patterns = [
            r'остались\s+(?:ли\s+)?(?:какие-?нибудь\s+)?вопросы',
            r'есть\s+(?:ли\s+)?вопросы',
            r'что\s+еще\s+(?:подсказать|рассказать)',
            r'понятно\s+(?:ли\s+)?все',
            r'все\s+(?:понятно|ясно)',
            r'уточнить\s+что-?нибудь'
        ]
        
        found = any(re.search(pattern, text) for pattern in patterns)
        status = "yes" if found else "no"
        score = 4 if found else 0
        
        self.results.append(KPIResult(
            key="ask_questions",
            name="Приглашение к вопросам",
            category="Управление встречей",
            score=score,
            max_score=4,
            status=status,
            details="Вопросы приглашены" if found else "Вопросы не приглашены"
        ))
        self.max_total_score += 4
        self.total_score += score
    
    # ===== ЭТАП 3: СБОР ИНФОРМАЦИИ =====
    
    def _check_volumes(self, text: str) -> None:
        """Проверка уточнения объёмов"""
        patterns = [
            r'сколько\s+(?:примерно\s+)?(?:отправ|заказ)',
            r'объем[ы]?\s+(?:отправок|отправлений)',
            r'в\s+(?:месяц|неделю|день)',
            r'планируете\s+(?:в\s+)?(?:месяц|неделю)',
            r'количество\s+(?:отправ|заказ)'
        ]
        
        found = any(re.search(pattern, text) for pattern in patterns)
        status = "yes" if found else "no"
        score = 3 if found else 0
        
        self.results.append(KPIResult(
            key="volumes",
            name="Уточнение объёмов",
            category="Сбор информации",
            score=score,
            max_score=3,
            status=status,
            details="Объёмы уточнены" if found else "Объёмы не уточнены"
        ))
        self.max_total_score += 3
        self.total_score += score
    
    def _check_cargo_type(self, text: str) -> None:
        """Проверка типа грузов"""
        patterns = [
            r'что\s+(?:именно\s+)?(?:отправ|груз)',
            r'(?:одежда|техника|хрупк|товар)',
            r'категория\s+(?:товаров|грузов)',
            r'какие\s+(?:товары|грузы|изделия)'
        ]
        
        found = any(re.search(pattern, text) for pattern in patterns)
        status = "yes" if found else "no"
        score = 3 if found else 0
        
        self.results.append(KPIResult(
            key="cargo_type",
            name="Тип грузов",
            category="Сбор информации",
            score=score,
            max_score=3,
            status=status,
            details="Тип грузов уточнен" if found else "Тип грузов не уточнен"
        ))
        self.max_total_score += 3
        self.total_score += score
    
    def _check_competitor_experience(self, text: str) -> None:
        """Проверка опыта с конкурентами"""
        patterns = [
            r'работали\s+(?:с\s+)?(?:другими|конкурентами)',
            r'(?:другие\s+)?логистическ[ие]',
            r'работа\s+с\s+(?:другими\s+)?компани',
            r'с\s+кем\s+(?:работали|работаете)'
        ]
        
        found = any(re.search(pattern, text) for pattern in patterns)
        status = "yes" if found else "no"
        score = 2 if found else 0
        
        self.results.append(KPIResult(
            key="competitor_experience",
            name="Опыт с конкурентами",
            category="Сбор информации",
            score=score,
            max_score=2,
            status=status,
            details="Опыт выяснен" if found else "Опыт не выяснен"
        ))
        self.max_total_score += 2
        self.total_score += score
    
    def _check_preferences(self, text: str) -> None:
        """Проверка приоритетов и предпочтений"""
        patterns = [
            r'что\s+(?:нравилось|понравилось)',
            r'что\s+не\s+нравилось',
            r'положитель',
            r'проблем[ы]?',
            r'что\s+нравится.*что\s+не\s+нравится'
        ]
        
        found = any(re.search(pattern, text) for pattern in patterns)
        status = "yes" if found else "no"
        score = 2 if found else 0
        
        self.results.append(KPIResult(
            key="preferences",
            name="Приоритеты и предпочтения",
            category="Сбор информации",
            score=score,
            max_score=2,
            status=status,
            details="Приоритеты выяснены" if found else "Приоритеты не выяснены"
        ))
        self.max_total_score += 2
        self.total_score += score
    
    # ===== ЭТАП 4: ДЕМОНСТРАЦИЯ УСЛУГ =====
    
    def _check_cost_parameters(self, text: str) -> None:
        """Проверка стоимости и параметров груза"""
        patterns = [
            r'(?:вес|габарит|размер)',
            r'город\s+(?:отправ|получ)',
            r'стоимость\s+(?:зависит|меняется)',
            r'расстояние\s+влияет'
        ]
        
        found = any(re.search(pattern, text) for pattern in patterns)
        status = "yes" if found else "no"
        score = 1 if found else 0
        
        self.results.append(KPIResult(
            key="cost_parameters",
            name="Стоимость и параметры груза",
            category="Демонстрация услуг",
            score=score,
            max_score=1,
            status=status,
            details="Параметры обсуждены" if found else "Параметры не обсуждены"
        ))
        self.max_total_score += 1
        self.total_score += score
    
    def _check_bulk_packaging(self, text: str) -> None:
        """Проверка оптовой упаковки"""
        patterns = [
            r'упаковка\s+(?:оптом|партий)',
            r'выгод[ноа](?:е|я)?\s+(?:покупать|приобрести)',
            r'поштучно\s+дороже',
            r'коробок|пакетов|расходник'
        ]
        
        found = any(re.search(pattern, text) for pattern in patterns)
        status = "yes" if found else "no"
        score = 2 if found else 0
        
        self.results.append(KPIResult(
            key="bulk_packaging",
            name="Оптовая упаковка",
            category="Демонстрация услуг",
            score=score,
            max_score=2,
            status=status,
            details="Упаковка обсуждена" if found else "Упаковка не обсуждена"
        ))
        self.max_total_score += 2
        self.total_score += score
    
    def _check_additional_services(self, text: str) -> None:
        """Проверка дополнительных услуг"""
        services = [
            r'упаковка',
            r'примерка',
            r'частичн(?:ая|). доставка',
            r'(?:уведомл|смс|-)?уведомл',
            r'подъ[её]м\s+на\s+этаж',
            r'ожидание\s+курьера',
            r'дополнительн[ые]?'
        ]
        
        found_services = sum(1 for service in services if re.search(service, text))
        found = found_services >= 2
        status = "yes" if found else "no"
        score = 2 if found else 0
        
        self.results.append(KPIResult(
            key="additional_services",
            name="Дополнительные услуги",
            category="Демонстрация услуг",
            score=score,
            max_score=2,
            status=status,
            details=f"Найдено {found_services} услуг" if found else "Услуги не обсуждены"
        ))
        self.max_total_score += 2
        self.total_score += score
    
    def _check_insurance(self, text: str) -> None:
        """Проверка страхования (объявленная стоимость)"""
        patterns = [
            r'объявленн[ая|.]?\s+стоимость',
            r'страховк[ау]?',
            r'компенсаци[я]',
            r'утер[яи]?\s+или\s+повреждение'
        ]
        
        found = any(re.search(pattern, text) for pattern in patterns)
        status = "yes" if found else "no"
        score = 5 if found else 0
        
        self.results.append(KPIResult(
            key="insurance",
            name="Страхование",
            category="Демонстрация услуг",
            score=score,
            max_score=5,
            status=status,
            details="Страхование объяснено" if found else "Страхование не объяснено"
        ))
        self.max_total_score += 5
        self.total_score += score
    
    def _check_cod_and_collection(self, text: str) -> None:
        """Проверка наложенного платежа и инкассации"""
        payoff_patterns = [
            r'наложенн[ый.]?\s+платеж',
            r'оплата\s+(?:при\s+)?получении',
            r'товар[ом]?\s+получении'
        ]
        commission_patterns = [
            r'комиссия\s+(?:банка|за\s+перевод)',
            r'комиссия'
        ]
        collection_patterns = [
            r'инкассаци[я]',
            r'получатель\s+(?:оплачивает|платит)',
            r'полная\s+стоимость\s+(?:придет|приходит)'
        ]
        
        payoff_found = any(re.search(pattern, text) for pattern in payoff_patterns)
        commission_found = any(re.search(pattern, text) for pattern in commission_patterns)
        collection_found = any(re.search(pattern, text) for pattern in collection_patterns)
        
        found = payoff_found and commission_found and collection_found
        status = "yes" if found else "no"
        score = 5 if found else 0
        
        self.results.append(KPIResult(
            key="cod_and_collection",
            name="Наложенный платёж и инкассация",
            category="Демонстрация услуг",
            score=score,
            max_score=5,
            status=status,
            details="Все условия выполнены" if found else f"Наложенный платеж: {payoff_found}, Комиссия: {commission_found}, Инкассация: {collection_found}"
        ))
        self.max_total_score += 5
        self.total_score += score
    
    def _check_paid_channels(self, text: str) -> None:
        """Проверка платных каналов связи"""
        chat_patterns = [
            r'(?:платный|выделенный)\s+(?:чат|канал)',
            r'приоритетн[ый|.]?\s+чат',
            r'проф.*просмотр'
        ]
        cost_patterns = [
            r'199\s+(?:рублей|руб|р\.)',
            r'в\s+месяц'
        ]
        
        chat_found = any(re.search(pattern, text) for pattern in chat_patterns)
        cost_found = any(re.search(pattern, text) for pattern in cost_patterns)
        
        found = chat_found and cost_found
        status = "yes" if found else "no"
        score = 5 if found else 0
        
        self.results.append(KPIResult(
            key="paid_channels",
            name="Платные каналы связи",
            category="Демонстрация услуг",
            score=score,
            max_score=5,
            status=status,
            details="Платный чат и стоимость упомянуты" if found else "Информация о платном канале неполная"
        ))
        self.max_total_score += 5
        self.total_score += score
    
    def _check_free_channels(self, text: str) -> None:
        """Проверка бесплатного канала (почта)"""
        patterns = [
            r'(?:основной|единственный)\s+(?:канал|способ|способо)',
            r'электрон[ная|.]?\s+почт',
            r'email',
            r'основн[ая|.]?\s+(?:поддержка|связь)'
        ]
        
        found = any(re.search(pattern, text) for pattern in patterns)
        # Исключаем простую фразу "будем по почте общаться"
        if re.search(r'будем\s+по\s+почте\s+(?:общаться|писать)', text) and not any(
            re.search(p, text) for p in [r'основной', r'единственный', r'основная']
        ):
            found = False
        
        status = "yes" if found else "no"
        score = 10 if found else 0
        
        self.results.append(KPIResult(
            key="free_channels",
            name="Бесплатный канал (почта)",
            category="Демонстрация услуг",
            score=score,
            max_score=10,
            status=status,
            details="Почта как основной канал упомянута" if found else "Почта не упомянута как основной канал"
        ))
        self.max_total_score += 10
        self.total_score += score
    
    def _check_payment_process(self, text: str) -> None:
        """Проверка процесса оплаты"""
        patterns = [
            r'деньги\s+(?:будут\s+)?списываться\s+с\s+\w+',
            r'баланс\s+(?:будет\s+)?списываться',
            r'списание\s+со\s+' r'счета'
        ]
        
        found = any(re.search(pattern, text) for pattern in patterns)
        status = "yes" if found else "no"
        score = 5 if found else 0
        
        self.results.append(KPIResult(
            key="payment_process",
            name="Процесс оплаты",
            category="Демонстрация услуг",
            score=score,
            max_score=5,
            status=status,
            details="Процесс оплаты объяснен" if found else "Процесс оплаты не объяснен"
        ))
        self.max_total_score += 5
        self.total_score += score
    
    def _check_deposit_end(self, text: str) -> None:
        """Проверка окончания депозита"""
        patterns = [
            r'депозит\s+(?:закончится|заканчивается)',
            r'счет.*почты|почт.*счет',
            r'эдо',
            r'3\s+(?:дня|дней|дн\.)'
        ]
        
        found = any(re.search(pattern, text) for pattern in patterns)
        status = "yes" if found else "no"
        score = 5 if found else 0
        
        self.results.append(KPIResult(
            key="deposit_end",
            name="Окончание депозита",
            category="Демонстрация услуг",
            score=score,
            max_score=5,
            status=status,
            details="Депозит и счета обсуждены" if found else "Информация о депозите не найдена"
        ))
        self.max_total_score += 5
        self.total_score += score
    
    def _check_special_services(self, text: str) -> None:
        """Проверка специальных услуг (фулфилмент, реверс, возврат)"""
        # Фулфилмент
        fulfillment_patterns = [
            r'фулфилмент',
            r'full\s+film',
            r'фулфил',
            r'фул\w*фил'
        ]
        fulfillment_found = any(re.search(pattern, text) for pattern in fulfillment_patterns)
        
        # Реверс
        reversal_patterns = [
            r'\bреверс\b',
        ]
        reversal_found = any(re.search(pattern, text) for pattern in reversal_patterns)
        
        # Клиентский возврат
        client_return_patterns = [
            r'клиентск[ий|.]?\s+возврат',
            r'клиентского\s+возврата'
        ]
        client_return_found = any(re.search(pattern, text) for pattern in client_return_patterns)
        
        # Расчет баллов за специальные услуги
        special_score = 0
        if fulfillment_found:
            special_score += 5
        if reversal_found:
            special_score += 5
        if client_return_found:
            special_score += 5
        
        self.results.append(KPIResult(
            key="fulfillment",
            name="Фулфилмент",
            category="Демонстрация услуг",
            score=5 if fulfillment_found else 0,
            max_score=5,
            status="yes" if fulfillment_found else "no",
            details="Фулфилмент упомянут" if fulfillment_found else "Фулфилмент не упомянут"
        ))
        
        self.results.append(KPIResult(
            key="reversal",
            name="Реверс",
            category="Демонстрация услуг",
            score=5 if reversal_found else 0,
            max_score=5,
            status="yes" if reversal_found else "no",
            details="Реверс упомянут" if reversal_found else "Реверс не упомянут"
        ))
        
        self.results.append(KPIResult(
            key="client_return",
            name="Клиентский возврат",
            category="Демонстрация услуг",
            score=5 if client_return_found else 0,
            max_score=5,
            status="yes" if client_return_found else "no",
            details="Клиентский возврат упомянут" if client_return_found else "Клиентский возврат не упомянут"
        ))
        
        self.max_total_score += 15
        self.total_score += special_score
    
    def _check_integration(self, text: str) -> None:
        """Проверка интеграции"""
        patterns = [
            r'интеграци[я]',
            r'интернет-?магазин',
            r'crm',
            r'cms',
            r'маркетплейс',
            r'api',
            r'модул',
            r'плагин',
            r'автоматизаци',
            r'трекинг'
        ]
        
        found = any(re.search(pattern, text) for pattern in patterns)
        status = "yes" if found else "no"
        score = 5 if found else 0
        
        self.results.append(KPIResult(
            key="integration",
            name="Интеграция",
            category="Демонстрация услуг",
            score=score,
            max_score=5,
            status=status,
            details="Интеграция обсуждена" if found else "Интеграция не обсуждена"
        ))
        self.max_total_score += 5
        self.total_score += score
    
    def _check_company_interaction(self, text: str) -> None:
        """Проверка упоминания взаимодействия с компанией"""
        patterns = [
            r'взаимодействи[е.]?\s+с\s+(?:нами|сдэк)',
            r'взаимодействи[е.]?\s+со\s+(?:сдэк)',
            r'опыт\s+(?:работы\s+)?с\s+(?:нами|сдэк)',
            r'работ[ы]?\s+(?:с\s+)?(?:нами|сдэк)'
        ]
        
        found = any(re.search(pattern, text) for pattern in patterns)
        status = "yes" if found else "no"
        score = 1 if found else 0
        
        self.results.append(KPIResult(
            key="company_interaction",
            name="Взаимодействие с компанией",
            category="Демонстрация услуг",
            score=score,
            max_score=1,
            status=status,
            details="Взаимодействие с компанией упомянуто" if found else "Взаимодействие с компанией не упомянуто"
        ))
        self.max_total_score += 1
        self.total_score += score
    
    # ===== ЭТАП 5: ФИНАЛИЗАЦИЯ =====
    
    def _check_telegram_warning(self, text: str) -> None:
        """Проверка предупреждения о задержках и Telegram"""
        wa_delay_patterns = [
            r'задержк[и]?\s+(?:в\s+работе\s+)?ва',
            r'задержк[и]?\s+(?:в\s+работе\s+)?(?:whatsapp|вацап)',
            r'проблем[ы]?\s+(?:с\s+)?ва'
        ]
        
        telegram_patterns = [
            r'telegram',
            r'телеграм'
        ]
        
        link_patterns = [
            r'(?:ссылк[ау]|направ|скину|пришлю|пошлю).*telegram',
            r'telegram.*(?:ссылк[а]|чат)'
        ]
        
        wa_found = any(re.search(pattern, text) for pattern in wa_delay_patterns)
        telegram_found = any(re.search(pattern, text) for pattern in telegram_patterns)
        link_found = any(re.search(pattern, text) for pattern in link_patterns)
        
        found = wa_found and telegram_found and link_found
        status = "yes" if found else "no"
        score = 3 if found else 0
        
        self.results.append(KPIResult(
            key="telegram_warning",
            name="Предупреждение о задержках ВА",
            category="Финализация",
            score=score,
            max_score=3,
            status=status,
            details="Все условия выполнены" if found else f"ВА: {wa_found}, Telegram: {telegram_found}, Ссылка: {link_found}"
        ))
        self.max_total_score += 3
        self.total_score += score
    
    def _check_materials_sending(self, text: str) -> None:
        """Проверка отправки материалов"""
        patterns = [
            r'(?:скину|направ|отправлю|пришлю|пошлю)\s+(?:ссылк[и]?|инструкци|видео|информаци)',
            r'отправлю\s+(?:вам|вас)',
            r'материал[ы]?\s+(?:отправлю|будут)',
            r'инструкци[и]?\s+(?:будут|отправлю)'
        ]
        
        found = any(re.search(pattern, text) for pattern in patterns)
        status = "yes" if found else "no"
        score = 2 if found else 0
        
        self.results.append(KPIResult(
            key="materials_sending",
            name="Отправка материалов",
            category="Финализация",
            score=score,
            max_score=2,
            status=status,
            details="Материалы будут отправлены" if found else "Обещание отправить материалы не найдено"
        ))
        self.max_total_score += 2
        self.total_score += score
    
    def _check_feedback_request(self, text: str) -> None:
        """Проверка запроса обратной связи"""
        patterns = [
            r'отзыв',
            r'обратн[ая]?\s+связь',
            r'как\s+вам\s+(?:понравилось|работа)',
            r'удовлетворены\s+ли',
            r'остались\s+ли\s+(?:довольны|довольны)'
        ]
        
        found = any(re.search(pattern, text) for pattern in patterns)
        status = "yes" if found else "no"
        score = 3 if found else 0
        
        self.results.append(KPIResult(
            key="feedback_request",
            name="Запрос обратной связи",
            category="Финализация",
            score=score,
            max_score=3,
            status=status,
            details="Обратная связь запрошена" if found else "Обратная связь не запрошена"
        ))
        self.max_total_score += 3
        self.total_score += score
    
    # ===== РАСЧЕТ ФИНАЛЬНОЙ ОЦЕНКИ =====
    
    def _calculate_final_score(self) -> Dict:
        """Расчитывает финальную оценку и возвращает результаты"""
        
        # Группируем результаты по категориям
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = {
                    'results': [],
                    'total_score': 0,
                    'max_score': 0
                }
            categories[result.category]['results'].append(result)
            categories[result.category]['total_score'] += result.score
            categories[result.category]['max_score'] += result.max_score
        
        # Определяем общую оценку
        if self.max_total_score > 0:
            score_percentage = (self.total_score / self.max_total_score) * 100
        else:
            score_percentage = 0
        
        # Определяем статус
        if score_percentage >= 91:
            overall_status = "Отлично"
            rating = "★★★★★"
        elif score_percentage >= 76:
            overall_status = "Хорошо"
            rating = "★★★★"
        elif score_percentage >= 51:
            overall_status = "Удовлетворительно"
            rating = "★★★"
        else:
            overall_status = "Требует развития"
            rating = "★★"
        
        return {
            'overall_score': round(self.total_score, 1),
            'max_possible_score': self.max_total_score,
            'score_percentage': round(score_percentage, 1),
            'overall_status': overall_status,
            'rating': rating,
            'categories': {
                category: {
                    'total_score': cat_data['total_score'],
                    'max_score': cat_data['max_score'],
                    'percentage': round((cat_data['total_score'] / cat_data['max_score'] * 100) if cat_data['max_score'] > 0 else 0, 1),
                    'items': [asdict(r) for r in cat_data['results']]
                }
                for category, cat_data in categories.items()
            },
            'all_items': [asdict(r) for r in self.results]
        }


def analyze_call(transcript: str) -> Dict:
    """
    Главная функция для анализа звонка
    
    Args:
        transcript: Текст транскрипции диалога
        
    Returns:
        Словарь с результатами анализа
    """
    analyzer = CallKPIAnalyzer()
    return analyzer.analyze(transcript)
