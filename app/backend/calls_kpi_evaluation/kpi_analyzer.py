"""
Call KPI Analyzer — Анализ звонков СДЭК по чеклисту
Проверяет соответствие диалога KPI критериям и выдает оценку из 100 баллов.

Чеклист: "Пример для СДЭК" — 29 оцениваемых критериев (Да/Нет) + 3 текстовых поля.
Максимум: 100 баллов.
"""

import re
import json
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict


@dataclass
class KPIResult:
    """Результат проверки одного KPI"""
    key: str
    name: str
    category: str
    score: int
    max_score: int
    status: str  # "Да" or "Нет"
    details: str = ""
    evidence: Optional[List[Dict]] = None  # [{"quote": "...", "timestamp": "00:01:23"}]


# ═══════════════════════════════════════════════════════════════════
# Определение всех 29 критериев СДЭК чеклиста
# Ключи и баллы точно соответствуют чеклисту
# ═══════════════════════════════════════════════════════════════════

CHECKLIST_CRITERIA = [
    # ── ЭТАП 1: УСТАНОВЛЕНИЕ КОНТАКТА ──
    {
        "key": "privetstvie_klienta",
        "name": "Приветствие клиента",
        "category": "Установление контакта",
        "max_score": 2,
        "description": (
            'Менеджер поприветствовал клиента. Принимаются фразы '
            '"Здравствуйте", "Добрый день", "Рада вас видеть", "Приветствую" и т.п. '
            'Фраза должна быть в начале разговора. '
            'Ответ "Нет", если использованы фразы без приветствия типа '
            '"Начнем", "Давайте обсудим" без обращения.'
        ),
    },
    {
        "key": "predstavilsyamenedzher",
        "name": "Представился менеджер",
        "category": "Установление контакта",
        "max_score": 2,
        "description": (
            "Менеджер назвал своё имя. Фраза должна быть в начале диалога."
        ),
    },
    {
        "key": "menedzher_utochnil_imya_klienta",
        "name": "Менеджер уточнил имя клиента",
        "category": "Установление контакта",
        "max_score": 2,
        "description": (
            'Менеджер спрашивает, как к клиенту обращаться, или уточняет правильность '
            'произношения имени, или явно спрашивает имя. Фраза должна быть в начале разговора. '
            '"Подскажите, как к Вам могу обращаться?", '
            '"Правильно произношу Ваше имя?", '
            '"Скажите пожалуйста, Ваше имя".'
        ),
    },
    {
        "key": "obrashchenie_po_imeni",
        "name": "Обращение по имени",
        "category": "Установление контакта",
        "max_score": 2,
        "description": (
            "Менеджер использует имя клиента более одного раза в ходе диалога. "
            "Обращения по имени распределены по разным частям встречи, а не только в одной фразе."
        ),
    },

    # ── ЭТАП 2: УПРАВЛЕНИЕ ВСТРЕЧЕЙ ──
    {
        "key": "vstrecha",
        "name": "Объяснение цели встречи",
        "category": "Управление встречей",
        "max_score": 2,
        "description": (
            "Менеджер явно проговаривает, ЗАЧЕМ проводится встреча: что по итогам клиент "
            "должен понять/получить. Явное проговаривание цели в начале разговора (первая треть). "
            'Если цель только подразумевается ("Давайте поговорим о ваших отправках") — '
            '"Да" только при более-менее внятной формулировке.'
        ),
    },
    {
        "key": "menedzher_ob_yasnil_kakoi_plan_vstrechi",
        "name": "План встречи",
        "category": "Управление встречей",
        "max_score": 2,
        "description": (
            "Менеджер озвучивает структуру разговора: какие блоки обсудим и в какой последовательности. "
            '"Сначала обсудим ваши отправки, затем покажу, как работать в личном кабинете, '
            'в конце отвечу на вопросы". Если менеджер говорит "Сейчас я вам всё расскажу" '
            "без структуры — это ответ \"Нет\"."
        ),
    },
    {
        "key": "itogi_vstrechi",
        "name": "Подведение итогов встречи",
        "category": "Управление встречей",
        "max_score": 5,
        "description": (
            "Анализ ТОЛЬКО завершающего блока (последние 5-10 реплик). "
            '"Да" если менеджер: явно подводит итоги (обобщает, что было разобрано), ИЛИ '
            "повторяет цель встречи из начала, ИЛИ связывает итоги с проблемами клиента. "
            "Только явные финальные формулировки менеджера. Реплики клиента не учитываются."
        ),
    },
    {
        "key": "voprosy_po_nakladnoi",
        "name": "Вопросы в конце встречи",
        "category": "Управление встречей",
        "max_score": 4,
        "description": (
            "Анализ ТОЛЬКО завершающего блока (последние 5-10 реплик). "
            '"Да" если менеджер задал открытый вопрос ("Остались ли вопросы?", "Что ещё подсказать?"). '
            "Не засчитываются вопросы, ограниченные конкретным блоком. "
            "После этого менеджер НЕ предоставляет клиенту новую информацию."
        ),
    },

    # ── ЭТАП 3: СБОР ИНФОРМАЦИИ ──
    {
        "key": "utochnenie_obemov",
        "name": "Уточнение объёмов",
        "category": "Сбор информации",
        "max_score": 3,
        "description": (
            "Менеджер спрашивает о количестве отправлений (в штуках, в месяц, в день, в неделю) "
            'в контексте работы по договору. "Сколько отправок примерно планируете в месяц?" '
            '"Какие сейчас объемы, и ожидаете ли рост?"'
        ),
    },
    {
        "key": "tip_gruzov_kotorye_planiruet_otpravlyat",
        "name": "Тип грузов",
        "category": "Сбор информации",
        "max_score": 3,
        "description": (
            "Менеджер уточняет, какие товары/грузы клиент будет отправлять. "
            '"Что именно отправляете?" "Это одежда, техника, хрупкие товары?"'
        ),
    },
    {
        "key": "opyt_rabot",
        "name": "Опыт с другими логист. компаниями",
        "category": "Сбор информации",
        "max_score": 2,
        "description": (
            'Менеджер спросил об опыте работы с другими компаниями. Фразы: '
            '"работали с другими логистическими операторами", "с кем именно", '
            '"на каких условиях".'
        ),
    },
    {
        "key": "utochnenie_pro_priority",
        "name": "Уточнение приоритетов",
        "category": "Сбор информации",
        "max_score": 2,
        "description": (
            'Уточнил, что больше всего нравилось и что не нравилось в работе с логист. компаниями. '
            "Вопрос должен включать явное упоминание как положительного, так и отрицательного опыта."
        ),
    },

    # ── ЭТАП 4: ДЕМОНСТРАЦИЯ УСЛУГ ──
    {
        "key": "vzaimodeistviya_s_nami",
        "name": "Взаимодействие с нами",
        "category": "Демонстрация услуг",
        "max_score": 1,
        "description": (
            'Проверка на наличие фраз: "взаимодействие с нами", '
            '"взаимодействие со СДЭК", "опыт работы с нами", "опыт работы со СДЭК". '
            "Точное совпадение. Допускаются разные падежи."
        ),
    },
    {
        "key": "geografiya_otpravok",
        "name": "Стоимость и направления (габариты)",
        "category": "Демонстрация услуг",
        "max_score": 1,
        "description": (
            "Демонстрация того, что стоимость доставки зависит от параметров груза. "
            "Менеджер запрашивает или демонстрирует ввод веса, габаритов и городов."
        ),
    },
    {
        "key": "prezentacia",
        "name": "Оптовая упаковка",
        "category": "Демонстрация услуг",
        "max_score": 2,
        "description": (
            "Менеджер предложил приобретать упаковку (коробки, пакеты) оптом или заранее. "
            "Покупать партиями выгоднее; предложил ссылку на заказ упаковки или расходников; "
            "упомянул, что при больших объемах лучше своя упаковка."
        ),
    },
    {
        "key": "dopolnitel_nye_uslugi",
        "name": "Дополнительные услуги (ЛК)",
        "category": "Демонстрация услуг",
        "max_score": 2,
        "description": (
            "Менеджер упомянул хотя бы 2-3 услуги из списка: упаковка, примерка, "
            "частичная доставка, уведомление (СМС), подъем на этаж, ожидание курьера."
        ),
    },
    {
        "key": "strahovanie",
        "name": "Страхование",
        "category": "Демонстрация услуг",
        "max_score": 5,
        "description": (
            'Менеджер объяснил смысл поля «Объявленная стоимость»: указанная сумма — это '
            "страховка, она будет компенсирована при утере/повреждении. "
            '"Это ваша страховка", "для компенсации нужно указывать реальную стоимость".'
        ),
    },
    {
        "key": "kanal_svyzi",
        "name": "Платные каналы связи",
        "category": "Демонстрация услуг",
        "max_score": 5,
        "description": (
            'Менеджер: 1) Упомянул, что чат является «платный». '
            "2) Указал точную стоимость подписки (199 рублей в месяц). "
            '"Да" только если оба условия выполнены.'
        ),
    },
    {
        "key": "besplatnyi_kanal_svyazi_pochta",
        "name": "Бесплатный канал связи (почта)",
        "category": "Демонстрация услуг",
        "max_score": 10,
        "description": (
            "Менеджер объяснил, что почта — основной/единственный канал связи "
            "в случае отсутствия платного канала. Строгие условия: "
            '1) Произнес слово «почта»/«электронный адрес» в контексте замены. '
            '2) Использовал слово «основной» или «единственный». '
            '3) Фраза «будем по почте общаться» НЕ СЧИТАЕТСЯ.'
        ),
    },
    {
        "key": "process_oplaty",
        "name": "Объяснение процесса оплаты",
        "category": "Демонстрация услуг",
        "max_score": 5,
        "description": (
            'Наличие фразы «деньги будут списываться с баланса» и т.п. '
            '"Да" ТОЛЬКО при точном лексическом совпадении.'
        ),
    },
    {
        "key": "okonchanie_depozita",
        "name": "Окончание депозита",
        "category": "Демонстрация услуг",
        "max_score": 5,
        "description": (
            "Менеджер рассказал: когда закончится депозит → счет на почту/ЭДО → срок оплаты 3 дня. "
            "Все три условия должны быть выполнены."
        ),
    },
    {
        "key": "fulfilment",
        "name": "Фулфилмент",
        "category": "Демонстрация услуг",
        "max_score": 5,
        "description": (
            'Произнесено слово «фулфилмент». Учитывать фонетически похожие: '
            '"Full Film", "Фулфил", "Фулил", "Фулфиллмент", "фуллфил". '
            "Упоминания «хранение на складе», «упаковка» НЕ засчитываются."
        ),
    },
    {
        "key": "inkassatsiya_s_poluchatelya",
        "name": "Инкассация с получателя",
        "category": "Демонстрация услуг",
        "max_score": 5,
        "description": (
            "ВСЕ условия: 1) Объяснён наложенный платеж (оплата получателем). "
            "2) Упомянута комиссия банка. 3) Объяснено, что получатель покрывает комиссию, "
            "чтобы отправитель получил полную сумму, ИЛИ использовано слово «инкассация»."
        ),
    },
    {
        "key": "klientskii_vozvrat",
        "name": "Клиентский возврат",
        "category": "Демонстрация услуг",
        "max_score": 5,
        "description": (
            'Наличие словосочетания «Клиентский возврат» (допускаются падежи). '
            "Слово «возврат» без прилагательного «клиентский» НЕ засчитывается."
        ),
    },
    {
        "key": "revers",
        "name": "Реверс",
        "category": "Демонстрация услуг",
        "max_score": 5,
        "description": (
            'Произнесено слово «Реверс» в контексте предоставляемых услуг. '
            "Точное лексическое совпадение."
        ),
    },
    {
        "key": "integratsiya",
        "name": "Интеграция",
        "category": "Демонстрация услуг",
        "max_score": 5,
        "description": (
            "Менеджер рассказывал про интеграцию: с интернет-магазином, CRM, CMS, маркетплейсом; "
            "через API, модули, плагины. "
            'Засчитывать упоминание слова «интеграция» в контексте ЛК.'
        ),
    },

    # ── ЭТАП 5: ФИНАЛИЗАЦИЯ ──
    {
        "key": "sledyushii_shag",
        "name": "Менеджер озвучил задержки в работе WA",
        "category": "Финализация",
        "max_score": 3,
        "description": (
            "Три условия одновременно: 1) Упомянуты задержки в работе «ВА»/WhatsApp. "
            "2) Рекомендация общаться в Telegram-чат. "
            "3) Обещание прислать ссылку на Telegram после встречи. "
            '"Нет" если Telegram просто как один из каналов без упоминания проблем ВА.'
        ),
    },
    {
        "key": "dopolnitelnaya_informatsiya",
        "name": "Дополнительная информация",
        "category": "Финализация",
        "max_score": 2,
        "description": (
            "Менеджер пообещал отправить клиенту доп. материалы (ссылки, инструкции, видео) "
            "после встречи. Любые формулировки: «скину ссылки», «направлю информацию», "
            "«пришлем инструкции», «отправлю в чат»."
        ),
    },
    {
        "key": "otzyv",
        "name": "Отзыв",
        "category": "Финализация",
        "max_score": 3,
        "description": (
            "Менеджер запросил у клиента обратную связь (отзыв) по работе со СДЭК "
            "или по прошедшей встрече."
        ),
    },
]


class CallKPIAnalyzer:
    """
    Анализатор звонков СДЭК по чеклисту.
    29 критериев Да/Нет, суммарно 100 баллов.
    """

    def __init__(self):
        self.results: List[KPIResult] = []
        self.total_score = 0
        self.max_total_score = 0
        # Текстовые поля (без оценки)
        self.text_fields: Dict[str, str] = {}
        # Сегменты с таймстемпами для сбора evidence
        self.segments: List[Dict] = []
        self.first_third_segments: List[Dict] = []
        self.last_block_segments: List[Dict] = []

    @staticmethod
    def _fmt_ts(seconds: float) -> str:
        """Форматирует секунды в HH:MM:SS."""
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = int(seconds % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _find_evidence(self, patterns: List[str], segments: Optional[List[Dict]] = None,
                       max_items: int = 3) -> List[Dict]:
        """
        Ищет паттерны в сегментах и возвращает evidence с цитатами и таймстемпами.
        Три уровня поиска:
        1. В каждом сегменте по отдельности
        2. В парах соседних сегментов (фраза разбита на 2 части)
        3. По склеенному тексту — находим позицию совпадения и возвращаем ближайший сегмент
        """
        if not segments:
            segments = self.segments
        if not segments:
            return []

        evidence = []
        seen_indices = set()

        # Уровень 1: поиск в отдельных сегментах
        for pattern in patterns:
            for i, seg in enumerate(segments):
                if i in seen_indices:
                    continue
                text = seg.get("text", "")
                if re.search(pattern, text.lower()):
                    evidence.append({
                        "quote": text.strip(),
                        "timestamp": self._fmt_ts(seg.get("start", 0)),
                    })
                    seen_indices.add(i)
                    if len(evidence) >= max_items:
                        return evidence

        # Уровень 2: поиск в парах соседних сегментов
        if not evidence:
            for pattern in patterns:
                for i in range(len(segments) - 1):
                    if i in seen_indices or (i + 1) in seen_indices:
                        continue
                    combined = segments[i].get("text", "") + " " + segments[i + 1].get("text", "")
                    if re.search(pattern, combined.lower()):
                        evidence.append({
                            "quote": combined.strip(),
                            "timestamp": self._fmt_ts(segments[i].get("start", 0)),
                        })
                        seen_indices.add(i)
                        seen_indices.add(i + 1)
                        if len(evidence) >= max_items:
                            return evidence

        # Уровень 3: поиск по склеенному тексту, возврат ближайшего сегмента
        if not evidence:
            # Строим карту позиций: для каждого символа в склеенном тексте знаем индекс сегмента
            seg_texts = [seg.get("text", "") for seg in segments]
            joined = " ".join(seg_texts).lower()
            char_to_seg = []
            for i, txt in enumerate(seg_texts):
                char_to_seg.extend([i] * len(txt))
                if i < len(seg_texts) - 1:
                    char_to_seg.append(i)  # пробел-разделитель

            for pattern in patterns:
                m = re.search(pattern, joined)
                if m:
                    seg_idx = char_to_seg[m.start()] if m.start() < len(char_to_seg) else len(segments) - 1
                    if seg_idx not in seen_indices:
                        evidence.append({
                            "quote": segments[seg_idx].get("text", "").strip(),
                            "timestamp": self._fmt_ts(segments[seg_idx].get("start", 0)),
                        })
                        seen_indices.add(seg_idx)
                        if len(evidence) >= max_items:
                            return evidence

        return evidence

    def analyze(self, transcript: str, segments: Optional[List[Dict]] = None) -> Dict:
        """
        Анализирует транскрипцию диалога по всем критериям СДЭК чеклиста.

        Args:
            transcript: Полный текст транскрипции диалога
            segments: Список сегментов [{start, end, text}, ...] для evidence

        Returns:
            Словарь с результатами анализа и итоговой оценкой из 100
        """
        self.results = []
        self.total_score = 0
        self.max_total_score = 0
        self.text_fields = {}
        self.segments = segments or []

        text = transcript.strip()
        text_lower = text.lower()

        # Разбиваем сегменты на позиционные части
        total_segs = len(self.segments)
        if total_segs > 0:
            ft_end = max(1, total_segs // 3)
            lb_start = max(0, total_segs - 10)
            self.first_third_segments = self.segments[:ft_end]
            self.last_block_segments = self.segments[lb_start:]
        else:
            self.first_third_segments = []
            self.last_block_segments = []

        # Разбиваем текст на части для позиционного анализа
        # Если есть сегменты — строим first_third/last_block из них,
        # чтобы позиции текста и сегментов совпадали
        if self.first_third_segments:
            first_third = ' '.join(
                seg.get("text", "") for seg in self.first_third_segments
            ).lower()
        else:
            lines = text.split('\n')
            total_lines = len(lines) if lines else 1
            first_third_end = max(1, total_lines // 3)
            first_third = '\n'.join(lines[:first_third_end]).lower()

        if self.last_block_segments:
            last_block = ' '.join(
                seg.get("text", "") for seg in self.last_block_segments
            ).lower()
        else:
            lines = text.split('\n')
            total_lines = len(lines) if lines else 1
            last_block_start = max(0, total_lines - 10)
            last_block = '\n'.join(lines[last_block_start:]).lower()

        # ═══ ЭТАП 1: УСТАНОВЛЕНИЕ КОНТАКТА ═══
        self._check_privetstvie(first_third)
        self._check_predstavilsya(first_third)
        self._check_utochnil_imya(first_third)
        self._check_obrashchenie_po_imeni(text_lower)

        # ═══ ЭТАП 2: УПРАВЛЕНИЕ ВСТРЕЧЕЙ ═══
        self._check_tsel_vstrechi(first_third)
        self._check_plan_vstrechi(first_third)
        self._check_itogi_vstrechi(last_block)
        self._check_voprosy_v_kontse(last_block)

        # ═══ ЭТАП 3: СБОР ИНФОРМАЦИИ ═══
        self._check_utochnenie_obemov(text_lower)
        self._check_tip_gruzov(text_lower)
        self._check_opyt_rabot(text_lower)
        self._check_utochnenie_prioritetov(text_lower)

        # ═══ ЭТАП 4: ДЕМОНСТРАЦИЯ УСЛУГ ═══
        self._check_vzaimodeistvie_s_nami(text_lower)
        self._check_stoimost_napravleniya(text_lower)
        self._check_optovaya_upakovka(text_lower)
        self._check_dop_uslugi(text_lower)
        self._check_strahovanie(text_lower)
        self._check_platnye_kanaly(text_lower)
        self._check_besplatnyi_kanal_pochta(text_lower)
        self._check_process_oplaty(text_lower)
        self._check_okonchanie_depozita(text_lower)
        self._check_fulfilment(text_lower)
        self._check_inkassatsiya(text_lower)
        self._check_klientskii_vozvrat(text_lower)
        self._check_revers(text_lower)
        self._check_integratsiya(text_lower)

        # ═══ ЭТАП 5: ФИНАЛИЗАЦИЯ ═══
        self._check_zaderzhki_wa(text_lower)
        self._check_dop_informatsiya(text_lower)
        self._check_otzyv(last_block)

        # ═══ ТЕКСТОВЫЕ ПОЛЯ (без баллов) ═══
        self._extract_text_fields(text)

        return self._build_result()

    # ─────────────────────────────────────────────
    # ЭТАП 1: УСТАНОВЛЕНИЕ КОНТАКТА
    # ─────────────────────────────────────────────

    def _check_privetstvie(self, first_third: str) -> None:
        """privetstvie_klienta — 2 балла"""
        patterns = [
            r'\bздравствуй',
            r'\bдобрый\s+(?:день|вечер|утро)',
            r'\bприветству',
            r'\bрад[а]?\s+(?:вас|вам)\s+(?:видеть|приветствовать)',
            r'\bдоброго\s+(?:дня|времени)',
        ]
        # Анти-паттерны (не считаем приветствием)
        anti_patterns = [
            r'^(?:начнем|давайте\s+обсудим)',
        ]

        found = any(re.search(p, first_third) for p in patterns)
        if found and any(re.search(p, first_third) for p in anti_patterns):
            pass

        evidence = self._find_evidence(patterns, self.first_third_segments, max_items=2)
        self._add_result("privetstvie_klienta", found, evidence=evidence)

    def _check_predstavilsya(self, first_third: str) -> None:
        """predstavilsyamenedzher — 2 балла"""
        patterns = [
            r'меня\s+зовут\s+\w+',
            r'мо[её]\s+имя\s+\w+',
            r'я\s+(?:\w+\s+)?(?:менеджер|специалист|консультант|сотрудник)',
            r'(?:представлюсь|представляюсь)',
            r'я\s+\w+,?\s+(?:менеджер|специалист|консультант)',
        ]
        found = any(re.search(p, first_third) for p in patterns)
        evidence = self._find_evidence(patterns, self.first_third_segments, max_items=1)
        self._add_result("predstavilsyamenedzher", found, evidence=evidence)

    def _check_utochnil_imya(self, first_third: str) -> None:
        """menedzher_utochnil_imya_klienta — 2 балла"""
        patterns = [
            r'как\s+(?:к\s+вам|вас)\s+(?:мог[ау]?\s+)?обращаться',
            r'как\s+(?:ваше\s+)?имя',
            r'подскажите.*имя',
            r'правильно\s+(?:ли\s+)?произношу\s+(?:ваше\s+)?имя',
            r'скажите.*ваше\s+имя',
            r'как\s+я\s+могу\s+(?:к\s+вам\s+)?обращаться',
            r'назовите.*имя',
        ]
        found = any(re.search(p, first_third) for p in patterns)
        evidence = self._find_evidence(patterns, self.first_third_segments, max_items=1)
        self._add_result("menedzher_utochnil_imya_klienta", found, evidence=evidence)

    def _check_obrashchenie_po_imeni(self, text_lower: str) -> None:
        """obrashchenie_po_imeni — 2 балла
        Проверяем, что менеджер использует имя клиента более одного раза.
        Эвристика: ищем паттерны персонального обращения.
        """
        # Ищем типичные имена/обращения в контексте менеджера
        # Паттерн: слово с заглавной буквы после запятой или в начале обращения
        # В нижнем регистре ищем паттерны обращения
        name_call_patterns = [
            r'(?:здравствуйте|добрый день|приветствую)\s*,?\s*([а-я]+)',
            r'(?:скажите|подскажите|расскажите)\s*,?\s*([а-я]+)\s*,',
            r',\s*([а-я]+)\s*,',
            r'([а-я]+)\s*,\s*(?:давайте|скажите|подскажите|расскажите)',
        ]
        all_names = []
        for pattern in name_call_patterns:
            matches = re.findall(pattern, text_lower)
            all_names.extend(matches)

        # Фильтруем служебные слова
        stop_words = {
            'да', 'нет', 'так', 'ну', 'вот', 'это', 'что', 'как', 'вы',
            'мы', 'он', 'она', 'они', 'тут', 'там', 'ещё', 'еще', 'уже',
            'давайте', 'скажите', 'подскажите', 'расскажите', 'здравствуйте',
            'пожалуйста', 'спасибо', 'хорошо', 'ладно', 'конечно',
        }
        real_names = [n for n in all_names if n not in stop_words and len(n) >= 3]

        # Если нашли минимум 2 обращения по имени
        found = len(real_names) >= 2
        details = f"Найдено {len(real_names)} обращений по имени" if found else "Менее 2 обращений по имени"
        # Evidence: ищем сегменты где есть обращения по имени
        evidence = self._find_evidence(name_call_patterns, self.segments, max_items=3)
        self._add_result("obrashchenie_po_imeni", found, details, evidence=evidence)

    # ─────────────────────────────────────────────
    # ЭТАП 2: УПРАВЛЕНИЕ ВСТРЕЧЕЙ
    # ─────────────────────────────────────────────

    def _check_tsel_vstrechi(self, first_third: str) -> None:
        """vstrecha — 2 балла"""
        patterns = [
            r'цель\s+(?:нашей\s+)?(?:сегодняшней\s+)?(?:встречи|разговора|беседы)',
            r'(?:сегодня|сейчас)\s+(?:мы\s+)?(?:расскажу|рассмотрим|разберем|обсудим|покажу)',
            r'по\s+итогам?\s+(?:вы\s+)?(?:поймете|получите|узнаете|сможете)',
            r'(?:вы\s+)?(?:поймёте|узнаете|сможете|получите).*(?:по\s+итогам|в\s+результате)',
            r'задача\s+(?:нашей\s+)?встречи',
            r'(?:хочу|хотела?\s+бы)\s+(?:вам\s+)?(?:рассказать|показать|продемонстрировать)',
        ]
        found = any(re.search(p, first_third) for p in patterns)
        evidence = self._find_evidence(patterns, self.first_third_segments, max_items=2)
        self._add_result("vstrecha", found, evidence=evidence)

    def _check_plan_vstrechi(self, first_third: str) -> None:
        """menedzher_ob_yasnil_kakoi_plan_vstrechi — 2 балла"""
        patterns = [
            r'сначала\s+(?:мы\s+)?(?:обсудим|разберем|рассмотрим|поговорим)',
            r'(?:затем|потом|после\s+этого)\s+(?:мы\s+)?(?:покажу|расскажу|перейдем|обсудим)',
            r'в\s+(?:конце|завершени)',
            r'план\s+(?:нашей\s+)?встречи',
            r'(?:первое|второе|третье)',
            r'по\s+(?:следующему\s+)?плану',
            r'сначала.*(?:затем|потом).*(?:в\s+конце|потом)',
        ]
        found = any(re.search(p, first_third) for p in patterns)
        evidence = self._find_evidence(patterns, self.first_third_segments, max_items=2)
        self._add_result("menedzher_ob_yasnil_kakoi_plan_vstrechi", found, evidence=evidence)

    def _check_itogi_vstrechi(self, last_block: str) -> None:
        """itogi_vstrechi — 5 баллов"""
        patterns = [
            r'подвед[ёе]м\s+итог',
            r'подводя\s+итог',
            r'(?:итак|значит|получается)\s*,?\s*(?:мы\s+)?(?:разобрали|обсудили|рассмотрели)',
            r'(?:мы\s+)?(?:с\s+вами\s+)?(?:разобрали|обсудили|рассмотрели)\s+(?:все|основн)',
            r'(?:закрепим|повторим|резюмируя)',
            r'в\s+итоге\s+(?:мы|вы)',
            r'давайте\s+(?:подведём|подведем|вспомним)',
        ]
        found = any(re.search(p, last_block) for p in patterns)
        evidence = self._find_evidence(patterns, self.last_block_segments, max_items=2)
        self._add_result("itogi_vstrechi", found, evidence=evidence)

    def _check_voprosy_v_kontse(self, last_block: str) -> None:
        """voprosy_po_nakladnoi — 4 балла"""
        patterns = [
            r'остались\s+(?:ли\s+)?(?:какие[\s-]?(?:то|нибудь|либо)\s+)?вопросы',
            r'(?:есть|имеются)\s+(?:ли\s+)?(?:ещё\s+|еще\s+)?вопросы',
            r'что\s+(?:ещё|еще)\s+(?:подсказать|рассказать|уточнить)',
            r'(?:хотите|хотели\s+бы)\s+(?:ещё\s+|еще\s+)?(?:что[\s-]?(?:то|нибудь)\s+)?(?:спросить|уточнить|узнать)',
            r'(?:все\s+)?(?:понятно|ясно)\s*\?',
        ]
        found = any(re.search(p, last_block) for p in patterns)
        evidence = self._find_evidence(patterns, self.last_block_segments, max_items=2)
        self._add_result("voprosy_po_nakladnoi", found, evidence=evidence)

    # ─────────────────────────────────────────────
    # ЭТАП 3: СБОР ИНФОРМАЦИИ
    # ─────────────────────────────────────────────

    def _check_utochnenie_obemov(self, text_lower: str) -> None:
        """utochnenie_obemov — 3 балла"""
        patterns = [
            r'сколько\s+(?:примерно\s+)?(?:отправ|посыл|заказ)',
            r'объ[её]м[ы]?\s+(?:отправок|отправлений|посылок)',
            r'(?:планируете|отправляете)\s+(?:в\s+)?(?:месяц|неделю|день)',
            r'количество\s+(?:отправ|заказ|посыл)',
            r'(?:какие|какой)\s+(?:сейчас\s+)?объ[её]м',
            r'ожидаете\s+(?:ли\s+)?рост',
        ]
        found = any(re.search(p, text_lower) for p in patterns)
        evidence = self._find_evidence(patterns, self.segments, max_items=2)
        self._add_result("utochnenie_obemov", found, evidence=evidence)

    def _check_tip_gruzov(self, text_lower: str) -> None:
        """tip_gruzov_kotorye_planiruet_otpravlyat — 3 балла"""
        patterns = [
            r'что\s+(?:именно\s+)?(?:отправляете|будете\s+отправлять)',
            r'какие?\s+(?:именно\s+)?(?:товар|груз|продукци)',
            r'(?:одежда|техника|хрупк|электроник|косметик|обувь)',
            r'категори[яи]\s+(?:товаров|грузов)',
            r'тип\s+(?:товаров|грузов|отправлений)',
            r'что\s+(?:за\s+)?(?:товар|груз|продукци)',
        ]
        found = any(re.search(p, text_lower) for p in patterns)
        evidence = self._find_evidence(patterns, self.segments, max_items=2)
        self._add_result("tip_gruzov_kotorye_planiruet_otpravlyat", found, evidence=evidence)

    def _check_opyt_rabot(self, text_lower: str) -> None:
        """opyt_rabot — 2 балла"""
        patterns = [
            r'работали\s+(?:с\s+)?(?:другими|иными)',
            r'(?:другие|иные)\s+логистическ',
            r'логистическ[ие]\??\s+(?:операторы|компании)',
            r'с\s+кем\s+(?:работали|работаете|сотруднича)',
            r'на\s+каких\s+условиях',
            r'(?:опыт|работа)\s+(?:с\s+)?(?:другими\s+)?(?:транспортн|логистическ|курьерск)',
        ]
        found = any(re.search(p, text_lower) for p in patterns)
        evidence = self._find_evidence(patterns, self.segments, max_items=2)
        self._add_result("opyt_rabot", found, evidence=evidence)

    def _check_utochnenie_prioritetov(self, text_lower: str) -> None:
        """utochnenie_pro_priority — 2 балла"""
        positive_patterns = [
            r'что\s+(?:больше\s+всего\s+)?(?:нравилось|понравилось|устраивало)',
            r'положительн[ые]?[ой]?\s+(?:момент|опыт|стороны)',
        ]
        negative_patterns = [
            r'(?:что\s+)?не\s+(?:нравилось|понравилось|устраивало)',
            r'(?:проблем|недостатк|минус)',
            r'хотели\s+бы\s+избежать',
        ]
        positive_found = any(re.search(p, text_lower) for p in positive_patterns)
        negative_found = any(re.search(p, text_lower) for p in negative_patterns)

        found = positive_found and negative_found
        details = (
            f"Положительный опыт: {'Да' if positive_found else 'Нет'}, "
            f"Отрицательный опыт: {'Да' if negative_found else 'Нет'}"
        )
        all_patterns = positive_patterns + negative_patterns
        evidence = self._find_evidence(all_patterns, self.segments, max_items=3)
        self._add_result("utochnenie_pro_priority", found, details, evidence=evidence)

    # ─────────────────────────────────────────────
    # ЭТАП 4: ДЕМОНСТРАЦИЯ УСЛУГ
    # ─────────────────────────────────────────────

    def _check_vzaimodeistvie_s_nami(self, text_lower: str) -> None:
        """vzaimodeistviya_s_nami — 1 балл"""
        patterns = [
            r'взаимодействи[еяю]\s+с\s+нами',
            r'взаимодействи[еяю]\s+со?\s+сдэк',
            r'опыт[а]?\s+работы\s+с\s+нами',
            r'опыт[а]?\s+работы\s+со?\s+сдэк',
        ]
        found = any(re.search(p, text_lower) for p in patterns)
        evidence = self._find_evidence(patterns, self.segments, max_items=2)
        self._add_result("vzaimodeistviya_s_nami", found, evidence=evidence)

    def _check_stoimost_napravleniya(self, text_lower: str) -> None:
        """geografiya_otpravok — 1 балл"""
        weight_size_patterns = [
            r'(?:вес|габарит|размер)',
        ]
        city_patterns = [
            r'(?:город|направлени|откуда|куда)',
        ]
        cost_patterns = [
            r'(?:стоимость|цена|тариф|рассчита)',
        ]
        weight_found = any(re.search(p, text_lower) for p in weight_size_patterns)
        city_found = any(re.search(p, text_lower) for p in city_patterns)
        cost_found = any(re.search(p, text_lower) for p in cost_patterns)

        found = (weight_found and city_found) or (weight_found and cost_found) or (city_found and cost_found)
        all_patterns = weight_size_patterns + city_patterns + cost_patterns
        evidence = self._find_evidence(all_patterns, self.segments, max_items=3)
        self._add_result("geografiya_otpravok", found, evidence=evidence)

    def _check_optovaya_upakovka(self, text_lower: str) -> None:
        """prezentacia — 2 балла"""
        patterns = [
            r'упаковк[аиу]\s+(?:оптом|партией|партиями)',
            r'(?:коробк|пакет|расходник).*(?:выгодн|дешевл|оптом)',
            r'(?:выгодн|дешевл).*(?:коробк|пакет|расходник)',
            r'(?:собственн|своя|свою)\s+упаковк',
            r'(?:закуп|купить|приобрести).*(?:упаковк|коробк|пакет)',
            r'ссылк[ау]\s+на\s+(?:заказ\s+)?упаковк',
        ]
        found = any(re.search(p, text_lower) for p in patterns)
        evidence = self._find_evidence(patterns, self.segments, max_items=2)
        self._add_result("prezentacia", found, evidence=evidence)

    def _check_dop_uslugi(self, text_lower: str) -> None:
        """dopolnitel_nye_uslugi — 2 балла"""
        services = [
            r'упаковка',
            r'примерка',
            r'частичн[аоуыей]+\s+(?:доставк|выдач)',
            r'(?:уведомлени|смс|sms)',
            r'подъ[её]м\s+на\s+этаж',
            r'ожидание\s+курьера',
        ]
        found_count = sum(1 for s in services if re.search(s, text_lower))
        found = found_count >= 2
        details = f"Найдено {found_count} из 6 услуг"
        evidence = self._find_evidence(services, self.segments, max_items=3)
        self._add_result("dopolnitel_nye_uslugi", found, details, evidence=evidence)

    def _check_strahovanie(self, text_lower: str) -> None:
        """strahovanie — 5 баллов"""
        patterns = [
            r'объявленн[аоуыей]+\s+стоимость',
            r'(?:это|ваша|является)\s+(?:ваша?\s+)?страховк',
            r'компенсаци[яию].*(?:утер|поврежд)',
            r'(?:утер|поврежд).*компенсаци',
            r'указывать\s+(?:реальную\s+)?стоимость',
            r'страхов[ка]+.*компенсир',
        ]
        found = any(re.search(p, text_lower) for p in patterns)
        evidence = self._find_evidence(patterns, self.segments, max_items=2)
        self._add_result("strahovanie", found, evidence=evidence)

    def _check_platnye_kanaly(self, text_lower: str) -> None:
        """kanal_svyzi — 5 баллов"""
        chat_patterns = [
            r'(?:платный|выделенный|приоритетный)\s+(?:чат|канал)',
            r'(?:чат|канал)\s+(?:является\s+)?(?:платн)',
        ]
        price_patterns = [
            r'199\s*(?:рублей|руб|р[\.\s])',
            r'(?:стоимость|цена|подписка).*199',
            r'199.*(?:в\s+месяц|ежемесячн)',
        ]
        chat_found = any(re.search(p, text_lower) for p in chat_patterns)
        price_found = any(re.search(p, text_lower) for p in price_patterns)
        found = chat_found and price_found
        details = f"Платный чат: {'Да' if chat_found else 'Нет'}, Стоимость 199₽: {'Да' if price_found else 'Нет'}"
        all_patterns = chat_patterns + price_patterns
        evidence = self._find_evidence(all_patterns, self.segments, max_items=3)
        self._add_result("kanal_svyzi", found, details, evidence=evidence)

    def _check_besplatnyi_kanal_pochta(self, text_lower: str) -> None:
        """besplatnyi_kanal_svyazi_pochta — 10 баллов"""
        mail_patterns = [
            r'(?:почт|электронн[ыойаяе]+\s+адрес)',
        ]
        main_patterns = [
            r'(?:основн[ойыаяе]+|единственн[ойыаяе]+)',
        ]
        # Анти-паттерн: "будем по почте общаться" без слов основной/единственный
        anti_pattern = r'будем\s+по\s+почте\s+(?:общаться|писать|тогда)'

        mail_found = any(re.search(p, text_lower) for p in mail_patterns)
        main_found = any(re.search(p, text_lower) for p in main_patterns)

        # Проверяем что «почта» и «основной/единственный» используются в контексте канала связи
        context_patterns = [
            r'(?:основн|единственн)[а-я]*\s+(?:канал|способ|средств)',
            r'(?:почт|email)[а-я]*\s+(?:является|будет|станет)\s+(?:основн|единственн)',
            r'(?:основн|единственн)[а-я]*.*(?:почт|email)',
            r'(?:почт|email).*(?:основн|единственн)',
        ]
        context_found = any(re.search(p, text_lower) for p in context_patterns)

        # Если есть анти-паттерн и нет слов основной/единственный — не засчитываем
        has_anti = bool(re.search(anti_pattern, text_lower))
        if has_anti and not main_found:
            found = False
        else:
            found = mail_found and main_found and context_found

        details = (
            f"Почта: {'Да' if mail_found else 'Нет'}, "
            f"Основной/единственный: {'Да' if main_found else 'Нет'}, "
            f"Контекст канала: {'Да' if context_found else 'Нет'}"
        )
        all_patterns = mail_patterns + main_patterns + context_patterns
        evidence = self._find_evidence(all_patterns, self.segments, max_items=3)
        self._add_result("besplatnyi_kanal_svyazi_pochta", found, details, evidence=evidence)

    def _check_process_oplaty(self, text_lower: str) -> None:
        """process_oplaty — 5 баллов"""
        patterns = [
            r'деньги\s+(?:будут\s+)?списываться\s+(?:с|со)\s+баланс',
            r'списани[ея]\s+(?:с|со)\s+баланс',
            r'средства\s+(?:будут\s+)?списываться\s+(?:с|со)\s+баланс',
            r'(?:с|со)\s+баланса\s+(?:будут\s+)?списываться',
            r'(?:оплата|списание)\s+(?:происходит|идет|будет)\s+(?:с|со|из)\s+баланс',
        ]
        found = any(re.search(p, text_lower) for p in patterns)
        evidence = self._find_evidence(patterns, self.segments, max_items=2)
        self._add_result("process_oplaty", found, evidence=evidence)

    def _check_okonchanie_depozita(self, text_lower: str) -> None:
        """okonchanie_depozita — 5 баллов"""
        deposit_patterns = [
            r'(?:депозит|аванс)\s+(?:закончится|заканчивается|когда.*закон)',
            r'(?:закон|истеч).*(?:депозит|аванс)',
        ]
        invoice_patterns = [
            r'(?:счет|счёт).*(?:почт|эдо|едо)',
            r'(?:почт|эдо|едо).*(?:счет|счёт)',
            r'(?:будет\s+)?(?:приходить|приходит|формируется)\s+(?:счет|счёт)',
        ]
        days_patterns = [
            r'(?:срок|в\s+течени).*3\s+(?:дня|дней|рабочих)',
            r'3\s+(?:дня|дней|рабочих).*(?:оплат|срок)',
        ]

        deposit_found = any(re.search(p, text_lower) for p in deposit_patterns)
        invoice_found = any(re.search(p, text_lower) for p in invoice_patterns)
        days_found = any(re.search(p, text_lower) for p in days_patterns)

        found = deposit_found and invoice_found and days_found
        details = (
            f"Депозит: {'Да' if deposit_found else 'Нет'}, "
            f"Счет почта/ЭДО: {'Да' if invoice_found else 'Нет'}, "
            f"Срок 3 дня: {'Да' if days_found else 'Нет'}"
        )
        all_patterns = deposit_patterns + invoice_patterns + days_patterns
        evidence = self._find_evidence(all_patterns, self.segments, max_items=3)
        self._add_result("okonchanie_depozita", found, details, evidence=evidence)

    def _check_fulfilment(self, text_lower: str) -> None:
        """fulfilment — 5 баллов"""
        patterns = [
            r'фулфилмент',
            r'фулфиллмент',
            r'full\s*fil[lm]?',
            r'фулфил\b',
            r'фулил\b',
            r'фуллфил',
        ]
        found = any(re.search(p, text_lower) for p in patterns)
        evidence = self._find_evidence(patterns, self.segments, max_items=2)
        self._add_result("fulfilment", found, evidence=evidence)

    def _check_inkassatsiya(self, text_lower: str) -> None:
        """inkassatsiya_s_poluchatelya — 5 баллов"""
        payoff_patterns = [
            r'наложенн[а-яё]*\s+плат[её]ж',
            r'оплат[аеу]\s+(?:при\s+)?получении',
        ]
        commission_patterns = [
            r'комисси[яию]\s+банка',
            r'комисси[яию]\s+(?:за\s+)?перевод',
            r'банковск[а-яё]*\s+комисси',
        ]
        collection_patterns = [
            r'инкассаци[яию]',
            r'получатель\s+(?:оплачивает|покрывает|платит)\s+комисси',
            r'полн[а-яё]*\s+(?:стоимость|сумм[аеу])\s+(?:пришла|приходит|придёт|придет)',
            r'(?:отправител[ьюя]|вам|вы)\s+(?:получ[а-яё]+|получите)\s+(?:ровн|полн)',
        ]

        payoff = any(re.search(p, text_lower) for p in payoff_patterns)
        commission = any(re.search(p, text_lower) for p in commission_patterns)
        collection = any(re.search(p, text_lower) for p in collection_patterns)

        found = payoff and commission and collection
        details = (
            f"Наложенный платеж: {'Да' if payoff else 'Нет'}, "
            f"Комиссия: {'Да' if commission else 'Нет'}, "
            f"Инкассация: {'Да' if collection else 'Нет'}"
        )
        all_patterns = payoff_patterns + commission_patterns + collection_patterns
        evidence = self._find_evidence(all_patterns, self.segments, max_items=3)
        self._add_result("inkassatsiya_s_poluchatelya", found, details, evidence=evidence)

    def _check_klientskii_vozvrat(self, text_lower: str) -> None:
        """klientskii_vozvrat — 5 баллов"""
        patterns = [
            r'клиентск[а-яё]*\s+возврат',
        ]
        found = any(re.search(p, text_lower) for p in patterns)
        evidence = self._find_evidence(patterns, self.segments, max_items=2)
        self._add_result("klientskii_vozvrat", found, evidence=evidence)

    def _check_revers(self, text_lower: str) -> None:
        """revers — 5 баллов"""
        # Только в контексте услуг, не в общих словах
        patterns = [
            r'\bреверс\b',
        ]
        found = any(re.search(p, text_lower) for p in patterns)
        evidence = self._find_evidence(patterns, self.segments, max_items=2)
        self._add_result("revers", found, evidence=evidence)

    def _check_integratsiya(self, text_lower: str) -> None:
        """integratsiya — 5 баллов"""
        patterns = [
            r'\bинтеграци[яию]\b',
            r'интеграци[яию]\s+(?:с\s+)?(?:интернет|crm|cms|маркетплейс|api)',
        ]
        found = any(re.search(p, text_lower) for p in patterns)
        evidence = self._find_evidence(patterns, self.segments, max_items=2)
        self._add_result("integratsiya", found, evidence=evidence)

    # ─────────────────────────────────────────────
    # ЭТАП 5: ФИНАЛИЗАЦИЯ
    # ─────────────────────────────────────────────

    def _check_zaderzhki_wa(self, text_lower: str) -> None:
        """sledyushii_shag — 3 балла"""
        wa_delay_patterns = [
            r'задержк[а-яё]*\s+(?:в\s+работе\s+)?(?:ва|whatsapp|вацап|ватсап|вотсап)',
            r'(?:ва|whatsapp|вацап|ватсап|вотсап)\s+(?:работает\s+)?(?:с\s+задержк|нестабильн|плохо)',
        ]
        telegram_patterns = [
            r'telegram',
            r'телеграм',
        ]
        link_patterns = [
            r'(?:ссылк|направ|скину|пришлю|отправлю).*(?:telegram|телеграм)',
            r'(?:telegram|телеграм).*(?:ссылк|чат)',
        ]

        wa_found = any(re.search(p, text_lower) for p in wa_delay_patterns)
        tg_found = any(re.search(p, text_lower) for p in telegram_patterns)
        link_found = any(re.search(p, text_lower) for p in link_patterns)

        found = wa_found and tg_found and link_found
        details = (
            f"Задержки ВА: {'Да' if wa_found else 'Нет'}, "
            f"Telegram: {'Да' if tg_found else 'Нет'}, "
            f"Ссылка: {'Да' if link_found else 'Нет'}"
        )
        all_patterns = wa_delay_patterns + telegram_patterns + link_patterns
        evidence = self._find_evidence(all_patterns, self.segments, max_items=3)
        self._add_result("sledyushii_shag", found, details, evidence=evidence)

    def _check_dop_informatsiya(self, text_lower: str) -> None:
        """dopolnitelnaya_informatsiya — 2 балла"""
        patterns = [
            r'(?:скину|направлю|отправлю|пришлю|пошлю)\s+(?:вам\s+)?(?:ссылк|инструкци|видео|информаци|материал)',
            r'(?:ссылк|инструкци|видео|материал).*(?:скину|направлю|отправлю|пришлю)',
            r'(?:после\s+встречи|в\s+чат|после\s+звонка).*(?:скину|направлю|отправлю|пришлю)',
        ]
        found = any(re.search(p, text_lower) for p in patterns)
        evidence = self._find_evidence(patterns, self.segments, max_items=2)
        self._add_result("dopolnitelnaya_informatsiya", found, evidence=evidence)

    def _check_otzyv(self, last_block: str) -> None:
        """otzyv — 3 балла"""
        patterns = [
            r'отзыв',
            r'обратн[а-яё]*\s+связь',
            r'(?:оцен|оставьте).*(?:отзыв|оценку)',
            r'(?:как\s+вам|понравил).*(?:встреча|работа|обслуживание)',
        ]
        found = any(re.search(p, last_block) for p in patterns)
        evidence = self._find_evidence(patterns, self.last_block_segments, max_items=2)
        self._add_result("otzyv", found, evidence=evidence)

    # ─────────────────────────────────────────────
    # ТЕКСТОВЫЕ ПОЛЯ (без оценки)
    # ─────────────────────────────────────────────

    def _extract_text_fields(self, text: str) -> None:
        """Извлекает текстовые поля: next_step, summary, detailed_summary"""
        # Краткий итог — последние предложения разговора
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        self.text_fields["summary"] = '. '.join(sentences[-3:]) + '.' if sentences else ""
        self.text_fields["detailed_summary"] = text[:2000] if text else ""

        # Следующий шаг — ищем обещания действий
        next_step_patterns = [
            r'(?:я\s+)?(?:скину|направлю|отправлю|пришлю).*',
            r'(?:давайте|договорились).*(?:свяжемся|созвонимся|напишу).*',
            r'(?:следующий\s+шаг|далее|дальше).*',
        ]
        next_steps = []
        for pattern in next_step_patterns:
            matches = re.findall(pattern, text.lower())
            next_steps.extend(matches[:2])
        self.text_fields["next_step"] = '; '.join(next_steps[:3]) if next_steps else ""

    # ─────────────────────────────────────────────
    # ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ─────────────────────────────────────────────

    def _add_result(self, key: str, found: bool, details: str = "",
                     evidence: Optional[List[Dict]] = None) -> None:
        """Добавляет результат проверки критерия"""
        criterion = next((c for c in CHECKLIST_CRITERIA if c["key"] == key), None)
        if criterion is None:
            return

        score = criterion["max_score"] if found else 0
        status = "Да" if found else "Нет"
        if not details:
            details = criterion["description"][:100]

        self.results.append(KPIResult(
            key=key,
            name=criterion["name"],
            category=criterion["category"],
            score=score,
            max_score=criterion["max_score"],
            status=status,
            details=details,
            evidence=evidence if evidence else None,
        ))
        self.max_total_score += criterion["max_score"]
        self.total_score += score

    def _build_result(self) -> Dict:
        """Формирует итоговый результат анализа"""
        # Группируем по категориям
        categories = {}
        for r in self.results:
            if r.category not in categories:
                categories[r.category] = {
                    "results": [],
                    "total_score": 0,
                    "max_score": 0,
                }
            categories[r.category]["results"].append(r)
            categories[r.category]["total_score"] += r.score
            categories[r.category]["max_score"] += r.max_score

        # Процент
        pct = (self.total_score / self.max_total_score * 100) if self.max_total_score > 0 else 0

        # Светофор
        if pct >= 91:
            traffic_light = "green"
            overall_status = "Отлично"
        elif pct >= 76:
            traffic_light = "yellow"
            overall_status = "Хорошо"
        elif pct >= 51:
            traffic_light = "orange"
            overall_status = "Удовлетворительно"
        else:
            traffic_light = "red"
            overall_status = "Требует развития"

        return {
            "overall_score": self.total_score,
            "max_possible_score": self.max_total_score,
            "score_percentage": round(pct, 1),
            "overall_status": overall_status,
            "traffic_light": traffic_light,
            "categories": {
                cat_name: {
                    "total_score": cat_data["total_score"],
                    "max_score": cat_data["max_score"],
                    "percentage": round(
                        (cat_data["total_score"] / cat_data["max_score"] * 100)
                        if cat_data["max_score"] > 0 else 0, 1
                    ),
                    "items": [asdict(r) for r in cat_data["results"]],
                }
                for cat_name, cat_data in categories.items()
            },
            "all_items": [asdict(r) for r in self.results],
            "text_fields": self.text_fields,
            "checklist_name": "Пример для СДЭК",
        }


def analyze_call(transcript: str, segments: Optional[List[Dict]] = None) -> Dict:
    """
    Главная функция для анализа звонка по СДЭК чеклисту.

    Args:
        transcript: Текст транскрипции диалога
        segments: Список сегментов [{start, end, text}, ...] для evidence

    Returns:
        Словарь с результатами анализа (из 100 баллов)
    """
    analyzer = CallKPIAnalyzer()
    return analyzer.analyze(transcript, segments=segments)
