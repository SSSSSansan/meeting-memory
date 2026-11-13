"""РОЛЬ A. Метрики одного ответа без LLM-судьи. Чистые функции: сеть и модель не нужны.

Что считаем:
  correct_meetings — цитаты ссылаются только на встречи из source_meetings (и цитата есть);
  events_found     — сколько событий эталона (decision_ids) найдено цитатами. Правило как в
                     scripts/compare_gold.py: цитата той же встречи близка по словам к цитате эталона
                     ИЛИ её таймкод в пределах WINDOW_S от start_s эталона. Несколько цитат на одно
                     событие — одно совпадение;
  citation_valid   — цитата реально есть в тексте встречи (normalize_quote из scripts/build_gold.py);
  abstained        — система сказала «данных нет» (эвристика по фразам, см. ABSTAIN_RE; цитаты не важны).
Метрики v2 (тоже без модели, только для вопросов с ответом; для answerable=false — None):
  citation_recall    — |цитируемые встречи ∩ source_meetings| / |source_meetings|;
  citation_precision — |цитируемые ∩ source_meetings| / |цитируемые|; ответ без цитат = 0.0 (как correct_meetings:
                       нет цитаты — не засчитано), иначе система, которая ничего не цитирует, выпала бы из среднего;
  retrieval_recall   — |retrieved_meetings ∩ source_meetings| / |source_meetings|, где retrieved_meetings — встречи,
                       попавшие в контекст ДО генерации. api.ask() их пока не отдаёт: тогда None («не измерено»).
Точность содержания ответа здесь не считается: её считает судья (judge.py, judge_facts.py).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from scripts.build_gold import MeetingText, load_segments
from src.extract.compare import _match
from src.schemas import Citation
from src.utils.text import normalize

# То же окно, что WINDOW_S в scripts/compare_gold.py. Импортировать его оттуда нельзя:
# файл исполняет сверку на уровне модуля. Если B поменяет окно, поменять и здесь.
WINDOW_S = 20

ABSTAIN_RE = re.compile(
    r"в контексте нет|нет информации|нет данных|данных нет|информации нет|не найден[оа]?\b|не упоминается"
    r"|контексте жоқ|ақпарат жоқ|мәлімет жоқ|дерек жоқ|жоқ екен"
)


def ts_to_sec(ts: str) -> Optional[int]:
    try:
        h, m, s = (int(x) for x in ts.split(":"))
    except ValueError:
        return None
    return h * 3600 + m * 60 + s


def meeting_number(meeting_id: str) -> Optional[int]:
    m = re.fullmatch(r"m_(\d+)", meeting_id)
    return int(m.group(1)) if m else None


def load_gold_events(path: Path) -> dict[str, dict]:
    return {e["id"]: e for e in json.loads(Path(path).read_text(encoding="utf-8"))}


class MeetingTexts:
    """Тексты встреч из data/raw/*.transcript.json; читаются по требованию и кэшируются."""

    def __init__(self, raw_dir: Path) -> None:
        self.raw_dir = Path(raw_dir)
        self._cache: dict[str, Optional[MeetingText]] = {}

    def get(self, meeting_id: str) -> Optional[MeetingText]:
        if meeting_id not in self._cache:
            path = self.raw_dir / f"{meeting_id}.transcript.json"
            self._cache[meeting_id] = MeetingText(load_segments(path)) if path.exists() else None
        return self._cache[meeting_id]


def citation_is_valid(c: Citation, texts: MeetingTexts) -> bool:
    text = texts.get(c.meeting_id)
    return text is not None and text.find(c.quote) is not None


def events_found(citations: list[Citation], event_ids: list[str], gold: dict[str, dict]) -> int:
    """Сколько событий из event_ids найдено; событие считается один раз, сколько бы цитат на него ни пришло."""
    found = 0
    for ev_id in event_ids:
        ev = gold.get(ev_id)
        if ev is None:
            continue
        for c in citations:
            sec = ts_to_sec(c.ts)
            near = sec is not None and abs(sec - float(ev["start_s"])) <= WINDOW_S
            if c.meeting_id == ev["meeting"] and (_match(c.quote, ev["quote"]) or near):
                found += 1
                break
    return found


def is_abstention(answer: str) -> bool:
    """«Данных нет»: в ответе есть фраза отказа. Приложенные цитаты не мешают: система часто цитирует
    ближайшую реплику, хотя честно пишет «нет информации» (q_34)."""
    return ABSTAIN_RE.search(normalize(answer)) is not None


def citation_recall(cited: Sequence[str], source: Sequence[str]) -> Optional[float]:
    src = set(source)
    return len(src & set(cited)) / len(src) if src else None


def citation_precision(cited: Sequence[str], source: Sequence[str]) -> Optional[float]:
    if not source:
        return None
    return len(set(source) & set(cited)) / len(set(cited)) if cited else 0.0


def retrieval_recall(retrieved: Optional[Sequence[str]], source: Sequence[str]) -> Optional[float]:
    """None — «не измерено»: система не отдала список встреч, попавших в контекст (это не то же самое, что 0.0)."""
    if retrieved is None or not source:
        return None
    return len(set(source) & set(retrieved)) / len(set(source))


@dataclass
class AnswerMetrics:
    cited_meetings: list[str]
    correct_meetings: Optional[bool]    # None: вопрос без ответа
    meetings_hit: Optional[float]       # доля source_meetings, на которые сослались
    cited_after_as_of: int              # цитат на встречи позже as_of: утечка будущего
    events_found: Optional[int]
    events_total: Optional[int]
    citations_total: int
    citations_valid: int
    abstained: bool
    citation_recall: Optional[float] = None       # метрики v2, см. docstring модуля
    citation_precision: Optional[float] = None
    retrieval_recall: Optional[float] = None


def score_answer(*, answer: str, citations: list[Citation], answerable: bool, source_meetings: list[str],
                 decision_ids: list[str], as_of: str, gold: dict[str, dict],
                 texts: MeetingTexts, retrieved_meetings: Optional[Sequence[str]] = None) -> AnswerMetrics:
    cited = sorted({c.meeting_id for c in citations})
    limit = meeting_number(as_of)
    after = sum(1 for c in citations if limit is not None and (meeting_number(c.meeting_id) or 0) > limit)
    valid = sum(1 for c in citations if citation_is_valid(c, texts))
    base = dict(cited_meetings=cited, cited_after_as_of=after, citations_total=len(citations),
                citations_valid=valid, abstained=is_abstention(answer))
    if not answerable:
        return AnswerMetrics(correct_meetings=None, meetings_hit=None, events_found=None, events_total=None, **base)
    source = set(source_meetings)
    return AnswerMetrics(
        correct_meetings=bool(cited) and set(cited) <= source,
        meetings_hit=len(source & set(cited)) / len(source) if source else None,
        events_found=events_found(citations, decision_ids, gold),
        events_total=len(decision_ids),
        citation_recall=citation_recall(cited, source_meetings),
        citation_precision=citation_precision(cited, source_meetings),
        retrieval_recall=retrieval_recall(retrieved_meetings, source_meetings),
        **base,
    )
