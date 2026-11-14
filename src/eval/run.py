"""РОЛЬ A. Eval-harness (черновик): прогон вопросов через src.api.ask и простые метрики, без судьи.

    python -m src.eval.run --mode baseline|graph|both --split dev|test|all [--as-of m_008] [--repeats 1..5]
    python -m src.eval.run --mode both --split dev --dry-run       # только план и оценка стоимости
    python -m src.eval.run --mode both --split test --unlock-test --repeats 3 --judge --dry-run   # test: нужен флаг
    python -m src.eval.run --mode both --as-of m_008 --questions-file data/gold/questions_m008.json   # m_008

Вопросы — data/gold/questions.json; в систему уходят только текст вопроса и его as_of_date, эталон она не видит:
api.ask(question, mode, as_of=<as_of_date>) — встречи с датой не позже as_of_date включительно (src/eval/systems.py).
Режимы: baseline | graph | hybrid | both (baseline + graph) | all (+ hybrid). Сколько чанков в контексте — CONTEXT_K
из конфига C, своего k у harness нет. Индекс baseline (--index-dir, по умолчанию runs/index/baseline) только читается:
нет его, старый формат или не хватает встреч — «нужна переиндексация», baseline и hybrid — stub. hybrid с
графовой частью-заглушкой помечается «graph part: stub» и в итоговую таблицу не входит. В сводке — проверка симметрии
(max и среднее элементов контекста по режимам, предупреждение при > CONTEXT_K) и предупреждения по retrieved_meetings.
--as-of m_008 оставляет только вопросы с этим as_of (по умолчанию — все).
Контрольные режимы A5 — только явно: --mode bm25, --mode longctx или список через запятую (--mode baseline,graph,bm25);
all их не включает. bm25 (поиск по словам по тем же чанкам, тот же CONTEXT_K) входит в итоговую таблицу, longctx (все
встречи до as_of, CONTEXT_K не применяется) — нет; в симметрии longctx помечен «контроль, бюджет контекста
не ограничен».

Выход: results/eval_<дата>.csv (по вопросам) и results/summary_<дата>.md (по типам и по языку, режимы рядом,
dev и test отдельно; вопросы as_of m_008 — отдельной секцией, в основные таблицы не входят). Если задан --as-of или
нестандартный --questions-file, к имени файлов добавляется суффикс (eval_<дата>_m_008.csv), чтобы прогоны одного дня
не затирали друг друга. Правило из memory.md: чиним по dev, в отчёт идёт test.
Повторы (--repeats N, 1..5): ответы системы на каждом повторе идут мимо кэша llm_client (свой каталог кэша на повтор,
src/llm_client.py не меняется); в таблицах mean ± std по повторам (std выборочное, n−1; по вопросам std не считается).
Температура отвечающей системы 0.0 (зашита в src/answer/answerer.py): если ответы повторов совпали, std=0 и в сводке
предупреждение, разброс не имитируется. Судья оценивает каждый повтор отдельно, но одинаковый вход — один раз.
Задержка в таблицах — только по ответам без кэша (по строкам runs/calls.jsonl: from_cache); если определить нельзя,
колонка помечена «может включать кэш».
Для режима both и вопросов test (без --limit) дополнительно пишется results/final_table_<дата>.md (при stub — с
суффиксом _stub, чтобы не затереть платный результат): T1–T5 и none, baseline и graph рядом, mean ± std.
--judge (по умолчанию выключен): после каждого ответа судья v2 (src/eval/judge.py, с транскриптами) ставит
judge_score 0/1/2 и judge_fabrication 0/1; в сводке accuracy = среднее judge_score/2, fabrication_rate = доля
ответов с judge_fabrication=1. Судью зовут одинаково для всех режимов: ему уходят вопрос, эталон, ответ, answerable
и тексты встреч, но не режим, тип и split. Одинаковый вход судья не получает дважды (хэш входа; плюс кэш llm_client
на диске). Без --judge accuracy и fabrication_rate = n/a.

--judge-facts (по умолчанию выключен): судья по фактам (src/eval/judge_facts.py, judge-facts-v1, без транскриптов)
заполняет fact_recall, fact_precision, fact_precision_strict, fact_f1, forbidden_hits; --judge-repeats N (1..5) —
N вызовов судьи на ответ, метрики усредняются, печатается std (каждый повтор судьи идёт мимо кэша llm_client).
Без судьи по фактам эти метрики null. Метрики без модели (citation_recall, citation_precision, retrieval_recall) —
всегда. retrieval_recall = null («не измерено»), если api.ask() не отдал retrieved_meetings; сам список пишется и в csv
(через запятую, плюс retrieved_count), и в json.
Рядом с csv и md пишется results/eval_<run_id>.json (контракт — src/eval/results_v2.py; run_id по умолчанию
<YYYYMMDD_HHMM>, свой — --run-id). После каждого ответа строка дописывается в results/eval_<run_id>.partial.jsonl;
упавший прогон продолжается: --resume --run-id <тот же>. В сводке «Метрики v2» — бутстрап по вопросам
(src/eval/bootstrap.py): интервалы режимов и разности graph − baseline.

Защита test: если среди выбранных вопросов есть split=test (--split test, --split all или --questions-file с
test-вопросами), без --unlock-test harness завершается с ошибкой (код 4) до оценки стоимости и вызовов модели, в
том числе при --dry-run. С --unlock-test печатается предупреждение, а в results/test_runs.jsonl дописывается запись
(время, commit, режимы, число вопросов, dry_run). В сводке — «test запускался N раз» (без dry-run).
Правило из memory.md: test открываем один раз, в воскресенье.

Если LLM_BACKEND не stub, прогон платный: сначала печатается оценка (ответы + судья), запуск — только с --confirm-paid.
Лимит расходов держит scripts/cost_guard через src/llm_client (complete/embed).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, ContextManager, Iterator, Optional

from scripts import cost_guard
from scripts.check_questions import Question, load_json
from src import llm_client  # читает .env до того, как мы смотрим на LLM_BACKEND
from src.eval import estimate as est_mod
from src.eval import judge as judge_mod
from src.eval import judge_facts as facts_mod
from src.eval import results_v2, systems
from src.eval.metrics import MeetingTexts, load_gold_events, score_answer
from src.retrieval.baseline import CONTEXT_K, DEFAULT_INDEX_DIR, STEP, WINDOW
from src.schemas import Answer

ROOT = Path(__file__).resolve().parent.parent.parent
QUESTIONS = ROOT / "data" / "gold" / "questions.json"
GOLD = ROOT / "data" / "gold" / "decisions.json"
RAW_DIR = ROOT / "data" / "raw"
INDEX_ROOT = DEFAULT_INDEX_DIR    # индекс baseline только читается (systems.index_problem), не пересобирается
RESULTS = ROOT / "results"
TYPES = ["T1", "T2", "T3", "T4", "T5", "none"]
LANGS = ["ru", "mixed", "kk"]
MAX_REPEATS = 5
HISTORICAL_AS_OF = "m_008"      # вопросы на состояние m_008 (q_16, questions_m008.json) — отдельная секция
ANSWER_TEMPERATURE = 0.0        # src/answer/answerer.py::generate зовёт complete(..., temperature=0.0)
MODES = {"baseline": ["baseline"], "graph": ["graph"], "hybrid": ["hybrid"], "both": ["baseline", "graph"],
         "all": ["baseline", "graph", "hybrid"], "bm25": ["bm25"], "longctx": ["longctx"]}
FINAL_MODES = ("baseline", "graph")     # итоговая таблица: baseline и graph рядом
FINAL_EXTRA_MODES = ("bm25",)           # контроль A5: в итоговую таблицу, если запускался; longctx — нет
UNBOUNDED_NOTE = "контроль, бюджет контекста не ограничен"
TEST_RUNS_FILE = "test_runs.jsonl"      # журнал запусков на test, лежит в RESULTS
TEST_LOCKED_CODE = 4
TEST_UNLOCK_WARNING = "Запуск на test: после него код, промпты и настройки не меняются"


@dataclass
class Row:
    question_id: str
    type: str
    split: str
    lang: str
    as_of: str
    mode: str
    system: str                     # real | stub
    backend: str                    # LLM_BACKEND; при stub цифры качества ничего не значат
    status: str                     # ok | stub | no_answer | invalid_format | error
    error: str = ""
    answerable: bool = True
    answer: str = ""
    cited_meetings: str = ""
    source_meetings: str = ""
    correct_meetings: Optional[bool] = None
    meetings_hit: Optional[float] = None
    cited_after_as_of: Optional[int] = None
    events_found: Optional[int] = None
    events_total: Optional[int] = None
    citations_total: Optional[int] = None
    citations_valid: Optional[int] = None
    citation_valid: Optional[float] = None
    abstained: Optional[bool] = None
    judge_score: Optional[int] = None          # 0 | 1 | 2; None — судья не запускался или не вернул вердикт
    judge_fabrication: Optional[int] = None    # 0 | 1
    judge_reason: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    latency_ms: Optional[int] = None
    from_cache: Optional[bool] = None   # хотя бы один вызов модели за ответ взят из кэша llm_client; None — не известно
    repeat: int = 1
    # --- метрики v2 (только добавлены). None — не измерено / не посчитано, а не 0 ---
    citation_recall: Optional[float] = None     # цитируемые встречи ∩ source_meetings / source_meetings
    citation_precision: Optional[float] = None  # то же / цитируемые встречи
    retrieval_recall: Optional[float] = None    # retrieved_meetings ∩ source_meetings / source_meetings
    fact_recall: Optional[float] = None         # судья по фактам (--judge-facts)
    fact_precision: Optional[float] = None
    fact_precision_strict: Optional[float] = None
    fact_f1: Optional[float] = None
    forbidden_hits: Optional[float] = None      # int при одном повторе судьи, иначе среднее
    # токены контекста в промпте ответа: usage.context_tokens, иначе usage.prompt_tokens (None — нет ответа)
    context_tokens: Optional[int] = None
    # --- as_of и контекст (только добавлены) ---
    as_of_date: str = ""                        # ISO-дата из вопроса, ушла в api.ask(as_of=...)
    retrieved_count: Optional[int] = None       # len(retrieved_meetings); None — api.ask() список не отдал
    context_chunks: Optional[int] = None        # чанков транскрипта в контексте (baseline, hybrid)
    context_graph_items: Optional[int] = None   # пунктов графа в контексте (graph, hybrid)
    context_elements: Optional[int] = None      # сумма двух выше; сравнивается с CONTEXT_K (проверка симметрии)
    mode_flag: str = ""                         # пометка режима: у hybrid «graph part: stub»
    context_meetings: Optional[int] = None      # longctx: встреч в контексте (элементы не считаются, бюджета нет)
    # Только для json и --resume, в csv не попадают:
    context_tokens_source: str = ""             # какое поле usage использовано
    citations: list = field(default_factory=list)               # [{meeting_id, ts, quote}]
    retrieved_meetings: Optional[list] = None   # None: api.ask() список не отдал
    facts_detail: Optional[dict] = None         # разбор судьи по фактам (первый повтор)
    metrics_std: Optional[dict] = None          # std метрик по повторам судьи (если их больше одного)
    judge_usage: Optional[dict] = None          # токены и цена вызовов судьи по фактам для этой строки


NOT_IN_CSV = {"context_tokens_source", "citations", "facts_detail", "metrics_std", "judge_usage"}
CSV_FIELDS = [f.name for f in fields(Row) if f.name not in NOT_IN_CSV]

JudgeFn = Callable[..., judge_mod.JudgeResult]


@dataclass
class JudgeRunner:
    """Судья v2 в harness. Один вызов на уникальный вход; режим, тип и split судье не передаются."""
    raw_dir: Path
    fn: JudgeFn = judge_mod.judge
    memo: dict[str, Any] = field(default_factory=dict)  # хэш входа → JudgeResult | текст ошибки формата
    calls: int = 0      # разных входов с непустым ответом (пустой ответ судья оценивает без модели)
    reused: int = 0     # вердикт взят из этого же прогона
    failed: int = 0     # судья не вернул вердикт в нужном формате

    def apply(self, row: "Row", q: Question) -> None:
        if row.status != "ok":  # система не ответила: это неверный ответ, а не «пропуск»
            row.judge_score, row.judge_fabrication, row.judge_reason = 0, 0, f"нет ответа системы ({row.status})"
            return
        cited = [m for m in row.cited_meetings.split(",") if m]
        meetings = judge_mod.select_meetings(q.source_meetings, cited, q.as_of, self.raw_dir).docs
        key = judge_mod.input_hash(q.question, q.expected_answer, row.answer, q.answerable, meetings=meetings)
        if key in self.memo:
            self.reused += 1
        else:
            if row.answer.strip():
                self.calls += 1
            try:
                self.memo[key] = self.fn(q.question, q.expected_answer, row.answer, q.answerable,
                                         with_transcripts=True, meetings=meetings)
            except judge_mod.JudgeFormatError as e:
                self.memo[key] = str(e)
        verdict = self.memo[key]
        if isinstance(verdict, str):
            self.failed += 1
            row.judge_reason = f"ОШИБКА ФОРМАТА: {verdict}"
        else:
            row.judge_score, row.judge_fabrication, row.judge_reason = verdict


@dataclass
class JudgeInfo:
    version: str
    model: str
    calls: int
    reused: int
    failed: int


def coerce_answer(result: Any) -> tuple[Optional[Answer], str, str]:
    """Что вернул ask() → (Answer | None, статус, пояснение). Мусор не роняет прогон."""
    if result is None:
        return None, "no_answer", "ask() вернул None"
    if isinstance(result, Answer):
        return result, "ok", ""
    if isinstance(result, dict):
        try:
            return Answer.model_validate(result), "ok", ""
        except Exception as e:
            return None, "invalid_format", f"не формат 4.6: {str(e).splitlines()[0]}"
    return None, "invalid_format", f"ожидался Answer, пришёл {type(result).__name__}"


class CallLog:
    """Строки runs/calls.jsonl, которые llm_client дописал за время одного ответа: так видно, взят ли ответ из кэша.
    Читаем только хвост файла с запомненной позиции; любой сбой чтения = «не известно» (None), а не «без кэша»."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path

    def _file(self) -> Path:
        return self.path or (llm_client.RUNS / "calls.jsonl")

    def mark(self) -> Optional[int]:
        try:
            f = self._file()
            return f.stat().st_size if f.exists() else 0
        except OSError:
            return None

    def answer_calls_since(self, mark: Optional[int]) -> Optional[list[dict]]:
        """Вызовы чата отвечающей системы (purpose answer_*) после mark; судья и эмбеддинги не считаются."""
        if mark is None:
            return None
        try:
            f = self._file()
            size = f.stat().st_size if f.exists() else 0
            if size < mark:  # файл пересоздали
                return None
            if size == mark:
                return []
            with f.open("rb") as fh:
                fh.seek(mark)
                tail = fh.read().decode("utf-8")
            rows = [json.loads(line) for line in tail.splitlines() if line.strip()]
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        return [r for r in rows if r.get("kind", "chat") == "chat" and str(r.get("purpose", "")).startswith("answer")]


@contextmanager
def isolated_cache(path: Path) -> Iterator[None]:
    """На время вызова подменяет каталог кэша llm_client. complete() читает llm_client.CACHE в момент вызова,
    поэтому src/llm_client.py трогать не нужно; у каждого повтора свой каталог — ответ повтора не берётся из кэша."""
    saved = llm_client.CACHE
    llm_client.CACHE = Path(path)
    try:
        yield
    finally:
        llm_client.CACHE = saved


def run_one(q: Question, system: systems.System, *, gold: dict, texts: MeetingTexts, backend: str,
            repeat: int = 1, ask_ctx: Optional[Callable[[], ContextManager]] = None,
            call_log: Optional[CallLog] = None) -> Row:
    row = Row(question_id=q.id, type=q.type, split=q.split, lang=q.lang, as_of=q.as_of, mode=system.mode,
              system="stub" if system.is_stub else "real", backend=backend, status="ok",
              answerable=q.answerable, source_meetings=",".join(q.source_meetings), repeat=repeat,
              as_of_date=getattr(q, "as_of_date", None) or "", mode_flag=getattr(system, "flag", ""))
    if system.is_stub:
        row.status, row.error, row.answer = "stub", system.note, ""
        return row
    mark = call_log.mark() if call_log else None
    system.last_context = None
    t0 = time.perf_counter()
    try:
        with (ask_ctx() if ask_ctx else _no_ctx()):
            # только текст вопроса и его дата as_of_date: будущее система отрезает сама
            result = (system.ask(q.question, as_of=row.as_of_date or None) if system.uses_as_of
                      else system.ask(q.question))
    except cost_guard.BudgetExceeded:
        raise  # лимит расходов — стоп всего прогона, а не «ошибка вопроса»
    except Exception as e:
        result, row.status, row.error = None, "error", f"{type(e).__name__}: {e}"
    wall_ms = int((time.perf_counter() - t0) * 1000)
    calls = call_log.answer_calls_since(mark) if call_log else None
    row.from_cache = None if calls is None else any(bool(c.get("from_cache")) for c in calls)

    answer = None
    if row.status != "error":
        answer, row.status, row.error = coerce_answer(result)
    row.latency_ms = (answer.latency_ms if answer and answer.latency_ms else wall_ms)
    row.answer = answer.answer if answer else ""
    citations = answer.citations if answer else []
    ctx = system.last_context
    if ctx is not None:
        row.context_chunks, row.context_graph_items = ctx.get("chunks"), ctx.get("graph_items")
        row.context_elements = systems.context_elements(ctx)
        row.context_meetings = ctx.get("meetings")
    # retrieved_meetings: встречи, попавшие в контекст до генерации (Answer 4.6, аддитивно); None — «не измерено».
    retrieved = getattr(answer, "retrieved_meetings", None) if answer else None
    m = score_answer(answer=row.answer, citations=citations, answerable=q.answerable,
                     source_meetings=q.source_meetings, decision_ids=q.decision_ids, as_of=q.as_of,
                     gold=gold, texts=texts, retrieved_meetings=retrieved)
    row.citations = [c.model_dump() for c in citations]
    row.retrieved_meetings = None if retrieved is None else list(retrieved)
    row.retrieved_count = None if retrieved is None else len(retrieved)
    row.citation_recall, row.citation_precision = m.citation_recall, m.citation_precision
    row.retrieval_recall = m.retrieval_recall
    row.cited_meetings = ",".join(m.cited_meetings)
    row.correct_meetings, row.meetings_hit = m.correct_meetings, m.meetings_hit
    row.cited_after_as_of = m.cited_after_as_of
    row.events_found, row.events_total = m.events_found, m.events_total
    row.citations_total, row.citations_valid = m.citations_total, m.citations_valid
    row.citation_valid = round(m.citations_valid / m.citations_total, 4) if m.citations_total else None
    row.abstained = m.abstained
    usage = answer.usage if answer else None
    row.prompt_tokens = usage.prompt_tokens if usage else 0
    row.completion_tokens = usage.completion_tokens if usage else 0
    row.cost_usd = round(usage.cost_usd, 6) if usage else 0.0
    if usage is not None:   # без ответа (ошибка, нет ответа) контекста не было: None, а не 0
        own = getattr(usage, "context_tokens", None)    # в Usage (4.6) пока нет: тогда prompt_tokens
        row.context_tokens = own if own is not None else usage.prompt_tokens
        row.context_tokens_source = results_v2.CONTEXT_FIELD if own is not None else results_v2.CONTEXT_FALLBACK
    return row


@contextmanager
def _no_ctx() -> Iterator[None]:
    yield


RowKey = tuple[str, str, str, int]      # (question_id, mode, as_of, repeat)


def row_key(row: Row) -> RowKey:
    return (row.question_id, row.mode, row.as_of, row.repeat)


def apply_facts(runner: facts_mod.JudgeFactsRunner, row: Row, q: Question) -> None:
    """Судья по фактам: ему уходят вопрос, ответ и списки фактов — ни режима, ни типа, ни эталонного ответа."""
    if not q.required_facts and not q.forbidden_facts:
        row.facts_detail = {"skipped": "у вопроса нет required_facts и forbidden_facts"}
        return
    out = runner.evaluate(question=q.question, answer=row.answer, required_facts=q.required_facts,
                          forbidden_facts=q.forbidden_facts, answerable=q.answerable, answered=row.status == "ok")
    row.judge_usage = {"prompt_tokens": out.prompt_tokens, "completion_tokens": out.completion_tokens,
                       "cost_usd": round(out.cost_usd, 6)}
    if out.metrics is None:     # ни один повтор не вернул вердикт: строка без оценки, в средние не входит
        row.facts_detail = {"error": out.errors}
        return
    m = out.metrics
    row.fact_recall, row.fact_precision = m["fact_recall"], m["fact_precision"]
    row.fact_precision_strict, row.fact_f1 = m["fact_precision_strict"], m["fact_f1"]
    hits = m["forbidden_hits"]
    row.forbidden_hits = int(hits) if hits is not None and float(hits).is_integer() else hits
    row.facts_detail, row.metrics_std = out.detail, out.std
    if out.repeats_failed:
        row.facts_detail = {**(out.detail or {}), "repeats_failed": out.repeats_failed, "errors": out.errors}


def run_eval(questions: list[Question], modes: list[str], build: Callable[[str, str], systems.System], *,
             gold: dict, texts: MeetingTexts, backend: str, judge: Optional[JudgeRunner] = None,
             repeats: int = 1, repeat_cache: Optional[Callable[[int], ContextManager]] = None,
             call_log: Optional[CallLog] = None, facts: Optional[facts_mod.JudgeFactsRunner] = None,
             on_row: Optional[Callable[[Row], None]] = None,
             done: Optional[dict[RowKey, Row]] = None) -> list[Row]:
    """build(mode, as_of) → System. Каждая пара (as_of, режим) строится один раз и отвечает на все свои вопросы,
    repeats раз. repeat_cache(n) — менеджер контекста, в котором идёт ответ n-го повтора (изоляция кэша).
    Заглушка (stub) отвечает один раз: повторять нечего.
    on_row(row) вызывается после каждой готовой строки (ответ и все оценки): харнес дописывает её в промежуточный
    файл. done — строки уже сделанной части прогона (--resume): берутся как есть, вопрос не задаётся заново, а
    группа, где всё готово, систему не строит (индекс baseline стоит эмбеддингов)."""
    done = done or {}
    by_asof: dict[str, list[Question]] = defaultdict(list)
    for q in questions:
        by_asof[q.as_of].append(q)
    rows: list[Row] = []

    def finish(row: Row, q: Question) -> None:
        if judge and row.status != "stub":
            judge.apply(row, q)
        if facts and row.status != "stub":
            apply_facts(facts, row, q)
        rows.append(row)
        if on_row:
            on_row(row)

    for as_of, qs in sorted(by_asof.items()):
        for mode in modes:
            all_done = all((q.id, mode, as_of, rep) in done for q in qs for rep in range(1, repeats + 1))
            stub_done = all(((q.id, mode, as_of, 1) in done and done[(q.id, mode, as_of, 1)].status == "stub")
                            for q in qs)
            if all_done or stub_done:
                rows += [done[(q.id, mode, as_of, rep)] for rep in range(1, 2 if stub_done else repeats + 1)
                         for q in qs]
                continue
            try:
                system = build(mode, as_of)
            except cost_guard.BudgetExceeded:
                raise
            except Exception as e:
                for rep in range(1, repeats + 1):
                    for q in qs:
                        if (q.id, mode, as_of, rep) in done:
                            rows.append(done[(q.id, mode, as_of, rep)])
                            continue
                        row = Row(question_id=q.id, type=q.type, split=q.split, lang=q.lang, as_of=as_of,
                                  mode=mode, system="real", backend=backend, status="error",
                                  error=f"{type(e).__name__}: {e}", answerable=q.answerable,
                                  source_meetings=",".join(q.source_meetings), repeat=rep)
                        finish(row, q)
                continue
            for rep in range(1, 2 if system.is_stub else repeats + 1):
                ask_ctx = (lambda rep=rep: repeat_cache(rep)) if repeat_cache else None
                for q in qs:
                    if (q.id, mode, as_of, rep) in done:
                        rows.append(done[(q.id, mode, as_of, rep)])
                        continue
                    row = run_one(q, system, gold=gold, texts=texts, backend=backend, repeat=rep,
                                  ask_ctx=ask_ctx, call_log=call_log)
                    finish(row, q)
    return rows


# ---------- сводка ----------

KEYS = ("correct", "events", "cite", "abst", "acc", "fab", "pt", "ct", "cost", "lat")


@dataclass
class Stat:
    """Метрика группы по повторам: mean и std (выборочное, n−1). text — «stub» / «n/a» вместо числа."""
    mean: Optional[float] = None
    std: Optional[float] = None
    n: int = 0
    text: Optional[str] = None


def _mean(xs: list[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _std(xs: list[float]) -> Optional[float]:
    if len(xs) < 2:
        return None
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def fmt_stat(st: Stat, spec: str = ".2f") -> str:
    if st.text is not None:
        return st.text
    if st.mean is None:
        return "—"
    if st.n < 2 or st.std is None:
        return format(st.mean, spec)
    return f"{format(st.mean, spec)} ± {format(st.std, spec)}"


def _agg_one(rows: list[Row], qtype: str, judge_on: bool) -> dict[str, Any]:
    """Метрики группы строк одного режима и ОДНОГО повтора: число, None («—») или «n/a»."""
    judged = [r for r in rows if r.judge_score is not None]
    ans = [r for r in rows if r.answerable]
    ev_tot = sum(r.events_total or 0 for r in ans)
    cit_tot = sum(r.citations_total or 0 for r in rows)
    # abstained: для none — доля отказов (хорошо, когда высокая); для T1–T5 — ложные отказы (хорошо, когда низкая)
    abst_rows = rows if qtype == "none" else ans if qtype != "all" else []
    cold = [r for r in rows if r.from_cache is False and r.latency_ms is not None]  # задержка — только без кэша
    return {
        "correct": _mean([float(bool(r.correct_meetings)) for r in ans]),
        "events": sum(r.events_found or 0 for r in ans) / ev_tot if ev_tot else None,
        "cite": sum(r.citations_valid or 0 for r in rows) / cit_tot if cit_tot else None,
        "abst": _mean([float(bool(r.abstained)) for r in abst_rows]),
        # без судьи — n/a; с судьёй, но без вердиктов в группе — «—»
        "acc": _mean([r.judge_score / 2 for r in judged]) if judge_on else "n/a",
        "fab": _mean([float(r.judge_fabrication == 1) for r in judged]) if judge_on else "n/a",
        "pt": _mean([float(r.prompt_tokens or 0) for r in rows]),
        "ct": _mean([float(r.completion_tokens or 0) for r in rows]),
        "cost": _mean([float(r.cost_usd or 0) for r in rows]),
        "lat": _mean([float(r.latency_ms) for r in cold]),
    }


def agg_repeats(rows: list[Row], qtype: str, judge_on: bool = False) -> dict[str, Stat]:
    """Метрики группы строк одного режима: сначала по каждому повтору, затем mean ± std по повторам.
    Если все строки — заглушка, вместо цифр «stub»; смешанные группы считаются без заглушек."""
    if rows and all(r.status == "stub" for r in rows):
        return {k: Stat(text="stub") for k in KEYS}
    by_rep: dict[int, list[Row]] = defaultdict(list)
    for r in rows:
        if r.status != "stub":
            by_rep[r.repeat].append(r)
    per = [_agg_one(rs, qtype, judge_on) for _, rs in sorted(by_rep.items())]
    out: dict[str, Stat] = {}
    for k in KEYS:
        vals = [p[k] for p in per]
        strs = [v for v in vals if isinstance(v, str)]
        nums = [v for v in vals if isinstance(v, float)]
        out[k] = Stat(text=strs[0]) if strs else Stat(_mean(nums), _std(nums), len(nums)) if nums else Stat()
    return out


COLUMNS = [("correct", "correct_meetings"), ("events", "events_found"), ("cite", "citation_valid"),
           ("abst", "abstained*"), ("acc", "accuracy"), ("fab", "fabrication_rate"), ("pt", "prompt_tok"),
           ("ct", "compl_tok"), ("cost", "cost_usd/вопрос")]
LAT_NO_CACHE = "latency_ms (без кэша)"
LAT_MAY_CACHE = "latency_ms (может включать кэш)"
SPECS = {"pt": ".0f", "ct": ".0f", "cost": ".4f", "lat": ".0f"}


def latency_title(rows: list[Row]) -> str:
    """Колонка задержки честная, только если про каждую настоящую строку известно, брала ли она ответ из кэша."""
    real = [r for r in rows if r.status != "stub"]
    return LAT_MAY_CACHE if any(r.from_cache is None for r in real) else LAT_NO_CACHE


def slice_table(rows: list[Row], modes: list[str], judge_on: bool = False, *, by: str = "type") -> list[str]:
    """Таблица по типу вопроса (by="type") или по языку (by="lang"); режимы в ячейках через « / »."""
    head = ["Тип" if by == "type" else "Язык", "n"] + [title for _, title in COLUMNS] + [latency_title(rows)]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    keys = [c[0] for c in COLUMNS] + ["lat"]
    for value in (TYPES if by == "type" else LANGS) + ["all"]:
        group = [r for r in rows if value == "all" or getattr(r, by) == value]
        n_q = len({r.question_id for r in group})
        if not n_q:
            continue
        # в срезе по языку смешаны ответимые и нет: abstained* не определён
        per_mode = [agg_repeats([r for r in group if r.mode == m], value if by == "type" else "all", judge_on)
                    for m in modes]
        cells = [" / ".join(fmt_stat(a[k], SPECS.get(k, ".2f")) for a in per_mode) for k in keys]
        out.append("| " + " | ".join([value, str(n_q)] + cells) + " |")
    return out


def split_table(rows: list[Row], modes: list[str], judge_on: bool = False) -> list[str]:
    return slice_table(rows, modes, judge_on, by="type")


def lang_notes(rows: list[Row]) -> list[str]:
    n_by_lang = {lang: len({r.question_id for r in rows if r.lang == lang}) for lang in sorted({r.lang for r in rows})}
    out = ["_Языки в вопросах этого раздела: " + ", ".join(f"{k} — {v}" for k, v in n_by_lang.items()) + "._"]
    if "mixed" not in n_by_lang:
        out.append("_Значения mixed в вопросах нет: срез ru / mixed / kk из TESTING.md здесь только ru и kk "
                   "(смешанная речь есть во встречах, но не в языке вопроса)._")
    small = [k for k, v in n_by_lang.items() if v < 3]
    if small:
        out.append("_В срезе " + ", ".join(small) + " меньше 3 вопросов: это иллюстрация, не статистика._")
    return out


def repeat_spread(rows: list[Row]) -> tuple[int, int]:
    """(сколько пар «режим × as_of × вопрос» имеют ≥2 повторов, у скольких из них ответы повторов различаются)."""
    groups: dict[tuple, dict[int, str]] = defaultdict(dict)
    for r in rows:
        if r.status != "stub":
            groups[(r.mode, r.as_of, r.question_id)][r.repeat] = " ".join(r.answer.split())
    multi = [g for g in groups.values() if len(g) > 1]
    return len(multi), sum(1 for g in multi if len(set(g.values())) > 1)


def mode_flags(rows: list[Row]) -> dict[str, str]:
    """Пометки режимов из строк: {"hybrid": "graph part: stub"} — режим настоящий, но не тот, что задуман."""
    return {r.mode: r.mode_flag for r in rows if r.mode_flag}


def mode_title(mode: str, flags: dict[str, str]) -> str:
    return f"{mode} [{flags[mode]}]" if mode in flags else mode


def symmetry_stats(rows: list[Row], modes: list[str], context_k: int = CONTEXT_K) -> dict[str, dict]:
    """По режиму: максимум и среднее число элементов контекста (чанки + пункты графа) по ответам, где оно измерено."""
    out = {}
    for m in modes:
        xs = [r.context_elements for r in rows if r.mode == m and r.status != "stub" and r.context_elements is not None]
        out[m] = {"max": max(xs) if xs else None, "mean": round(sum(xs) / len(xs), 2) if xs else None, "n": len(xs),
                  "over_k": sum(1 for x in xs if x > context_k), "context_k": context_k}
    return out


def unbounded_stats(rows: list[Row], mode: str) -> dict:
    """longctx: сколько встреч и токенов было в контексте (элементы не считаются — бюджета нет)."""
    real = [r for r in rows if r.mode == mode and r.status != "stub"]
    ms = [r.context_meetings for r in real if r.context_meetings is not None]
    ts = [r.context_tokens for r in real if r.context_tokens is not None]
    return {"max": max(ms) if ms else None, "tokens": round(sum(ts) / len(ts)) if ts else None, "n": len(real)}


def symmetry_lines(rows: list[Row], modes: list[str], context_k: int = CONTEXT_K) -> list[str]:
    """Проверка симметрии: сколько контекста получает каждый режим. Больше CONTEXT_K — предупреждение."""
    stats = symmetry_stats(rows, modes, context_k)
    flags = mode_flags(rows)
    lines = ["## Симметрия контекста", "",
             f"Элементов контекста на ответ (чанки транскрипта + пункты графа), CONTEXT_K = {context_k} "
             "(конфиг C, src/retrieval/baseline.py; harness свой k не задаёт).", "",
             "| Режим | max | среднее | ответов | больше CONTEXT_K |", "|---|---|---|---|---|"]
    for m in modes:
        st = stats[m]
        if m in systems.UNBOUNDED_MODES:
            ub = unbounded_stats(rows, m)
            lines.append(f"| {m} ({UNBOUNDED_NOTE}) | встреч max {'—' if ub['max'] is None else ub['max']} | "
                         f"токенов ≈{'—' if ub['tokens'] is None else ub['tokens']} | {ub['n']} | — |")
            continue
        lines.append(f"| {mode_title(m, flags)} | {'—' if st['max'] is None else st['max']} | "
                     f"{'—' if st['mean'] is None else format(st['mean'], '.1f')} | {st['n']} | {st['over_k']} |")
    over = [m for m in modes if m not in systems.UNBOUNDED_MODES
            and stats[m]["max"] is not None and stats[m]["max"] > context_k]
    if over:
        lines += ["", "> **Предупреждение: контекст больше CONTEXT_K у режимов " + ", ".join(
            f"{m} (max {stats[m]['max']}, среднее {stats[m]['mean']:.1f})" for m in over)
            + ". Режимы получают разный объём контекста — разница в качестве может объясняться им, а не "
              "архитектурой.**"]
    if any(m in systems.UNBOUNDED_MODES for m in modes):
        lines += ["", f"_longctx — {UNBOUNDED_NOTE}: в контексте все встречи до as_of, CONTEXT_K не применяется; "
                      "вместо элементов — встречи и средние токены промпта ответа (usage.prompt_tokens)._"]
    if any(stats[m]["n"] == 0 for m in modes if m not in systems.UNBOUNDED_MODES):
        lines += ["", "_«—» — размер контекста не измерен (заглушка или ответов нет)._"]
    return lines


def retrieved_lines(rows: list[Row]) -> list[str]:
    """Предупреждения по retrieved_meetings: пустой список при непустом ответе; список не отдан вовсе."""
    real = [r for r in rows if r.status == "ok"]
    empty = [r for r in real if r.retrieved_meetings == [] and r.answer.strip()]
    missing = [r for r in real if r.retrieved_meetings is None]
    out = []
    if empty:
        refusals = sum(1 for r in empty if r.abstained)
        by_mode = Counter(r.mode for r in empty)
        out.append(f"> **Предупреждение: retrieved_meetings пуст при непустом ответе — {len(empty)} из {len(real)} "
                   "ответов (" + ", ".join(f"{m}: {c}" for m, c in sorted(by_mode.items()))
                   + f"); из них отказов «нет в контексте»: {refusals}.** Ответ без найденных встреч: "
                   "проверить вручную.")
    if missing:
        out.append(f"> api.ask() не отдал retrieved_meetings в {len(missing)} ответах: retrieval_recall для них "
                   "не измерен.")
    return out


def _git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                           timeout=10)
        return r.stdout.strip() or "n/a"
    except Exception:
        return "n/a"


def _meta_lines(rows: list[Row], modes: list[str], *, k: int, depth: int, judge: Optional[JudgeInfo]) -> list[str]:
    """Заголовок результатов (TESTING.md, правило 5) и всё, что нужно знать про повторы и судью."""
    backend = os.environ.get("LLM_BACKEND", "stub")
    repeats = max((r.repeat for r in rows), default=1)
    lines = [f"- commit: {_git_commit()}, backend: {backend}, model: {os.environ.get('LLM_MODEL', 'stub-model')}, "
             f"embed: {os.environ.get('EMBED_MODEL') or 'n/a'}",
             f"- PROMPT_VERSION: в коде не задан; k={k} (CONTEXT_K из конфига C), глубина обхода графа={depth}, "
             f"чанк={WINDOW} реплик со сдвигом {STEP}; температура отвечающей системы {ANSWER_TEMPERATURE} "
             "(src/answer/answerer.py)",
             f"- as_of в вопросах: {', '.join(sorted({r.as_of for r in rows})) or '—'}; as_of_date (уходит в "
             "api.ask, встречи не позже неё включительно): "
             f"{', '.join(sorted({r.as_of_date for r in rows if r.as_of_date})) or '—'}; повторов: {repeats}",
             "- режимы (в ячейках через « / » в этом порядке): "
             f"{' / '.join(mode_title(m, mode_flags(rows)) for m in modes)}"]
    if repeats > 1:
        real = [r for r in rows if r.status != "stub"]
        cached = sum(1 for r in real if r.from_cache is True)
        unknown = sum(1 for r in real if r.from_cache is None)
        lines.append("- повторы: ответы системы идут мимо кэша llm_client (свой каталог кэша на повтор); "
                     "mean ± std считается по повторам, не по вопросам, std выборочное (n−1). "
                     f"Ответов из кэша: {cached} из {len(real)}" + (f", неизвестно: {unknown}" if unknown else ""))
        n_cmp, n_diff = repeat_spread(rows)
        if n_cmp and n_diff == 0:
            lines.append(f"> **Предупреждение: ответы всех повторов совпали дословно ({n_cmp} пар режим × вопрос, "
                         f"температура {ANSWER_TEMPERATURE}). Разброса нет, std = 0 — это не мера устойчивости "
                         "системы, а следствие детерминированной генерации; разброс не имитируется.**")
        elif n_cmp:
            lines.append(f"- ответы повторов различаются в {n_diff} из {n_cmp} пар режим × вопрос")
        if cached:
            lines.append("> **Предупреждение: часть ответов повторов взята из кэша — это не независимые повторы.**")
    if judge is None:
        lines.append("- судья: выключен (--judge не задан), accuracy и fabrication_rate = n/a")
    else:
        same = (" — та же модель, что у системы, возможное смещение в пользу своих ответов"
                if judge.model == os.environ.get("LLM_MODEL", "stub-model") else "")
        lines.append(f"- судья: {judge.version} (с транскриптами встреч), модель {judge.model}{same}; "
                     f"вызовов {judge.calls}, повторных входов пропущено {judge.reused}, "
                     f"без вердикта (ошибка формата) {judge.failed}"
                     + ("; каждый повтор оценивается отдельно, одинаковый вход — один вызов (кэш по хэшу)"
                        if repeats > 1 else ""))
    return lines


def render_summary(rows: list[Row], modes: list[str], *, date: str, k: int, depth: int, notes: dict[str, str],
                   splits: list[str], judge: Optional[JudgeInfo] = None, v2: Optional[list[str]] = None,
                   test_runs: Optional[int] = None) -> str:
    backend = os.environ.get("LLM_BACKEND", "stub")
    lines = [f"# Eval summary {date}", ""] + _meta_lines(rows, modes, k=k, depth=depth, judge=judge)
    if test_runs is not None:
        lines.append(f"- test запускался {test_runs} раз (по {TEST_RUNS_FILE}, без dry-run)")
    if backend == "stub":
        lines += ["", "> **LLM_BACKEND=stub**: модель и эмбеддинги — заглушки. Таблица проверяет только "
                      "работу harness, цифры качества ничего не значат."]
    for mode in modes:
        if mode in notes:
            lines += ["", f"> **{mode}: stub** — {notes[mode]}. В таблице вместо цифр «stub»."]
    for mode, flag in mode_flags(rows).items():
        lines += ["", f"> **{mode}: {flag}** — графовая часть hybrid — временная реализация C (контекст режима graph "
                      "через retrieve_subgraph), обход роли B (B6) не подставлен. Ответы настоящие, но это не "
                      "задуманный hybrid: в итоговую таблицу не входит."]
    if any(r.status != "stub" for r in rows):
        lines += [""] + symmetry_lines(rows, modes)
    warn = retrieved_lines(rows)
    if warn:
        lines += [""] + warn
    errs = Counter((r.mode, r.status) for r in rows if r.status not in ("ok", "stub"))
    if errs:
        lines += ["", "> Строки без нормального ответа (считаются в n, метрики по ним нулевые): "
                      + ", ".join(f"{m}/{s}: {c}" for (m, s), c in sorted(errs.items()))]
    real = [r for r in rows if r.status != "stub"]
    n_cached, n_unknown = sum(1 for r in real if r.from_cache is True), sum(1 for r in real if r.from_cache is None)
    notes_md = [
        "correct_meetings — доля ответимых вопросов, где цитаты ссылаются только на встречи из source_meetings "
        "(цитата есть); events_found — найденные события эталона / всего (несколько цитат на событие = 1); "
        "citation_valid — цитаты, найденные в тексте встречи, / все цитаты.",
        "abstained*: для none — доля отказов «данных нет» (выше лучше); для T1–T5 — ложные отказы (ниже лучше). "
        "Определяется по фразам, см. ABSTAIN_RE; в срезе по языку — «—».",
        "accuracy — среднее judge_score / 2 (2 верно или честный отказ, 1 частично, 0 неверно или не ответили); "
        "fabrication_rate — доля ответов, где судья нашёл выдумку (judge_fabrication = 1); без --judge обе n/a. "
        "Токены, стоимость и задержка — среднее на вопрос (стоимость судьи не входит: она в runs/calls.jsonl, "
        "purpose=judge).",
        "Задержка — среднее только по ответам, где ни один вызов модели не взят из кэша llm_client "
        f"(определяется по runs/calls.jsonl); исключено из-за кэша: {n_cached} из {len(real)} строк"
        + (f"; про {n_unknown} строк известно не было — колонка помечена «может включать кэш»" if n_unknown else "")
        + ".",
        "Вопросы as_of m_008 (q_16, questions_m008.json) считаются отдельной секцией: у них другое состояние графа "
        "и они не смешиваются с основной таблицей."]
    main = [r for r in rows if r.as_of != HISTORICAL_AS_OF]
    hist = [r for r in rows if r.as_of == HISTORICAL_AS_OF]
    for split, title in (("test", "test — цифры для отчёта"), ("dev", "dev — для отладки, в отчёт не идут")):
        if split not in splits:
            continue
        part = [r for r in main if r.split == split]
        lines += ["", f"## {title}", ""]
        if not part:
            lines.append("_нет вопросов_")
            continue
        if split == "test":
            lines += ["### По типу вопроса", ""]
        lines += split_table(part, modes, judge is not None)
        if split == "test":
            lines += ["", "_На каждый тип приходится 1–2 тестовых вопроса: это иллюстрация, не статистика._",
                      "", "### По языку вопроса", ""]
            lines += slice_table(part, modes, judge is not None, by="lang") + [""] + lang_notes(part)
    if hist:
        lines += ["", f"## as_of {HISTORICAL_AS_OF} — отдельно, в основные таблицы не входит", ""]
        for split in ("test", "dev"):
            part = [r for r in hist if r.split == split]
            if part and split in splits:
                lines += [f"Раздел {HISTORICAL_AS_OF}, split={split}:", ""]
                lines += split_table(part, modes, judge is not None) + [""]
    if v2:
        lines += [""] + v2
    lines += [""] + [f"{i}. {t}" for i, t in enumerate(notes_md, 1)]
    return "\n".join(lines) + "\n"


FINAL_METRICS = [("acc", "accuracy"), ("fab", "fabrication_rate"), ("correct", "correct_meetings"),
                 ("events", "events_found"), ("cost", "cost_usd/вопрос")]


def final_rows(rows: list[Row]) -> list[Row]:
    """Строки итоговой таблицы: только split=test и только основной as_of (m_008 отдельно)."""
    return [r for r in rows if r.split == "test" and r.as_of != HISTORICAL_AS_OF]


def render_final_table(rows: list[Row], modes: list[str], *, date: str, k: int, depth: int, notes: dict[str, str],
                       judge: Optional[JudgeInfo] = None) -> str:
    """Итоговая таблица для отчёта: split=test, T1–T5 и none, baseline и graph рядом, mean ± std по повторам.
    Режим-заглушка — «stub» (цифр не выдумываем), режим, который не запускали, — «не запускался»."""
    part = final_rows(rows)
    judge_on = judge is not None
    lat = latency_title(part)
    columns = FINAL_METRICS + [("lat", lat)]
    cols = table_modes(modes)
    head = ["Тип", "n"]
    for _, title in columns:
        head += [f"{title}: {m}" for m in cols]
    lines = [f"# Итоговая таблица {date} (split=test)", ""] + _meta_lines(part, modes, k=k, depth=depth, judge=judge)
    if os.environ.get("LLM_BACKEND", "stub") == "stub":
        lines += ["", "> **LLM_BACKEND=stub**: цифры ничего не значат, таблица проверяет только работу harness."]
    for mode in FINAL_MODES:
        if mode in notes:
            lines += ["", f"> **{mode}: stub** — {notes[mode]}. В таблице «stub», цифры не приводятся."]
        elif mode not in modes:
            lines += ["", f"> **{mode}**: в этом прогоне не запускался."]
    flags = mode_flags(rows)
    for mode in (m for m in modes if m in FINAL_EXTRA_MODES):
        lines += ["", f"> **{mode}**: контрольный режим A5 (поиск по словам по тем же чанкам, CONTEXT_K)."]
    for mode in (m for m in modes if m not in cols):
        if mode in flags:
            why = f" ({flags[mode]}, не настоящий режим)."
        elif mode in systems.UNBOUNDED_MODES:
            why = f" ({UNBOUNDED_NOTE})."
        else:
            why = " (таблица — baseline против graph)."
        lines += ["", f"> **{mode}**: в таблицу не входит" + why]
    lines += ["", "| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for qtype in TYPES:
        group = [r for r in part if r.type == qtype]
        n_q = len({r.question_id for r in group})
        if not n_q:
            continue
        per_mode = {m: (agg_repeats([r for r in group if r.mode == m], qtype, judge_on) if m in modes else None)
                    for m in cols}
        cells = ["не запускался" if per_mode[m] is None else fmt_stat(per_mode[m][key], SPECS.get(key, ".2f"))
                 for key, _ in columns for m in cols]
        lines.append("| " + " | ".join([qtype, str(n_q)] + cells) + " |")
    real = [r for r in part if r.status != "stub"]
    n_cached = sum(1 for r in real if r.from_cache is True)
    lines += ["", "- mean ± std — по повторам (при одном повторе только среднее); на каждый тип приходится 1–2 "
                  "вопроса, это иллюстрация, не статистика.",
              "- cost_usd/вопрос — стоимость ответов системы; судья отдельно (runs/calls.jsonl, purpose=judge).",
              "- latency — среднее только по ответам без кэша llm_client; исключено из-за кэша: "
              f"{n_cached} из {len(real)} строк.",
              "- Вопросы as_of m_008 (q_16, questions_m008.json) в таблицу не входят: отдельная секция сводки.",
              "- accuracy/fabrication_rate — n/a, если прогон был без --judge."]
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:  # BOM: Excel открывает кириллицу без плясок
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")  # json-only поля Row в csv не пишем
        w.writeheader()
        w.writerows(csv_record(r) for r in rows)


def csv_record(row: Row) -> dict:
    d = asdict(row)
    # список встреч в csv — через запятую; пустой список от None отличает retrieved_count (0 против пусто)
    d["retrieved_meetings"] = "" if row.retrieved_meetings is None else ",".join(row.retrieved_meetings)
    return d


# ---------- CLI ----------

def select_questions(path: Path, split: str, as_of: Optional[str], limit: Optional[int]) -> list[Question]:
    raw = load_json(path)
    questions = [Question.model_validate(x) for x in raw]
    questions = [q for q in questions if split == "all" or q.split == split]
    if as_of:
        questions = [q for q in questions if q.as_of == as_of]
    return questions[:limit] if limit else questions


def journal_path() -> Path:
    # RESULTS читается в момент вызова: тесты подменяют его на tmp
    return RESULTS / TEST_RUNS_FILE


def has_test_questions(questions: list[Question]) -> bool:
    return any(q.split == "test" for q in questions)


def record_test_run(*, modes: list[str], split: str, questions: list[Question], questions_file: Path,
                    dry_run: bool, path: Optional[Path] = None) -> dict:
    """Дописывает запись о запуске на test в журнал; журнал коммитится (git add -f), чтобы запуски было видно всем."""
    rec = {"time": dt.datetime.now().astimezone().isoformat(timespec="seconds"), "commit": _git_commit(),
           "modes": modes, "split": split, "questions_file": Path(questions_file).name,
           "n_questions": len(questions), "n_test": sum(1 for q in questions if q.split == "test"),
           "dry_run": dry_run}
    path = path or journal_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return rec


def count_test_runs(path: Optional[Path] = None) -> int:
    """Сколько раз test запускался по-настоящему (dry-run не считается); битые строки пропускаются."""
    path = path or journal_path()
    if not path.exists():
        return 0
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and not rec.get("dry_run"):
            n += 1
    return n


def print_plan(est: est_mod.Estimate, jest: Optional[est_mod.JudgeEstimate], backend: str,
               dry_run: bool = False, fest: Optional[est_mod.FactsJudgeEstimate] = None) -> None:
    """Оценка прогона: ответы, судья (v2 и/или по фактам) отдельно, итог и сравнение с остатком бюджета (в --dry-run —
    при любом backend: план нужен, чтобы решить, запускать ли платный прогон)."""
    print("Оценка ответов системы:\n  " + "\n  ".join(est.lines()))
    if jest is None and fest is None:
        return
    if jest is not None:
        print("Оценка судьи (judge-v2, модель " + judge_mod.judge_model() + "):\n  " + "\n  ".join(jest.lines()))
    if fest is not None:
        print(f"Оценка судьи по фактам ({facts_mod.JUDGE_FACTS_PROMPT_VERSION}, модель " + judge_mod.judge_model()
              + "):\n  " + "\n  ".join(fest.lines()))
    total = est_mod.total_cost(est, jest, fest)
    judge_calls = (jest.calls if jest else 0) + (fest.calls if fest else 0)
    calls = f"{est.chat_calls_min}–{est.chat_calls_max} вызовов ответов + до {judge_calls} вызовов судьи"
    print(f"Итого (ответы + судья): {calls}; " + (
        f"≈${total[0]:.4f} без ретраев, до ≈${total[1]:.4f} с ретраями" if total else "стоимость не оценить (нет цен)"))
    if total and (backend != "stub" or dry_run):
        left = est_mod.remaining_budget()
        if total[0] > left:
            print(f"ВНИМАНИЕ: оценка ${total[0]:.4f} больше остатка бюджета ${left:.4f}: даже без ретраев не хватит, "
                  "cost_guard остановит прогон на лимите, и результаты не запишутся.")
        elif total[1] > left:
            print(f"ВНИМАНИЕ: оценка с ретраями ${total[1]:.4f} больше остатка бюджета ${left:.4f}: "
                  "при ретраях может не хватить.")


def mode_arg(value: str) -> str:
    """Имя из MODES или список одиночных режимов через запятую: «baseline,graph,bm25»."""
    if value in MODES:
        return value
    parts = [p.strip() for p in value.split(",") if p.strip()]
    bad = [p for p in parts if p not in systems.MODES]
    if not parts or bad or len(set(parts)) != len(parts):
        raise argparse.ArgumentTypeError(
            f"{value!r}: нужен один из {', '.join(MODES)} или список без повторов из {', '.join(systems.MODES)}")
    return ",".join(parts)


def modes_of(mode: str) -> list[str]:
    return list(MODES[mode]) if mode in MODES else mode.split(",")


def table_modes(modes: list[str]) -> tuple[str, ...]:
    """Столбцы итоговой таблицы: baseline и graph всегда, контроль bm25 — если запускался."""
    return FINAL_MODES + tuple(m for m in FINAL_EXTRA_MODES if m in modes)


def repeats_arg(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r}: нужно целое число от 1 до {MAX_REPEATS}") from None
    if not 1 <= n <= MAX_REPEATS:
        raise argparse.ArgumentTypeError(
            f"{n}: допустимо от 1 до {MAX_REPEATS} (каждый повтор — полный платный проход)")
    return n


def out_suffix(questions_file: Path, as_of: Optional[str]) -> str:
    """Суффикс имён результатов: прогоны одного дня с другим набором вопросов или as_of не затирают друг друга."""
    parts = ([] if Path(questions_file) == QUESTIONS else [Path(questions_file).stem]) + ([as_of] if as_of else [])
    return "_" + "_".join(parts) if parts else ""


def judge_repeats_arg(value: str) -> int:
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"{value!r}: нужно целое число от 1 до {MAX_REPEATS}") from None
    if not 1 <= n <= MAX_REPEATS:
        raise argparse.ArgumentTypeError(
            f"{n}: допустимо от 1 до {MAX_REPEATS} (каждый повтор судьи — отдельный платный вызов)")
    return n


RUN_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*")


def run_header(a: argparse.Namespace, run_id: str, questions: list[Question]) -> dict:
    """Что должно совпасть при --resume: тот же прогон, а не просто тот же run_id."""
    return {"run_id": run_id, "modes": modes_of(a.mode), "split": a.split, "as_of_filter": a.as_of, "limit": a.limit,
            "questions_file": Path(a.questions).name, "questions": [q.id for q in questions], "repeats": a.repeats,
            "judge": a.judge, "judge_facts": a.judge_facts, "judge_repeats": a.judge_repeats,
            "model": os.environ.get("LLM_MODEL", "stub-model"), "judge_model": judge_mod.judge_model(),
            "backend": os.environ.get("LLM_BACKEND", "stub"), "k": CONTEXT_K, "depth": a.depth,
            "JUDGE_FACTS_PROMPT_VERSION": facts_mod.JUDGE_FACTS_PROMPT_VERSION}


def annotate_result(result: dict, rows: list[Row], modes: list[str]) -> dict:
    """Аддитивно к контракту results_v2: CONTEXT_K и пометки режимов в config, размер контекста и as_of_date в
    per_question, проверка симметрии в aggregate. per_question идёт в порядке rows."""
    flags = mode_flags(rows)
    result["config"]["context_k"] = CONTEXT_K
    result["config"]["mode_flags"] = flags
    result["config"]["final_table_modes"] = [m for m in table_modes(modes) if m in modes]
    for rec, r in zip(result["per_question"], rows):
        rec.update({"as_of_date": r.as_of_date, "retrieved_count": r.retrieved_count,
                    "context_chunks": r.context_chunks, "context_graph_items": r.context_graph_items,
                    "context_elements": r.context_elements, "mode_flag": r.mode_flag,
                    "context_meetings": r.context_meetings})
    result["aggregate"]["context_symmetry"] = symmetry_stats(rows, modes)
    return result


def readiness_lines(modes: list[str], *, raw_dir: Path, index_dir: Path, as_of_dates: list[str]) -> list[str]:
    """Что помешает режимам ответить по-настоящему: индекс, граф, графовая часть hybrid. Ничего не пересобирает."""
    out = []
    if any(m in systems.INDEX_MODES for m in modes):
        problem = systems.index_problem(index_dir, raw_dir, as_of_dates)
        if problem:
            users = ", ".join(m for m in modes if m in systems.INDEX_MODES)
            out.append(f"ИНДЕКС ({users}): {problem}. Harness индекс не пересобирает: соберите его через UI "
                       f"(«Демо-встречи») или api.index_batches в {index_dir}; до этого {users} — stub. "
                       + est_mod.reindex_note(raw_dir))
        else:
            out.append(f"Индекс baseline: {index_dir} готов (chunk_format {systems.CHUNK_FORMAT}).")
    if any(m in systems.GRAPH_MODES for m in modes) and as_of_dates:
        problem = systems.graph_problem(raw_dir, max(as_of_dates))
        out.append(f"ГРАФ: {problem}" if problem else "Граф: все встречи до as_of_date есть.")
    if "hybrid" in modes and systems.hybrid_graph_part_is_stub():
        out.append(f"hybrid: {systems.HYBRID_GRAPH_STUB} — graph_part() пока контекст режима graph, обход B (B6) не "
                   "подставлен; в итоговую таблицу hybrid не войдёт.")
    return out


def row_from_dict(d: dict) -> Row:
    names = {f.name for f in fields(Row)}
    return Row(**{k: v for k, v in d.items() if k in names})


def main(argv: Optional[list[str]] = None, judge_fn: Optional[JudgeFn] = None,
         facts_fn: Optional[Callable[..., facts_mod.FactsVerdict]] = None) -> int:
    ap = argparse.ArgumentParser(description="Eval-harness: вопросы → api.ask → метрики, повторы, срезы")
    ap.add_argument("--mode", type=mode_arg, default="both",
                    help=f"{' | '.join(MODES)} или список через запятую (контроли bm25, longctx — только явно)")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="test")
    ap.add_argument("--as-of", dest="as_of", help="только вопросы с этим as_of (m_008); по умолчанию все")
    ap.add_argument("--repeats", type=repeats_arg, default=1,
                    help=f"повторов 1..{MAX_REPEATS}; ответы повторов идут мимо кэша llm_client и оплачиваются")
    ap.add_argument("--limit", type=int, help="первые N вопросов — для отладки")
    ap.add_argument("--depth", type=int, default=2, help="глубина обхода графа")
    ap.add_argument("--questions-file", "--questions", dest="questions", type=Path, default=QUESTIONS,
                    help="файл вопросов (по умолчанию data/gold/questions.json; для m_008 — questions_m008.json)")
    ap.add_argument("--gold", type=Path, default=GOLD)
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="встречи для baseline и проверки цитат")
    ap.add_argument("--index-dir", type=Path, default=INDEX_ROOT,
                    help="готовый индекс baseline (только чтение; harness его не пересобирает)")
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    ap.add_argument("--judge", action="store_true",
                    help="оценить ответы судьёй v2: заполнить accuracy и fabrication_rate (платные вызовы)")
    ap.add_argument("--judge-facts", action="store_true",
                    help="судья по фактам (judge-facts-v1): fact_recall, fact_precision, fact_f1, forbidden_hits "
                         "(платные вызовы); по умолчанию выключен")
    ap.add_argument("--judge-repeats", type=judge_repeats_arg, default=1,
                    help=f"повторов судьи по фактам 1..{MAX_REPEATS}: метрики усредняются, печатается std; "
                         "каждый повтор мимо кэша llm_client и оплачивается")
    ap.add_argument("--run-id", help="идентификатор прогона (по умолчанию <YYYYMMDD_HHMM>): имена "
                                     "results/eval_<run_id>.json и .partial.jsonl")
    ap.add_argument("--resume", action="store_true",
                    help="продолжить прерванный прогон с тем же --run-id по промежуточному файлу")
    ap.add_argument("--dry-run", action="store_true", help="план и оценка стоимости, без прогона")
    ap.add_argument("--confirm-paid", action="store_true", help="разрешить прогон с платным backend")
    ap.add_argument("--unlock-test", action="store_true",
                    help="разрешить вопросы split=test (в том числе --dry-run); запуск пишется в "
                         "results/test_runs.jsonl")
    a = ap.parse_args(argv)
    if a.judge_repeats > 1 and not a.judge_facts:
        ap.error("--judge-repeats имеет смысл только вместе с --judge-facts")
    if a.resume and not a.run_id:
        ap.error("--resume требует --run-id: какой прогон продолжать")
    if a.run_id and not RUN_ID_RE.fullmatch(a.run_id):
        ap.error("--run-id: только латиница, цифры, «_» и «-»")

    modes = modes_of(a.mode)
    questions = select_questions(a.questions, a.split, a.as_of, a.limit)
    if not questions:
        print("Нет вопросов под выбранные --split/--as-of.", file=sys.stderr)
        return 2
    if has_test_questions(questions):
        if not a.unlock_test:
            n_test = sum(1 for q in questions if q.split == "test")
            print(f"Среди выбранных вопросов {n_test} с split=test (--split {a.split}, файл {Path(a.questions).name}). "
                  "test открываем один раз, в воскресенье, после заморозки кода, промптов и настроек: запуск — с "
                  "флагом --unlock-test (и для --dry-run тоже). Для отладки — --split dev.", file=sys.stderr)
            return TEST_LOCKED_CODE
        print(TEST_UNLOCK_WARNING)
    try:    # as_of_date — из вопроса; у исторических файлов без поля — дата встречи as_of
        questions = [q.model_copy(update={"as_of_date": systems.question_as_of_date(q, a.raw_dir)})
                     for q in questions]
    except ValueError as e:
        print(f"Вопросы без даты as_of_date: {e}", file=sys.stderr)
        return 2
    as_of_dates = sorted({q.as_of_date for q in questions})
    backend = os.environ.get("LLM_BACKEND", "stub")
    plan: dict[tuple[str, str], list[str]] = defaultdict(list)
    for m in modes:
        for q in questions:
            plan[(m, q.as_of_date)].append(q.question)

    print(f"Вопросов: {len(questions)} (split={a.split}), режимы: {', '.join(modes)}, повторов: {a.repeats}, "
          f"backend: {backend}, as_of_date: {', '.join(as_of_dates)}, CONTEXT_K={CONTEXT_K}")
    print(cost_guard.status())
    for line in readiness_lines(modes, raw_dir=a.raw_dir, index_dir=a.index_dir, as_of_dates=as_of_dates):
        print(line)
    estimate = est_mod.estimate_run(plan, raw_dir=a.raw_dir, k=CONTEXT_K, depth=a.depth, repeats=a.repeats)
    judge_estimate = (est_mod.estimate_judge(questions, modes, raw_dir=a.raw_dir, repeats=a.repeats)
                      if a.judge else None)
    facts_estimate = (est_mod.estimate_judge_facts(questions, modes, raw_dir=a.raw_dir, repeats=a.repeats,
                                                   judge_repeats=a.judge_repeats) if a.judge_facts else None)
    print_plan(estimate, judge_estimate, backend, dry_run=a.dry_run, fest=facts_estimate)
    test_run = has_test_questions(questions)

    def log_test_run(dry_run: bool) -> None:
        if test_run:
            record_test_run(modes=modes, split=a.split, questions=questions, questions_file=a.questions,
                            dry_run=dry_run)

    if a.dry_run:
        log_test_run(True)
        return 0
    if ((a.judge and judge_fn is None) or (a.judge_facts and facts_fn is None)) and backend == "stub":
        print("--judge / --judge-facts при LLM_BACKEND=stub: заглушка не оценивает ответы. Для проверки без денег — "
              "--dry-run; для настоящего прогона поставьте LLM_BACKEND=openai.", file=sys.stderr)
        return 2
    if backend != "stub" and not a.confirm_paid:
        print(f"\nLLM_BACKEND={backend} — прогон платный. Подтвердите числа выше и запустите с --confirm-paid.",
              file=sys.stderr)
        return 3

    built: dict[str, systems.System] = {}

    def build(mode: str, as_of: str) -> systems.System:
        # одна система на режим для всех as_of: будущее отрезает api.ask(as_of=as_of_date вопроса)
        if mode not in built:
            built[mode] = systems.build_system(mode, raw_dir=a.raw_dir, index_dir=a.index_dir,
                                               as_of_dates=as_of_dates, depth=a.depth)
        return built[mode]

    # --- run_id, промежуточный файл и --resume ---
    now = dt.datetime.now()
    run_id = a.run_id or now.strftime("%Y%m%d_%H%M")
    json_path, part_path = results_v2.result_path(a.out_dir, run_id), results_v2.partial_path(a.out_dir, run_id)
    header = run_header(a, run_id, questions)
    done: dict[RowKey, Row] = {}
    if part_path.exists():
        if not a.resume:
            print(f"Есть промежуточный файл {part_path}: прогон {run_id} не завершён. Продолжить — добавьте "
                  "--resume, начать заново — другой --run-id.", file=sys.stderr)
            return 2
        saved_header, saved = results_v2.read_partial(part_path)
        if saved_header != header:
            diff = sorted(k for k in {**header, **(saved_header or {})} if (saved_header or {}).get(k) != header.get(k))
            print(f"--resume: параметры прогона изменились ({', '.join(diff)}); продолжить нельзя, чтобы не смешать "
                  "разные прогоны. Верните прежние параметры или возьмите другой --run-id.", file=sys.stderr)
            return 2
        done = {row_key(r): r for r in map(row_from_dict, saved)}
        print(f"--resume: из {part_path.name} взято {len(done)} готовых строк; остальные будут посчитаны.")
    elif a.resume and json_path.exists():
        print(f"--resume: прогон {run_id} уже завершён: {json_path}")
        return 0
    elif a.resume:
        print(f"--resume: промежуточного файла для {run_id} нет, начинаю прогон заново.")
    if not done and json_path.exists():
        print(f"Результат {json_path} уже есть: возьмите другой --run-id.", file=sys.stderr)
        return 2
    if not part_path.exists():
        results_v2.append_jsonl(part_path, {"header": header})

    log_test_run(False)    # запись — только когда прогон действительно стартует (все проверки выше пройдены)

    def save_row(row: Row) -> None:
        results_v2.append_jsonl(part_path, {"row": asdict(row)})

    # Повторы без кэша: у каждого свой каталог (под runs/cache/, он в .gitignore), новый на каждый запуск.
    stamp = now.strftime("%Y%m%dT%H%M%S")
    cache_root = llm_client.CACHE / "eval_repeats" / stamp
    repeat_cache = (lambda rep: isolated_cache(cache_root / f"rep{rep}")) if a.repeats > 1 else None
    judge_cache = ((lambda rep: isolated_cache(cache_root / f"judge_rep{rep}")) if a.judge_repeats > 1 else None)

    from src import api
    runner = JudgeRunner(raw_dir=a.raw_dir, fn=judge_fn or judge_mod.judge) if a.judge else None
    facts_runner = (facts_mod.JudgeFactsRunner(judge_repeats=a.judge_repeats, fn=facts_fn or facts_mod.judge_facts,
                                               repeat_cache=judge_cache) if a.judge_facts else None)
    try:
        rows = run_eval(questions, modes, build, gold=load_gold_events(a.gold), texts=MeetingTexts(a.raw_dir),
                        backend=backend, judge=runner, repeats=a.repeats, repeat_cache=repeat_cache,
                        call_log=CallLog(), facts=facts_runner, on_row=save_row, done=done)
    except BaseException:
        print(f"\nПрогон прерван. Готовые строки сохранены в {part_path}. "
              f"Продолжить: те же аргументы и --resume --run-id {run_id}", file=sys.stderr)
        raise
    finally:
        api.use_index_dir(api.DEFAULT_INDEX_DIR)  # UI-индекс не должен остаться подменённым

    date = dt.date.today().isoformat()
    tag = out_suffix(a.questions, a.as_of)
    notes = {r.mode: r.error for r in rows if r.status == "stub"}
    csv_path, md_path = a.out_dir / f"eval_{date}{tag}.csv", a.out_dir / f"summary_{date}{tag}.md"
    write_csv(csv_path, rows)
    info = JudgeInfo(judge_mod.JUDGE_PROMPT_VERSION_V2, judge_mod.judge_model(), runner.calls, runner.reused,
                     runner.failed) if runner else None
    # json: aggregate считает харнес, дашборд только читает
    config = results_v2.build_config(
        modes=modes, split=a.split, n_questions=len({q.id for q in questions}),
        as_of=results_v2.as_of_label(q.as_of for q in questions), judge_repeats=a.judge_repeats,
        repeats=a.repeats, backend=backend, commit=_git_commit(), judge_facts=a.judge_facts)
    result = annotate_result(results_v2.build_result(
        run_id=run_id, timestamp=now.astimezone().isoformat(timespec="seconds"), config=config, rows=rows), rows, modes)
    v2 =results_v2.render_v2_section(
        result["aggregate"], result["per_question"], modes, judge_facts=a.judge_facts,
        judge_repeats=a.judge_repeats, split=a.split,
        header_lines=([f"- судья по фактам: {facts_mod.JUDGE_FACTS_PROMPT_VERSION} (без транскриптов), модель "
                       f"{judge_mod.judge_model()}; вызовов {facts_runner.calls}, повторных входов пропущено "
                       f"{facts_runner.reused}, без вердикта (ошибка формата) {facts_runner.failed}. "
                       "Судья и система по умолчанию на одной модели; судья без транскриптов не проверяет факты "
                       "вне списка, поэтому extra не штрафуется."] if facts_runner else []))
    summary = render_summary(rows, modes, date=date, k=CONTEXT_K, depth=a.depth, notes=notes,
                             splits=["dev", "test"] if a.split == "all" else [a.split], judge=info, v2=v2,
                             test_runs=count_test_runs())
    md_path.write_text(summary, encoding="utf-8")
    results_v2.write_result(json_path, result)
    part_path.unlink(missing_ok=True)   # итог записан целиком, промежуточный файл больше не нужен
    print(f"\n{summary}\nCSV: {csv_path}\nСводка: {md_path}\nJSON: {json_path}")
    # Итоговая таблица — только полный прогон both по test (отладка с --limit её не затирает); при stub — отдельное
    # имя, чтобы заглушка не перезаписала платный результат того же дня.
    if set(FINAL_MODES) <= set(modes) and not a.limit and final_rows(rows):
        final_path = a.out_dir / f"final_table_{date}{'_stub' if backend == 'stub' else ''}.md"
        final_path.write_text(render_final_table(rows, modes, date=date, k=CONTEXT_K, depth=a.depth, notes=notes,
                                                 judge=info), encoding="utf-8")
        print(f"Итоговая таблица: {final_path}")
    else:
        print("Итоговая таблица не записана: нужны режимы baseline и graph (both, all или список), вопросы "
              "split=test с основным as_of и без --limit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
