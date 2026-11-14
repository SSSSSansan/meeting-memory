"""РОЛЬ A. Адаптеры к src.api.ask: система получает текст вопроса и дату as_of_date, эталон не видит.

Будущее отрезает сама система (роль C, memory.md 2026-11-13): api.ask(question, mode, as_of=<ISO-дата>) видит
только встречи с датой не позже as_of_date включительно — во всех режимах (baseline, graph, hybrid). Поэтому harness
больше не строит индекс на каждый as_of: один индекс baseline (по умолчанию runs/index/baseline) на все вопросы.
Индекс harness не пересобирает: если его нет, он другого формата (CHUNK_FORMAT) или другой модели эмбеддингов, или в
нём не хватает встреч — режимы baseline и hybrid становятся заглушкой с причиной «нужна переиндексация».
Сколько чанков уходит в контекст, решает конфиг C (CONTEXT_K); harness свой k не задаёт.

hybrid: пока графовая часть (src.retrieval.hybrid.graph_part) — временная реализация C на retrieve_subgraph, а не
обход роли B (B6), ответы hybrid настоящие, но режим помечается HYBRID_GRAPH_STUB и в итоговую таблицу не входит.
Заглушки не изображают ответ системы: у них нет метрик, в таблице стоит «stub».

Контрольные режимы (A5) читают тот же индекс baseline: bm25 — поиск по словам по тем же чанкам и с тем же CONTEXT_K;
longctx — все встречи до as_of без поиска (CONTEXT_K не применяется, переполнение окна модели — ошибка вопроса).
"""
from __future__ import annotations

import inspect
import json
import re
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

import src.answer.answerer as answerer
from src import api
from src.eval.metrics import meeting_number
from src.retrieval import hybrid as hybrid_mod
from src.retrieval.baseline import CHUNK_FORMAT, CONTEXT_K, BaselineIndex
from src.schemas import Answer, Transcript

AskFn = Callable[..., Answer]  # ask(question) или, при uses_as_of, ask(question, as_of=<ISO-дата>)

MODES = ("baseline", "graph", "hybrid", "bm25", "longctx")
INDEX_MODES = ("baseline", "hybrid", "bm25", "longctx")     # читают индекс baseline
CONTROL_MODES = ("bm25", "longctx")      # контроли A5: запускаются только явно, не входят в all
UNBOUNDED_MODES = ("longctx",)           # CONTEXT_K не применяется: все встречи до as_of
GRAPH_MODES = ("graph", "hybrid")        # читают граф
REINDEX = "нужна переиндексация"
HYBRID_GRAPH_STUB = "graph part: stub"
ISO_DATE = re.compile(r"\d{4}-\d{2}-\d{2}")

__all__ = ["CONTEXT_K", "MODES", "System", "build_system"]


@dataclass
class System:
    mode: str
    ask: AskFn
    is_stub: bool = False
    note: str = ""
    uses_as_of: bool = False     # ask принимает as_of=<ISO-дата вопроса>
    flag: str = ""               # пометка настоящего режима: у hybrid — HYBRID_GRAPH_STUB
    last_context: Optional[dict] = field(default=None, repr=False)   # {"chunks", "graph_items"} последнего ответа


# ---------- даты встреч и as_of ----------

def meeting_dates(raw_dir: Path) -> dict[str, str]:
    """meeting_id → дата встречи, как у системы: started_at[:10] из <id>.meeting.json (локальная дата, +05:00)."""
    out = {}
    for p in sorted(Path(raw_dir).glob("*.meeting.json")):
        try:
            started = json.loads(p.read_text(encoding="utf-8")).get("started_at") or ""
        except (OSError, json.JSONDecodeError):
            continue
        if ISO_DATE.match(started):
            out[p.name.split(".")[0]] = started[:10]
    return out


def as_of_date_of(as_of: str, raw_dir: Path) -> Optional[str]:
    """ISO-дата как есть; meeting_id — дата этой встречи в raw_dir; иначе None."""
    if ISO_DATE.fullmatch(as_of or ""):
        return as_of
    return meeting_dates(raw_dir).get(as_of)


def question_as_of_date(q: Any, raw_dir: Path) -> str:
    """as_of_date берётся из вопроса. У исторических файлов без поля (questions_m008.json) — дата встречи as_of."""
    d = getattr(q, "as_of_date", None) or as_of_date_of(q.as_of, raw_dir)
    if not d or not ISO_DATE.fullmatch(d):
        raise ValueError(f"{q.id}: нет as_of_date и дату встречи {q.as_of} не найти в {raw_dir}")
    return d


def raw_meeting_ids(raw_dir: Path, as_of: str) -> list[str]:
    """Встречи из raw_dir с датой не позже as_of включительно (as_of — ISO-дата или meeting_id).

    Правило A (scripts/check_questions.py): «позже» — по дате встречи, не по номеру. Если дату не найти, по номеру.
    """
    ids = {p.name.split(".")[0] for p in Path(raw_dir).glob("*.transcript.json")}
    dates, limit = meeting_dates(raw_dir), as_of_date_of(as_of, raw_dir)
    if limit is not None:
        keep = {m for m in ids if m in dates and dates[m] <= limit}
    else:
        n = meeting_number(as_of)
        keep = {m for m in ids if n is not None and (meeting_number(m) or 10**9) <= n}
    return sorted(keep, key=lambda m: (meeting_number(m) or 0, m))


def load_transcripts(raw_dir: Path, as_of: str) -> list[Transcript]:
    wanted = set(raw_meeting_ids(raw_dir, as_of))
    files = {p.name: p.read_bytes() for p in sorted(Path(raw_dir).glob("*.json"))
             if p.name.split(".")[0] in wanted}
    pairs, errors = api.pair_scribal_files(files)
    if errors:
        raise ValueError("; ".join(errors))
    return [api.load_scribal(*p) for p in pairs]


# ---------- готовность индекса и графа ----------

def index_problem(index_dir: Path, raw_dir: Path, as_of_dates: list[str]) -> str:
    """Почему индекс baseline нельзя использовать; пустая строка — можно. Сам индекс не трогается."""
    path = Path(index_dir) / "chunks.json"
    if not path.exists() or not (Path(index_dir) / "vectors.npy").exists():
        return f"{REINDEX}: нет индекса в {index_dir}"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        meta = data["meta"]
        indexed = {c["meeting_id"] for c in data["chunks"]}
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as e:
        return f"{REINDEX}: {path} не читается ({e})"
    if meta.get("chunk_format") != CHUNK_FORMAT:
        return (f"{REINDEX}: индекс старого формата (chunk_format={meta.get('chunk_format') or 'нет'}, "
                f"нужен {CHUNK_FORMAT})")
    now = BaselineIndex().meta
    if meta != now:
        return f"{REINDEX}: индекс собран для {meta}, а сейчас {now}"
    need = set(raw_meeting_ids(raw_dir, max(as_of_dates))) if as_of_dates else set()
    if need - indexed:
        return f"{REINDEX}: в индексе нет встреч {', '.join(sorted(need - indexed))}"
    return ""


def graph_problem(raw_dir: Path, as_of: str) -> str:
    """Почему режим graph нельзя честно запустить; пустая строка — можно.

    Встречи после as_of граф может содержать: api.ask(as_of=...) их отрезает. Не хватать встреч до as_of не должно.
    """
    try:
        info = api.graph_info()
    except Exception as e:  # граф не читается — для eval это тоже «нечем отвечать»
        return f"граф не читается: {e}"
    if info["is_sample"]:
        return (f"нет data/graph/graph.json: режим graph читает заглушку {info['path']} "
                "(старый сюжет Атлас/Горизонт)")
    missing = set(raw_meeting_ids(raw_dir, as_of)) - set(info["meetings"])
    if missing:
        return f"в графе нет встреч {', '.join(sorted(missing))}; нужен граф по всем встречам до {as_of}"
    return ""


def hybrid_graph_part_is_stub() -> bool:
    """Графовая часть hybrid — временная реализация C (контекст режима graph через retrieve_subgraph), пока роль B
    не подставила свой обход (B6). Явного флага в hybrid.py нет, поэтому признак — код graph_part(); не прочитать
    код — считаем заглушкой (так безопаснее для таблицы)."""
    try:
        return "retrieve_subgraph" in inspect.getsource(hybrid_mod.graph_part)
    except (OSError, TypeError):
        return True


# ---------- размер контекста ----------

@contextmanager
def count_context() -> Iterator[dict]:
    """Сколько элементов ушло в контекст одного ответа: чанков транскрипта и пунктов графа.

    В Answer (4.6) этого числа нет, поэтому на время вызова оборачиваются точки answerer, через которые проходит
    контекст: format_context (чанки baseline и hybrid), retrieve_subgraph (пункты graph), retrieve_hybrid (пункты и
    чанки hybrid). Логика ответа не меняется; None — ветка не вызывалась.
    """
    seen: dict = {"chunks": None, "graph_items": None}
    saved = answerer.format_context, answerer.retrieve_subgraph, answerer.retrieve_hybrid
    saved_long = answerer.build_longctx

    def fmt(hits, *a, **kw):
        seen["chunks"] = len(hits)
        return saved[0](hits, *a, **kw)

    def sub(*a, **kw):
        out = saved[1](*a, **kw)
        seen["graph_items"] = len(out["items"])
        return out

    def hyb(*a, **kw):
        out = saved[2](*a, **kw)
        seen["graph_items"], seen["chunks"] = len(out["graph_items"]), len(out["chunks"])
        return out

    def long(*a, **kw):
        # longctx: элементы не считаются (бюджета нет), вместо них — встречи и токены контекста
        out = saved_long(*a, **kw)
        seen["meetings"], seen["context_tokens"] = len(out["meetings"]), out["context_tokens"]
        return out

    answerer.format_context, answerer.retrieve_subgraph, answerer.retrieve_hybrid = fmt, sub, hyb
    answerer.build_longctx = long
    try:
        yield seen
    finally:
        answerer.format_context, answerer.retrieve_subgraph, answerer.retrieve_hybrid = saved
        answerer.build_longctx = saved_long


def context_elements(ctx: Optional[dict]) -> Optional[int]:
    if not ctx or (ctx.get("chunks") is None and ctx.get("graph_items") is None):
        return None
    return (ctx.get("chunks") or 0) + (ctx.get("graph_items") or 0)


# ---------- системы ----------

def stub_system(mode: str, reason: str) -> System:
    def ask(question: str, as_of: Optional[str] = None) -> Answer:
        return Answer(answer=f"[stub:{mode}] {reason}", mode=mode)  # type: ignore[arg-type]
    return System(mode=mode, ask=ask, is_stub=True, note=reason)


def _real_system(mode: str, depth: int, flag: str = "") -> System:
    system = System(mode=mode, ask=lambda *a, **kw: None, uses_as_of=True, flag=flag)  # type: ignore[arg-type]

    def ask(question: str, as_of: Optional[str] = None) -> Answer:
        # Не ISO-дата сравнилась бы со строкой даты чанка как текст («2026-…» <= «m_015» — всегда да) и
        # тихо отключила бы фильтр будущего. Поэтому без настоящей даты — ошибка вопроса, а не ответ с утечкой.
        if not ISO_DATE.fullmatch(as_of or ""):
            raise ValueError(f"as_of={as_of!r}: нужна ISO-дата as_of_date вопроса")
        with count_context() as ctx:
            result = api.ask(question, mode, depth=depth, as_of=as_of)  # k не передаём: CONTEXT_K из конфига C
        system.last_context = dict(ctx)
        return result

    system.ask = ask
    return system


def build_system(mode: str, *, raw_dir: Path, index_dir: Path, as_of_dates: list[str], depth: int = 2) -> System:
    """Одна система на режим для всех вопросов; as_of_dates — даты вопросов (по ним проверяется полнота индекса
    и графа). Индекс только читается: api.use_index_dir(index_dir), без сохранения."""
    if mode not in MODES:
        raise ValueError(f"mode={mode!r}: допустимо {' | '.join(MODES)}")
    if mode in INDEX_MODES:
        reason = index_problem(index_dir, raw_dir, as_of_dates)
        if reason:
            return stub_system(mode, reason)
        api.use_index_dir(Path(index_dir))
    if mode in GRAPH_MODES and as_of_dates:
        reason = graph_problem(raw_dir, max(as_of_dates))
        if reason:
            return stub_system(mode, reason)
    flag = HYBRID_GRAPH_STUB if mode == "hybrid" and hybrid_graph_part_is_stub() else ""
    return _real_system(mode, depth, flag)
