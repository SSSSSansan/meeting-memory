"""РОЛЬ A. Адаптеры к src.api.ask: система получает только текст вопроса и видит встречи не позже as_of.

api.ask(question, mode) не умеет ограничивать встречи, поэтому ограничение делает harness:
  baseline — индекс строится только по встречам data/raw до as_of включительно (свой каталог индекса
             на каждый as_of, UI-индекс runs/index/baseline не трогаем);
  graph    — api.ask читает граф из data/graph/graph.json и отфильтровать его не даёт. Режим запускается,
             только если в графе РОВНО встречи до as_of; иначе вместо него заглушка с причиной в таблице.
Заглушки не изображают ответ системы: у них нет метрик, в таблице стоит «stub».
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from src import api
from src.eval.metrics import meeting_number
from src.schemas import Answer, Transcript

AskFn = Callable[[str], Answer]  # в систему уходит только текст вопроса, эталон ей не передаётся


@dataclass
class System:
    mode: str
    ask: AskFn
    is_stub: bool = False
    note: str = ""


def raw_meeting_ids(raw_dir: Path, as_of: str) -> list[str]:
    """Встречи из raw_dir до as_of включительно, по порядку номеров."""
    limit = meeting_number(as_of)
    ids = {p.name.split(".")[0] for p in Path(raw_dir).glob("*.transcript.json")}
    return sorted((m for m in ids if limit is not None and (meeting_number(m) or 10**9) <= limit),
                  key=lambda m: meeting_number(m) or 0)


def load_transcripts(raw_dir: Path, as_of: str) -> list[Transcript]:
    wanted = set(raw_meeting_ids(raw_dir, as_of))
    files = {p.name: p.read_bytes() for p in sorted(Path(raw_dir).glob("*.json"))
             if p.name.split(".")[0] in wanted}
    pairs, errors = api.pair_scribal_files(files)
    if errors:
        raise ValueError("; ".join(errors))
    return [api.load_scribal(*p) for p in pairs]


def graph_problem(raw_dir: Path, as_of: str) -> str:
    """Почему режим graph нельзя честно запустить на этом as_of; пустая строка — можно."""
    try:
        info = api.graph_info()
    except Exception as e:  # граф не читается — для eval это тоже «нечем отвечать»
        return f"граф не читается: {e}"
    if info["is_sample"]:
        return (f"нет data/graph/graph.json: режим graph читает заглушку {info['path']} "
                "(старый сюжет Атлас/Горизонт)")
    have, need = set(info["meetings"]), set(raw_meeting_ids(raw_dir, as_of))
    if have - need:
        return f"граф содержит встречи после {as_of}: {', '.join(sorted(have - need))}; нужен граф только до {as_of}"
    if need - have:
        return f"в графе нет встреч {', '.join(sorted(need - have))}; нужен граф по всем встречам до {as_of}"
    return ""


def stub_system(mode: str, reason: str) -> System:
    def ask(question: str) -> Answer:
        return Answer(answer=f"[stub:{mode}] {reason}", mode=mode)  # type: ignore[arg-type]
    return System(mode=mode, ask=ask, is_stub=True, note=reason)


def build_system(mode: str, as_of: str, *, raw_dir: Path, index_root: Path, k: int = 5, depth: int = 2) -> System:
    if mode == "baseline":
        transcripts = load_transcripts(raw_dir, as_of)
        if not transcripts:
            return stub_system(mode, f"в {raw_dir} нет встреч до {as_of}")
        api.use_index_dir(Path(index_root) / as_of)
        api.reset_index()  # индекс прошлого прогона мог быть собран по другим данным
        for _ in api.index_batches(transcripts):
            pass
        seen = {m["meeting_id"] for m in api.indexed_meetings()}
        if seen != {t.meeting_id for t in transcripts}:
            raise RuntimeError(f"в индексе {sorted(seen)}, ожидались встречи до {as_of}")
        return System(mode, lambda q: api.ask(q, "baseline", k=k))
    if mode == "graph":
        reason = graph_problem(raw_dir, as_of)
        if reason:
            return stub_system(mode, reason)
        return System(mode, lambda q: api.ask(q, "graph", depth=depth))
    raise ValueError(f"mode={mode!r}: допустимо baseline | graph")
