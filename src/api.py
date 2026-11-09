"""РОЛЬ C. Функции для UI и eval — вся логика здесь, в app/ только отображение.

    from src import api
    pairs = api.pair_scribal_files({"m_001.transcript.json": b"...", "m_001.meeting.json": b"..."})
    transcripts = [api.load_scribal(*p) for p in pairs]
    for done, total in api.index_batches(transcripts): ...     # батчами по 5, индекс сохраняется после каждого
    api.ask("Когда релиз Zhol?")                              # → src.schemas.Answer

Загрузка — пара файлов Scribal на встречу: `<meeting_id>.transcript.json` (raw) и
`<meeting_id>.meeting.json` (ответ GET /meetings/{id}), как в data/raw/.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterator, Optional

from src.answer.answerer import answer
from src.ingest.scribal_adapter import convert
from src.retrieval.baseline import DEFAULT_INDEX_DIR, ROOT, BaselineIndex
from src.retrieval.graph_rag import SAMPLE_GRAPH_PATH, default_graph_path, get_aliases, get_graph
from src.retrieval.router import Route
from src.retrieval.router import route as _route
from src.schemas import Answer, ScribalMeeting, ScribalTranscript, Transcript

BATCH_SIZE = 5  # решение куратора: встречи обрабатываются батчами по 5
DEMO_DIR = ROOT / "data" / "raw"

_SCRIBAL_NAME = re.compile(r"^(?P<mid>.+)\.(?P<kind>transcript|meeting)\.json$")

_index: Optional[BaselineIndex] = None
_index_dir: Path = DEFAULT_INDEX_DIR


def pair_scribal_files(files: dict[str, bytes]) -> tuple[list[tuple[str, bytes, bytes]], list[str]]:
    """Имя файла → содержимое. Возвращает ([(meeting_id, transcript, meeting)], [ошибки])."""
    found: dict[str, dict[str, bytes]] = {}
    errors = []
    for name, content in files.items():
        m = _SCRIBAL_NAME.match(Path(name).name)
        if not m:
            errors.append(f"{name}: ожидается <id>.transcript.json или <id>.meeting.json")
            continue
        found.setdefault(m["mid"], {})[m["kind"]] = content
    pairs = []
    for mid, kinds in sorted(found.items()):
        missing = {"transcript", "meeting"} - kinds.keys()
        if missing:
            errors.append(f"{mid}: не хватает файла {', '.join(f'{mid}.{k}.json' for k in sorted(missing))}")
        else:
            pairs.append((mid, kinds["transcript"], kinds["meeting"]))
    return pairs, errors


def load_scribal(meeting_id: str, transcript_json: bytes, meeting_json: bytes,
                 source: str = "scribal") -> Transcript:
    raw = ScribalTranscript.model_validate(json.loads(transcript_json))
    meta = ScribalMeeting.model_validate(json.loads(meeting_json))
    return convert(raw, meta, meeting_id, source)


def demo_files() -> dict[str, bytes]:
    """Синтетические встречи роли A из data/raw/ — те же файлы, что пользователь загрузил бы руками."""
    return {p.name: p.read_bytes() for p in sorted(DEMO_DIR.glob("*.json"))}


def use_index_dir(path: Path) -> None:
    """Куда сохранять индекс (тесты подставляют временную папку)."""
    global _index, _index_dir
    _index, _index_dir = None, Path(path)


def get_index() -> BaselineIndex:
    """Индекс с диска; если его нет или он собран другой моделью эмбеддингов — пустой."""
    global _index
    if _index is None:
        try:
            _index = BaselineIndex.load(_index_dir)
        except (FileNotFoundError, ValueError):
            _index = BaselineIndex()
    return _index


def indexed_meetings() -> list[dict]:
    seen: dict[str, dict] = {}
    for c in get_index().chunks:
        m = seen.setdefault(c["meeting_id"], {"meeting_id": c["meeting_id"], "date": c["date"],
                                              "title": c["title"], "chunks": 0})
        m["chunks"] += 1
    return sorted(seen.values(), key=lambda m: (m["date"], m["meeting_id"]))


def index_batches(transcripts: list[Transcript], batch_size: int = BATCH_SIZE) -> Iterator[tuple[int, int]]:
    """Индексирует по batch_size встреч и сохраняет индекс после каждого батча.

    Отдаёт (обработано встреч, всего) после каждого батча — для прогресса в UI.
    Уже проиндексированные встречи пропускаются, поэтому прерванную загрузку можно повторить.
    """
    index = get_index()
    for i in range(0, len(transcripts), batch_size):
        if index.add(transcripts[i:i + batch_size]):
            index.save(_index_dir)
        yield min(i + batch_size, len(transcripts)), len(transcripts)


def reset_index() -> None:
    global _index
    _index = BaselineIndex()
    for name in ("chunks.json", "vectors.npy"):
        (_index_dir / name).unlink(missing_ok=True)


def ask(question: str, mode: str = "baseline", k: int = 5, depth: int = 2) -> Answer:
    """Одна точка для UI и eval. baseline ищет по индексу встреч, загруженных в UI;
    graph — по графу роли B (graph_info()); auto — ветку выбирает роутер (см. route()).
    Какой веткой реально ответили — Answer.mode."""
    return answer(question, mode=mode, k=k, depth=depth, index=get_index())


def route(question: str) -> Route:
    """Какую ветку выберет auto и почему — без вызова LLM, для подписи в UI."""
    return _route(question)


def graph_info() -> dict:
    """Какой граф читает режим graph и что в нём есть — чтобы в UI было видно расхождение с индексом."""
    path = default_graph_path()
    G = get_graph(path)
    meetings = sorted({d["meeting_id"] for _, d in G.nodes(data=True) if d.get("type") == "Meeting"})
    return {"path": path.relative_to(ROOT).as_posix(), "is_sample": path == SAMPLE_GRAPH_PATH, "meetings": meetings,
            "aliases": len(get_aliases(path))}
