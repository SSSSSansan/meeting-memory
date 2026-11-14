"""Общее для CLI-тестов eval (роль A): harness индекс baseline не строит, поэтому тест готовит его сам."""
import os
from pathlib import Path

from src import api
from src.eval import systems

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
GRAPH_STUB_REASON = "граф до m_003 не готов (посылка теста: graph — stub)"


def build_index(path: Path, upto: str = "m_008", raw: Path = RAW) -> Path:
    """Индекс по встречам raw до upto включительно — как перед настоящим прогоном (UI / api.index_batches).

    Только на LLM_BACKEND=stub: на платном backend тесту индекс не нужен (там --dry-run или отказ без
    --confirm-paid), а эмбеддинги ушли бы в сеть.
    """
    if os.environ.get("LLM_BACKEND", "stub") != "stub":
        return path
    api.use_index_dir(path)
    try:
        for _ in api.index_batches(systems.load_transcripts(raw, upto)):
            pass
    finally:
        api.use_index_dir(api.DEFAULT_INDEX_DIR)
    return path


def graph_as_stub(monkeypatch) -> None:
    """Тесты судьи, повторов и resume писались, когда graph на m_003 был заглушкой; посылку фиксируем явно."""
    monkeypatch.setattr(systems, "graph_problem", lambda raw_dir, as_of: GRAPH_STUB_REASON)
