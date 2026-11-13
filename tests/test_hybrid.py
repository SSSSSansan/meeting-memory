"""C4: гибрид = контекст режима graph + реплики вокруг его пунктов + обычный поиск, без повторов."""
import json

import pytest

import src.answer.answerer as ans
from src.graph.queries import load_graph
from src.llm_client import LLMResponse
from src.retrieval.baseline import BaselineIndex
from src.retrieval.graph_rag import retrieve_subgraph
from src.retrieval.hybrid import backfill_candidates, retrieve_hybrid, ts_seconds
from src.schemas import Segment, Transcript

QUESTION = "Как менялся срок релиза Атласа?"


@pytest.fixture(autouse=True)
def _stub_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_BACKEND", "stub")
    monkeypatch.setenv("EMBED_MODEL", "")
    monkeypatch.setattr(ans, "REPORT", tmp_path / "answer_report.jsonl")


@pytest.fixture
def G():
    return load_graph(ans.ROOT / "data/sample/graph.json")


def _ts(sec: int) -> str:
    return f"{sec // 3600:02d}:{sec % 3600 // 60:02d}:{sec % 60:02d}"


def _meeting(mid: str, date: str, texts_at: dict[int, str]) -> Transcript:
    """Реплики каждые 20 секунд с 0 до 15 минут; texts_at подменяет текст в нужные секунды."""
    segs = [Segment(ts=_ts(s), speaker="Айгерим", text=texts_at.get(s, f"Обсуждаем пункт {mid} {s}."),
                    start_s=float(s)) for s in range(0, 900, 20)]
    return Transcript(meeting_id=mid, title=f"Атлас {mid}", date=date, participants=[], segments=segs)


@pytest.fixture
def index():
    idx = BaselineIndex(projects=["Атлас", "Горизонт"])
    idx.add([_meeting("m_001", "2026-02-03", {240: "давайте перенесём релиз на двадцатое апреля"}),
             _meeting("m_002", "2026-02-10", {60: "Переносим релиз Атласа на пятое мая"})])
    return idx


def test_ts_seconds():
    assert ts_seconds("00:04:12") == 252 and ts_seconds("01:00:00") == 3600


def test_backfill_takes_nearest_chunk_within_window(index):
    (c,) = backfill_candidates(index, [("m_001", "00:04:12")], as_of=None)
    starts = [s["start_s"] for s in c["segments"]]
    assert c["meeting_id"] == "m_001" and min(abs(s - 252) for s in starts) <= 45
    assert backfill_candidates(index, [("m_001", "02:00:00")], as_of=None) == []      # дальше окна — ничего
    assert backfill_candidates(index, [("m_002", "00:01:05")], as_of="2026-02-05") == []  # будущее отрезано


def test_backfill_has_no_repeats(index):
    got = backfill_candidates(index, [("m_001", "00:04:12"), ("m_001", "00:04:13")], as_of=None)
    assert len(got) == 1


def test_graph_part_is_exactly_graph_mode_context(index, G):
    h = retrieve_hybrid(QUESTION, index, G=G)
    sub = retrieve_subgraph(QUESTION, G=G)
    assert h["graph_context"] == sub["context"] and h["graph_items"] == sub["items"]


def test_budget_and_no_duplicates(index, G):
    h = retrieve_hybrid(QUESTION, index, G=G, max_chunks=5, backfill_max=2)
    ids = [c["chunk_id"] for c in h["chunks"]]
    assert len(ids) == len(set(ids)) == 5
    assert h["n_backfill"] == 2 and h["n_vector"] == 3
    assert [c["via"] for c in h["chunks"]] == ["graph"] * 2 + ["vector"] * 3


def test_graph_chunk_found_again_by_vector_takes_one_place(index, G):
    h = retrieve_hybrid(QUESTION, index, G=G, max_chunks=40, backfill_max=40)
    lines = [(c["meeting_id"], s["ts"]) for c in h["chunks"] for s in c["segments"]]
    assert h["n_vector"] > 0
    graph_ids = {c["chunk_id"] for c in h["chunks"] if c["via"] == "graph"}
    assert not graph_ids & {c["chunk_id"] for c in h["chunks"] if c["via"] == "vector"}
    assert len(lines) <= len(set(lines)) * 2  # соседние чанки делят одну реплику, но не больше


def test_as_of_cuts_graph_and_text(index, G):
    h = retrieve_hybrid(QUESTION, index, G=G, as_of="2026-02-05")
    assert h["retrieved_meetings"] == ["m_001"]
    assert all(c["meeting_id"] == "m_001" for c in h["chunks"]) and "m_002" not in h["graph_context"]


def test_retrieved_meetings_graph_first_then_text(index, G):
    h = retrieve_hybrid(QUESTION, index, G=G)
    assert h["retrieved_meetings"] == ["m_001", "m_002"]


def test_empty_index_still_gives_graph(G):
    h = retrieve_hybrid(QUESTION, BaselineIndex(projects=[]), G=G)
    assert h["chunks"] == [] and h["graph_items"] and h["retrieved_meetings"] == ["m_001", "m_002"]


def test_answer_hybrid_verifies_quotes_from_both_parts(monkeypatch, index, G):
    calls = []

    def fake(messages, **kw):
        calls.append((messages, kw))
        return LLMResponse(text=json.dumps({"answer": "20 апреля, потом 5 мая.", "citations": [
            {"meeting_id": "m_001", "ts": "00:04:12", "quote": "перенесём релиз на двадцатое апреля"},
            {"meeting_id": "m_002", "ts": "00:01:00", "quote": "Переносим релиз Атласа на пятое мая"}]},
            ensure_ascii=False))

    monkeypatch.setattr(ans, "complete", fake)
    a = ans.answer(QUESTION, mode="hybrid", index=index, graph=G)
    user = calls[0][0][1]["content"]
    assert a.mode == "hybrid" and len(a.citations) == 2 and "Удалено" not in a.answer
    assert "##### ГРАФ РЕШЕНИЙ И ЗАДАЧ #####" in user and "##### ФРАГМЕНТЫ ТРАНСКРИПТОВ #####" in user
    assert user.index("ГРАФ РЕШЕНИЙ") < user.index("ФРАГМЕНТЫ ТРАНСКРИПТОВ")
    assert calls[0][1]["purpose"] == "answer_hybrid"
    assert a.retrieved_meetings == ["m_001", "m_002"]
    row = json.loads(ans.REPORT.read_text(encoding="utf-8").splitlines()[-1])
    assert row["mode"] == "hybrid" and set(row["context_parts"]) == {"graph_items", "backfill_chunks",
                                                                      "vector_chunks"}
