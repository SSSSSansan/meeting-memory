"""C4: гибрид = контекст режима graph + реплики вокруг его пунктов + обычный поиск, без повторов."""
import json

import pytest

import src.answer.answerer as ans
from src.graph.queries import load_graph
from src.llm_client import LLMResponse
from src.retrieval.baseline import CONTEXT_K, BaselineIndex
from src.retrieval.graph_rag import question_focus, retrieve_subgraph, select_items
from src.retrieval.hybrid import BACKFILL_MAX, HYBRID_GRAPH_K, backfill_candidates, retrieve_hybrid, ts_seconds
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


def test_graph_part_is_graph_mode_ranking_cut_to_its_share(index, G):
    h = retrieve_hybrid(QUESTION, index, G=G)
    sub = retrieve_subgraph(QUESTION, G=G, max_items=HYBRID_GRAPH_K)
    assert h["graph_context"] == sub["context"] and h["graph_items"] == sub["items"]


def test_budget_and_no_duplicates(index, G):
    h = retrieve_hybrid(QUESTION, index, G=G, context_k=7, graph_k=2, backfill_max=2)
    ids = [c["chunk_id"] for c in h["chunks"]]
    assert len(h["graph_items"]) == 2 and len(ids) == len(set(ids)) == 5   # всего 7 = context_k
    assert h["n_backfill"] == 2 and h["n_vector"] == 3
    assert [c["via"] for c in h["chunks"]] == ["graph"] * 2 + ["vector"] * 3


def test_hybrid_total_never_exceeds_context_k(index, G):
    h = retrieve_hybrid(QUESTION, index, G=G)
    assert len(h["graph_items"]) + len(h["chunks"]) == CONTEXT_K
    assert len(h["graph_items"]) <= HYBRID_GRAPH_K and h["n_backfill"] <= BACKFILL_MAX


# ---------- бюджет контекста графа: тот же CONTEXT_K, что у baseline ----------

def test_select_items_keeps_supersede_chain_whole(G):
    # d_m002_01 заменяет d_m001_01: цепочка из двух
    ranked = ["d_m002_01", "t_m001_01", "q_m001_01", "d_m001_01"]
    assert select_items(G, ranked, 3) == ["d_m001_01", "d_m002_01", "t_m001_01"]
    assert select_items(G, ranked, 1) == ["d_m002_01"]       # цепочка не влезает даже одна — её начало
    assert select_items(G, ranked, None) == ranked


def test_graph_context_cut_to_context_k_on_real_graph():
    big = load_graph(ans.ROOT / "data/graph_raw/graph.json")
    sub = retrieve_subgraph("Как менялся срок релиза Zhol?", G=big)
    assert sub["n_found"] > CONTEXT_K and len(sub["items"]) == CONTEXT_K
    assert {"d_m001_01", "d_m006_01", "d_m011_01"} <= set(sub["items"])   # вся цепочка переносов срока
    assert len(retrieve_subgraph("Как менялся срок релиза Zhol?", G=big, max_items=None)["items"]) == sub["n_found"]


@pytest.mark.parametrize("question, focus", [
    ("Как менялся срок релиза Zhol?", "T2"),
    ("Есть ли конфликты по загрузке людей?", "T4"),
    ("Что по Qamqor ещё не закрыто?", "T5"),
    ("Кто делает экран статистики?", None),
])
def test_question_focus_uses_router_words(question, focus):
    assert question_focus(question) == focus


def test_change_question_keeps_whole_chain_even_in_hybrid_share():
    """q_05: без учёта типа вопроса графовая часть hybrid (4 пункта) теряла цепочку и покрывала 1 встречу из 4."""
    big = load_graph(ans.ROOT / "data/graph_raw/graph.json")
    sub = retrieve_subgraph("Как менялся срок релиза Zhol?", G=big, max_items=HYBRID_GRAPH_K)
    assert {"d_m001_01", "d_m006_01", "d_m011_01"} <= set(sub["items"])


def test_graph_chunk_found_again_by_vector_takes_one_place(index, G):
    h = retrieve_hybrid(QUESTION, index, G=G, context_k=44, backfill_max=40)
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
