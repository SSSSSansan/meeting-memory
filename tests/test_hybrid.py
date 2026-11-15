"""Режимы graph и hybrid на обходе роли B (src.graph.traverse): бюджет CONTEXT_K, as_of без утечек, без повторов.

graph  — первые CONTEXT_K пунктов обхода B, у каждого реплики вокруг таймкода;
hybrid — первые HYBRID_GRAPH_K пунктов того же обхода + обычный поиск до CONTEXT_K элементов.
"""
import json

import pytest

import src.answer.answerer as ans
from src.graph.queries import load_graph
from src.llm_client import LLMResponse
from src.retrieval.baseline import CONTEXT_K, BaselineIndex, is_near_duplicate
from src.retrieval.graph_rag import question_focus, retrieve_subgraph
from src.retrieval.hybrid import HYBRID_GRAPH_K, graph_part, retrieve_hybrid
from src.schemas import Segment, Transcript

BIG = ans.ROOT / "data/graph_raw/graph.json"   # граф B по 14 встречам: его встречи совпадают с data/raw
ZHOL_CHAIN = {"d_m001_01", "d_m006_01", "d_m011_01"}  # 15 сентября → 6 октября → 10 ноября

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




@pytest.fixture(scope="module")
def big():
    return load_graph(BIG)


@pytest.fixture
def big_index():
    # уровень теста, а не модуля: иначе фикстура строится до autouse-переключения на stub и берёт backend из .env
    from src import api
    pairs, _ = api.pair_scribal_files(api.demo_files())
    idx = BaselineIndex()
    idx.add([api.load_scribal(*p) for p in pairs])
    return idx


# ---------- graph: обход B с бюджетом CONTEXT_K ----------

def test_graph_mode_is_b_traversal_cut_to_context_k(big):
    from src.graph.traverse import traverse

    sub = retrieve_subgraph("Как менялся срок релиза Zhol?", G=big)
    order = traverse("Как менялся срок релиза Zhol?", G=big, with_text=False)["items"]
    assert set(sub["items"]) == set(order[:CONTEXT_K]) and len(sub["items"]) <= CONTEXT_K
    assert ZHOL_CHAIN <= set(sub["items"])                    # цепочка переносов срока целиком
    assert "Реплики рядом:" in sub["context"]                  # третий слой обхода: текст вокруг пункта
    assert all(sub["details"][n].get("tier") for n in sub["items"])


def test_change_question_keeps_whole_chain_in_hybrid_share(big):
    sub = retrieve_subgraph("Как менялся срок релиза Zhol?", G=big, max_items=HYBRID_GRAPH_K)
    assert ZHOL_CHAIN <= set(sub["items"]) and len(sub["items"]) <= HYBRID_GRAPH_K


def test_no_future_statuses_or_links_leak_into_context(big):
    """На 1 сентября перенос срока (m_006, 24 августа) уже был, а следующий (m_011, 21 сентября) — ещё нет."""
    ctx = retrieve_subgraph("Как менялся срок релиза Zhol?", G=big, as_of="2026-09-01")["context"]
    assert "d_m011_01" not in ctx                              # ни пункта, ни ссылки на него
    block = ctx.split("Решение d_m006_01")[1].split("\n\n")[0]
    assert "[действует" in block and "ЗАМЕНЕНО" not in block


def test_sample_graph_does_not_borrow_replicas_of_other_meetings(G):
    """У заглушки data/sample те же id m_001/m_002, но другой сюжет: реплики из data/raw к ней не подходят."""
    assert "Реплики рядом:" not in retrieve_subgraph(QUESTION, G=G)["context"]


@pytest.mark.parametrize("bad", ["m_015", "2026-9-1", "15.09.2026"])
def test_as_of_must_be_iso_date_everywhere(bad, index, G):
    """Номер встречи вместо даты молча отключал фильтр будущего: «2026-…» <= «m_015» всегда верно (нашла A)."""
    with pytest.raises(ValueError):
        index.search("релиз", as_of=bad)
    with pytest.raises(ValueError):
        retrieve_subgraph(QUESTION, G=G, as_of=bad)
    with pytest.raises(ValueError):
        ans.answer(QUESTION, mode="hybrid", index=index, graph=G, as_of=bad)


def test_graph_item_without_date_is_not_let_through_by_as_of(G):
    H = G.copy()
    del H.nodes["d_m002_01"]["date"]
    assert "d_m002_01" in retrieve_subgraph(QUESTION, G=H)["items"]
    assert "d_m002_01" not in retrieve_subgraph(QUESTION, G=H, as_of="2026-12-31")["items"]


@pytest.mark.parametrize("question, focus", [
    ("Как менялся срок релиза Zhol?", "T2"),
    ("Есть ли конфликты по загрузке людей?", "T4"),
    ("Что по Qamqor ещё не закрыто?", "T5"),
    ("Кто делает экран статистики?", None),
])
def test_question_focus_uses_router_words(question, focus):
    assert question_focus(question) == focus


# ---------- hybrid: пункты обхода B + поиск, всего CONTEXT_K ----------

def test_graph_part_is_the_same_traversal_cut_to_its_share(index, G):
    h = retrieve_hybrid(QUESTION, index, G=G)
    sub = retrieve_subgraph(QUESTION, G=G, max_items=HYBRID_GRAPH_K)
    assert h["graph_context"] == sub["context"] and h["graph_items"] == sub["items"]


def test_budget_split(index, G):
    h = retrieve_hybrid(QUESTION, index, G=G, context_k=7, graph_k=2)
    ids = [c["chunk_id"] for c in h["chunks"]]
    assert len(h["graph_items"]) == 2 and len(ids) == len(set(ids)) == 5
    assert {c["via"] for c in h["chunks"]} == {"vector"} and h["n_vector"] == 5


def test_hybrid_total_equals_context_k(index, G):
    h = retrieve_hybrid(QUESTION, index, G=G)
    assert len(h["graph_items"]) + len(h["chunks"]) == CONTEXT_K
    assert len(h["graph_items"]) <= HYBRID_GRAPH_K


def test_search_chunk_repeating_graph_replicas_takes_no_place(big, big_index):
    question = "Как менялся срок релиза Zhol?"
    part = graph_part(question, 2, big, None)
    h = retrieve_hybrid(question, big_index, G=big)
    assert h["n_graph_text"] == len(part["raw_lines"]) > 0
    assert not any(is_near_duplicate(c, part["raw_lines"]) for c in h["chunks"])
    assert len(h["graph_items"]) + len(h["chunks"]) == CONTEXT_K


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
    assert row["mode"] == "hybrid" and set(row["context_parts"]) == {"graph_items", "graph_items_with_text",
                                                                      "vector_chunks"}
