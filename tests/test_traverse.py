"""Тесты обхода графа (src/graph/traverse.py, роль B). Без LLM и без сети."""
from __future__ import annotations

import networkx as nx
import pytest

from src.graph import traverse as tr
from src.graph.traverse import EXPAND_EDGES, expand, find_start_entities, raw_text_around, traverse


def _item(G, nid, kind, date, ts, project, statement, **extra):
    G.add_node(nid, type=kind, date=date, ts=ts, meeting_id="m_" + nid.split("_")[1][1:], project=project,
               statement=statement, quote=statement, status=extra.pop("status", "active"), **extra)
    G.add_edge(nid, f"project:{project}", type="BELONGS_TO")


@pytest.fixture()
def G() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.add_node("project:Alpha", type="Project", label="Alpha")
    g.add_node("project:Beta", type="Project", label="Beta")
    # цепочка замен срока: d_m001_01 <- d_m002_01 <- d_m004_01
    _item(g, "d_m001_01", "Decision", "2026-08-01", "00:01:00", "Alpha", "Срок релиза пятнадцатое сентября",
          status="superseded")
    _item(g, "d_m002_01", "Decision", "2026-08-10", "00:01:00", "Alpha", "Срок релиза шестое октября",
          status="superseded")
    _item(g, "d_m004_01", "Decision", "2026-09-20", "00:01:00", "Alpha", "Срок релиза десятое ноября")
    g.add_edge("d_m002_01", "d_m001_01", type="SUPERSEDES")
    g.add_edge("d_m004_01", "d_m002_01", type="SUPERSEDES")
    # посторонние пункты того же проекта и пункт чужого проекта
    _item(g, "d_m003_01", "Decision", "2026-08-15", "00:02:00", "Alpha", "Закупаем лицензии на монитор")
    _item(g, "d_m003_02", "Decision", "2026-08-15", "00:03:00", "Alpha", "Переходим на новый шрифт")
    _item(g, "d_m005_01", "Decision", "2026-09-25", "00:02:00", "Beta", "Николай работает на Beta", in_conflict=True)
    _item(g, "d_m006_01", "Decision", "2026-10-01", "00:02:00", "Alpha", "Николай работает на Alpha", in_conflict=True)
    g.add_edge("d_m006_01", "d_m005_01", type="CONTRADICTS")
    return g


def test_belongs_to_is_not_an_expansion_edge():
    assert "BELONGS_TO" not in EXPAND_EDGES


def test_expand_does_not_pull_whole_project(G):
    # от пункта про шрифт нет ни одной связи между пунктами: соседи по проекту не подтягиваются
    assert expand(G, ["d_m003_02"], depth=3) == {}


def test_expand_follows_supersedes_and_contradicts(G):
    found = expand(G, ["d_m006_01"], depth=1)
    assert found == {"d_m005_01": {"dist": 1, "via": "CONTRADICTS", "from": "d_m006_01"}}


def test_chain_is_taken_whole_and_before_unrelated_items(G):
    r = traverse("Как менялся срок релиза Alpha?", G=G, aliases={}, with_text=False)
    chain = ["d_m001_01", "d_m002_01", "d_m004_01"]
    assert set(chain) <= set(r["items"][:3])
    assert r["details"]["d_m001_01"]["chain_id"] == "d_m001_01"
    # пункт про шрифт к вопросу не подходит: в конце, ярус 4
    assert r["details"]["d_m003_02"]["tier"] == 4
    assert r["items"].index("d_m003_02") > r["items"].index("d_m004_01")


def test_result_is_not_the_whole_graph(G):
    r = traverse("Как менялся срок релиза Alpha?", G=G, aliases={}, with_text=False)
    assert "d_m005_01" not in r["items"] or r["details"]["d_m005_01"]["tier"] == 3
    assert r["details"]["d_m003_01"]["tier"] == 4


def test_as_of_cuts_future_and_chain(G):
    r = traverse("Как менялся срок релиза Alpha?", G=G, aliases={}, as_of="2026-08-31", with_text=False)
    assert "d_m004_01" not in r["items"]
    assert {"d_m001_01", "d_m002_01"} <= set(r["items"])
    assert all(G.nodes[n]["date"] <= "2026-08-31" for n in r["items"])


def test_as_of_must_be_iso(G):
    with pytest.raises(ValueError):
        traverse("Как менялся срок релиза Alpha?", G=G, aliases={}, as_of="m_015", with_text=False)


def test_undated_item_does_not_pass_as_of(G):
    G.add_node("d_m009_01", type="Decision", date="", ts="00:01:00", meeting_id="m_009", project="Alpha",
               statement="Срок релиза не меняем", quote="Срок релиза не меняем", status="active")
    G.add_edge("d_m009_01", "project:Alpha", type="BELONGS_TO")
    r = traverse("Как менялся срок релиза Alpha?", G=G, aliases={}, as_of="2026-12-31", with_text=False)
    assert "d_m009_01" not in r["items"]


def test_question_without_entities_words_or_type_returns_nothing(G):
    r = traverse("Расскажи про погоду?", G=G, aliases={}, with_text=False)
    assert r["items"] == [] and r["fallback"] is True


def test_conflict_question_without_entities_finds_conflicts(G):
    r = traverse("Есть ли конфликты по загрузке людей?", G=G, aliases={}, with_text=False)
    assert set(r["items"]) == {"d_m005_01", "d_m006_01"}
    assert r["focus"] == "T4"


def test_text_seed_finds_item_without_project_name(G):
    start = find_start_entities(G, "Когда мы закупаем лицензии?", {}, None)
    assert "d_m003_01" in start["seeds"] and start["entities"] == []


def test_entity_name_words_do_not_count_as_text_match(G):
    # «Alpha» есть в названии проекта, но не должна делать подходящими все его пункты
    start = find_start_entities(G, "Что решили по Alpha?", {}, None)
    assert start["entities"] == ["project:Alpha"]
    assert start["seeds"] == {}


def test_details_explain_every_item(G):
    r = traverse("Как менялся срок релиза Alpha?", G=G, aliases={}, with_text=False)
    assert set(r["details"]) == set(r["items"])
    assert all(d["why"] and d["tier"] in (1, 2, 3, 4) for d in r["details"].values())


# --- слой 3: сырой текст (на настоящих встречах data/raw, они в репозитории) ---

def test_raw_text_window_and_label_format():
    out = raw_text_around("m_001", "00:09:37", 45)
    if out.get("missing"):
        pytest.skip("нет data/raw/m_001")
    assert out["lines"], "в окне ±45 с должны быть реплики"
    centre = 9 * 60 + 37
    assert all(line["start_s"] <= centre + 45 for line in out["lines"])
    assert out["text"].count("[m_001 ") == len(out["lines"])
    # те же имена, что у индекса baseline и извлечения: их даёт одна и та же apply_speakers (где она не уверена,
    # остаётся SPEAKER_XX — это не ошибка слоя 3)
    from pathlib import Path

    from src.ingest.scribal_adapter import convert_files
    from src.ingest.speakers import apply_speakers

    base = Path("data/raw")
    t = apply_speakers(convert_files(base / "m_001.transcript.json", base / "m_001.meeting.json", "m_001"))
    expected = {s.ts: s.speaker for s in t.segments}
    assert all(expected[line["ts"]] == line["speaker"] for line in out["lines"])


def test_raw_text_missing_meeting_is_empty_not_error():
    out = raw_text_around("m_999", "00:00:10")
    assert out["missing"] is True and out["lines"] == [] and out["text"] == ""


# --- на настоящем графе 14 встреч ---

def test_real_graph_chain_whole_and_not_whole_graph():
    from src.retrieval.graph_rag import get_graph

    real = get_graph()
    if "d_m011_01" not in real:
        pytest.skip("нет графа по 14 встречам")
    r = traverse("Как менялся срок релиза Zhol?", with_text=False)
    assert {"d_m001_01", "d_m006_01", "d_m011_01"} <= set(r["items"][:7])
    n_items = sum(1 for _, d in real.nodes(data=True) if d.get("type") in ("Decision", "Task", "OpenQuestion"))
    assert r["n_found"] < n_items
    cut = traverse("Как менялся срок релиза Zhol?", as_of="2026-09-20", with_text=False)
    assert "d_m011_01" not in cut["items"]


# --- размер сырого текста и неделимость цепочки на границе K ---

def test_raw_text_nearest_lines_and_char_cap():
    out = raw_text_around("m_001", "00:09:37", 45, max_lines=3)
    if out.get("missing"):
        pytest.skip("нет data/raw/m_001")
    assert 1 <= len(out["lines"]) <= 3
    assert [x["start_s"] for x in out["lines"]] == sorted(x["start_s"] for x in out["lines"])
    capped = raw_text_around("m_001", "00:09:37", 45, max_lines=5, max_chars=120)
    assert len(capped["lines"]) == 1 or len(capped["text"]) <= 120
    nearest = min(raw_text_around("m_001", "00:09:37", 45)["lines"], key=lambda x: abs(x["start_s"] - 577))
    assert nearest in capped["lines"], "самая близкая к таймкоду реплика остаётся, режутся дальние"


def test_traverse_default_text_is_chunk_sized():
    r = traverse("Как менялся срок релиза Zhol?")
    if not r["raw"] or all(v.get("missing") for v in r["raw"].values()):
        pytest.skip("нет data/raw")
    assert all(len(v["lines"]) <= 3 for v in r["raw"].values())
    assert all(len(v["lines"]) == 1 or len(v["text"]) <= 340 for v in r["raw"].values())


def _toy(n_dates: list[str]) -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    for i, d in enumerate(n_dates):
        g.add_node(f"x{i}", type="Decision", date=d, ts="00:00:01")
    return g


def test_arrange_chain_stays_whole_at_k_boundary():
    from src.graph.traverse import _arrange

    g = _toy(["2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04", "2026-08-05", "2026-08-06"])
    # важность: s0, звенья цепочки c1..c3, потом s4, s5. k=3: цепочка из 3 после s0 не влезает
    by_importance = ["x0", "x1", "x2", "x3", "x4", "x5"]
    chain_id = {"x1": "x1", "x2": "x1", "x3": "x1"}
    order = _arrange(g, by_importance, chain_id, k=3)
    head = set(order[:3])
    assert head & {"x1", "x2", "x3"} in (set(), {"x1", "x2", "x3"}), "цепочка не режется на границе"
    assert order[:3] == ["x0", "x4", "x5"]
    assert order[3:] == ["x1", "x2", "x3"]
    assert set(order) == set(by_importance)


def test_arrange_chain_is_contiguous_and_lifted_to_best_member():
    from src.graph.traverse import _arrange

    g = _toy(["2026-08-01", "2026-08-02", "2026-08-03", "2026-08-04"])
    by_importance = ["x2", "x0", "x3", "x1"]          # звенья x2 и x1 далеко друг от друга по важности
    order = _arrange(g, by_importance, {"x2": "x1", "x1": "x1"}, k=10)
    assert order == ["x1", "x2", "x0", "x3"], "цепочка там, где лучшее звено, звенья по датам"


def test_arrange_chain_longer_than_k_takes_most_important_links():
    from src.graph.traverse import _arrange

    g = _toy(["2026-08-01", "2026-08-02", "2026-08-03"])
    order = _arrange(g, ["x2", "x0", "x1"], {"x0": "x0", "x1": "x0", "x2": "x0"}, k=2)
    assert order[:2] == ["x0", "x2"] and order[2:] == ["x1"]


# ---------- два проекта в вопросе ----------

def _two_project_graph():
    G = nx.MultiDiGraph()
    for p in ("Alpha", "Beta"):
        G.add_node(f"project:{p}", type="Project", label=p)
    for m, projects in (("m_001", ["Alpha", "Beta"]), ("m_002", ["Beta"])):
        G.add_node(f"meeting:{m}", type="Meeting", label=m, date="2026-01-1" + m[-1])
        for p in projects:
            G.add_edge(f"meeting:{m}", f"project:{p}", key="DISCUSSES", type="DISCUSSES")
    for nid, p, m in (("d_m001_01", "Alpha", "m_001"), ("d_m001_02", "Beta", "m_001"),
                          ("d_m002_01", "Beta", "m_002")):
        G.add_node(nid, type="Decision", label=nid, statement="решили сделать", project=p, meeting_id=m,
                   date="2026-01-1" + m[-1], ts="00:00:10")
        G.add_edge(nid, f"project:{p}", key="BELONGS_TO", type="BELONGS_TO")
    return G


def test_two_projects_second_only_from_meetings_discussing_first():
    G = _two_project_graph()
    items = ["d_m001_01", "d_m001_02", "d_m002_01"]
    out = tr._link_two_projects(G, ["project:Alpha", "project:Beta"], "что по Alpha влияет на Beta", None, items)
    assert out == ["d_m001_01", "d_m001_02"]          # d_m002_01: встреча m_002 не обсуждала Alpha


def test_two_projects_order_follows_question():
    G = _two_project_graph()
    items = ["d_m001_01", "d_m001_02", "d_m002_01"]
    out = tr._link_two_projects(G, ["project:Alpha", "project:Beta"], "что по Beta влияет на Alpha", None, items)
    assert set(out) == set(items)


def test_one_project_untouched():
    G = _two_project_graph()
    items = ["d_m001_01", "d_m002_01"]
    assert tr._link_two_projects(G, ["project:Beta"], "что по Beta", None, items) == items
