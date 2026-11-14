"""SQLite-хранилище отдаёт то же, что src/graph/queries.py на NetworkX (на нашем графе), плюс окно реплик и алиасы."""
from __future__ import annotations

from pathlib import Path

import networkx as nx
import pytest

from src.graph import queries as q
from src.store import sqlite_store as st

ROOT = Path(__file__).resolve().parent.parent
GRAPH = ROOT / "data" / "graph" / "graph.json"
RAW = ROOT / "data" / "raw"


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    if not GRAPH.exists():
        pytest.skip("нет data/graph/graph.json")
    db = st.build_db(GRAPH, RAW, tmp_path_factory.mktemp("store") / "t.sqlite")
    conn = st.connect(db)
    yield q.load_graph(GRAPH), conn
    conn.close()


def _small_graph():
    G = nx.MultiDiGraph()
    G.add_node("project:P", type="Project", label="P")
    G.add_node("meeting:m_001", type="Meeting", label="m_001", meeting_id="m_001", date="2026-01-01")
    G.add_edge("meeting:m_001", "project:P", key="DISCUSSES", type="DISCUSSES")
    G.add_node("d_m001_01", type="Decision", project="P", meeting_id="m_001", date="2026-01-01", ts="00:01:00",
               statement="a", status="superseded")
    G.add_node("d_m002_01", type="Decision", project="P", meeting_id="m_002", date="2026-01-08", ts="00:02:00",
               statement="b", status="active")
    for n in ("d_m001_01", "d_m002_01"):
        G.add_edge(n, "project:P", key="BELONGS_TO", type="BELONGS_TO")
    G.add_edge("d_m002_01", "d_m001_01", key="SUPERSEDES", type="SUPERSEDES", confidence=0.9, rationale="r")
    G.add_node("t_m001_01", type="Task", project="P", meeting_id="m_001", date="2026-01-01", ts="00:03:00",
               statement="c", status="open", due="2026-01-05", owner="Анна")
    G.add_edge("t_m001_01", "project:P", key="BELONGS_TO", type="BELONGS_TO")
    return G


def _build_small(tmp_path):
    import json
    p = tmp_path / "g.json"
    p.write_text(json.dumps(nx.node_link_data(_small_graph(), edges="edges"), ensure_ascii=False), encoding="utf-8")
    db = st.build_db(p, tmp_path / "no_raw", tmp_path / "s.sqlite", aliases_path=tmp_path / "none.json")
    return st.connect(db)


def test_small_graph_queries_match_networkx(tmp_path):
    G, conn = _small_graph(), _build_small(tmp_path)
    assert st.project_decisions(conn, "P") == q.project_decisions(G, "P")
    assert st.project_decisions(conn, "P", only_active=True) == q.project_decisions(G, "P", only_active=True)
    assert st.project_decisions(conn, "P", last_meetings=1) == q.project_decisions(G, "P", last_meetings=1)
    assert st.decision_history(conn, "d_m002_01") == q.decision_history(G, "d_m002_01")
    assert st.decision_history(conn, "nope") == []
    assert st.open_tasks(conn) == q.open_tasks(G)
    assert [t["id"] for t in st.overdue_tasks(conn, "2026-01-10")] == ["t_m001_01"]
    assert st.overdue_tasks(conn, "2026-01-02") == []                     # срок ещё не прошёл


def test_real_graph_project_queries_match(real):
    G, conn = real
    for p in sorted(d["label"] for _, d in G.nodes(data=True) if d.get("type") == "Project"):
        assert st.project_meetings(conn, p) == q.project_meetings(G, p)
        assert st.project_decisions(conn, p) == q.project_decisions(G, p)
        assert st.project_decisions(conn, p, last_meetings=2) == q.project_decisions(G, p, last_meetings=2)
        assert st.project_decisions(conn, p, only_active=True) == q.project_decisions(G, p, only_active=True)
        assert st.open_tasks(conn, p) == q.open_tasks(G, p)
    assert st.open_tasks(conn) == q.open_tasks(G)
    assert st.project_decisions(conn, "нет такого") == []


def test_real_graph_history_and_conflicts_match(real):
    G, conn = real
    for n, d in G.nodes(data=True):
        if d.get("type") == "Decision":
            assert st.decision_history(conn, n) == q.decision_history(G, n), n
    key = lambda c: (c["a"]["id"], c["b"]["id"])  # noqa: E731  порядок рёбер в NetworkX зависит от порядка узлов
    assert sorted(st.conflicts(conn), key=key) == sorted(q.conflicts(G), key=key)
    assert sorted(st.conflicts(conn, "Zhol"), key=key) == sorted(q.conflicts(G, "Zhol"), key=key)


def test_real_graph_counts_equal(real):
    G, conn = real
    assert conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == G.number_of_nodes()
    assert conn.execute("SELECT COUNT(*) FROM edges").fetchone()[0] == G.number_of_edges()


def test_real_graph_overdue_tasks_follow_the_rule(real):
    G, conn = real
    as_of = "2026-09-10"
    expected = sorted(n for n, d in G.nodes(data=True) if d.get("type") == "Task" and d.get("status") == "open"
                      and d.get("due") and d["due"] < as_of and d["date"] <= as_of)
    assert sorted(t["id"] for t in st.overdue_tasks(conn, as_of)) == expected


def test_replicas_window_equals_raw_text_around(real):
    from src.graph.traverse import raw_text_around
    _, conn = real
    if not RAW.exists() or conn.execute("SELECT COUNT(*) FROM replicas").fetchone()[0] == 0:
        pytest.skip("нет data/raw")
    for mid, ts in (("m_001", "00:09:37"), ("m_006", "00:06:02"), ("m_011", "00:05:53")):
        h, m, s = (int(x) for x in ts.split(":"))
        want = raw_text_around(mid, ts, 45, RAW)["lines"]
        got = st.replicas_window(conn, mid, h * 3600 + m * 60 + s, 45)
        assert [(x["ts"], x["speaker"], x["text"]) for x in got] == [(x["ts"], x["speaker"], x["text"]) for x in want]
        near = st.replicas_window(conn, mid, h * 3600 + m * 60 + s, 45, max_lines=2)
        assert [x["ts"] for x in near] == [x["ts"] for x in raw_text_around(mid, ts, 45, RAW, max_lines=2)["lines"]]


def test_replicas_keep_timecode_in_seconds(real):
    _, conn = real
    if conn.execute("SELECT COUNT(*) FROM replicas").fetchone()[0] == 0:
        pytest.skip("нет data/raw")
    bad = conn.execute("SELECT COUNT(*) FROM replicas WHERE start_s IS NULL OR end_s < start_s").fetchone()[0]
    assert bad == 0


def test_find_entity_by_alias(real):
    _, conn = real
    assert st.find_entity(conn, "Zhol")["canonical"] == "Zhol"
    assert st.find_entity(conn, "Коля")["canonical"] == "Николай"           # прежняя запись aliases.json
    assert st.find_entity(conn, "  АБДУ ")["canonical"] == "Абдурахман"      # нормализация регистра и пробелов
    assert st.find_entity(conn, "нет такого проекта") is None


def test_alias_variants_are_strict():
    from src.store.aliases import is_variant
    assert is_variant("Даниярдың", "Данияр") and is_variant("Абду", "Абдурахман") and is_variant("қамқор", "Qamqor")
    assert is_variant("Zhol-ға", "Zhol") and is_variant("жол", "Zhol")
    assert not is_variant("жел", "Zhol")          # похожее по звучанию слово не должно становиться алиасом
    assert not is_variant("день", "Данияр") and not is_variant("данные", "Данияр")
    assert not is_variant("Данияр", "Данияр")
