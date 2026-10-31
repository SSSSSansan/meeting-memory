from pathlib import Path

import networkx as nx

from src.graph.builder import build_graph, save_graph, _load_links
from src.graph.queries import (conflicts, decision_history, load_graph, open_tasks,
                               project_decisions, project_meetings)
from src.schemas import Link

ROOT = Path(__file__).resolve().parent.parent


def _sample(links=None, aliases=None):
    links = _load_links(ROOT / "data/sample/links.json") if links is None else links
    return build_graph(ROOT / "data/extracted", ROOT / "data/transcripts", links, aliases)


def test_node_and_edge_counts():
    G = _sample()
    types = [d["type"] for _, d in G.nodes(data=True)]
    assert types.count("Decision") == 3 and types.count("Meeting") == 2
    assert types.count("Project") == 2 and types.count("Task") == 2


def test_status_is_computed_from_supersedes():
    G = _sample()
    assert G.nodes["d_m001_01"]["status"] == "superseded"
    assert G.nodes["d_m002_01"]["status"] == "active"


def test_project_decisions_chronological_and_filters():
    G = _sample()
    ids = [d["id"] for d in project_decisions(G, "Атлас")]
    assert ids == ["d_m001_01", "d_m002_01"]
    assert [d["id"] for d in project_decisions(G, "Атлас", only_active=True)] == ["d_m002_01"]
    assert [d["id"] for d in project_decisions(G, "Атлас", last_meetings=1)] == ["d_m002_01"]


def test_cross_project_meeting_counts_for_both_projects():
    G = _sample()
    assert [m["meeting_id"] for m in project_meetings(G, "Горизонт")] == ["m_001", "m_002"]


def test_decision_history_chain():
    G = _sample()
    assert [d["id"] for d in decision_history(G, "d_m002_01")] == ["d_m001_01", "d_m002_01"]
    assert [d["id"] for d in decision_history(G, "d_m001_01")] == ["d_m001_01", "d_m002_01"]


def test_open_tasks():
    G = _sample()
    assert [t["id"] for t in open_tasks(G, "Атлас")] == ["t_m001_01"]


def test_conflicts_from_contradicts_link():
    link = Link.model_validate({"from": "d_m002_02", "to": "d_m001_01", "type": "contradicts",
                                "confidence": 0.8, "rationale": "тест"})
    G = _sample(links=[link])
    cs = conflicts(G)
    assert len(cs) == 1 and cs[0]["rationale"] == "тест"
    assert G.nodes["d_m002_02"]["in_conflict"] is True
    assert conflicts(_sample()) == []


def test_low_confidence_link_is_excluded():
    link = Link.model_validate({"from": "d_m002_01", "to": "d_m001_01", "type": "supersedes",
                                "confidence": 0.3, "rationale": "слабая"})
    G = _sample(links=[link])
    assert G.nodes["d_m001_01"]["status"] == "active"


def test_aliases_merge_entities():
    G = _sample(aliases={"Горизонт": "Атлас"})
    projects = [d["label"] for _, d in G.nodes(data=True) if d["type"] == "Project"]
    assert projects == ["Атлас"]


def test_save_and_load_roundtrip(tmp_path):
    G = _sample()
    save_graph(G, tmp_path / "g.json")
    G2 = load_graph(tmp_path / "g.json")
    assert set(G.nodes) == set(G2.nodes)
    assert sorted(G.edges(keys=True)) == sorted(G2.edges(keys=True))
    assert G2.nodes["d_m001_01"]["status"] == "superseded"
    assert isinstance(G2, nx.MultiDiGraph)


def test_committed_sample_graph_is_loadable():
    G = load_graph(ROOT / "data/sample/graph.json")
    assert "d_m001_01" in G
