"""B6: структура графа, собранного из всех встреч (data/graph_raw/graph.json)."""
import json
from pathlib import Path

import networkx as nx
import pytest

PATH = Path(__file__).resolve().parent.parent / "data" / "graph_raw" / "graph.json"
pytestmark = pytest.mark.skipif(not PATH.exists(), reason="граф не собран: python scripts/build_graph.py")


def load():
    return nx.node_link_graph(json.loads(PATH.read_text(encoding="utf-8")), edges="edges")


def test_node_types_and_meetings():
    G = load()
    types = {d["type"] for _, d in G.nodes(data=True)}
    assert {"Meeting", "Person", "Project", "Decision", "Task", "OpenQuestion"} <= types
    assert sum(1 for _, d in G.nodes(data=True) if d["type"] == "Meeting") == 10


def test_every_item_has_project_meeting_and_status():
    G = load()
    for n, d in G.nodes(data=True):
        if d["type"] in ("Decision", "Task", "OpenQuestion"):
            out = {e["type"] for _, _, e in G.out_edges(n, data=True)}
            assert {"BELONGS_TO", "STATED_IN"} <= out, n
        if d["type"] == "Decision":
            assert d["status"] in ("active", "superseded")


def test_no_speaker_labels_and_no_duplicate_entities():
    G = load()
    labels = [d["label"] for _, d in G.nodes(data=True) if d["type"] in ("Person", "Project")]
    assert not any("peaker" in x.lower() for x in labels)
    assert len(labels) == len(set(labels))
