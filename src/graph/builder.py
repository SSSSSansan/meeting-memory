"""РОЛЬ B. Сборка графа NetworkX из результатов извлечения и списка связей.

    python -m src.graph.builder                       # ЗАГЛУШКА для роли C: ручные файлы → data/sample/graph.json
    python -m src.graph.builder --extracted data/extracted_llm \\
        --links data/graph/links.json --out data/graph/graph.json    # настоящий граф

Формат графа описан в docs/GRAPH_FORMAT.md. Читать граф удобно через src/graph/queries.py.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import networkx as nx

from src.schemas import LINK_CONFIDENCE_THRESHOLD, Extraction, Link, Transcript

ROOT = Path(__file__).resolve().parent.parent.parent

LINK_TO_EDGE = {"supersedes": "SUPERSEDES", "contradicts": "CONTRADICTS",
                "depends_on": "DEPENDS_ON", "refines": "REFINES", "same_topic": "SAME_TOPIC"}


def _canon(name: str, aliases: dict[str, str]) -> str:
    return aliases.get(name, name)


def build_graph(extracted_dir: Path, transcripts_dir: Path, links: list[Link],
                aliases: Optional[dict[str, str]] = None) -> nx.MultiDiGraph:
    """Собирает граф. aliases — {как_прозвучало: каноническое_имя} для проектов и людей (шаг склейки сущностей)."""
    aliases = aliases or {}
    G = nx.MultiDiGraph()

    def person(name: str) -> str:
        name = _canon(name, aliases)
        nid = f"person:{name}"
        if nid not in G:
            G.add_node(nid, type="Person", label=name)
        return nid

    def project(name: str) -> str:
        name = _canon(name, aliases)
        nid = f"project:{name}"
        if nid not in G:
            G.add_node(nid, type="Project", label=name)
        return nid

    for path in sorted(Path(extracted_dir).glob("*.json")):
        ext = Extraction.model_validate_json(path.read_text(encoding="utf-8"))
        tr = Transcript.model_validate_json(
            (Path(transcripts_dir) / f"{ext.meeting_id}.json").read_text(encoding="utf-8"))
        mid = f"meeting:{ext.meeting_id}"
        G.add_node(mid, type="Meeting", label=tr.title, meeting_id=ext.meeting_id,
                   date=tr.date, participants=[_canon(p, aliases) for p in tr.participants])
        for p in tr.participants:
            G.add_edge(person(p), mid, key="PARTICIPATED_IN", type="PARTICIPATED_IN")
        for pr in ext.projects_discussed:
            G.add_edge(mid, project(pr), key="DISCUSSES", type="DISCUSSES")

        common = lambda it: dict(project=_canon(it.project, aliases), meeting_id=ext.meeting_id,
                                 date=tr.date, ts=it.source.ts, quote=it.source.quote,
                                 statement=it.statement)
        for d in ext.decisions:
            G.add_node(d.id, type="Decision", label=d.statement, confidence=d.confidence,
                       made_by=_canon(d.made_by, aliases) if d.made_by else None, **common(d))
            G.add_edge(d.id, project(d.project), key="BELONGS_TO", type="BELONGS_TO")
            G.add_edge(d.id, mid, key="STATED_IN", type="STATED_IN")
        for t in ext.tasks:
            G.add_node(t.id, type="Task", label=t.statement,
                       owner=_canon(t.owner, aliases) if t.owner else None,
                       due=t.due, status=t.status.value, **common(t))
            G.add_edge(t.id, project(t.project), key="BELONGS_TO", type="BELONGS_TO")
            G.add_edge(t.id, mid, key="STATED_IN", type="STATED_IN")
            if t.owner:
                G.add_edge(t.id, person(t.owner), key="ASSIGNED_TO", type="ASSIGNED_TO")
        for q in ext.open_questions:
            G.add_node(q.id, type="OpenQuestion", label=q.statement, **common(q))
            G.add_edge(q.id, project(q.project), key="BELONGS_TO", type="BELONGS_TO")
            G.add_edge(q.id, mid, key="STATED_IN", type="STATED_IN")

    for ln in links:
        if ln.confidence < LINK_CONFIDENCE_THRESHOLD:
            continue
        if ln.from_id not in G or ln.to_id not in G:
            continue
        if G.nodes[ln.from_id]["type"] != "Decision" or G.nodes[ln.to_id]["type"] != "Decision":
            continue
        et = LINK_TO_EDGE[ln.type]
        G.add_edge(ln.from_id, ln.to_id, key=et, type=et,
                   confidence=ln.confidence, rationale=ln.rationale)

    compute_statuses(G)
    return G


def compute_statuses(G: nx.MultiDiGraph) -> None:
    """Статус решения вычисляется, а не вводится вручную.

    superseded — на решение есть входящее ребро SUPERSEDES (его заменили более новым), иначе active.
    in_conflict — решение участвует в связи CONTRADICTS.
    """
    for n, data in G.nodes(data=True):
        if data.get("type") != "Decision":
            continue
        incoming = {d["type"] for _, _, d in G.in_edges(n, data=True)}
        outgoing = {d["type"] for _, _, d in G.out_edges(n, data=True)}
        data["status"] = "superseded" if "SUPERSEDES" in incoming else "active"
        data["in_conflict"] = "CONTRADICTS" in incoming or "CONTRADICTS" in outgoing


def save_graph(G: nx.MultiDiGraph, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = nx.node_link_data(G, edges="edges")
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def _load_links(path: Path) -> list[Link]:
    if not path.exists():
        return []
    return [Link.model_validate(x) for x in json.loads(path.read_text(encoding="utf-8"))]


def _load_aliases(path: Optional[Path]) -> dict[str, str]:
    if path and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted", default="data/extracted")
    ap.add_argument("--transcripts", default="data/transcripts")
    ap.add_argument("--links", default="data/sample/links.json")
    ap.add_argument("--aliases", default=None)
    ap.add_argument("--out", default="data/sample/graph.json")
    a = ap.parse_args()

    G = build_graph(ROOT / a.extracted, ROOT / a.transcripts, _load_links(ROOT / a.links),
                    _load_aliases(ROOT / a.aliases if a.aliases else None))
    save_graph(G, ROOT / a.out)

    types: dict[str, int] = {}
    for _, d in G.nodes(data=True):
        types[d["type"]] = types.get(d["type"], 0) + 1
    edge_types: dict[str, int] = {}
    for _, _, d in G.edges(data=True):
        edge_types[d["type"]] = edge_types.get(d["type"], 0) + 1
    print(f"Узлы: {dict(sorted(types.items()))}")
    print(f"Рёбра: {dict(sorted(edge_types.items()))}")
    print(f"Сохранено: {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
