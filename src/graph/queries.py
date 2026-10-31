"""Удобное чтение графа (для роли C и для оценки).

    from src.graph.queries import load_graph, project_decisions, decision_history, conflicts, open_tasks
    G = load_graph("data/sample/graph.json")
    project_decisions(G, "Атлас", last_meetings=3)   # «что решили по Атласу за последние три встречи»
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import networkx as nx


def load_graph(path) -> nx.MultiDiGraph:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return nx.node_link_graph(data, edges="edges", directed=True, multigraph=True)


def _node(G, nid: str) -> dict:
    return {"id": nid, **G.nodes[nid]}


def _sort_key(d: dict):
    return (d.get("date", ""), d.get("ts", ""), d["id"])


def project_meetings(G, project: str) -> list[dict]:
    """Встречи, на которых обсуждался проект (включая кросс-проектные), по дате."""
    pid = f"project:{project}"
    if pid not in G:
        return []
    meetings = [_node(G, u) for u, _, d in G.in_edges(pid, data=True) if d["type"] == "DISCUSSES"]
    return sorted(meetings, key=_sort_key)


def project_decisions(G, project: str, last_meetings: Optional[int] = None,
                      only_active: bool = False) -> list[dict]:
    """Решения по проекту в хронологическом порядке.

    last_meetings=N — только из N последних встреч, где обсуждался проект.
    Решение относится к проекту по полю project, а не по названию встречи.
    """
    pid = f"project:{project}"
    if pid not in G:
        return []
    allowed = None
    if last_meetings is not None:
        allowed = {m["meeting_id"] for m in project_meetings(G, project)[-last_meetings:]}
    out = []
    for u, _, d in G.in_edges(pid, data=True):
        if d["type"] != "BELONGS_TO" or G.nodes[u]["type"] != "Decision":
            continue
        node = _node(G, u)
        if allowed is not None and node["meeting_id"] not in allowed:
            continue
        if only_active and node["status"] != "active":
            continue
        out.append(node)
    return sorted(out, key=_sort_key)


def decision_history(G, decision_id: str) -> list[dict]:
    """Цепочка «заменяет» от самого раннего решения к самому позднему (через SUPERSEDES в обе стороны)."""
    if decision_id not in G:
        return []
    seen, stack = {decision_id}, [decision_id]
    while stack:
        n = stack.pop()
        neigh = [v for _, v, d in G.out_edges(n, data=True) if d["type"] == "SUPERSEDES"]
        neigh += [u for u, _, d in G.in_edges(n, data=True) if d["type"] == "SUPERSEDES"]
        for x in neigh:
            if x not in seen:
                seen.add(x)
                stack.append(x)
    return sorted((_node(G, n) for n in seen), key=_sort_key)


def conflicts(G, project: Optional[str] = None) -> list[dict]:
    """Пары противоречащих решений (CONTRADICTS) с обоснованием."""
    out = []
    for u, v, d in G.edges(data=True):
        if d["type"] != "CONTRADICTS":
            continue
        a, b = _node(G, u), _node(G, v)
        if project and project not in (a["project"], b["project"]):
            continue
        out.append({"a": a, "b": b, "confidence": d.get("confidence"), "rationale": d.get("rationale")})
    return out


def open_tasks(G, project: Optional[str] = None) -> list[dict]:
    """Задачи со статусом open (опционально по проекту)."""
    out = [_node(G, n) for n, d in G.nodes(data=True)
           if d.get("type") == "Task" and d.get("status") == "open"
           and (project is None or d["project"] == project)]
    return sorted(out, key=_sort_key)
