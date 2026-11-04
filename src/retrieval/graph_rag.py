"""РОЛЬ C. GraphRAG: вопрос → сущности → обход графа → подграф с цитатами в контекст.

    python -m src.retrieval.graph_rag "Как менялся срок релиза Атласа?" --depth 2

    from src.retrieval.graph_rag import retrieve_subgraph
    sub = retrieve_subgraph("Что по Атласу ещё не закрыто?", depth=2)
    sub["context"]      # текст для src.answer.answerer.generate(question, context, "graph")

Граф строит роль B (data/graph/graph.json, формат — docs/GRAPH_FORMAT.md); пока его нет,
читается заглушка data/sample/graph.json.

Обход: от узлов-сущностей (проекты и люди из вопроса) на depth шагов по рёбрам
BELONGS_TO / ASSIGNED_TO / IMPLEMENTS и связям решений (SUPERSEDES, CONTRADICTS, ...).
Через встречи не ходим: встреча связана со всем, что на ней звучало, и подграф стал бы всем графом.
Через чужие проекты и людей тоже не ходим — иначе на втором шаге подтянется весь соседний проект.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from collections import deque
from pathlib import Path
from typing import Optional

import networkx as nx

from src.graph.queries import load_graph, project_meetings
from src.utils.text import normalize

ROOT = Path(__file__).resolve().parent.parent.parent
GRAPH_PATH = ROOT / "data" / "graph" / "graph.json"
SAMPLE_GRAPH_PATH = ROOT / "data" / "sample" / "graph.json"

ITEM_TYPES = ("Decision", "Task", "OpenQuestion")
TRAVERSE_EDGES = {"BELONGS_TO", "ASSIGNED_TO", "IMPLEMENTS",
                  "SUPERSEDES", "CONTRADICTS", "DEPENDS_ON", "REFINES", "SAME_TOPIC"}
FALLBACK_MAX_ITEMS = 40  # вопрос без названных сущностей: весь граф в контекст не влезет

_NUMBERS = {"одн": 1, "две": 2, "двух": 2, "три": 3, "трех": 3, "трёх": 3, "четыр": 4,
            "пят": 5, "шест": 6, "сем": 7, "восьм": 8, "девят": 9, "десят": 10}
_LAST_N = re.compile(r"последн\w*\s+(\d+|[а-яё]+)\s+встреч", re.I)

_cache: dict[Path, tuple[float, nx.MultiDiGraph]] = {}


def default_graph_path() -> Path:
    return GRAPH_PATH if GRAPH_PATH.exists() else SAMPLE_GRAPH_PATH


def get_graph(path: Optional[Path] = None) -> nx.MultiDiGraph:
    """Граф с диска; перечитывается, только если файл изменился (роль B дописала батч)."""
    path = Path(path or default_graph_path())
    mtime = os.path.getmtime(path)
    if path not in _cache or _cache[path][0] != mtime:
        _cache[path] = (mtime, load_graph(path))
    return _cache[path][1]


# ---------- 1. Сущности из вопроса ----------

def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", normalize(text))


def _stem_match(label_token: str, word: str) -> bool:
    """«Атлас» ~ «Атласу», «Горизонт» ~ «Горизонту»: совпадает основа, расходится окончание."""
    if len(label_token) <= 3:
        return label_token == word
    common = len(os.path.commonprefix([label_token, word]))
    return common >= max(4, len(label_token) - 2) and len(word) - common <= 3


def _mentions(label: str, words: list[str], all_tokens: bool) -> bool:
    """Проект — все слова названия; человек — хватит одного слова имени длиннее трёх букв."""
    parts = _tokens(label)
    if not all_tokens:
        parts = [p for p in parts if len(p) > 3] or parts
    hits = [any(_stem_match(p, w) for w in words) for p in parts]
    return bool(parts) and (all(hits) if all_tokens else any(hits))


def find_entities(G: nx.MultiDiGraph, question: str) -> list[str]:
    words = _tokens(question)
    found = []
    for n, d in G.nodes(data=True):
        if d.get("type") == "Project" and _mentions(d.get("label", ""), words, all_tokens=True):
            found.append(n)
        elif d.get("type") == "Person" and _mentions(d.get("label", ""), words, all_tokens=False):
            found.append(n)
    return sorted(found)


def last_n_meetings(question: str) -> Optional[int]:
    m = _LAST_N.search(question)
    if not m:
        return None
    word = m.group(1).lower()
    if word.isdigit():
        return int(word)
    return next((n for stem, n in _NUMBERS.items() if word.startswith(stem)), None)


# ---------- 2. Обход ----------

def _traverse(G: nx.MultiDiGraph, seeds: list[str], depth: int) -> dict[str, int]:
    """BFS по разрешённым рёбрам в обе стороны. Возвращает {узел-пункт: на каком шаге найден}."""
    dist = {s: 0 for s in seeds}
    queue = deque(seeds)
    while queue:
        n = queue.popleft()
        if dist[n] >= depth:
            continue
        neighbours = [(v, d) for _, v, d in G.out_edges(n, data=True)] + \
                     [(u, d) for u, _, d in G.in_edges(n, data=True)]
        for m, d in neighbours:
            if d.get("type") not in TRAVERSE_EDGES or m in dist:
                continue
            if G.nodes[m].get("type") not in ITEM_TYPES:
                continue  # чужие проекты и люди — тупик
            dist[m] = dist[n] + 1
            queue.append(m)
    return {n: k for n, k in dist.items() if G.nodes[n].get("type") in ITEM_TYPES}


def _mentioned_person_items(G: nx.MultiDiGraph, person: str) -> list[str]:
    """Пункты, где человек назван в тексте, хотя ребра к нему нет: «Айгерим будет вторым бэкендером»."""
    label = G.nodes[person].get("label", "")
    out = []
    for n, d in G.nodes(data=True):
        if d.get("type") in ITEM_TYPES and n not in G.successors(person):
            text = f"{d.get('statement', '')} {d.get('quote', '')}"
            if d.get("made_by") == label or d.get("owner") == label or _mentions(label, _tokens(text), False):
                out.append(n)
    return out


def _fallback_items(G: nx.MultiDiGraph, limit: int) -> list[str]:
    """Без сущностей в вопросе: сначала конфликты, потом действующие решения и открытое, свежие раньше."""
    def priority(n: str) -> tuple:
        d = G.nodes[n]
        important = d.get("in_conflict") or d.get("status") in ("active", "open") or d["type"] == "OpenQuestion"
        return (not d.get("in_conflict"), not important, _neg(d.get("date", "")), n)

    items = [n for n, d in G.nodes(data=True) if d.get("type") in ITEM_TYPES]
    return sorted(items, key=priority)[:limit]


def _neg(date: str) -> str:
    # для сортировки «свежие раньше» без datetime: инвертируем цифры
    return date.translate(str.maketrans("0123456789", "9876543210"))


# ---------- 3. Контекст ----------

_LINK_TEXT = {"SUPERSEDES": ("заменяет", "заменено решением"),
              "CONTRADICTS": ("противоречит", "противоречит"),
              "DEPENDS_ON": ("зависит от", "от него зависит"),
              "REFINES": ("уточняет", "уточнено решением"),
              "SAME_TOPIC": ("на ту же тему, что", "на ту же тему, что")}
_KIND = {"Decision": "Решение", "Task": "Задача", "OpenQuestion": "Открытый вопрос"}


def _item_block(G: nx.MultiDiGraph, n: str, keep: set[str]) -> str:
    d = G.nodes[n]
    meeting = G.nodes.get(f"meeting:{d['meeting_id']}", {})
    head = f"{_KIND[d['type']]} {n} — встреча {d['meeting_id']} «{meeting.get('label', '')}», {d.get('date', '')}"
    marks = []
    if d["type"] == "Decision":
        marks.append("ЗАМЕНЕНО" if d.get("status") == "superseded" else "действует")
        if d.get("in_conflict"):
            marks.append("В КОНФЛИКТЕ")
        if d.get("made_by"):
            marks.append(f"решил: {d['made_by']}")
    elif d["type"] == "Task":
        marks.append({"open": "открыта", "done": "выполнена", "cancelled": "отменена"}.get(d.get("status"), d.get("status", "")))
        if d.get("owner"):
            marks.append(f"исполнитель: {d['owner']}")
        if d.get("due"):
            marks.append(f"срок: {d['due']}")
    marks = [m for m in marks if m]
    lines = [f"{head} [{'; '.join(marks)}]" if marks else head, f"Суть: {d.get('statement', '')}"]
    for _, v, e in G.out_edges(n, data=True):
        if e["type"] in _LINK_TEXT and v in G:
            lines.append(f"Связь: {_LINK_TEXT[e['type']][0]} {v}" + _rationale(e, v, keep))
    for u, _, e in G.in_edges(n, data=True):
        if e["type"] in _LINK_TEXT and e["type"] != "CONTRADICTS" and u in G:
            lines.append(f"Связь: {_LINK_TEXT[e['type']][1]} {u}" + _rationale(e, u, keep))
        elif e["type"] == "CONTRADICTS":
            lines.append(f"Связь: противоречит {u}" + _rationale(e, u, keep))
    # Строка-источник в том же формате, что у baseline: по ней answerer проверяет цитаты.
    lines.append(f"[{d['meeting_id']} {d.get('ts', '00:00:00')}] цитата: {d.get('quote', '')}")
    return "\n".join(lines)


def _rationale(edge: dict, other: str, keep: set[str]) -> str:
    out = f" ({edge['rationale']})" if edge.get("rationale") else ""
    return out + ("" if other in keep else " — этого пункта нет в контексте")


def format_context(G: nx.MultiDiGraph, items: list[str]) -> str:
    """Пункты по проектам, внутри проекта — по времени: так модель видит хронологию решений."""
    keep = set(items)
    by_project: dict[str, list[str]] = {}
    for n in items:
        by_project.setdefault(G.nodes[n].get("project", "без проекта"), []).append(n)
    blocks = []
    for project in sorted(by_project):
        nodes = sorted(by_project[project], key=lambda n: (G.nodes[n].get("date", ""), G.nodes[n].get("ts", ""), n))
        blocks.append(f"=== Проект {project} ===\n" + "\n\n".join(_item_block(G, n, keep) for n in nodes))
    return "\n\n".join(blocks)


# ---------- Точка входа ----------

def retrieve_subgraph(question: str, depth: int = 2, G: Optional[nx.MultiDiGraph] = None) -> dict:
    """Подграф по вопросу. depth — сколько шагов от сущностей (1: их пункты; 2: + связанные решения; 3: ещё шаг).

    Возвращает {"entities", "items", "last_meetings", "fallback", "context"}.
    """
    if depth < 1:
        raise ValueError("depth должен быть ≥ 1")
    G = G if G is not None else get_graph()
    seeds = find_entities(G, question)
    found = _traverse(G, seeds, depth)
    for s in seeds:
        if G.nodes[s]["type"] == "Person":
            found.update({n: 1 for n in _mentioned_person_items(G, s) if n not in found})

    fallback = not seeds
    items = _fallback_items(G, FALLBACK_MAX_ITEMS) if fallback else list(found)

    n_last = last_n_meetings(question)
    if n_last:
        projects = [G.nodes[s]["label"] for s in seeds if G.nodes[s]["type"] == "Project"]
        if projects:
            allowed = {m["meeting_id"] for p in projects for m in project_meetings(G, p)[-n_last:]}
        else:
            meetings = sorted((d.get("date", ""), d["meeting_id"]) for _, d in G.nodes(data=True)
                              if d.get("type") == "Meeting")
            allowed = {mid for _, mid in meetings[-n_last:]}
        items = [n for n in items if G.nodes[n].get("meeting_id") in allowed]

    items = sorted(items, key=lambda n: (G.nodes[n].get("date", ""), G.nodes[n].get("ts", ""), n))
    return {"entities": seeds, "items": items, "last_meetings": n_last, "fallback": fallback,
            "context": format_context(G, items)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--graph", default=None)
    a = ap.parse_args()
    sub = retrieve_subgraph(a.question, a.depth, get_graph(ROOT / a.graph if a.graph else None))
    print(f"Сущности: {sub['entities'] or '— (запасной режим)'}; пунктов: {len(sub['items'])}; "
          f"последних встреч: {sub['last_meetings']}\n")
    print(sub["context"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
