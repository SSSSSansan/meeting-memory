"""РОЛЬ C. GraphRAG: вопрос → сущности → обход графа → подграф с цитатами в контекст.

    python -m src.retrieval.graph_rag "Как менялся срок релиза Атласа?" --depth 2

    from src.retrieval.graph_rag import retrieve_subgraph
    sub = retrieve_subgraph("Что по Атласу ещё не закрыто?", depth=2)
    sub["context"]      # текст для src.answer.answerer.generate(question, context, "graph")

Граф строит роль B (формат — docs/GRAPH_FORMAT.md). Какой файл читать — default_graph_path():
MEETING_GRAPH из .env, иначе первый существующий из data/graph/graph.json (путь из контракта),
data/graph_raw/graph.json (туда сейчас пишет scripts/build_graph.py), data/sample/graph.json (заглушка).
Рядом с графом может лежать aliases.json от resolve роли B — варианты имён тоже ищутся в вопросе.

Обход: от узлов-сущностей (проекты и люди из вопроса) на depth шагов по рёбрам
BELONGS_TO / ASSIGNED_TO / IMPLEMENTS и связям решений (SUPERSEDES, CONTRADICTS, ...).
Через встречи не ходим: встреча связана со всем, что на ней звучало, и подграф стал бы всем графом.
Через чужие проекты и людей тоже не ходим — иначе на втором шаге подтянется весь соседний проект.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import deque
from pathlib import Path
from typing import Optional

import networkx as nx

from src.graph.queries import load_graph, project_meetings
from src.resolve.entities import resolve_people, resolve_projects
from src.retrieval.baseline import CONTEXT_K  # один бюджет контекста на все режимы
from src.utils.text import normalize

ROOT = Path(__file__).resolve().parent.parent.parent
GRAPH_PATH = ROOT / "data" / "graph" / "graph.json"
B_GRAPH_PATH = ROOT / "data" / "graph_raw" / "graph.json"
SAMPLE_GRAPH_PATH = ROOT / "data" / "sample" / "graph.json"
GRAPH_CANDIDATES = [GRAPH_PATH, B_GRAPH_PATH, SAMPLE_GRAPH_PATH]

ITEM_TYPES = ("Decision", "Task", "OpenQuestion")
TRAVERSE_EDGES = {"BELONGS_TO", "ASSIGNED_TO", "IMPLEMENTS",
                  "SUPERSEDES", "CONTRADICTS", "DEPENDS_ON", "REFINES", "SAME_TOPIC"}
FALLBACK_MAX_ITEMS = 40  # вопрос без названных сущностей: весь граф в контекст не влезет

_NUMBERS = {"одн": 1, "две": 2, "двух": 2, "три": 3, "трех": 3, "трёх": 3, "четыр": 4,
            "пят": 5, "шест": 6, "сем": 7, "восьм": 8, "девят": 9, "десят": 10}
_LAST_N = re.compile(r"последн\w*\s+(\d+|[а-яё]+)\s+встреч", re.I)

_cache: dict[Path, tuple[float, object]] = {}
_override: Optional[Path] = None


def use_graph_path(path: Optional[Path]) -> None:
    """Выбор графа из UI; None — обычный порядок (MEETING_GRAPH, затем GRAPH_CANDIDATES)."""
    global _override
    _override = Path(path) if path else None


def default_graph_path() -> Path:
    if _override is not None:
        return _override
    env = os.environ.get("MEETING_GRAPH")
    if env:
        return Path(env) if Path(env).is_absolute() else ROOT / env
    return next((p for p in GRAPH_CANDIDATES if p.exists()), SAMPLE_GRAPH_PATH)


def _cached(path: Path, load):
    """Перечитывает файл, только если он изменился (роль B дописала батч)."""
    mtime = os.path.getmtime(path)
    if path not in _cache or _cache[path][0] != mtime:
        _cache[path] = (mtime, load(path))
    return _cache[path][1]


def get_graph(path: Optional[Path] = None) -> nx.MultiDiGraph:
    return _cached(Path(path or default_graph_path()), load_graph)


def get_aliases(graph_path: Optional[Path] = None) -> dict[str, str]:
    """{как_прозвучало: каноническое_имя} из aliases.json рядом с графом; нет файла — пусто."""
    path = Path(graph_path or default_graph_path()).with_name("aliases.json")
    if not path.exists():
        return {}
    return _cached(path, lambda p: json.loads(p.read_text(encoding="utf-8")))


# ---------- 1. Сущности из вопроса ----------

def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", normalize(text))


def _stem_match(label_token: str, word: str) -> bool:
    """«Атлас» ~ «Атласу», «Горизонт» ~ «Горизонту»: совпадает основа, расходится окончание."""
    if len(label_token) <= 3:
        return label_token == word
    common = len(os.path.commonprefix([label_token, word]))
    # у коротких имён склоняется последняя буква: «Коля» → «Коли», «Коле»
    need = len(label_token) - 1 if len(label_token) <= 5 else max(4, len(label_token) - 2)
    return common >= need and len(word) - common <= 3


def _mentions(label: str, words: list[str], all_tokens: bool) -> bool:
    """Проект — все слова названия; человек — хватит одного слова имени длиннее трёх букв."""
    parts = _tokens(label)
    if not all_tokens:
        parts = [p for p in parts if len(p) > 3] or parts
    hits = [any(_stem_match(p, w) for w in words) for p in parts]
    return bool(parts) and (all(hits) if all_tokens else any(hits))


def _proper_nouns(question: str) -> list[str]:
    """Слова с заглавной не в начале предложения.

    Транслит и нечёткое сравнение resolve цепляют и обычные слова («Жалко» → Zhol), поэтому
    им отдаём только вероятные имена; точное название в начале фразы найдёт и сравнение по основе.
    """
    out = []
    for sentence in re.split(r"[.!?]+", unicodedata.normalize("NFC", question)):
        out += [w for w in re.findall(r"\w+", sentence)[1:] if len(w) >= 3 and w[0].isupper()]
    return out


def find_entities(G: nx.MultiDiGraph, question: str, aliases: Optional[dict[str, str]] = None) -> list[str]:
    """Проекты и люди из вопроса.

    1. Название узла или его вариант из aliases.json — по основе слова: «Атласу» → Атлас, «Коле» → Николай.
    2. Слова с заглавной — через resolve роли B (транслит + нечёткое сравнение): «Жол» → Zhol, «Нуралы» → Нұралы.
    """
    words = _tokens(question)
    by_label = {"Project": {}, "Person": {}}
    for n, d in G.nodes(data=True):
        if d.get("type") in by_label:
            by_label[d["type"]][d.get("label", "")] = n
    variants = {label: [label] for labels in by_label.values() for label in labels}
    for alias, canon in (aliases or {}).items():
        if canon in variants:
            variants[canon].append(alias)

    found = set()
    for kind, labels in by_label.items():
        for label, n in labels.items():
            if any(_mentions(v, words, all_tokens=kind == "Project") for v in variants[label]):
                found.add(n)
    names = _proper_nouns(question)
    if names:
        for kind, resolve in (("Project", resolve_projects), ("Person", resolve_people)):
            for canon in resolve(names, list(by_label[kind])).values():
                if canon in by_label[kind]:
                    found.add(by_label[kind][canon])
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


def _chain(G: nx.MultiDiGraph, n: str, pool: set[str]) -> list[str]:
    """Вся цепочка замен вокруг пункта (SUPERSEDES в обе стороны), только среди кандидатов pool."""
    seen, stack = {n}, [n]
    while stack:
        m = stack.pop()
        nbrs = [v for _, v, e in G.out_edges(m, data=True) if e.get("type") == "SUPERSEDES"] + \
               [u for u, _, e in G.in_edges(m, data=True) if e.get("type") == "SUPERSEDES"]
        for x in nbrs:
            if x in pool and x not in seen:
                seen.add(x)
                stack.append(x)
    return sorted(seen, key=lambda x: (G.nodes[x].get("date", ""), G.nodes[x].get("ts", ""), x))


def select_items(G: nx.MultiDiGraph, ranked: list[str], limit: Optional[int]) -> list[str]:
    """Не больше limit пунктов: бюджет контекста тот же, что у baseline (CONTEXT_K), иначе граф выигрывал бы объёмом.

    ranked — кандидаты по важности. Пункт идёт вместе со всей своей цепочкой замен, если она помещается:
    без середины цепочки ответ на «как менялся срок» неполон при формально верных цитатах (план, B6).
    Цепочка, которая не влезает целиком, пропускается в пользу следующих пунктов; если не влезает и в пустой
    контекст — берётся её начало по важности.
    """
    if limit is None or len(ranked) <= limit:
        return list(ranked)
    pool, out = set(ranked), []
    for n in ranked:
        if n in out:
            continue
        unit = [m for m in _chain(G, n, pool) if m not in out]
        if len(out) + len(unit) <= limit:
            out += unit
        elif not out:
            out = [m for m in ranked if m in set(unit)][:limit]
        if len(out) >= limit:
            break
    return out[:limit]


def question_focus(question: str) -> Optional[str]:
    """T2 (как менялось) | T4 (конфликты) | T5 (что не закрыто) | None — по тем же словам, что роутер auto, без LLM."""
    from src.retrieval.router import _PATTERNS, _has  # локально: router сам импортирует этот модуль

    text = normalize(question)
    return next((t for t in ("T4", "T2", "T5") if _has(text, _PATTERNS[t])), None)


def _rank(G: nx.MultiDiGraph, dist: dict[str, int], focus: Optional[str] = None) -> list[str]:
    """Важнее: то, о чём спрашивают (focus), затем ближе к сущностям вопроса, в конфликте, действующее или
    открытое, свежее.

    Без focus на «как менялся срок» конфликты и действующие пункты обгоняли заменённые решения, и в маленький
    бюджет hybrid цепочка замен не попадала: на q_05 графовая часть покрывала 1 встречу-источник из 4.
    """
    pool = set(dist)

    def asked(n: str) -> bool:
        d = G.nodes[n]
        if focus == "T2":
            return len(_chain(G, n, pool)) > 1
        if focus == "T4":
            return bool(d.get("in_conflict"))
        if focus == "T5":
            return d["type"] == "OpenQuestion" or (d["type"] == "Task" and d.get("status") == "open")
        return True

    def key(n: str) -> tuple:
        d = G.nodes[n]
        current = d.get("status") in ("active", "open") or d["type"] == "OpenQuestion"
        return (not asked(n), dist[n], not d.get("in_conflict"), not current, _neg(d.get("date", "")), n)
    return sorted(dist, key=key)


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
_TASK_STATUS = {"open": "открыта", "done": "выполнена", "cancelled": "отменена"}


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
        marks.append(_TASK_STATUS.get(d.get("status"), d.get("status", "")))
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

def retrieve_subgraph(question: str, depth: int = 2, G: Optional[nx.MultiDiGraph] = None,
                      aliases: Optional[dict[str, str]] = None, as_of: Optional[str] = None,
                      max_items: Optional[int] = CONTEXT_K) -> dict:
    """Подграф по вопросу. depth — сколько шагов от сущностей (1: их пункты; 2: + связанные решения; 3: ещё шаг).

    Без G читается граф по умолчанию вместе с его aliases.json.
    as_of (ISO-дата) — только пункты встреч не позже неё, как у baseline.
    max_items — бюджет контекста в пунктах, по умолчанию CONTEXT_K, как у baseline в чанках; None — без обрезки.
    Возвращает {"entities", "items", "n_found", "last_meetings", "fallback", "context"}; n_found — сколько
    пунктов нашёл обход до обрезки.
    """
    if depth < 1:
        raise ValueError("depth должен быть ≥ 1")
    if G is None:
        G, aliases = get_graph(), get_aliases() if aliases is None else aliases
    seeds = find_entities(G, question, aliases)
    found = _traverse(G, seeds, depth)
    for s in seeds:
        if G.nodes[s]["type"] == "Person":
            found.update({n: 1 for n in _mentioned_person_items(G, s) if n not in found})

    fallback = not seeds
    if fallback:
        items = [n for n in _fallback_items(G, len(G)) if as_of is None or G.nodes[n].get("date", "") <= as_of]
        items = items[:FALLBACK_MAX_ITEMS]
        dist = {n: i for i, n in enumerate(items)}  # порядок важности _fallback_items
    else:
        items = [n for n in found if as_of is None or G.nodes[n].get("date", "") <= as_of]
        dist = {n: found[n] for n in items}

    n_last = last_n_meetings(question)
    if n_last:
        projects = [G.nodes[s]["label"] for s in seeds if G.nodes[s]["type"] == "Project"]
        if projects:
            allowed = {m["meeting_id"] for p in projects
                       for m in [m for m in project_meetings(G, p)
                                 if as_of is None or m.get("date", "") <= as_of][-n_last:]}
        else:
            meetings = sorted((d.get("date", ""), d["meeting_id"]) for _, d in G.nodes(data=True)
                              if d.get("type") == "Meeting" and (as_of is None or d.get("date", "") <= as_of))
            allowed = {mid for _, mid in meetings[-n_last:]}
        items = [n for n in items if G.nodes[n].get("meeting_id") in allowed]

    n_found = len(items)
    items = select_items(G, _rank(G, {n: dist[n] for n in items}, question_focus(question)), max_items)
    items = sorted(items, key=lambda n: (G.nodes[n].get("date", ""), G.nodes[n].get("ts", ""), n))
    return {"entities": seeds, "items": items, "n_found": n_found, "last_meetings": n_last, "fallback": fallback,
            "context": format_context(G, items)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--graph", default=None, help="по умолчанию — default_graph_path()")
    a = ap.parse_args()
    path = ROOT / a.graph if a.graph else default_graph_path()
    sub = retrieve_subgraph(a.question, a.depth, get_graph(path), get_aliases(path))
    print(f"Граф: {path.relative_to(ROOT).as_posix()}")
    print(f"Сущности: {sub['entities'] or '— (запасной режим)'}; пунктов: {len(sub['items'])}; "
          f"последних встреч: {sub['last_meetings']}\n")
    print(sub["context"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
