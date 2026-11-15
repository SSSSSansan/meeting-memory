"""РОЛЬ C. GraphRAG: вопрос → сущности → обход графа → подграф с цитатами в контекст.

    python -m src.retrieval.graph_rag "Как менялся срок релиза Атласа?" --depth 2

    from src.retrieval.graph_rag import retrieve_subgraph
    sub = retrieve_subgraph("Что по Атласу ещё не закрыто?", depth=2)
    sub["context"]      # текст для src.answer.answerer.generate(question, context, "graph")

Граф строит роль B (формат — docs/GRAPH_FORMAT.md). Какой файл читать — default_graph_path():
MEETING_GRAPH из .env, иначе первый существующий из data/graph/graph.json (путь из контракта),
data/graph_raw/graph.json (туда сейчас пишет scripts/build_graph.py), data/sample/graph.json (заглушка).
Рядом с графом может лежать aliases.json от resolve роли B — варианты имён тоже ищутся в вопросе.

Обход — src.graph.traverse роли B (план B6, описание — docs/GRAPH_RETRIEVAL.md): сущности и пункты по словам и
типу вопроса → цепочки замен целиком → противоречия и закрытия → остальные пункты сущностей. Этот модуль режет
результат до бюджета контекста (CONTEXT_K, как у baseline) и собирает текст для модели: статусы на дату as_of,
связи и реплики вокруг каждого пункта.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Optional

import networkx as nx

from src.graph.queries import load_graph, project_meetings
from src.resolve.entities import resolve_people, resolve_projects
from src.retrieval.baseline import CONTEXT_K, check_as_of  # один бюджет контекста и одна проверка as_of
from src.utils.text import normalize


def _not_after(date: Optional[str], as_of: Optional[str]) -> bool:
    """Пункт или встреча не позже as_of. Без даты при заданном as_of — не проходит: «» <= любая дата."""
    return as_of is None or bool(date) and date <= as_of


ROOT = Path(__file__).resolve().parent.parent.parent
GRAPH_PATH = ROOT / "data" / "graph" / "graph.json"
B_GRAPH_PATH = ROOT / "data" / "graph_raw" / "graph.json"
SAMPLE_GRAPH_PATH = ROOT / "data" / "sample" / "graph.json"
GRAPH_CANDIDATES = [GRAPH_PATH, B_GRAPH_PATH, SAMPLE_GRAPH_PATH]

ITEM_TYPES = ("Decision", "Task", "OpenQuestion")
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


# ---------- 2. Обход — src.graph.traverse роли B; здесь только тип вопроса для него ----------

def question_focus(question: str) -> Optional[str]:
    """T2 (как менялось) | T4 (конфликты) | T5 (что не закрыто) | None — по тем же словам, что роутер auto, без LLM."""
    from src.retrieval.router import _PATTERNS, _has  # локально: router сам импортирует этот модуль

    text = normalize(question)
    return next((t for t in ("T4", "T2", "T5") if _has(text, _PATTERNS[t])), None)


# ---------- 3. Контекст ----------

_LINK_TEXT = {"SUPERSEDES": ("заменяет", "заменено решением"),
              "CONTRADICTS": ("противоречит", "противоречит"),
              "DEPENDS_ON": ("зависит от", "от него зависит"),
              "REFINES": ("уточняет", "уточнено решением"),
              "SAME_TOPIC": ("на ту же тему, что", "на ту же тему, что"),
              "CLOSES": ("закрывает", "закрыта пунктом")}
_KIND = {"Decision": "Решение", "Task": "Задача", "OpenQuestion": "Открытый вопрос"}
# Состояние задачи на дату вопроса — src.graph.status.task_state роли B (B7), а не итоговый status узла.
_TASK_STATE = {"closed": "выполнена", "open": "открыта", "overdue_mentioned": "ПРОСРОЧЕНА, упоминалась позже",
               "overdue_forgotten": "ПРОСРОЧЕНА, после выдачи не упоминалась"}


def _visible(G: nx.MultiDiGraph, n: str, as_of: Optional[str]) -> bool:
    return n in G and _not_after(G.nodes[n].get("date"), as_of)


def _item_block(G: nx.MultiDiGraph, n: str, keep: set[str], as_of: Optional[str] = None,
                raw: Optional[dict] = None) -> str:
    """Пункт для модели. Статусы и связи — только по тому, что видно на as_of: итоговый граф знает про будущие
    замены и закрытия, и без отсечки «ЗАМЕНЕНО» и id будущего решения утекали бы в ответ на вопрос о прошлом."""
    from src.graph.status import task_state  # локально: status тянет traverse, а тот — этот модуль

    d = G.nodes[n]
    meeting = G.nodes.get(f"meeting:{d['meeting_id']}", {})
    head = f"{_KIND[d['type']]} {n} — встреча {d['meeting_id']} «{meeting.get('label', '')}», {d.get('date', '')}"
    ins = [(u, e) for u, _, e in G.in_edges(n, data=True) if _visible(G, u, as_of)]
    outs = [(v, e) for _, v, e in G.out_edges(n, data=True) if _visible(G, v, as_of)]
    marks = []
    if d["type"] == "Decision":
        superseded = any(e["type"] == "SUPERSEDES" for _, e in ins)
        marks.append("ЗАМЕНЕНО" if superseded else "действует")
        if any(e["type"] == "CONTRADICTS" for _, e in ins + outs):
            marks.append("В КОНФЛИКТЕ")
        if d.get("made_by"):
            marks.append(f"решил: {d['made_by']}")
    else:
        state = task_state(G, n, as_of)
        marks.append(_TASK_STATE.get(state.get("state"), state.get("state", "")))
        if d.get("owner"):
            marks.append(f"исполнитель: {d['owner']}")
        if d.get("due"):
            marks.append(f"срок: {d['due']}")
    marks = [m for m in marks if m]
    lines = [f"{head} [{'; '.join(marks)}]" if marks else head, f"Суть: {d.get('statement', '')}"]
    for v, e in outs:
        if e["type"] in _LINK_TEXT:
            lines.append(f"Связь: {_LINK_TEXT[e['type']][0]} {v}" + _rationale(e, v, keep))
    for u, e in ins:
        if e["type"] == "CONTRADICTS":
            lines.append(f"Связь: противоречит {u}" + _rationale(e, u, keep))
        elif e["type"] in _LINK_TEXT:
            lines.append(f"Связь: {_LINK_TEXT[e['type']][1]} {u}" + _rationale(e, u, keep))
    # Строка-источник в том же формате, что у baseline: по ней answerer проверяет цитаты.
    lines.append(f"[{d['meeting_id']} {d.get('ts', '00:00:00')}] цитата: {d.get('quote', '')}")
    text = (raw or {}).get(n, {}).get("text")
    if text:  # третий слой обхода B: реплики вокруг пункта — то, что экстракт мог не поймать
        lines.append("Реплики рядом:\n" + text)
    return "\n".join(lines)


def _rationale(edge: dict, other: str, keep: set[str]) -> str:
    out = f" ({edge['rationale']})" if edge.get("rationale") else ""
    return out + ("" if other in keep else " — этого пункта нет в контексте")


def format_context(G: nx.MultiDiGraph, items: list[str], as_of: Optional[str] = None,
                   raw: Optional[dict] = None) -> str:
    """Пункты по проектам, внутри проекта — по времени: так модель видит хронологию решений."""
    keep = set(items)
    by_project: dict[str, list[str]] = {}
    for n in items:
        by_project.setdefault(G.nodes[n].get("project", "без проекта"), []).append(n)
    blocks = []
    for project in sorted(by_project):
        nodes = sorted(by_project[project], key=lambda n: (G.nodes[n].get("date", ""), G.nodes[n].get("ts", ""), n))
        blocks.append(f"=== Проект {project} ===\n" +
                      "\n\n".join(_item_block(G, n, keep, as_of, raw) for n in nodes))
    return "\n\n".join(blocks)


@lru_cache(maxsize=64)
def _raw_title(meeting_id: str) -> Optional[str]:
    path = ROOT / "data" / "raw" / f"{meeting_id}.meeting.json"
    try:
        return json.loads(path.read_text(encoding="utf-8")).get("title")
    except (OSError, json.JSONDecodeError):
        return None


def _raw_matches(G: nx.MultiDiGraph, meeting_id: str) -> bool:
    """Реплики из data/raw подходят графу, только если это та же встреча: у заглушки data/sample те же id m_001,
    m_002, но другой сюжет (Атлас), и чужие реплики сбивали бы проверку цитат."""
    label = G.nodes.get(f"meeting:{meeting_id}", {}).get("label")
    return bool(label) and label == _raw_title(meeting_id)


# ---------- Точка входа ----------

def retrieve_subgraph(question: str, depth: int = 2, G: Optional[nx.MultiDiGraph] = None,
                      aliases: Optional[dict[str, str]] = None, as_of: Optional[str] = None,
                      max_items: Optional[int] = CONTEXT_K, with_text: bool = True) -> dict:
    """Подграф по вопросу — обход роли B (src.graph.traverse, план B6) с бюджетом контекста режима graph.

    Обход B: стартовые сущности и пункты по словам и типу вопроса → цепочки замен целиком → связи (противоречия,
    закрытия, уточнения) → остальные пункты сущностей; первые max_items уже собраны так, чтобы цепочка не резалась.
    Здесь — обрезка до бюджета, фильтр «последние N встреч» и контекст для модели: статусы на дату as_of, связи и
    реплики вокруг каждого пункта (третий слой обхода, ±45 с, размером с чанк baseline).

    Без G читается граф по умолчанию вместе с его aliases.json. as_of (ISO-дата) — только пункты встреч не позже неё.
    max_items — бюджет контекста в пунктах, по умолчанию CONTEXT_K, как у baseline в чанках; None — без обрезки.
    with_text=False — без реплик вокруг пунктов (рисунок графа в UI).
    Возвращает {"entities", "items", "n_found", "last_meetings", "fallback", "context", "raw", "details"};
    n_found — сколько пунктов нашёл обход до обрезки, details — почему пункт попал (ярус обхода B).
    """
    from src.graph.traverse import traverse

    if depth < 1:
        raise ValueError("depth должен быть ≥ 1")
    check_as_of(as_of)
    if G is None:
        G, aliases = get_graph(), get_aliases() if aliases is None else aliases
    r = traverse(question, G=G, aliases=aliases, as_of=as_of, depth=depth, with_text=False, k=max_items)
    return graph_context(G, question, r, as_of, max_items, with_text)


def graph_context(G: nx.MultiDiGraph, question: str, traversal: dict, as_of: Optional[str],
                  max_items: Optional[int], with_text: bool = True) -> dict:
    """Результат traverse() → то, что уходит модели: фильтр «последние N встреч», обрезка до max_items, реплики
    вокруг пунктов, текст со статусами на as_of. Общий для режима graph и графовой части hybrid."""
    from src.graph.traverse import LINES_PER_ITEM, MAX_CHARS, WINDOW_S, raw_text_around

    seeds, fallback, items = traversal["entities"], traversal["fallback"], list(traversal["items"])
    n_last = last_n_meetings(question)
    if n_last:
        projects = [G.nodes[s]["label"] for s in seeds if G.nodes[s]["type"] == "Project"]
        if projects:
            allowed = {m["meeting_id"] for p in projects
                       for m in [m for m in project_meetings(G, p)
                                 if _not_after(m.get("date"), as_of)][-n_last:]}
        else:
            meetings = sorted((d.get("date", ""), d["meeting_id"]) for _, d in G.nodes(data=True)
                              if d.get("type") == "Meeting" and _not_after(d.get("date"), as_of))
            allowed = {mid for _, mid in meetings[-n_last:]}
        items = [n for n in items if G.nodes[n].get("meeting_id") in allowed]

    n_found = len(items)
    kept = items if max_items is None else items[:max_items]
    raw = {}
    if with_text:
        for n in kept:
            d = G.nodes[n]
            if d.get("meeting_id") and d.get("ts") and _raw_matches(G, d["meeting_id"]):
                raw[n] = raw_text_around(d["meeting_id"], d["ts"], WINDOW_S, None, LINES_PER_ITEM, MAX_CHARS)
    kept = sorted(kept, key=lambda n: (G.nodes[n].get("date", ""), G.nodes[n].get("ts", ""), n))
    return {"entities": seeds, "items": kept, "n_found": n_found, "last_meetings": n_last, "fallback": fallback,
            "context": format_context(G, kept, as_of, raw), "raw": raw,
            "details": {n: traversal["details"].get(n, {}) for n in kept}}


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
