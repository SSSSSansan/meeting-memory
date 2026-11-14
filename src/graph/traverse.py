"""РОЛЬ B. Обход графа (план B6): стартовые сущности → расширение по рёбрам → сырой текст вокруг узла.

    from src.graph.traverse import traverse
    r = traverse("Как менялся срок релиза Zhol?", as_of="2026-10-26")
    r["items"]      # id пунктов, уже упорядоченные по важности (обрезку до CONTEXT_K делает вызывающий)
    r["details"]    # на каждый пункт: tier, почему попал, chain_id, оценка по тексту вопроса
    r["raw"]        # на каждый пункт: реплики ±45 с вокруг его таймкода, из data/raw

Три слоя, каждый — отдельная функция с финальным именем:
  find_start_entities(G, question, aliases, as_of) — проекты/люди из вопроса и пункты, прямо относящиеся к вопросу
  expand(G, seeds, as_of, depth)                   — расширение ТОЛЬКО по связям между пунктами
  raw_text_around(meeting_id, ts, window_s)         — реплики транскрипта вокруг таймкода

Почему расширение не идёт по BELONGS_TO. Связь «принадлежит проекту» нужна, чтобы найти стартовые пункты проекта.
Если по ней же расширяться, то любой пункт проекта тянет все остальные пункты проекта, и «подграф» превращается
в весь проект (так и было в первой версии: 35 пунктов из 35 при бюджете 10).

Порядок результата (вызывающий режет по первым K):
  1. пункты про сущности из вопроса и пункты, прямо подходящие к вопросу по словам или по типу вопроса;
  2. цепочки замен (SUPERSEDES) этих пунктов целиком, цепочка идёт подряд по датам;
  3. связанные через CONTRADICTS и CLOSES (затем REFINES, DEPENDS_ON, IMPLEMENTS);
  4. остальные пункты тех же сущностей, свежие раньше.

Отсечка по as_of — та же, что у baseline: пункт без даты при заданном as_of не проходит, цепочка замен строится
только из пунктов, видимых на эту дату, поэтому будущее в неё не попадает.

Поиск пунктов по словам вопроса — лексический (основы слов, вес по редкости), а не по эмбеддингам: он
детерминирован, бесплатен и не требует платных вызовов. План допускал эмбеддинги; это ограничение названо в docs.
"""
from __future__ import annotations

import math
import os
import re
from collections import deque
from functools import lru_cache
from pathlib import Path
from typing import Optional

import networkx as nx

from src.utils.text import normalize

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / "data" / "raw"

ITEM_TYPES = ("Decision", "Task", "OpenQuestion")
# BELONGS_TO сюда не входит намеренно (см. докстринг модуля). SAME_TOPIC тоже: это связь «на ту же тему», не причинная.
EXPAND_EDGES = ("SUPERSEDES", "CONTRADICTS", "CLOSES", "REFINES", "DEPENDS_ON", "IMPLEMENTS")
_EDGE_RANK = {"SUPERSEDES": 0, "CONTRADICTS": 1, "CLOSES": 1, "REFINES": 2, "DEPENDS_ON": 2, "IMPLEMENTS": 2}
WINDOW_S = 45
LINES_PER_ITEM = 2         # по просьбе: пункт с текстом должен быть не больше чанка baseline (окно 3 реплики)
MAX_CHARS = 340            # средний чанк baseline в контексте ≈ 280 символов, верхняя граница ≈ 340
TEXT_SEED_MAX = 5          # сколько пунктов по словам вопроса берём как стартовые
TEXT_SEED_REL = 0.5        # и только те, у кого оценка не меньше половины лучшей
TEXT_SEED_MIN = 1.0        # слабее этой оценки совпадение случайное (один частый корень)

_QUESTION_WORDS = {
    "какие", "какой", "какая", "какое", "каких", "кто", "что", "где", "когда", "как", "почему", "зачем", "куда",
    "есть", "были", "было", "была", "будет", "будут", "этого", "этой", "этом", "всех", "всё", "нет", "или",
    "проекту", "проекте", "проекта", "проект", "проекты", "встреч", "встречах", "встрече", "встреча", "решили",
    "решали", "решил", "решила", "решение", "решения", "решений", "задачи", "задач", "задача", "менялся",
    "менялась", "менялось", "поменялся", "изменился", "изменилось", "ещё", "еще", "сейчас", "потом", "после",
    "перед", "между", "который", "которые", "которая", "которое", "сказал", "сказала", "говорили", "обсуждали",
}
_KK_FOLD = str.maketrans("ұүқғңәөіһё", "уукгнаоихе")


def _fold(s: str) -> str:
    return normalize(s).translate(_KK_FOLD)


def _words(text: str) -> list[str]:
    return [w for w in re.findall(r"\w+", _fold(text)) if len(w) >= 4 and w not in _QUESTION_WORDS]


def _same_stem(a: str, b: str) -> bool:
    """«статистики» ~ «статистика», «срок» ~ «сроку»: общее начало, разные окончания."""
    if a == b:
        return True
    common = len(os.path.commonprefix([a, b]))
    return common >= max(4, min(len(a), len(b)) - 2) and abs(len(a) - len(b)) <= 3


def _not_after(date: Optional[str], as_of: Optional[str]) -> bool:
    return as_of is None or (bool(date) and date <= as_of)


def _items(G: nx.MultiDiGraph, as_of: Optional[str]) -> list[str]:
    return sorted(n for n, d in G.nodes(data=True)
                  if d.get("type") in ITEM_TYPES and _not_after(d.get("date"), as_of))


def _sort_date(G: nx.MultiDiGraph, n: str) -> tuple:
    d = G.nodes[n]
    return (d.get("date", ""), d.get("ts", ""), n)


# ---------- слой 1: стартовые сущности ----------

def _item_text(d: dict) -> str:
    return f"{d.get('statement', '')} {d.get('quote', '')}"


def text_scores(G: nx.MultiDiGraph, question: str, skip_words: set[str]) -> dict[str, float]:
    """Оценка совпадения вопроса с текстом пункта: сумма редкости совпавших слов вопроса. Слова из skip_words
    (названия найденных проектов и людей) не считаются: иначе любой пункт проекта «подходит» к вопросу о проекте."""
    docs = {n: {_fold(w) for w in _words(_item_text(d))} for n, d in G.nodes(data=True)
            if d.get("type") in ITEM_TYPES}
    if not docs:
        return {}
    q_words = [w for w in dict.fromkeys(_words(question)) if not any(_same_stem(w, s) for s in skip_words)]
    scores: dict[str, float] = {}
    for q in q_words:
        hit = [n for n, ws in docs.items() if any(_same_stem(q, w) for w in ws)]
        if not hit:
            continue
        idf = math.log(1 + len(docs) / (1 + len(hit)))
        for n in hit:
            scores[n] = scores.get(n, 0.0) + idf
    return scores


def _short_name(label: str, word: Optional[str]) -> bool:
    """«Абду» — это «Абдурахман»: полное имя начинается с короткого. Короче трёх букв не сравниваем.

    Работает для сокращений, не для диминутивов вроде «Коля» → «Николай» (их решает resolve и aliases.json).
    """
    if not word:
        return False
    a, b = _fold(label), _fold(word)
    return len(b) >= 3 and (a == b or a.startswith(b))


def _entity_items(G: nx.MultiDiGraph, entity: str) -> list[str]:
    d = G.nodes[entity]
    label = d.get("label", "")
    if d["type"] == "Project":
        return [u for u, _, e in G.in_edges(entity, data=True) if e["type"] == "BELONGS_TO"]
    out = [u for u, _, e in G.in_edges(entity, data=True) if e["type"] == "ASSIGNED_TO"]
    out += [n for n, x in G.nodes(data=True)
            if x.get("type") in ITEM_TYPES and label
            and (_short_name(label, x.get("made_by")) or _short_name(label, x.get("owner"))
                 or any(_short_name(label, w) for w in re.findall(r"\w+", _item_text(x))))]
    return list(dict.fromkeys(out))


def _chain(G: nx.MultiDiGraph, n: str, pool: set[str]) -> list[str]:
    """Вся цепочка замен вокруг пункта (SUPERSEDES в обе стороны) среди пунктов pool, по датам."""
    seen, stack = {n}, [n]
    while stack:
        m = stack.pop()
        nbrs = [v for _, v, e in G.out_edges(m, data=True) if e.get("type") == "SUPERSEDES"]
        nbrs += [u for u, _, e in G.in_edges(m, data=True) if e.get("type") == "SUPERSEDES"]
        for x in nbrs:
            if x in pool and x not in seen:
                seen.add(x)
                stack.append(x)
    return sorted(seen, key=lambda x: _sort_date(G, x))


def _asked_by_focus(G: nx.MultiDiGraph, n: str, focus: Optional[str], pool: set[str],
                    states: Optional[dict] = None) -> bool:
    d = G.nodes[n]
    if focus == "T2":
        return len(_chain(G, n, pool)) > 1
    if focus == "T4":
        return bool(d.get("in_conflict"))
    if focus == "T5":
        if states is not None:  # статус на дату вопроса: закрытое после as_of ещё не закрыто, просроченное — важнее
            return n in states and states[n]["state"] != "closed"
        return d["type"] == "OpenQuestion" or (d["type"] == "Task" and d.get("status") == "open")
    return False


def _mention_pos(G: nx.MultiDiGraph, entity: str, question: str, aliases: Optional[dict[str, str]]) -> int:
    """Позиция (номер слова) первого упоминания сущности в вопросе; нет — в конец."""
    label = G.nodes[entity].get("label", "")
    names = [label] + [a for a, canon in (aliases or {}).items() if canon == label]
    qw = _words(question)
    hits = [i for i, w in enumerate(qw) for nm in names for x in _words(nm) if _same_stem(w, x)]
    return min(hits, default=len(qw))


def _link_two_projects(G: nx.MultiDiGraph, entities: list[str], question: str, aliases: Optional[dict[str, str]],
                       items: list[str]) -> list[str]:
    """Вопрос назвал два проекта («что по Qamqor влияет на Squad»): пункты второго берём только со встреч,
    где обсуждали первый (ребро DISCUSSES), а не все его пункты. Порядок проектов — как в вопросе.
    Это единственное место, где DISCUSSES сужает выборку, а не расширяет её."""
    projects = sorted((e for e in entities if G.nodes[e]["type"] == "Project"),
                      key=lambda e: _mention_pos(G, e, question, aliases))
    if len(projects) != 2:
        return items
    first, second = projects
    joint = {u for u, _, e in G.in_edges(first, data=True) if e["type"] == "DISCUSSES"}
    second_items = set(_entity_items(G, second))
    return [n for n in items if n not in second_items or f"meeting:{G.nodes[n].get('meeting_id')}" in joint
            or n in set(_entity_items(G, first))]


def find_start_entities(G: nx.MultiDiGraph, question: str, aliases: Optional[dict[str, str]] = None,
                        as_of: Optional[str] = None, raw_dir: Optional[Path] = None) -> dict:
    """Стартовые узлы обхода.

    Возвращает {"entities": [...], "entity_items": [...], "seeds": {item_id: {"score", "reason"}}}.
    entity_items — все пункты найденных проектов и людей (для яруса 4); seeds — те из них и те из других проектов,
    что прямо подходят к вопросу (ярус 1).
    Связь BELONGS_TO используется только здесь, для выбора пунктов проекта.
    """
    from src.retrieval.graph_rag import find_entities, question_focus  # локально: graph_rag сам тянет src.graph

    visible = set(_items(G, as_of))
    entities = find_entities(G, question, aliases)
    entity_items = [n for e in entities for n in _entity_items(G, e) if n in visible]
    entity_items = _link_two_projects(G, entities, question, aliases, list(dict.fromkeys(entity_items)))

    skip = {w for e in entities for w in _words(G.nodes[e].get("label", ""))}
    for alias, canon in (aliases or {}).items():
        if any(G.nodes[e].get("label") == canon for e in entities):
            skip.update(_words(alias))
    scores = {n: s for n, s in text_scores(G, question, skip).items() if n in visible}
    focus = question_focus(question)
    states = None
    if focus == "T5":
        from src.graph.status import task_states  # локально: status сам импортирует этот модуль
        states = {s["id"]: s for s in task_states(G, as_of, raw_dir=raw_dir)}

    seeds: dict[str, dict] = {}
    person_items = {n for e in entities if G.nodes[e]["type"] == "Person" for n in _entity_items(G, e)}
    for n in entity_items:
        if n in person_items:
            seeds[n] = {"score": scores.get(n, 0.0), "reason": "пункт названного человека"}
        elif scores.get(n, 0.0) > 0:
            seeds[n] = {"score": scores[n], "reason": "пункт названного проекта, слова вопроса совпали"}
        elif _asked_by_focus(G, n, focus, visible, states):
            seeds[n] = {"score": 0.0, "reason": f"пункт названного проекта, подходит к типу вопроса {focus}"}

    # вопрос назвал проект или человека — слова ищем только среди их пунктов: «первой версии Zhol» не должно
    # поднимать в ярус 1 «первую версию Squad». Чужие пункты придут только по связям (ярус 2–3).
    pool = {n: s for n, s in scores.items() if n in set(entity_items)} if entities else scores
    best = max(pool.values(), default=0.0)
    if best >= TEXT_SEED_MIN:
        ranked = sorted(pool.items(), key=lambda kv: (-kv[1], kv[0]))[:TEXT_SEED_MAX]
        for n, s in ranked:
            if s >= max(TEXT_SEED_MIN, TEXT_SEED_REL * best) and n not in seeds:
                seeds[n] = {"score": s, "reason": "слова вопроса совпали с текстом пункта"}
    if not entities and focus:
        for n in sorted(visible):
            if n not in seeds and _asked_by_focus(G, n, focus, visible, states):
                seeds[n] = {"score": scores.get(n, 0.0), "reason": f"вопрос типа {focus} без названных сущностей"}
    return {"entities": entities, "entity_items": entity_items, "seeds": seeds, "focus": focus, "states": states}


# ---------- слой 2: расширение ----------

def expand(G: nx.MultiDiGraph, seeds: list[str], as_of: Optional[str] = None, depth: int = 2) -> dict[str, dict]:
    """Расширение от пунктов-стартов по EXPAND_EDGES (в обе стороны), не дальше depth шагов.

    Возвращает {пункт: {"dist": шагов, "via": тип первой связи, "from": откуда пришли}} без самих стартов.
    По BELONGS_TO, STATED_IN, DISCUSSES и рёбрам к людям и проектам не идём: они ведут к «всему проекту».
    """
    visible = set(_items(G, as_of))
    dist = {s: 0 for s in seeds}
    found: dict[str, dict] = {}
    queue = deque(seeds)
    while queue:
        n = queue.popleft()
        if dist[n] >= depth:
            continue
        edges = [(v, e["type"]) for _, v, e in G.out_edges(n, data=True)]
        edges += [(u, e["type"]) for u, _, e in G.in_edges(n, data=True)]
        for m, kind in sorted(edges, key=lambda x: (_EDGE_RANK.get(x[1], 9), x[0])):
            if kind not in EXPAND_EDGES or m in dist or m not in visible:
                continue
            dist[m] = dist[n] + 1
            found[m] = {"dist": dist[m], "via": kind, "from": n}
            queue.append(m)
    return found


# ---------- слой 3: сырой текст ----------

def _ts_to_s(ts: str) -> int:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


@lru_cache(maxsize=32)
def _load_meeting(meeting_id: str, raw_dir: str, mtimes: tuple):
    from src.ingest.scribal_adapter import convert_files
    from src.ingest.speakers import apply_speakers

    base = Path(raw_dir)
    paths = base / f"{meeting_id}.transcript.json", base / f"{meeting_id}.meeting.json"
    if not all(p.exists() for p in paths):
        return None
    # те же имена говорящих, что в индексе baseline и в графе: api.load_scribal делает ровно это
    return apply_speakers(convert_files(paths[0], paths[1], meeting_id))


def raw_text_around(meeting_id: str, ts: str, window_s: int = WINDOW_S, raw_dir: Optional[Path] = None,
                    max_lines: Optional[int] = None, max_chars: Optional[int] = None) -> dict:
    """Реплики встречи в окне ±window_s секунд от таймкода ts.

    max_lines — взять только столько реплик, ближайших по времени к таймкоду (None — все в окне);
    max_chars — потом выбросить самые дальние реплики, пока текст не уложится в лимит (но одна, ближайшая, остаётся).
    Реплики отдаются в порядке времени. Целые реплики, не обрезанные: цитата ответа должна быть подстрокой реплики.
    Возвращает {"meeting_id", "ts", "window_s", "lines": [{"ts","start_s","speaker","text"}], "text"}; text — те же
    строки в формате «[m_001 00:09:37] Имя: реплика», как в чанках baseline: по метке answerer проверяет цитаты.
    Нет файлов встречи — пустой результат с "missing": True (а не исключение: граф бывает шире, чем data/raw).
    """
    base = Path(raw_dir or RAW_DIR)
    names = (base / f"{meeting_id}.transcript.json", base / f"{meeting_id}.meeting.json")
    mtimes = tuple(p.stat().st_mtime if p.exists() else 0 for p in names)
    t = _load_meeting(meeting_id, str(base), mtimes)
    out = {"meeting_id": meeting_id, "ts": ts, "window_s": window_s, "lines": [], "text": ""}
    if t is None:
        out["missing"] = True
        return out
    center = _ts_to_s(ts)
    cand = []
    for seg in t.segments:
        start = seg.start_s if seg.start_s is not None else _ts_to_s(seg.ts)
        end = seg.end_s if seg.end_s is not None else start
        if end >= center - window_s and start <= center + window_s:
            cand.append({"ts": seg.ts, "start_s": start, "speaker": seg.speaker, "text": seg.text})

    def line_len(x: dict) -> int:
        return len(f"[{meeting_id} {x['ts']}] {x['speaker']}: {x['text']}") + 1

    by_distance = sorted(cand, key=lambda x: (abs(x["start_s"] - center), x["start_s"]))
    if max_lines is not None:
        by_distance = by_distance[:max_lines]
    if max_chars is not None:
        while len(by_distance) > 1 and sum(line_len(x) for x in by_distance) - 1 > max_chars:
            by_distance.pop()  # самая дальняя от таймкода
    out["lines"] = sorted(by_distance, key=lambda x: x["start_s"])
    out["text"] = "\n".join(f"[{meeting_id} {x['ts']}] {x['speaker']}: {x['text']}" for x in out["lines"])
    return out


def _context_k() -> int:
    from src.retrieval.baseline import CONTEXT_K
    return CONTEXT_K


def _arrange(G: nx.MultiDiGraph, by_importance: list[str], chain_id: dict[str, str], k: int) -> list[str]:
    """Звенья одной цепочки замен — неделимый блок: стоит там, где стоит самое важное звено, звенья по датам.

    Первые k позиций заполняются блоками целиком. Блок, который не помещается в остаток, откладывается за позицию k,
    а его место занимает следующий (так же делает select_items у C). Если не помещается и в пустой контекст
    (цепочка длиннее k) — берутся самые важные её звенья. Тогда вызывающий, режущий по первым k, никогда
    не получает половину цепочки, если она в принципе влезает.
    """
    units: dict[str, list[str]] = {}
    for n in by_importance:                                   # порядок появления = важность лучшего звена
        units.setdefault(chain_id.get(n, n), []).append(n)
    blocks = [sorted(m, key=lambda x: _sort_date(G, x)) for m in units.values()]
    head: list[str] = []
    deferred: list[list[str]] = []
    for b in blocks:
        if len(head) + len(b) <= k:
            head += b
        elif not head and len(b) > k:
            keep = [m for m in by_importance if m in set(b)][:k]
            head += sorted(keep, key=lambda x: _sort_date(G, x))
            deferred.append([m for m in b if m not in keep])
        else:
            deferred.append(b)
    return head + [m for b in deferred for m in b]


# ---------- сборка результата ----------

def _tier_key(G: nx.MultiDiGraph, n: str, info: dict) -> tuple:
    d = G.nodes[n]
    newest_first = d.get("date", "").translate(str.maketrans("0123456789", "9876543210"))
    return (info["tier"], info.get("sub", 0), -info.get("score", 0.0), newest_first, n)


def traverse(question: str, *, G: Optional[nx.MultiDiGraph] = None, aliases: Optional[dict[str, str]] = None,
             as_of: Optional[str] = None, depth: int = 2, window_s: int = WINDOW_S, with_text: bool = True,
             raw_dir: Optional[Path] = None, lines_per_item: Optional[int] = LINES_PER_ITEM,
             max_chars: Optional[int] = MAX_CHARS, k: Optional[int] = None) -> dict:
    """Обход графа по вопросу.

    Результат не обрезается, но первые k позиций собраны так, чтобы вызывающий мог резать строго по первым k:
    цепочка замен идёт подряд и целиком помещается в первые k, а не оказывается разрезанной на границе.
    k по умолчанию — CONTEXT_K из baseline (10); гибрид передаёт свою долю (HYBRID_GRAPH_K).
    lines_per_item и max_chars — размер сырого текста на пункт (ближайшие к таймкоду реплики), чтобы пункт с текстом
    был размером с чанк baseline, а не 16–22 реплики окна ±window_s; None — без ограничения.

    Возвращает {
      "entities": [...],        найденные проекты и люди
      "items": [...],           id пунктов, упорядочены по важности (ярусы 1..4, см. докстринг модуля)
      "details": {id: {"tier", "why", "score", "dist", "chain_id", "via"}},
      "raw": {id: raw_text_around(...)},   пусто, если with_text=False
      "n_found": int, "fallback": bool,    fallback — ни сущностей, ни слов, ни типа вопроса не нашлось: items пуст
      "focus": "T2" | "T4" | "T5" | None
    }
    """
    from src.retrieval.baseline import check_as_of
    from src.retrieval.graph_rag import get_aliases, get_graph

    if depth < 1:
        raise ValueError("depth должен быть ≥ 1")
    check_as_of(as_of)
    if G is None:
        G, aliases = get_graph(), get_aliases() if aliases is None else aliases
    visible = set(_items(G, as_of))

    start = find_start_entities(G, question, aliases, as_of, raw_dir)
    seeds = start["seeds"]
    states = start["states"]
    state_rank = {"overdue_forgotten": 0, "overdue_mentioned": 1, "open": 2, "closed": 3}
    info: dict[str, dict] = {}
    for n, s in seeds.items():
        info[n] = {"tier": 1, "score": s["score"], "why": s["reason"], "dist": 0, "via": None}
        if states is not None and n in states:  # «что не закрыто»: забытое → просроченное → открытое
            info[n]["sub"] = state_rank[states[n]["state"]]

    # ярус 2: цепочки замен стартовых пунктов целиком (в пределах as_of)
    for n in list(seeds):
        for m in _chain(G, n, visible):
            if m not in info:
                info[m] = {"tier": 2, "score": info[n]["score"], "why": f"цепочка замен пункта {n}",
                           "dist": 1, "via": "SUPERSEDES"}

    # ярус 3: связи между пунктами (противоречия и закрытия раньше уточнений и зависимостей)
    for m, e in expand(G, list(info), as_of, depth).items():
        if m not in info:
            kind = e["via"]
            info[m] = {"tier": 3, "sub": _EDGE_RANK.get(kind, 9), "score": info[e["from"]]["score"],
                       "why": f"связь {kind} с пунктом {e['from']}", "dist": e["dist"], "via": kind}

    # ярус 4: остальные пункты найденных сущностей
    for n in start["entity_items"]:
        if n not in info:
            info[n] = {"tier": 4, "score": 0.0, "why": "пункт названной сущности, к вопросу прямо не подходит",
                       "dist": 1, "via": None}

    # цепочка замен целиком: у любого пункта в результате должны быть все звенья, видимые на as_of
    for n in list(info):
        for m in _chain(G, n, visible):
            if m not in info:
                info[m] = {"tier": info[n]["tier"], "sub": info[n].get("sub", 0), "score": info[n]["score"],
                           "why": f"цепочка замен пункта {n}", "dist": 1, "via": "SUPERSEDES"}

    chain_id: dict[str, str] = {}
    for n in sorted(info):
        chain = _chain(G, n, visible)
        if len(chain) > 1:
            chain_id[n] = chain[0]
    by_importance = sorted(info, key=lambda n: _tier_key(G, n, info[n]))
    order = _arrange(G, by_importance, chain_id, k if k is not None else _context_k())
    for n in order:
        rank_before = by_importance.index(n)
        info[n]["lifted"] = chain_id.get(n) is not None and rank_before > order.index(n)

    from src.graph.status import closed_by, task_state  # локально: status сам импортирует этот модуль

    details = {n: {**info[n], "chain_id": chain_id.get(n)} for n in order}
    for n in order:
        if G.nodes[n]["type"] in ("Task", "OpenQuestion"):
            details[n]["task_state"] = (states or {}).get(n) or task_state(G, n, as_of, raw_dir)
            cb = closed_by(G, n, as_of)
            if cb:
                details[n]["closed_by"] = cb
    raw = {}
    if with_text:
        for n in order:
            d = G.nodes[n]
            if d.get("meeting_id") and d.get("ts"):
                raw[n] = raw_text_around(d["meeting_id"], d["ts"], window_s, raw_dir, lines_per_item, max_chars)
    return {"entities": start["entities"], "items": order, "details": details, "raw": raw,
            "n_found": len(order), "fallback": not order, "focus": start["focus"]}
