"""РОЛЬ B. Статус задач и открытых вопросов на дату вопроса (T5: «что ещё не закрыто», «что просрочено»).

    from src.graph.status import task_states
    for s in task_states(G, as_of="2026-10-26", project="Squad"):
        s["id"], s["state"], s["reason"]

Статус считается на дату as_of, а не хранится: решение «просрочено» зависит от того, когда спрашивают, а закрытие,
случившееся после as_of, для вопроса на более раннюю дату ещё не существует.

Состояния:
  closed             задача выполнена: у неё есть входящее CLOSES от пункта не позже as_of, или она сама отмечена
                     в извлечении как done/cancelled
  overdue_forgotten  срок прошёл относительно as_of, и после выдачи задача нигде не упоминалась — «забытое обещание»
  overdue_mentioned  срок прошёл, задача упоминалась позже, но закрытия не найдено
  open               срок не наступил или срока нет, закрытия нет
Для открытого вопроса (OpenQuestion) срока нет: бывает только closed (есть входящее CLOSES) или open.

Путь 1 (этот модуль без закрытий) детерминирован и бесплатен: срок берётся из извлечения, упоминание ищется по
основам слов в репликах позднейших встреч того же проекта (вход системы — data/raw, не эталон).
Путь 2 (связь CLOSES) добавляет входящие рёбра, и тот же код начинает видеть закрытия.
"""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import networkx as nx

from src.graph.traverse import RAW_DIR, _fold, _load_meeting, _not_after, _same_stem, _words

TASK_TYPES = ("Task", "OpenQuestion")
OPEN_STATES = ("overdue_forgotten", "overdue_mentioned", "open")
# слова-«шум» постановки задачи: по ним нельзя узнать задачу в чужой реплике
_TASK_STOP = {
    "сделать", "делает", "делаем", "делать", "сделал", "сделала", "нужно", "надо", "должен", "должна", "должны",
    "будет", "будут", "готов", "готова", "готово", "дальше", "потом", "когда", "чтобы", "этого", "этот", "первый",
    "первая", "первой", "первую", "второй", "третий", "января", "февраля", "марта", "апреля", "августа", "сентября",
    "октября", "ноября", "декабря", "первого", "второго", "третьего", "четвертого", "четвёртого", "пятого",
    "шестого", "седьмого", "восьмого", "девятого", "десятого", "двадцатого", "тридцатого", "недели", "неделю",
}
# глаголы постановки («показать», «отправить»): есть в любой задаче, суть задачи не определяют
_ACTION_STEMS = ("показ", "отправ", "подключ", "сдела", "дела", "провест", "провод", "написа", "подготов", "провер",
                 "добав", "обнов", "довест", "довод", "запуст", "запуск", "собра", "собир", "довед", "оформ", "выпол")
MENTION_SNIPPET = 160
MAX_MENTIONS = 3


def _reference_date(G: nx.MultiDiGraph, as_of: Optional[str]) -> str:
    """Дата «сейчас» для вопроса: as_of; без него — дата последней встречи графа."""
    if as_of:
        return as_of
    return max((d.get("date", "") for _, d in G.nodes(data=True) if d.get("type") == "Meeting"), default="")


def _keywords(G: nx.MultiDiGraph, n: str) -> list[str]:
    d = G.nodes[n]
    names = " ".join(str(d.get(k) or "") for k in ("project", "owner", "made_by"))
    skip = _words(names)
    words = [w for w in dict.fromkeys(_words(d.get("statement", "")))
             if w not in _TASK_STOP and not w.startswith(_ACTION_STEMS)]
    return [w for w in words if not any(_same_stem(w, s) for s in skip)]


def _project_meetings(G: nx.MultiDiGraph, project: str) -> list[tuple[str, str]]:
    pid = f"project:{project}"
    if pid not in G:
        return []
    return sorted((G.nodes[u].get("date", ""), G.nodes[u]["meeting_id"]) for u, _, e in G.in_edges(pid, data=True)
                  if e["type"] == "DISCUSSES")


def find_mentions(G: nx.MultiDiGraph, task: str, ref_date: str, raw_dir: Optional[Path] = None,
                  limit: int = MAX_MENTIONS) -> list[dict]:
    """Реплики позднейших встреч проекта задачи, где снова звучит суть задачи.

    Совпадением считается реплика, где нашлись основы не меньше чем двух ключевых слов задачи (если слово одно —
    его). Встречи берутся строго после встречи выдачи и не позже ref_date; сама встреча выдачи не считается: там
    задача и формулируется. Возвращает [{"meeting_id", "ts", "snippet"}] по времени.
    """
    d = G.nodes[task]
    keys = _keywords(G, task)
    if not keys:
        return []
    need = min(2, len(keys))
    base = Path(raw_dir or RAW_DIR)
    out = []
    for mdate, mid in _project_meetings(G, d.get("project", "")):
        if not (mdate > d.get("date", "") and _not_after(mdate, ref_date)):
            continue
        paths = (base / f"{mid}.transcript.json", base / f"{mid}.meeting.json")
        mtimes = tuple(p.stat().st_mtime if p.exists() else 0 for p in paths)
        t = _load_meeting(mid, str(base), mtimes)
        if t is None:
            continue
        for seg in t.segments:
            seg_words = set(_words(seg.text))
            hits = sum(any(_same_stem(k, w) for w in seg_words) for k in keys)
            if hits >= need:
                out.append({"meeting_id": mid, "ts": seg.ts, "snippet": seg.text[:MENTION_SNIPPET]})
                break  # одной реплики на встречу достаточно: нужен факт упоминания, а не их число
        if len(out) >= limit:
            break
    return out


def _closers(G: nx.MultiDiGraph, task: str, ref_date: str) -> list[str]:
    return [u for u, _, e in G.in_edges(task, data=True)
            if e["type"] == "CLOSES" and _not_after(e.get("date") or G.nodes[u].get("date"), ref_date)]


def closed_by(G: nx.MultiDiGraph, task: str, as_of: Optional[str] = None) -> Optional[dict]:
    """Чем закрыта задача на дату as_of: {by, quote, ts, meeting_id, date, confidence, rationale} или None.
    Цитата и таймкод берутся с ребра CLOSES, поэтому не зависят от того, какой узел его породил."""
    ref = _reference_date(G, as_of)
    best = None
    for u, _, e in G.in_edges(task, data=True):
        when = e.get("date") or G.nodes[u].get("date")
        if e["type"] == "CLOSES" and _not_after(when, ref) and (best is None or when < best["date"]):
            src = G.nodes[u]
            best = {"by": u, "quote": e.get("quote") or src.get("quote", ""), "ts": e.get("ts") or src.get("ts", ""),
                    "meeting_id": e.get("meeting_id") or src.get("meeting_id", ""), "date": when,
                    "confidence": e.get("confidence"), "rationale": e.get("rationale", "")}
    return best


def _days(a: str, b: str) -> int:
    return (date.fromisoformat(a) - date.fromisoformat(b)).days


def task_state(G: nx.MultiDiGraph, task: str, as_of: Optional[str] = None,
               raw_dir: Optional[Path] = None) -> dict:
    """Состояние одной задачи или открытого вопроса на дату as_of. См. модульный докстринг."""
    d = G.nodes[task]
    ref = _reference_date(G, as_of)
    out = {"id": task, "type": d["type"], "project": d.get("project"), "owner": d.get("owner"),
           "due": d.get("due"), "issued": d.get("date"), "statement": d.get("statement"),
           "closed_by": None, "mentions": [], "days_overdue": None}

    closers = _closers(G, task, ref)
    if closers:
        out.update(state="closed", closed_by=sorted(closers, key=lambda x: G.nodes[x].get("date", ""))[0],
                   reason=f"закрыта пунктом {closers[0]}")
        return out
    # closed_at ставит closes --apply: статус done, выставленный закрытием позже даты вопроса, на эту дату не виден
    if d["type"] == "Task" and d.get("status") in ("done", "cancelled") and _not_after(d.get("closed_at") or "0", ref):
        out.update(state="closed", reason=f"в извлечении отмечена как {d['status']}")
        return out
    if d["type"] == "OpenQuestion" or not d.get("due"):
        out.update(state="open", reason="срока нет, закрытия не найдено" if d["type"] == "Task"
                   else "вопрос без ответа в данных до этой даты")
        return out
    if d["due"] >= ref:
        out.update(state="open", reason=f"срок {d['due']} ещё не прошёл на {ref}")
        return out
    out["days_overdue"] = _days(ref, d["due"])
    out["mentions"] = find_mentions(G, task, ref, raw_dir)
    if out["mentions"]:
        m = out["mentions"][0]
        out.update(state="overdue_mentioned",
                   reason=f"срок {d['due']} прошёл, упоминалась позже ({m['meeting_id']} {m['ts']}), закрытия нет")
    else:
        out.update(state="overdue_forgotten",
                   reason=f"срок {d['due']} прошёл на {out['days_overdue']} дн., после выдачи задача не упоминалась")
    return out


def task_states(G: nx.MultiDiGraph, as_of: Optional[str] = None, project: Optional[str] = None,
                raw_dir: Optional[Path] = None) -> list[dict]:
    """Состояния всех задач и открытых вопросов, видимых на as_of (по проекту, если задан). Порядок по важности для
    вопроса «что не закрыто»: забытые, просроченные с упоминанием, открытые, закрытые."""
    rank = {"overdue_forgotten": 0, "overdue_mentioned": 1, "open": 2, "closed": 3}
    out = []
    for n, d in G.nodes(data=True):
        if d.get("type") not in TASK_TYPES or not _not_after(d.get("date"), as_of):
            continue
        if project and _fold(d.get("project", "")) != _fold(project):
            continue
        out.append(task_state(G, n, as_of, raw_dir))
    return sorted(out, key=lambda s: (rank[s["state"]], s["due"] or "9999", s["issued"] or "", s["id"]))
