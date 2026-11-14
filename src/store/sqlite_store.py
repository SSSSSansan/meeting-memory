"""РОЛЬ B (B3). Граф и транскрипты в SQLite: те же запросы, что в src/graph/queries.py, плюс окно реплик и алиасы.

    python -m src.store.sqlite_store      # data/graph/graph.json + data/raw → data/store/meeting_memory.sqlite
    python -m src.store.sqlite_store --db /tmp/x.sqlite

Таблицы (поля только добавляются, существующие не переименовываются):
  nodes(id, type, label, project, meeting_id, date, ts, ts_s, statement, status, owner, made_by, due, closed_at,
        origin, props)               — props: все атрибуты узла JSON-ом, колонки выше — копии для индексов и фильтров
  edges(id, source, target, type, key, confidence, rationale, origin, props)
  meetings(meeting_id, title, date, started_at, language, participants)
  replicas(meeting_id, idx, ts, ts_s, start_s, end_s, speaker, speaker_raw, text, lang)  — таймкод и в секундах
  aliases(alias, alias_norm, canonical, node_id, kind, meeting_id, occurrences, source)

Ответы функций совпадают с queries.py: словари {"id": ..., **атрибуты узла}, тот же порядок (дата, время, id).
Хранилище строится из graph.json, поэтому после пересборки графа или `closes --apply` его нужно построить заново.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DB = ROOT / "data" / "store" / "meeting_memory.sqlite"

SCHEMA = """
CREATE TABLE nodes (
  id TEXT PRIMARY KEY, type TEXT NOT NULL, label TEXT, project TEXT, meeting_id TEXT,
  date TEXT NOT NULL DEFAULT '', ts TEXT NOT NULL DEFAULT '', ts_s INTEGER, statement TEXT, status TEXT,
  owner TEXT, made_by TEXT, due TEXT, closed_at TEXT, origin TEXT, props TEXT NOT NULL);
CREATE INDEX idx_nodes_type ON nodes(type, project);
CREATE INDEX idx_nodes_meeting ON nodes(meeting_id);
CREATE TABLE edges (
  id INTEGER PRIMARY KEY AUTOINCREMENT, source TEXT NOT NULL, target TEXT NOT NULL, type TEXT NOT NULL, key TEXT,
  confidence REAL, rationale TEXT, origin TEXT, props TEXT NOT NULL);
CREATE INDEX idx_edges_source ON edges(source, type);
CREATE INDEX idx_edges_target ON edges(target, type);
CREATE TABLE meetings (
  meeting_id TEXT PRIMARY KEY, title TEXT, date TEXT, started_at TEXT, language TEXT, participants TEXT);
CREATE TABLE replicas (
  meeting_id TEXT NOT NULL, idx INTEGER NOT NULL, ts TEXT NOT NULL, ts_s INTEGER NOT NULL, start_s REAL NOT NULL,
  end_s REAL NOT NULL, speaker TEXT, speaker_raw TEXT, text TEXT NOT NULL, lang TEXT, PRIMARY KEY (meeting_id, idx));
CREATE INDEX idx_replicas_time ON replicas(meeting_id, start_s);
CREATE TABLE aliases (
  alias TEXT NOT NULL, alias_norm TEXT NOT NULL, canonical TEXT NOT NULL, node_id TEXT, kind TEXT NOT NULL,
  meeting_id TEXT NOT NULL DEFAULT '', occurrences INTEGER NOT NULL DEFAULT 0, source TEXT,
  PRIMARY KEY (alias, kind, meeting_id));
CREATE INDEX idx_aliases_norm ON aliases(alias_norm);
"""

_NODE_COLS = ("label", "project", "meeting_id", "statement", "status", "owner", "made_by", "due", "closed_at", "origin")


def _ts_to_s(ts: str) -> int:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def connect(db: Optional[Path] = None) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db or DEFAULT_DB))
    conn.row_factory = sqlite3.Row
    return conn


def _node(row: sqlite3.Row) -> dict:
    return {"id": row["id"], **json.loads(row["props"])}


def _load_nodes_edges(conn: sqlite3.Connection, graph_path: Path) -> None:
    data = json.loads(graph_path.read_text(encoding="utf-8"))
    for n in data["nodes"]:
        attrs = {k: v for k, v in n.items() if k != "id"}
        ts = attrs.get("ts") or ""
        conn.execute(
            "INSERT INTO nodes(id, type, label, project, meeting_id, date, ts, ts_s, statement, status, owner, made_by,"
            " due, closed_at, origin, props) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (n["id"], n["type"], attrs.get("label"), attrs.get("project"), attrs.get("meeting_id"),
             attrs.get("date") or "", ts, _ts_to_s(ts) if ts else None,
             *(attrs.get(c) for c in ("statement", "status", "owner", "made_by", "due", "closed_at", "origin")),
             json.dumps(attrs, ensure_ascii=False)))
    for e in data["edges"]:
        attrs = {k: v for k, v in e.items() if k not in ("source", "target")}
        conn.execute("INSERT INTO edges(source, target, type, key, confidence, rationale, origin, props) "
                     "VALUES (?,?,?,?,?,?,?,?)",
                     (e["source"], e["target"], e["type"], e.get("key"), e.get("confidence"), e.get("rationale"),
                      e.get("origin"), json.dumps(attrs, ensure_ascii=False)))


def _load_meetings(conn: sqlite3.Connection, raw_dir: Path) -> int:
    from src.ingest.scribal_adapter import convert_files
    from src.ingest.speakers import apply_speakers

    n = 0
    for f in sorted(raw_dir.glob("*.transcript.json")):
        mid = f.name.split(".")[0]
        meta = raw_dir / f"{mid}.meeting.json"
        if not meta.exists():
            continue
        t = apply_speakers(convert_files(f, meta, mid))  # те же имена, что в графе и baseline
        conn.execute("INSERT INTO meetings VALUES (?,?,?,?,?,?)",
                     (mid, t.title, t.date, t.started_at, t.language, json.dumps(t.participants, ensure_ascii=False)))
        for i, s in enumerate(t.segments):
            start = s.start_s if s.start_s is not None else float(_ts_to_s(s.ts))
            end = s.end_s if s.end_s is not None else start
            conn.execute("INSERT INTO replicas VALUES (?,?,?,?,?,?,?,?,?,?)",
                         (mid, i, s.ts, _ts_to_s(s.ts), start, end, s.speaker, s.speaker_raw, s.text, s.lang))
        n += 1
    return n


def build_db(graph_path: Optional[Path] = None, raw_dir: Optional[Path] = None, db: Optional[Path] = None,
             aliases_path: Optional[Path] = None) -> Path:
    """Строит базу заново (старый файл заменяется). Нет data/raw — таблицы meetings и replicas остаются пустыми."""
    from src.store.aliases import build_alias_rows

    graph_path = Path(graph_path or ROOT / "data" / "graph" / "graph.json")
    raw_dir = Path(raw_dir or ROOT / "data" / "raw")
    db = Path(db or DEFAULT_DB)
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()
    conn = connect(db)
    with conn:
        conn.executescript(SCHEMA)
        _load_nodes_edges(conn, graph_path)
        if raw_dir.exists():
            _load_meetings(conn, raw_dir)
        rows = build_alias_rows(conn, aliases_path or graph_path.with_name("aliases.json"), raw_dir)
        conn.executemany("INSERT INTO aliases VALUES (?,?,?,?,?,?,?,?)", rows)
    conn.close()
    return db


# ---------- те же запросы, что в src/graph/queries.py ----------

_ORDER = "ORDER BY date, ts, id"


def project_meetings(conn: sqlite3.Connection, project: str) -> list[dict]:
    rows = conn.execute(
        "SELECT m.* FROM nodes m JOIN edges e ON e.source = m.id AND e.type = 'DISCUSSES' "
        "WHERE e.target = ? ORDER BY m.date, m.ts, m.id", (f"project:{project}",)).fetchall()
    return [_node(r) for r in rows]


def project_decisions(conn: sqlite3.Connection, project: str, last_meetings: Optional[int] = None,
                      only_active: bool = False) -> list[dict]:
    allowed = None
    if last_meetings is not None:
        allowed = {m["meeting_id"] for m in project_meetings(conn, project)[-last_meetings:]}
    rows = conn.execute(
        "SELECT n.* FROM nodes n JOIN edges e ON e.source = n.id AND e.type = 'BELONGS_TO' "
        "WHERE e.target = ? AND n.type = 'Decision' ORDER BY n.date, n.ts, n.id", (f"project:{project}",)).fetchall()
    out = [_node(r) for r in rows]
    if allowed is not None:
        out = [n for n in out if n["meeting_id"] in allowed]
    if only_active:
        out = [n for n in out if n["status"] == "active"]
    return out


def decision_history(conn: sqlite3.Connection, decision_id: str) -> list[dict]:
    """Цепочка «заменяет» от раннего к позднему: рекурсивный запрос по SUPERSEDES в обе стороны."""
    if conn.execute("SELECT 1 FROM nodes WHERE id = ?", (decision_id,)).fetchone() is None:
        return []
    rows = conn.execute(
        "WITH RECURSIVE chain(id) AS (SELECT ? UNION "
        "  SELECT CASE WHEN e.source = chain.id THEN e.target ELSE e.source END FROM chain "
        "  JOIN edges e ON e.type = 'SUPERSEDES' AND (e.source = chain.id OR e.target = chain.id)) "
        "SELECT n.* FROM nodes n JOIN chain ON chain.id = n.id ORDER BY n.date, n.ts, n.id", (decision_id,)).fetchall()
    return [_node(r) for r in rows]


def conflicts(conn: sqlite3.Connection, project: Optional[str] = None) -> list[dict]:
    rows = conn.execute(
        "SELECT a.id AS aid, b.id AS bid, e.confidence, e.rationale FROM edges e "
        "JOIN nodes a ON a.id = e.source JOIN nodes b ON b.id = e.target "
        "WHERE e.type = 'CONTRADICTS' ORDER BY e.id").fetchall()
    out = []
    for r in rows:
        a = _node(conn.execute("SELECT * FROM nodes WHERE id = ?", (r["aid"],)).fetchone())
        b = _node(conn.execute("SELECT * FROM nodes WHERE id = ?", (r["bid"],)).fetchone())
        if project and project not in (a["project"], b["project"]):
            continue
        out.append({"a": a, "b": b, "confidence": r["confidence"], "rationale": r["rationale"]})
    return out


def open_tasks(conn: sqlite3.Connection, project: Optional[str] = None) -> list[dict]:
    rows = conn.execute(
        "SELECT * FROM nodes WHERE type = 'Task' AND status = 'open' AND (? IS NULL OR project = ?) " + _ORDER,
        (project, project)).fetchall()
    return [_node(r) for r in rows]


def overdue_tasks(conn: sqlite3.Connection, as_of: str, project: Optional[str] = None) -> list[dict]:
    """Открытые задачи, срок которых раньше as_of и выданные не позже as_of. Без учёта упоминаний (это status.py)."""
    rows = conn.execute(
        "SELECT * FROM nodes WHERE type = 'Task' AND status = 'open' AND due IS NOT NULL AND due < ? AND date <= ? "
        "AND (? IS NULL OR project = ?) " + _ORDER, (as_of, as_of, project, project)).fetchall()
    return [_node(r) for r in rows]


# ---------- реплики и алиасы ----------

def replicas_window(conn: sqlite3.Connection, meeting_id: str, second: float, window_s: float = 45,
                    max_lines: Optional[int] = None) -> list[dict]:
    """Реплики в окне ±window_s секунд от секунды second, по времени. Правило пересечения то же, что в
    traverse.raw_text_around; max_lines оставляет столько реплик, ближайших к second."""
    rows = conn.execute(
        "SELECT idx, ts, ts_s, start_s, end_s, speaker, text FROM replicas WHERE meeting_id = ? "
        "AND end_s >= ? AND start_s <= ? ORDER BY start_s, idx",
        (meeting_id, second - window_s, second + window_s)).fetchall()
    out = [dict(r) for r in rows]
    if max_lines is not None:
        near = sorted(out, key=lambda x: (abs(x["start_s"] - second), x["start_s"]))[:max_lines]
        out = sorted(near, key=lambda x: (x["start_s"], x["idx"]))
    return out


def find_entity(conn: sqlite3.Connection, text: str, meeting_id: str = "") -> Optional[dict]:
    """Узел проекта или человека по любому написанию из таблицы алиасов (нормализация как normalize).
    Метки SPEAKER_XX привязаны к встрече, поэтому для них нужен meeting_id."""
    from src.utils.text import normalize

    row = conn.execute(
        "SELECT canonical, node_id, kind FROM aliases WHERE alias_norm = ? AND (meeting_id = '' OR meeting_id = ?) "
        "ORDER BY meeting_id = '', occurrences DESC LIMIT 1", (normalize(text), meeting_id)).fetchone()
    return dict(row) if row else None


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--graph", default=None)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--db", default=None)
    a = ap.parse_args(argv)
    path = build_db(a.graph and Path(a.graph), a.raw and Path(a.raw), a.db and Path(a.db))
    conn = connect(path)
    counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0] for t in
              ("nodes", "edges", "meetings", "replicas", "aliases")}
    print(f"{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}: {counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
