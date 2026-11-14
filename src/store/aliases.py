"""РОЛЬ B (B4). Таблица алиасов «канон → все написания» из graph.json и транскриптов.

    python -m src.store.aliases      # строит базу, пишет data/graph/aliases.json и data/graph/alias_table.json

Источники строк: узлы Project/Person графа (канон), прежний data/graph/aliases.json (записи сохраняются как есть),
написания из реплик (кириллица ↔ латиница, суффикс через дефис «Zhol-ға», склонение «Даниярдың», краткая форма «Абду»,
уменьшительное «Коля») и метки SPEAKER_XX → имя по каждой встрече (если имя определено).
aliases.json остаётся плоским {написание: канон}: так его читает src/retrieval/graph_rag.get_aliases. Метки
SPEAKER_XX в плоский файл не попадают: они действуют только внутри одной встречи, они лежат в alias_table.json и в
таблице aliases (kind='speaker', meeting_id заполнен).
"""
from __future__ import annotations

import collections
import json
import re
import sqlite3
import sys
from pathlib import Path
from typing import Optional

from src.resolve.entities import _CYR, _DIMINUTIVES
from src.utils.text import normalize

ROOT = Path(__file__).resolve().parent.parent.parent
_WORD = re.compile(r"[\w'-]{3,}")
_END = "йьая"


def _stem(canon: str) -> str:
    s = canon.lower()
    return s.rstrip(_END) if len(s.rstrip(_END)) >= 4 else s


def _translit(s: str) -> str:
    """Строгая транслитерация без слияния гласных (в отличие от resolve.key): «жел» не должно стать «Zhol»."""
    return "".join(_CYR.get(ch, ch) for ch in s.lower()).replace("q", "k")


def is_variant(token: str, canon: str) -> bool:
    """Написание token — вариант канона canon (но не сам канон)."""
    t, c = token.lower(), canon.lower()
    if t == c:
        return False
    if _translit(t) == _translit(c) or _DIMINUTIVES.get(t) == c:
        return True
    base = t.split("-")[0]
    if base != t and base != c and _translit(base) == _translit(c):
        return True
    stem = _stem(canon)
    if len(stem) >= 4 and t.startswith(stem) and len(t) <= len(stem) + 4:
        return True
    return len(t) >= 4 and c.startswith(t)


def build_alias_rows(conn: sqlite3.Connection, aliases_path: Path, raw_dir: Path) -> list[tuple]:
    entities = [(r["id"], r["label"], r["type"].lower()) for r in
                conn.execute("SELECT id, label, type FROM nodes WHERE type IN ('Project', 'Person') ORDER BY id")]
    node_of = {label: nid for nid, label, _ in entities}
    kind_of = {label: kind for _, label, kind in entities}
    tokens: collections.Counter = collections.Counter()
    for (text,) in conn.execute("SELECT text FROM replicas"):
        for w in _WORD.findall(text):
            tokens[w] += 1
    low: collections.Counter = collections.Counter()
    for w, n in tokens.items():
        low[w.lower()] += n

    rows: dict[tuple, tuple] = {}

    def add(alias, canon, kind, source, meeting="", occurrences=0):
        rows[(alias, kind, meeting)] = (alias, normalize(alias), canon, node_of.get(canon), kind, meeting,
                                        occurrences, source)

    for _, label, kind in entities:
        add(label, label, kind, "graph", occurrences=low.get(label.lower(), 0))
    if aliases_path.exists():
        for alias, canon in json.loads(aliases_path.read_text(encoding="utf-8")).items():
            if canon in kind_of:
                add(alias, canon, kind_of[canon], "aliases.json", occurrences=low.get(alias.lower(), 0))
    for _, label, kind in entities:
        seen: dict[str, str] = {}
        for w in tokens:
            if is_variant(w, label):
                seen.setdefault(w.lower(), w)
        for lw, w in seen.items():
            if (w, kind, "") not in rows:
                add(max((x for x in tokens if x.lower() == lw), key=lambda x: tokens[x]), label, kind, "transcripts",
                    occurrences=low[lw])
    for r in conn.execute("SELECT meeting_id, speaker_raw, speaker, COUNT(*) AS n FROM replicas "
                          "WHERE speaker_raw IS NOT NULL AND speaker_raw != speaker "
                          "GROUP BY meeting_id, speaker_raw, speaker"):
        if r["speaker"] in kind_of and not r["speaker"].startswith("SPEAKER_"):
            add(r["speaker_raw"], r["speaker"], "speaker", "speakers", r["meeting_id"], r["n"])
    return sorted(rows.values(), key=lambda x: (x[2], x[4], x[5], x[0]))


def export(conn: sqlite3.Connection, out_dir: Path) -> tuple[int, int]:
    """Пишет плоский aliases.json (с сохранением прежних записей) и полный alias_table.json."""
    flat_path = out_dir / "aliases.json"
    flat = json.loads(flat_path.read_text(encoding="utf-8")) if flat_path.exists() else {}
    table: dict[str, dict] = {}
    for r in conn.execute("SELECT * FROM aliases ORDER BY canonical, kind, meeting_id, alias"):
        entry = table.setdefault(r["canonical"], {"kind": None, "node_id": r["node_id"], "variants": [],
                                                  "speaker_labels": {}})
        if r["kind"] == "speaker":
            entry["speaker_labels"][r["meeting_id"]] = r["alias"]
            continue
        entry["kind"] = r["kind"]
        entry["variants"].append({"alias": r["alias"], "occurrences": r["occurrences"], "source": r["source"]})
        if r["alias"] != r["canonical"]:
            flat.setdefault(r["alias"], r["canonical"])
    flat_path.write_text(json.dumps(flat, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "alias_table.json").write_text(json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(flat), len(table)


def main(argv: Optional[list[str]] = None) -> int:
    from src.store.sqlite_store import build_db, connect

    db = build_db()
    conn = connect(db)
    n_flat, n_canon = export(conn, ROOT / "data" / "graph")
    for r in conn.execute("SELECT canonical, kind, alias, occurrences, source FROM aliases WHERE kind != 'speaker' "
                          "AND alias != canonical ORDER BY canonical, occurrences DESC"):
        print(f"{r['canonical']:12} {r['kind']:8} {r['alias']:14} ×{r['occurrences']:<3} {r['source']}")
    print(f"\naliases.json: {n_flat} записей, alias_table.json: {n_canon} канонов")
    return 0


if __name__ == "__main__":
    sys.exit(main())
