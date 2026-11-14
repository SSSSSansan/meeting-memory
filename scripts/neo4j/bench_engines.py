"""РОЛЬ B. NetworkX против Neo4j на четырёх запросах к графу встреч.

    python scripts/neo4j/bench_engines.py --engine networkx            # только NetworkX, Neo4j не нужен
    python scripts/neo4j/bench_engines.py --engine both                # нужен запущенный Neo4j (см. docker-compose.yml)
    python scripts/neo4j/bench_engines.py --engine both --scales 1 10 100 1000

Сценарии (одна и та же выдача в обоих движках, сравнивается как множество id):
  S1 история решения: цепочка SUPERSEDES в обе стороны (переменная длина пути);
  S2 кросс-проектный: в каких проектах человеку поручено и что (ASSIGNED_TO + BELONGS_TO);
  S3 противоречия: все пары CONTRADICTS;
  S4 открытые и просроченные задачи на дату as_of.
Рост: граф размножается ×N непересекающимися копиями (id получают суффикс `~k`), запросы всегда про копию 0,
поэтому размер ответа не меняется и видно только стоимость поиска в большом графе.
Честная оговорка: на ~60 узлах разница в миллисекундах и тонет в шуме; различает движки только кривая роста.
Python-версий две: «перебор» (как в queries.py, без индексов) и «индекс» (словари строятся один раз, время
построения меряется отдельно). Neo4j тут с индексом по `nid`, как и положено. S3 и S4 по определению возвращают
больше строк на больших графах (копий больше), S1 и S2 про копию 0. Результат: results/graph_engines.json.
"""
from __future__ import annotations

import argparse
import inspect
import json
import statistics as st
import sys
import time
from pathlib import Path

import networkx as nx

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.graph.queries import load_graph  # noqa: E402

Q_DECISION, Q_PERSON, Q_ASOF = "d_m001_01", "person:Николай", "2026-10-26"
WARM_RUNS = 20


# ---------- NetworkX: полный перебор (как в queries.py) ----------

def py_s1(G, decision_id):
    seen, stack = {decision_id}, [decision_id]
    while stack:
        n = stack.pop()
        neigh = [v for _, v, d in G.out_edges(n, data=True) if d["type"] == "SUPERSEDES"]
        neigh += [u for u, _, d in G.in_edges(n, data=True) if d["type"] == "SUPERSEDES"]
        for x in neigh:
            if x not in seen:
                seen.add(x)
                stack.append(x)
    return sorted(seen)


def py_s2(G, person):
    """Кросс-проектный: в каких проектах человеку что-то поручено; строки «проект|пункт»."""
    out = []
    for u, _, d in G.in_edges(person, data=True):
        if d["type"] == "ASSIGNED_TO":
            out += [f"{v}|{u}" for _, v, e in G.out_edges(u, data=True) if e["type"] == "BELONGS_TO"]
    return sorted(out)


def py_s3(G):
    return sorted(f"{u}->{v}" for u, v, d in G.edges(data=True) if d["type"] == "CONTRADICTS")


def py_s4(G, as_of):
    """Открытые и просроченные задачи: «id|open» или «id|overdue» (срок раньше as_of)."""
    return sorted(f"{n}|{'overdue' if d.get('due') and d['due'] < as_of else 'open'}"
                  for n, d in G.nodes(data=True) if d.get("type") == "Task" and d.get("status") == "open")


# ---------- NetworkX: со словарями-индексами (строятся один раз) ----------

class NxIndex:
    """Тот же ответ, но без перебора всех узлов и рёбер. Время построения индекса измеряется отдельно."""

    def __init__(self, G):
        self.G = G
        self.contradicts = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "CONTRADICTS"]
        self.open_tasks = [n for n, d in G.nodes(data=True) if d.get("type") == "Task" and d.get("status") == "open"]
        self.assigned: dict[str, list] = {}
        for u, v, d in G.edges(data=True):
            if d["type"] == "ASSIGNED_TO":
                self.assigned.setdefault(v, []).append(u)

    def s1(self, decision_id):
        return py_s1(self.G, decision_id)            # обход по соседям и так локальный, индекс не нужен

    def s2(self, person):
        return sorted(f"{v}|{u}" for u in self.assigned.get(person, [])
                      for _, v, e in self.G.out_edges(u, data=True) if e["type"] == "BELONGS_TO")

    def s3(self):
        return sorted(f"{u}->{v}" for u, v in self.contradicts)

    def s4(self, as_of):
        G = self.G
        return sorted(f"{n}|{'overdue' if G.nodes[n].get('due') and G.nodes[n]['due'] < as_of else 'open'}"
                      for n in self.open_tasks)


# ---------- Cypher ----------

CYPHER = {
    "S1": ("MATCH (a:Node {nid:$id})-[:SUPERSEDES*0..]-(m) RETURN DISTINCT m.nid AS r ORDER BY r",
           lambda: {"id": Q_DECISION}),
    "S2": ("MATCH (t)-[:ASSIGNED_TO]->(p:Node {nid:$p}), (t)-[:BELONGS_TO]->(pr) "
           "RETURN pr.nid + '|' + t.nid AS r ORDER BY r",
           lambda: {"p": Q_PERSON}),
    "S3": ("MATCH (a)-[:CONTRADICTS]->(b) RETURN a.nid + '->' + b.nid AS r ORDER BY r", dict),
    "S4": ("MATCH (t:Node {type:'Task', status:'open'}) RETURN t.nid + '|' + "
           "CASE WHEN t.due <> '' AND t.due < $t THEN 'overdue' ELSE 'open' END AS r ORDER BY r",
           lambda: {"t": Q_ASOF}),
}
PY = {"S1": lambda G, ix: py_s1(G, Q_DECISION), "S2": lambda G, ix: py_s2(G, Q_PERSON),
      "S3": lambda G, ix: py_s3(G), "S4": lambda G, ix: py_s4(G, Q_ASOF)}
PYX = {"S1": lambda ix: ix.s1(Q_DECISION), "S2": lambda ix: ix.s2(Q_PERSON), "S3": lambda ix: ix.s3(),
       "S4": lambda ix: ix.s4(Q_ASOF)}
PY_SRC = {"S1": py_s1, "S2": py_s2, "S3": py_s3, "S4": py_s4}
PYX_SRC = {"S1": NxIndex.s1, "S2": NxIndex.s2, "S3": NxIndex.s3, "S4": NxIndex.s4}


def scale_graph(G: nx.MultiDiGraph, k: int) -> nx.MultiDiGraph:
    """k непересекающихся копий; копия 0 сохраняет исходные id."""
    if k == 1:
        return G
    out = nx.MultiDiGraph()
    for i in range(k):
        suf = "" if i == 0 else f"~{i}"
        for n, d in G.nodes(data=True):
            out.add_node(n + suf, **d)
        for u, v, key, d in G.edges(keys=True, data=True):
            out.add_edge(u + suf, v + suf, key=key, **d)
    return out


def loc(fn_or_text) -> int:
    text = fn_or_text if isinstance(fn_or_text, str) else inspect.getsource(fn_or_text)
    return sum(1 for ln in text.splitlines() if ln.strip() and not ln.strip().startswith(("#", '"""')))


def timed(fn):
    t = time.perf_counter()
    res = fn()
    return res, (time.perf_counter() - t) * 1000


def measure(run) -> dict:
    res, cold = timed(run)
    warm = [timed(run)[1] for _ in range(WARM_RUNS)]
    return {"result": res, "cold_ms": round(cold, 3), "warm_median_ms": round(st.median(warm), 3)}


# ---------- Neo4j ----------

def neo4j_load(driver, G: nx.MultiDiGraph) -> float:
    t = time.perf_counter()
    nodes = [{"nid": n, "type": d.get("type", ""), "project": d.get("project", ""), "date": d.get("date", ""),
             "status": d.get("status") or "", "due": d.get("due") or ""}
             for n, d in G.nodes(data=True)]
    by_type: dict[str, list] = {}
    for e in G.edges(data=True):
        by_type.setdefault(e[2]["type"], []).append({"s": e[0], "t": e[1]})
    with driver.session() as s:
        s.run("MATCH (n) DETACH DELETE n")
        s.run("CREATE INDEX node_nid IF NOT EXISTS FOR (n:Node) ON (n.nid)")
        s.run("CREATE INDEX dec_project IF NOT EXISTS FOR (n:Decision) ON (n.project)")
        for i in range(0, len(nodes), 5000):
            s.run("UNWIND $rows AS r CREATE (n:Node {nid:r.nid, type:r.type, project:r.project, date:r.date, "
            "status:r.status, due:r.due})",
                  rows=nodes[i:i + 5000])
        s.run("MATCH (n:Node {type:'Decision'}) SET n:Decision")
        for typ, rows in by_type.items():
            for i in range(0, len(rows), 5000):
                s.run(f"UNWIND $rows AS r MATCH (a:Node {{nid:r.s}}), (b:Node {{nid:r.t}}) CREATE (a)-[:{typ}]->(b)",
                      rows=rows[i:i + 5000])
    return (time.perf_counter() - t) * 1000


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", choices=["networkx", "both"], default="networkx")
    ap.add_argument("--scales", type=int, nargs="+", default=[1, 10, 100, 1000])
    ap.add_argument("--graph", default="data/graph/graph.json")
    ap.add_argument("--uri", default="bolt://localhost:7687")
    ap.add_argument("--auth", default="neo4j/k02-local-pass")
    a = ap.parse_args()

    base = load_graph(ROOT / a.graph)
    driver = None
    if a.engine == "both":
        from neo4j import GraphDatabase  # pip install neo4j
        user, pwd = a.auth.split("/", 1)
        driver = GraphDatabase.driver(a.uri, auth=(user, pwd))
        driver.verify_connectivity()

    report = {"scales": {}, "loc": {}, "notes": __doc__.split("Честная оговорка:")[1].split("Результат")[0].strip()}
    report["loc"] = {s: {"python_scan": loc(PY_SRC[s]), "python_index": loc(PYX_SRC[s]), "cypher": loc(CYPHER[s][0])}
                     for s in CYPHER}
    for k in a.scales:
        t0 = time.perf_counter()
        G = scale_graph(base, k)
        entry = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(), "queries": {}}
        entry["networkx_build_ms"] = round((time.perf_counter() - t0) * 1000, 1)
        t1 = time.perf_counter()
        ix = NxIndex(G)
        entry["networkx_index_build_ms"] = round((time.perf_counter() - t1) * 1000, 1)
        if driver:
            entry["neo4j_load_ms"] = round(neo4j_load(driver, G), 1)
        for s in CYPHER:
            row = {"networkx": measure(lambda s=s: PY[s](G, ix)), "networkx_index": measure(lambda s=s: PYX[s](ix))}
            if driver:
                cy, params = CYPHER[s]

                def run(cy=cy, params=params):
                    with driver.session() as sess:
                        return [r["r"] for r in sess.run(cy, **params())]
                row["neo4j"] = measure(run)
                row["agree"] = (sorted(row["networkx"]["result"]) == sorted(row["neo4j"]["result"])
                                == sorted(row["networkx_index"]["result"]))
            else:
                row["agree"] = row["networkx"]["result"] == row["networkx_index"]["result"]
            row["n_results"] = len(row["networkx"]["result"])
            for e in ("networkx", "networkx_index", "neo4j"):
                if e in row:
                    row[e].pop("result")
            entry["queries"][s] = row
        report["scales"][str(k)] = entry
        line = " ".join(f"{s}: nx {r['networkx']['warm_median_ms']}/idx {r['networkx_index']['warm_median_ms']}мс"
                        + (f" / neo4j {r['neo4j']['warm_median_ms']}мс" if "neo4j" in r else "")
                        + (" ✓" if r.get("agree") else (" ✗ РАСХОЖДЕНИЕ" if "agree" in r else ""))
                        for s, r in entry["queries"].items())
        print(f"×{k} ({entry['nodes']} узлов): {line}")
    out = ROOT / "results" / "graph_engines.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"LoC: {report['loc']}\nЗаписано: {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
