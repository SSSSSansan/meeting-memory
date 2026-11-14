"""Таблица для docs/GRAPH_ENGINES.md из results/graph_engines.json: python scripts/neo4j/engines_table.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
r = json.loads((ROOT / "results" / "graph_engines.json").read_text(encoding="utf-8"))
names = {"S1": "S1 эволюция решения", "S2": "S2 кросс-проектный", "S3": "S3 противоречия",
         "S4": "S4 открытые и просроченные"}
print("| сценарий | узлов | строк в ответе | NetworkX перебор, мс | NetworkX индекс, мс | Neo4j, мс "
      "| индекс быстрее Neo4j в |")
print("|---|---:|---:|---:|---:|---:|---:|")
for s in names:
    for k, e in r["scales"].items():
        q = e["queries"][s]
        a, b, c = (q[m]["warm_median_ms"] for m in ("networkx", "networkx_index", "neo4j"))
        print(f"| {names[s]} | {e['nodes']} | {q['n_results']} | {a:.3f} | {b:.3f} | {c:.3f} | ×{c / b:.0f} |")
