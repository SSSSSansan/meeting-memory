"""РОЛЬ B. Точность и полнота связей CLOSES против эталона data/gold/closures.json (6 закрытий).

    python -m src.graph.eval_closes                  # читает data/graph/closes.json и closes_rejected.json

Эталон используется только здесь, для измерения. В промпты, отбор кандидатов и обход он не попадает.
Предсказание верно, если задача найдена в той же встрече и по тексту совпадает с эталонной (общие основы слов),
а закрытие сказано на встрече эталонного закрытия. Строка метрик отдельная от извлечения и от связей между решениями.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from src.graph.queries import load_graph
from src.graph.traverse import _same_stem, _words

ROOT = Path(__file__).resolve().parent.parent.parent


def _similar(a: str, b: str) -> float:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return 0.0
    return sum(any(_same_stem(x, y) for y in wb) for x in wa) / len(wa)


def _gold_target(G, gold: dict) -> str | None:
    """Узел графа, соответствующий эталонной задаче: лучшее совпадение текста среди Task/OpenQuestion той встречи."""
    best = max(((_similar(gold["task_statement"], d.get("statement", "")), n) for n, d in G.nodes(data=True)
                if d.get("meeting_id") == gold["task_meeting"] and d.get("type") in ("Task", "OpenQuestion")),
               default=(0.0, None))
    return best[1] if best[0] >= 0.5 else None


def evaluate(G, closes: list[dict], gold: list[dict]) -> dict:
    targets = {g["task_event_id"]: _gold_target(G, g) for g in gold}
    by_target = {c["to"]: c for c in closes}
    tp, rows = 0, []
    for g in gold:
        t = targets[g["task_event_id"]]
        pred = by_target.get(t) if t else None
        ok = pred is not None and pred["meeting_id"] == g["closing_meeting"]
        tp += ok
        rows.append({"gold": g["task_statement"], "target": t, "predicted_meeting": pred and pred["meeting_id"],
                     "gold_meeting": g["closing_meeting"], "ok": ok})
    gold_targets = {t for t in targets.values() if t}
    # лишнее: закрытие у задачи, которой нет в эталоне, или не на той встрече
    right = {(targets[g["task_event_id"]], g["closing_meeting"]) for g in gold}
    wrong = [c for c in closes if c["to"] not in gold_targets or (c["to"], c["meeting_id"]) not in right]
    n_pred = len(closes)
    return {"gold": len(gold), "predicted": n_pred, "tp": tp,
            "precision": tp / n_pred if n_pred else None, "recall": tp / len(gold), "wrong": [c["to"] for c in wrong],
            "rows": rows}


def main() -> int:
    G = load_graph(ROOT / "data/graph/graph.json")
    p = ROOT / "data/graph/closes.json"
    closes = json.loads(p.read_text(encoding="utf-8")) if p.exists() else []
    gold = json.loads((ROOT / "data/gold/closures.json").read_text(encoding="utf-8"))
    r = evaluate(G, closes, gold)
    for row in r["rows"]:
        print("OK  " if row["ok"] else "MISS", row["gold"][:45], "→", row["target"],
              "| найдено на", row["predicted_meeting"], "| эталон", row["gold_meeting"])
    prec = "—" if r["precision"] is None else f"{r['precision']:.2f}"
    print(f"\nCLOSES: предсказано {r['predicted']}, верно {r['tp']} из {r['gold']} эталонных; "
          f"precision {prec}, recall {r['recall']:.2f}; лишние: {r['wrong'] or 'нет'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
