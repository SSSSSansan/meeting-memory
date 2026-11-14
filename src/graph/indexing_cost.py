"""РОЛЬ B. Стоимость индексации по этапам из runs/calls.jsonl → results/indexing_cost.json.

    python -m src.graph.indexing_cost

Берётся последний полный прогон каждого этапа (не все эксперименты в логе), токены пересчитываются по ценам модели:
в логе до 11 ноября цена gpt-4o записана по тарифу mini (`_price` не смотрит на модель),
поэтому «по логу» ниже реальной.
Этап «closes» — оценка (платный прогон ещё не делали), её даёт `python -m src.graph.closes --dry-run`.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
# $ за 1M токенов (вход, выход). Источник: публичный прайс OpenAI на дату замера; меняйте здесь, если тариф другой.
PRICES = {"gpt-4o": (2.5, 10.0), "gpt-4o-mini": (0.15, 0.60), "text-embedding-3-small": (0.02, 0.0)}


def _cost(model: str, tin: int, tout: int) -> float:
    pi, po = PRICES[model]
    return (tin * pi + tout * po) / 1_000_000


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]


def stage_rows(calls: list[dict], purpose: str, day: str | None = None) -> list[dict]:
    return [r for r in calls if r["purpose"] == purpose and not r["from_cache"]
            and (day is None or r["ts"].startswith(day))]


def summarize(rows: list[dict]) -> dict:
    tin, tout = sum(r["prompt_tokens"] for r in rows), sum(r["completion_tokens"] for r in rows)
    return {"calls": len(rows), "tokens_in": tin, "tokens_out": tout,
            "usd_logged": round(sum(r["cost_usd"] for r in rows), 5),
            "usd_repriced": round(sum(_cost(r["model"], r["prompt_tokens"], r["completion_tokens"]) for r in rows), 5),
            "models": sorted({r["model"] for r in rows})}


def main() -> int:
    calls = _jsonl(ROOT / "runs" / "calls.jsonl")
    report = _jsonl(ROOT / "runs" / "extract_report.jsonl")
    last = {}
    for r in report:
        last[r["meeting_id"]] = r
    # последний прогон извлечения по каждой встрече: строка лога с теми же токенами и временем
    extract_rows, per_meeting = [], {}
    pool = stage_rows(calls, "extract")
    for mid, r in sorted(last.items()):
        row = next((c for c in pool if c["ts"] == r["ts"] and c["prompt_tokens"] == r["prompt_tokens"]), None)
        if row:
            extract_rows.append(row)
            per_meeting[mid] = summarize([row])
    n = len(last)
    link_day = max(r["ts"][:10] for r in calls if r["purpose"] == "link")
    stages = {"extract": summarize(extract_rows), "link": summarize(stage_rows(calls, "link", link_day))}
    embed_day = max(r["ts"][:10] for r in calls if r["purpose"] == "embed_index")
    baseline = summarize(stage_rows(calls, "embed_index", embed_day))
    closes_est = {"calls": 9, "usd_estimate": 0.002, "note": "оценка dry-run, платный прогон не запускали"}
    graph_total = stages["extract"]["usd_repriced"] + stages["link"]["usd_repriced"] + closes_est["usd_estimate"]
    out = {"meetings": n, "stages": stages, "closes": closes_est, "baseline_index": baseline,
           "per_meeting_usd": {"extract": round(stages["extract"]["usd_repriced"] / n, 5),
                               "link": round(stages["link"]["usd_repriced"] / n, 5),
                               "closes": round(closes_est["usd_estimate"] / n, 5),
                               "graph_total": round(graph_total / n, 5),
                               "baseline_index": round(baseline["usd_repriced"] / n, 6)},
           "corpus_usd": {"graph_total": round(graph_total, 4), "baseline_index": baseline["usd_repriced"],
                          "ratio_graph_to_baseline": round(graph_total / baseline["usd_repriced"])},
           "extract_per_meeting": per_meeting,
           "notes": ["link: среднее на встречу, у вызовов нет meeting_id (вызов = пара решений)",
                     "usd_logged для gpt-4o занижена до 11.11: цена в логе по тарифу mini; usd_repriced — по токенам",
                     "baseline-индекс: только эмбеддинги чанков, без вызовов LLM"]}
    (ROOT / "results").mkdir(exist_ok=True)
    text = json.dumps(out, ensure_ascii=False, indent=2)
    (ROOT / "results" / "indexing_cost.json").write_text(text, encoding="utf-8")
    pm, cu = out["per_meeting_usd"], out["corpus_usd"]
    print(f"{'этап':14}{'вызовов':>8}{'$ корпус':>10}{'$ / встреча':>13}")
    for k, v in (("extract", stages["extract"]), ("link", stages["link"])):
        print(f"{k:14}{v['calls']:>8}{v['usd_repriced']:>10.4f}{pm[k]:>13.5f}")
    print(f"{'closes (оценка)':14}{closes_est['calls']:>8}{closes_est['usd_estimate']:>10.4f}{pm['closes']:>13.5f}")
    print(f"{'граф итого':14}{'':>8}{cu['graph_total']:>10.4f}{pm['graph_total']:>13.5f}")
    print(f"{'baseline-индекс':14}{baseline['calls']:>8}{baseline['usd_repriced']:>10.4f}{pm['baseline_index']:>13.6f}")
    print(f"\nВстреч: {n}. Граф дороже baseline-индекса примерно в {cu['ratio_graph_to_baseline']} раз.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
