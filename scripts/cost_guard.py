"""Жёсткий лимит расходов. Падает с исключением при превышении BUDGET_USD.

Счётчик лежит в runs/budget.json. Сбросить: python -m scripts.cost_guard --reset
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUDGET_FILE = ROOT / "runs" / "budget.json"


class BudgetExceeded(RuntimeError):
    pass


def _limit() -> float:
    return float(os.environ.get("BUDGET_USD", "3.0"))


def _read() -> dict:
    if BUDGET_FILE.exists():
        return json.loads(BUDGET_FILE.read_text(encoding="utf-8"))
    return {"spent_usd": 0.0}


def check_budget() -> None:
    spent = _read()["spent_usd"]
    if spent >= _limit():
        raise BudgetExceeded(
            f"Потрачено ${spent:.4f} из лимита ${_limit():.2f} (BUDGET_USD в .env). "
            "Остановлено, чтобы не сжечь бюджет."
        )


def record_spend(usd: float) -> None:
    BUDGET_FILE.parent.mkdir(exist_ok=True)
    data = _read()
    data["spent_usd"] = round(data["spent_usd"] + usd, 6)
    BUDGET_FILE.write_text(json.dumps(data), encoding="utf-8")


if __name__ == "__main__":
    if "--reset" in sys.argv:
        BUDGET_FILE.write_text(json.dumps({"spent_usd": 0.0}), encoding="utf-8")
        print("Счётчик сброшен.")
    else:
        print(f"Потрачено ${_read()['spent_usd']:.4f} из ${_limit():.2f}")
