"""Жёсткий лимит расходов. Падает с исключением при превышении BUDGET_USD.

Счётчик лежит в runs/budget.json.
    python -m scripts.cost_guard --status  # показать потраченное и лимит (так же без флагов)
    python -m scripts.cost_guard --reset   # обнулить счётчик
    python scripts/cost_guard.py --status  # то же, запуск файлом
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:  # при запуске файлом в sys.path только scripts/, и src не импортируется
    sys.path.insert(0, str(ROOT))

import src.llm_client  # noqa: F401,E402  — при импорте читает .env, чтобы BUDGET_USD брался оттуда

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
    if not usd:
        return
    BUDGET_FILE.parent.mkdir(exist_ok=True)
    data = _read()
    # Без округления: эмбеддинг стоит ~1e-7 $, округление до 6 знаков обнуляло бы такие траты.
    data["spent_usd"] = data["spent_usd"] + usd
    BUDGET_FILE.write_text(json.dumps(data), encoding="utf-8")


def status() -> str:
    spent, limit = _read()["spent_usd"], _limit()
    return f"Потрачено ${spent:.6f} из ${limit:.2f}, осталось ${max(limit - spent, 0):.6f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Счётчик расходов на LLM")
    group = ap.add_mutually_exclusive_group()
    group.add_argument("--status", action="store_true", help="показать потраченное и лимит (по умолчанию)")
    group.add_argument("--reset", action="store_true", help="обнулить счётчик")
    a = ap.parse_args(argv)
    if a.reset:
        BUDGET_FILE.parent.mkdir(exist_ok=True)
        BUDGET_FILE.write_text(json.dumps({"spent_usd": 0.0}), encoding="utf-8")
        print("Счётчик сброшен.")
    else:
        print(status())
    return 0


if __name__ == "__main__":
    sys.exit(main())
