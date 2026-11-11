"""РОЛЬ A. Лист для второго рецензента эталона: случайные 10% записей decisions.json. Модель не вызывается.

    python scripts/make_second_review_sheet.py
    python scripts/make_second_review_sheet.py --fraction 0.1 --seed 42 --out-dir results/second_review

Выборка: max(4, ceil(10% от числа записей)), random.Random(42) по записям, отсортированным по id.
Файлы в results/second_review/ (utf-8-sig, Excel открывает кириллицу и казахский):
  sheet.csv       — review_id, meeting, quote, statement (из world.yaml), type (метка эталона),
                    reviewer_type и reviewer_agrees заполняет второй рецензент:
                    reviewer_type — decision/task/open_question/status_update/recap/non_decision,
                    reviewer_agrees — 1, если согласен с типом эталона из колонки type, иначе 0.
  key_private.csv — review_id, event_id, type: связь с эталоном для score_second_review.py.
Листы идут в порядке выборки, не в порядке встреч. Расхождения потом разбираются и фиксируются
в data/gold/what_is_a_decision.md.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DECISIONS = ROOT / "data" / "gold" / "decisions.json"
DEFAULT_WORLD = ROOT / "data" / "gold" / "world.yaml"
DEFAULT_OUT_DIR = ROOT / "results" / "second_review"

SEED = 42
FRACTION = 0.10
MIN_ROWS = 4
TYPES = ("decision", "task", "open_question", "status_update", "recap", "non_decision")
SHEET_COLUMNS = ["review_id", "meeting", "quote", "statement", "type", "reviewer_type", "reviewer_agrees"]
KEY_COLUMNS = ["review_id", "event_id", "type"]


def sample_size(n: int, fraction: float = FRACTION) -> int:
    return min(n, max(MIN_ROWS, math.ceil(fraction * n - 1e-9)))


def select(records: list[dict], seed: int = SEED, fraction: float = FRACTION) -> list[dict]:
    pool = sorted(records, key=lambda r: r["id"])
    return random.Random(seed).sample(pool, sample_size(len(pool), fraction))


def load_statements(world_path: Path) -> dict[str, str]:
    world = yaml.safe_load(world_path.read_text(encoding="utf-8"))
    return {e["id"]: e["statement"] for e in world["events"]}


def build_tables(chosen: list[dict], statements: dict[str, str]) -> tuple[list[dict], list[dict]]:
    sheet, key = [], []
    for i, rec in enumerate(chosen, 1):
        rid = f"R{i:02d}"
        sheet.append({"review_id": rid, "meeting": rec["meeting"], "quote": rec["quote"],
                      "statement": statements[rec["id"]], "type": rec["type"],
                      "reviewer_type": "", "reviewer_agrees": ""})
        key.append({"review_id": rid, "event_id": rec["id"], "type": rec["type"]})
    return sheet, key


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    ap.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--fraction", type=float, default=FRACTION)
    args = ap.parse_args(argv)

    records = json.loads(args.decisions.read_text(encoding="utf-8"))
    statements = load_statements(args.world)
    missing = [r["id"] for r in records if r["id"] not in statements]
    if missing:
        print(f"ошибка: нет формулировки в world.yaml для {missing}", file=sys.stderr)
        return 1
    sheet, key = build_tables(select(records, args.seed, args.fraction), statements)
    write_csv(args.out_dir / "sheet.csv", SHEET_COLUMNS, sheet)
    write_csv(args.out_dir / "key_private.csv", KEY_COLUMNS, key)
    print(f"записей в эталоне: {len(records)}; в листе: {len(sheet)} (сид {args.seed})")
    print(f"записано: {args.out_dir / 'sheet.csv'}, {args.out_dir / 'key_private.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
