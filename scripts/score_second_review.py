"""РОЛЬ A. Совпадение слепой оценки второго рецензента с эталоном. Модель не вызывается.

    python scripts/score_second_review.py
    python scripts/score_second_review.py --sheet results/second_review/sheet.csv --out agreement.md

Читает заполненный sheet.csv и key_private.csv (оба из make_second_review_sheet.py) и сравнивает
reviewer_type с типом эталона из ключа. Показатель — доля строк, где reviewer_type == тип эталона.
Расхождения выводятся с формулировкой из ключа и обеими оценками. Незаполненные строки в расчёт не идут.
Файл можно сохранять из Excel с разделителем «;». Результат: results/second_review/agreement.md.
Код возврата: 0 — посчитано, 1 — ошибка данных или нет ни одной заполненной строки.
"""
from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # запуск файлом: в sys.path только scripts/
    sys.path.insert(0, str(ROOT))

from scripts.make_second_review_sheet import DEFAULT_OUT_DIR, TYPES  # noqa: E402

DEFAULT_SHEET = DEFAULT_OUT_DIR / "sheet.csv"
DEFAULT_KEY = DEFAULT_OUT_DIR / "key_private.csv"
DEFAULT_OUT = DEFAULT_OUT_DIR / "agreement.md"


class ScoreError(ValueError):
    """Некорректный лист: сообщение показывается пользователю."""


def read_csv(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    header = text.splitlines()[0] if text else ""
    delimiter = ";" if header.count(";") > header.count(",") else ","
    return list(csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter))


def pct(x: float) -> str:
    return f"{x * 100:.1f}%".replace(".", ",")


def score(sheet: list[dict], key: list[dict]) -> dict:
    gold = {r["review_id"]: r for r in key}
    unknown = [r["review_id"] for r in sheet if r["review_id"] not in gold]
    if unknown:
        raise ScoreError(f"review_id нет в key_private.csv: {unknown}")
    filled, empty, mismatches = [], [], []
    for r in sheet:
        rtype = (r.get("reviewer_type") or "").strip().lower()
        if not rtype:
            empty.append(r["review_id"])
            continue
        if rtype not in TYPES:
            raise ScoreError(f"{r['review_id']}: reviewer_type «{rtype}» не из {', '.join(TYPES)}")
        g = gold[r["review_id"]]
        row = {**r, "reviewer_type": rtype, "event_id": g["event_id"], "gold_type": g["type"],
               "statement": g.get("statement", "")}
        filled.append(row)
        if rtype != g["type"]:
            mismatches.append(row)
    n = len(filled)
    return {"total": len(sheet), "filled": n, "empty": empty, "mismatches": mismatches,
            "agreement": (n - len(mismatches)) / n if n else 0.0}


def render(res: dict) -> str:
    n, bad = res["filled"], len(res["mismatches"])
    out = ["# Согласие второго рецензента с эталоном", "",
           f"- Строк в листе: {res['total']}; заполнено: {n}; не заполнено: {len(res['empty'])}.",
           f"- **Совпадение типа (reviewer_type == эталон): {n - bad} из {n} = {pct(res['agreement'])}.**",
           f"- Расхождений: {bad}.", ""]
    if res["mismatches"]:
        out += ["## Расхождения", "",
                "| review_id | Событие | Встреча | Эталон | Рецензент | Цитата | Формулировка | Комментарий |",
                "|---|---|---|---|---|---|---|---|"]
        for r in res["mismatches"]:
            out.append(f"| {r['review_id']} | {r['event_id']} | {r['meeting']} | {r['gold_type']} | "
                       f"{r['reviewer_type']} | {r['quote'].replace('|', '/')} | {r['statement'].replace('|', '/')} | "
                       f"{(r.get('reviewer_comment') or '').replace('|', '/')} |")
        out += ["", "Разобрать и зафиксировать итог в `data/gold/what_is_a_decision.md`.", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    ap.add_argument("--key", type=Path, default=DEFAULT_KEY)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    try:
        res = score(read_csv(args.sheet), read_csv(args.key))
    except (ScoreError, FileNotFoundError) as e:
        print(f"ошибка: {e}", file=sys.stderr)
        return 1
    if not res["filled"]:
        print("ошибка: в листе нет заполненных строк", file=sys.stderr)
        return 1
    text = render(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"совпадение: {res['filled'] - len(res['mismatches'])} из {res['filled']} = {pct(res['agreement'])}")
    for r in res["mismatches"]:
        print(f"  расхождение {r['review_id']} ({r['event_id']}): "
              f"эталон {r['gold_type']}, рецензент {r['reviewer_type']}")
    print(f"записано: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
