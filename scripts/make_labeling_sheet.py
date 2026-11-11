"""РОЛЬ A. Лист для ручной разметки 20 ответов (калибровка LLM-судьи, A8). Модель не вызывается.

    python scripts/make_labeling_sheet.py
    python scripts/make_labeling_sheet.py --eval-csv results/eval_2026-11-11.csv --out-dir results/judge_labeling

Из eval-CSV берутся строки mode=baseline, status=ok; выборка детерминирована (сид 42).
Выходные файлы (папка results/judge_labeling/):
  labeling_sheet.csv — то, что видит разметчик: label_id, question, expected_answer, system_answer,
                       cited_meetings и три пустые колонки human_score, human_fabrication, comment.
                       Ни метрик, ни типа вопроса, ни split, ни question_id в листе нет.
  key_private.csv    — label_id, question_id, type, split, lang. Связь с листом только через label_id.
Оба файла utf-8-sig: Excel открывает кириллицу и казахский без перекодировки.

Состав выборки: none 3, T1 3, T2 4, T3 3, T4 3, T5 4; не меньше 3 вопросов lang=kk и не меньше 6 из split=test;
около половины — «проблемные» строки (correct_meetings=False, не все цитаты валидны или abstained не
совпадает с answerable), остальное случайно. Обязательные вопросы: q_20, q_31, q_34, q_12.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_CSV = ROOT / "results" / "eval_2026-11-11.csv"
DEFAULT_QUESTIONS = ROOT / "data" / "gold" / "questions.json"
DEFAULT_OUT_DIR = ROOT / "results" / "judge_labeling"

SEED = 42
QUOTAS = {"none": 3, "T1": 3, "T2": 4, "T3": 3, "T4": 3, "T5": 4}
MANDATORY = ("q_20", "q_31", "q_34", "q_12")
MIN_KK = 3
MIN_TEST = 6
MAX_ATTEMPTS = 2000

SHEET_COLUMNS = [
    "label_id", "question", "expected_answer", "system_answer", "cited_meetings",
    "human_score", "human_fabrication", "comment",
]
KEY_COLUMNS = ["label_id", "question_id", "type", "split", "lang"]


def is_problem(row: dict) -> bool:
    """Строка, где метрики подозревают ошибку: именно такие ответы калибровке нужны больше всего."""
    answerable = row["answerable"] == "True"
    abstained = row["abstained"] == "True"
    bad_citations = int(row["citations_valid"] or 0) < int(row["citations_total"] or 0)
    return row["correct_meetings"] == "False" or bad_citations or abstained == answerable


def load_rows(eval_csv: Path) -> list[dict]:
    with open(eval_csv, encoding="utf-8-sig", newline="") as f:
        rows = [r for r in csv.DictReader(f) if r["mode"] == "baseline" and r["status"] == "ok"]
    # Несколько повторов одного вопроса: берём первый, иначе вопрос попадёт в лист дважды.
    seen: dict[str, dict] = {}
    for r in sorted(rows, key=lambda r: (r["question_id"], int(r.get("repeat") or 1))):
        seen.setdefault(r["question_id"], r)
    return list(seen.values())


def _draw(rng: random.Random, pool: list[dict], quotas: dict[str, int], mandatory: tuple[str, ...],
          target_problems: int) -> list[dict]:
    chosen = [r for r in pool if r["question_id"] in mandatory]
    left = dict(quotas)
    for r in chosen:
        left[r["type"]] -= 1
    rest = [r for r in pool if r["question_id"] not in mandatory]
    problems = [r for r in rest if is_problem(r)]
    others = [r for r in rest if not is_problem(r)]
    rng.shuffle(problems)
    rng.shuffle(others)
    n_problems = sum(is_problem(r) for r in chosen)
    for r in problems:
        if n_problems >= target_problems:
            break
        if left.get(r["type"], 0) > 0:
            chosen.append(r)
            left[r["type"]] -= 1
            n_problems += 1
    for r in others + [p for p in problems if p not in chosen]:
        if left.get(r["type"], 0) > 0:
            chosen.append(r)
            left[r["type"]] -= 1
    return chosen


def select(rows: list[dict], *, seed: int = SEED, quotas: dict[str, int] = QUOTAS,
           mandatory: tuple[str, ...] | None = None, min_kk: int = MIN_KK, min_test: int = MIN_TEST) -> list[dict]:
    """Перебор с повторными жеребьёвками из одного генератора: результат зависит только от сида и данных."""
    mandatory = MANDATORY if mandatory is None else mandatory
    pool = sorted(rows, key=lambda r: r["question_id"])
    missing = [q for q in mandatory if q not in {r["question_id"] for r in pool}]
    if missing:
        raise ValueError(f"обязательных вопросов нет среди baseline/ok строк: {missing}")
    total = sum(quotas.values())
    target_problems = total // 2
    rng = random.Random(seed)
    for _ in range(MAX_ATTEMPTS):
        chosen = _draw(rng, pool, quotas, mandatory, target_problems)
        counts = {t: sum(r["type"] == t for r in chosen) for t in quotas}
        if counts != quotas:
            raise ValueError(f"не хватает вопросов для квот: нужно {quotas}, получилось {counts}")
        if sum(r["lang"] == "kk" for r in chosen) >= min_kk and sum(r["split"] == "test" for r in chosen) >= min_test:
            return chosen
    raise ValueError(f"за {MAX_ATTEMPTS} попыток не набрались kk>={min_kk} и test>={min_test}")


def build_tables(chosen: list[dict], questions: dict[str, dict], seed: int = SEED) -> tuple[list[dict], list[dict]]:
    order = sorted(chosen, key=lambda r: r["question_id"])
    random.Random(seed).shuffle(order)
    sheet, key = [], []
    for i, r in enumerate(order, 1):
        label_id = f"L{i:02d}"
        q = questions[r["question_id"]]
        sheet.append({
            "label_id": label_id,
            "question": q["question"],
            "expected_answer": q["expected_answer"],
            "system_answer": r["answer"],
            "cited_meetings": r["cited_meetings"],
            "human_score": "",
            "human_fabrication": "",
            "comment": "",
        })
        key.append({"label_id": label_id, "question_id": r["question_id"], "type": r["type"],
                    "split": r["split"], "lang": r["lang"]})
    return sheet, key


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-csv", type=Path, default=DEFAULT_EVAL_CSV)
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    questions = {q["id"]: q for q in json.loads(args.questions.read_text(encoding="utf-8"))}
    try:
        chosen = select(load_rows(args.eval_csv), seed=args.seed)
    except ValueError as e:
        print(f"ошибка: {e}", file=sys.stderr)
        return 1
    sheet, key = build_tables(chosen, questions, args.seed)
    write_csv(args.out_dir / "labeling_sheet.csv", SHEET_COLUMNS, sheet)
    write_csv(args.out_dir / "key_private.csv", KEY_COLUMNS, key)

    by_type = {t: sum(r["type"] == t for r in chosen) for t in QUOTAS}
    print(f"строк: {len(chosen)}; по типам: {by_type}; kk: {sum(r['lang'] == 'kk' for r in chosen)}; "
          f"test: {sum(r['split'] == 'test' for r in chosen)}; проблемных: {sum(is_problem(r) for r in chosen)}")
    print(f"записано: {args.out_dir / 'labeling_sheet.csv'}, {args.out_dir / 'key_private.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
