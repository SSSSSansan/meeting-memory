"""РОЛЬ A. Лист ручной разметки фактов для калибровки судьи judge-facts-v1 (A8). Модель не вызывается.

    python scripts/make_facts_sheet.py
    python scripts/make_facts_sheet.py --eval-json results/eval_20261114_0935.json --seed 42

Берёт ответы baseline (repeat 1, status ok, с вердиктом судьи) из json-результата прогона с --judge-facts и
выбирает 30 (сид 42, детерминировано): все ответы на вопросы с answerable=false, остальное — по кругу по типам
T1–T5 (внутри типа порядок перемешан сидом), чтобы типы были представлены ровно.
На каждый ответ — по строке на каждый факт вопроса: required (есть ли в ответе по смыслу) и forbidden
(утверждает ли ответ). Тексты фактов — из data/gold/questions_dev.json.

Выход (utf-8-sig, папка results/judge_facts_labeling/):
  sheet.csv        — row_id, label_id, question, system_answer, fact_kind, fact, human (пусто), comment.
                     Вердиктов судьи, типа вопроса и question_id в листе нет.
  key_private.csv  — row_id, label_id, question_id, type, judge_value (0/1). Связь с листом — через row_id.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_JSON = ROOT / "results" / "eval_20261114_0935.json"
DEFAULT_QUESTIONS = ROOT / "data" / "gold" / "questions_dev.json"
DEFAULT_OUT_DIR = ROOT / "results" / "judge_facts_labeling"

SEED = 42
N_ANSWERS = 30
MODE = "baseline"

SHEET_COLUMNS = ["row_id", "label_id", "question", "system_answer", "fact_kind", "fact", "human", "comment"]
KEY_COLUMNS = ["row_id", "label_id", "question_id", "type", "judge_value"]


class SheetError(ValueError):
    """Данных не хватает или они не сходятся с эталоном: сообщение показывается пользователю."""


def load_answers(eval_json: Path, mode: str = MODE) -> list[dict]:
    """Ответы режима mode с вердиктом судьи по фактам; один ответ на вопрос (первый повтор)."""
    data = json.loads(eval_json.read_text(encoding="utf-8"))
    rows = [r for r in data["per_question"]
            if r.get("mode") == mode and r.get("status") == "ok" and r.get("judge_facts")]
    if not rows:
        raise SheetError(f"в {eval_json} нет ответов {mode} с вердиктом судьи (прогон без --judge-facts?)")
    seen: dict[str, dict] = {}
    for r in sorted(rows, key=lambda r: (r["question_id"], int(r.get("repeat") or 1))):
        seen.setdefault(r["question_id"], r)
    return list(seen.values())


def select(answers: list[dict], *, n: int = N_ANSWERS, seed: int = SEED) -> list[dict]:
    """Все answerable=false + круговой обход по типам среди остальных, пока не наберётся n."""
    pool = sorted(answers, key=lambda r: r["question_id"])
    chosen = [r for r in pool if not r["answerable"]]
    if len(chosen) > n:
        raise SheetError(f"ответов с answerable=false ({len(chosen)}) больше, чем размер выборки {n}")
    rng = random.Random(seed)
    by_type: dict[str, list[dict]] = {}
    for r in pool:
        if r["answerable"]:
            by_type.setdefault(r["type"], []).append(r)
    for rs in by_type.values():
        rng.shuffle(rs)
    types = sorted(by_type)
    while len(chosen) < n and any(by_type.values()):
        for t in types:
            if by_type[t] and len(chosen) < n:
                chosen.append(by_type[t].pop())
    if len(chosen) < n:
        raise SheetError(f"ответов не хватает: нужно {n}, есть {len(chosen)}")
    return chosen


def _facts_with_verdicts(r: dict, q: dict) -> list[tuple[str, str, int]]:
    """[(fact_kind, факт, вердикт судьи 0/1)] в порядке эталона; факты эталона и вердикта должны совпасть."""
    jf = r["judge_facts"]
    out = []
    for kind, gold_key, flag in (("required", "required_facts", "present"),
                                 ("forbidden", "forbidden_facts", "asserted")):
        gold, judged = q.get(gold_key) or [], jf.get(kind) or []
        if [x["fact"] for x in judged] != list(gold):
            raise SheetError(f"{r['question_id']}: {kind}-факты вердикта не совпадают с эталоном "
                             "(вопросы правили после прогона?)")
        out += [(kind, x["fact"], int(bool(x[flag]))) for x in judged]
    return out


def build_tables(chosen: list[dict], questions: dict[str, dict], seed: int = SEED) -> tuple[list[dict], list[dict]]:
    order = sorted(chosen, key=lambda r: r["question_id"])
    random.Random(seed).shuffle(order)  # порядок не выдаёт тип и номер вопроса
    sheet, key = [], []
    for i, r in enumerate(order, 1):
        label_id = f"L{i:02d}"
        q = questions.get(r["question_id"])
        if q is None:
            raise SheetError(f"{r['question_id']} нет в файле вопросов")
        for kind, fact, judge_value in _facts_with_verdicts(r, q):
            row_id = f"R{len(sheet) + 1:03d}"
            sheet.append({"row_id": row_id, "label_id": label_id, "question": q["question"],
                          "system_answer": r["answer"], "fact_kind": kind, "fact": fact, "human": "", "comment": ""})
            key.append({"row_id": row_id, "label_id": label_id, "question_id": r["question_id"], "type": r["type"],
                        "judge_value": judge_value})
    return sheet, key


def write_csv(path: Path, columns: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--eval-json", type=Path, default=DEFAULT_EVAL_JSON)
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--n", type=int, default=N_ANSWERS)
    ap.add_argument("--seed", type=int, default=SEED)
    args = ap.parse_args(argv)

    questions = {q["id"]: q for q in json.loads(args.questions.read_text(encoding="utf-8"))}
    try:
        chosen = select(load_answers(args.eval_json), n=args.n, seed=args.seed)
        sheet, key = build_tables(chosen, questions, args.seed)
    except SheetError as e:
        print(f"ошибка: {e}", file=sys.stderr)
        return 1
    write_csv(args.out_dir / "sheet.csv", SHEET_COLUMNS, sheet)
    write_csv(args.out_dir / "key_private.csv", KEY_COLUMNS, key)

    types: dict[str, int] = {}
    for r in chosen:
        types[r["type"]] = types.get(r["type"], 0) + 1
    n_req = sum(r["fact_kind"] == "required" for r in sheet)
    print(f"ответов: {len(chosen)}; по типам: {dict(sorted(types.items()))}; "
          f"answerable=false: {sum(not r['answerable'] for r in chosen)}")
    print(f"строк: {len(sheet)} (required {n_req}, forbidden {len(sheet) - n_req})")
    print(f"записано: {args.out_dir / 'sheet.csv'}, {args.out_dir / 'key_private.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
