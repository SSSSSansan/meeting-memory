"""РОЛЬ A. Согласие ручной разметки фактов с судьёй judge-facts-v1 (A8). Модель не вызывается.

    python scripts/score_facts_calibration.py
    python scripts/score_facts_calibration.py --sheet results/judge_facts_labeling/sheet.csv --out agreement.md

Читает заполненный sheet.csv и key_private.csv (оба из make_facts_sheet.py) и сравнивает human с judge_value.
Печатает и пишет в agreement.md: % совпадения и Cohen's kappa отдельно для required, forbidden и в целом;
матрицы ошибок 2×2 (строки — человек, столбцы — судья); список расхождений.

fact_f1 на уровне ответов: recall по required-фактам считается дважды — по разметке человека и по судье;
precision у обоих одна и та же — судьи из json-прогона (metrics.fact_precision), потому что утверждения ответа
(claims) человек не размечает. Разница fact_f1 поэтому показывает только расхождение по required.
Ответ идёт в fact_f1, только если все его required-строки размечены.
Незаполненные строки в расчёт не идут. Код возврата: 0 — посчитано, 1 — ошибка данных или ничего не размечено.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from pathlib import Path
from typing import Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # запуск файлом: в sys.path только scripts/
    sys.path.insert(0, str(ROOT))

from scripts.calibrate_judge import cohen_kappa, confusion_matrix  # noqa: E402
from scripts.make_facts_sheet import DEFAULT_EVAL_JSON, DEFAULT_OUT_DIR, MODE  # noqa: E402
from src.eval.judge_facts import harmonic  # noqa: E402

DEFAULT_SHEET = DEFAULT_OUT_DIR / "sheet.csv"
DEFAULT_KEY = DEFAULT_OUT_DIR / "key_private.csv"
DEFAULT_OUT = DEFAULT_OUT_DIR / "agreement.md"
KINDS = ("required", "forbidden")
LABELS = (0, 1)
ANSWER_SHORT = 160


class ScoreError(ValueError):
    """Некорректный лист или ключ: сообщение показывается пользователю."""


def read_csv(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8-sig")
    header = text.splitlines()[0] if text else ""
    delimiter = ";" if header.count(";") > header.count(",") else ","
    return list(csv.DictReader(io.StringIO(text, newline=""), delimiter=delimiter))


def parse_label(value: Optional[str], where: str) -> Optional[int]:
    v = (value or "").strip()
    if not v:
        return None
    if v in ("0", "1"):
        return int(v)
    raise ScoreError(f"{where}: human «{v}» — допустимо только пусто, 0 или 1")


def load_precision(eval_json: Optional[Path], mode: str = MODE) -> dict[str, float]:
    """question_id → fact_precision судьи (первый повтор режима mode)."""
    if eval_json is None or not eval_json.exists():
        return {}
    data = json.loads(eval_json.read_text(encoding="utf-8"))
    out: dict[str, float] = {}
    for r in sorted(data["per_question"], key=lambda r: int(r.get("repeat") or 1)):
        p = (r.get("metrics") or {}).get("fact_precision")
        if r.get("mode") == mode and p is not None:
            out.setdefault(r["question_id"], float(p))
    return out


def join(sheet: list[dict], key: list[dict]) -> tuple[list[dict], int]:
    """Строки листа, соединённые с ключом по row_id; (размеченные строки, всего строк)."""
    gold = {r["row_id"]: r for r in key}
    unknown = [r["row_id"] for r in sheet if r["row_id"] not in gold]
    if unknown:
        raise ScoreError(f"row_id нет в key_private.csv: {unknown[:10]}")
    rows = []
    for r in sheet:
        g = gold[r["row_id"]]
        if r.get("fact_kind") not in KINDS:
            raise ScoreError(f"{r['row_id']}: fact_kind «{r.get('fact_kind')}» не из {', '.join(KINDS)}")
        if g["label_id"] != r["label_id"]:
            raise ScoreError(f"{r['row_id']}: label_id в листе и ключе разный (лист пересобран?)")
        human = parse_label(r.get("human"), r["row_id"])
        judge = parse_label(g.get("judge_value"), f"{r['row_id']} (ключ)")
        if judge is None:
            raise ScoreError(f"{r['row_id']}: в ключе нет judge_value")
        rows.append({**r, "human": human, "judge": judge, "question_id": g["question_id"], "type": g["type"]})
    return rows, len(sheet)


def agreement_block(rows: Sequence[dict]) -> dict:
    h = [r["human"] for r in rows]
    j = [r["judge"] for r in rows]
    n = len(rows)
    same = sum(a == b for a, b in zip(h, j))
    return {"n": n, "same": same, "agreement": same / n if n else None,
            "kappa": cohen_kappa(h, j, LABELS), "matrix": confusion_matrix(h, j, LABELS)}


def answer_f1(all_rows: Sequence[dict], precision: dict[str, float]) -> dict:
    """fact_f1 по ответам: человек против судьи, precision общая (судьи)."""
    by_label: dict[str, list[dict]] = {}
    for r in all_rows:
        if r["fact_kind"] == "required":
            by_label.setdefault(r["label_id"], []).append(r)
    items, skipped = [], []
    for label_id, req in sorted(by_label.items()):
        qid = req[0]["question_id"]
        if any(r["human"] is None for r in req) or qid not in precision:
            skipped.append(label_id)
            continue
        p = precision[qid]
        h_rec = sum(r["human"] for r in req) / len(req)
        j_rec = sum(r["judge"] for r in req) / len(req)
        items.append({"label_id": label_id, "question_id": qid, "type": req[0]["type"], "precision": p,
                      "human_recall": h_rec, "judge_recall": j_rec,
                      "human_f1": harmonic(h_rec, p), "judge_f1": harmonic(j_rec, p)})
    n = len(items)
    res: dict = {"items": items, "skipped": skipped, "n": n}
    if n:
        res["human_mean"] = sum(i["human_f1"] for i in items) / n
        res["judge_mean"] = sum(i["judge_f1"] for i in items) / n
        res["diff"] = res["judge_mean"] - res["human_mean"]
        res["mean_abs_diff"] = sum(abs(i["judge_f1"] - i["human_f1"]) for i in items) / n
        res["n_differ"] = sum(abs(i["judge_f1"] - i["human_f1"]) > 1e-9 for i in items)
    return res


def score(sheet: list[dict], key: list[dict], precision: dict[str, float]) -> dict:
    all_rows, total = join(sheet, key)
    filled = [r for r in all_rows if r["human"] is not None]
    if not filled:
        raise ScoreError("ни одна строка не размечена (колонка human пустая)")
    return {
        "total": total, "filled": len(filled),
        "by_kind": {k: agreement_block([r for r in filled if r["fact_kind"] == k]) for k in KINDS},
        "overall": agreement_block(filled),
        "mismatches": [r for r in filled if r["human"] != r["judge"]],
        "f1": answer_f1(all_rows, precision),
    }


def pct(x: Optional[float]) -> str:
    return "—" if x is None else f"{x * 100:.1f}%"


def fmt(x: Optional[float], nd: int = 3) -> str:
    return "не определён" if x is None else f"{x:.{nd}f}"


def _cell(text: str, n: int = ANSWER_SHORT) -> str:
    t = " ".join((text or "").split()).replace("|", "/")
    return t if len(t) <= n else t[: n - 1] + "…"


def render(res: dict, *, sheet_path: str = "", eval_json: str = "") -> str:
    names = {"required": "required (факт есть в ответе)", "forbidden": "forbidden (ответ утверждает)"}
    out = ["# Согласие ручной разметки фактов с судьёй judge-facts-v1", "",
           f"Лист: `{sheet_path}`; вердикты судьи: `{eval_json}` (режим {MODE}). Модель при подсчёте не вызывалась.",
           "",
           f"- Строк в листе: {res['total']}; размечено: {res['filled']}; "
           f"не размечено: {res['total'] - res['filled']}.",
           "", "## Совпадение и kappa", "",
           "| Факты | Строк | Совпало | % совпадения | Cohen's kappa |", "|---|---|---|---|---|"]
    for name, b in [*((names[k], res["by_kind"][k]) for k in KINDS), ("**в целом**", res["overall"])]:
        out.append(f"| {name} | {b['n']} | {b['same']} | {pct(b['agreement'])} | {fmt(b['kappa'])} |")
    out += ["", "Kappa «не определён» — у человека и судьи одна и та же единственная метка (или строк нет).", "",
            "## Матрицы ошибок (строки — человек, столбцы — судья)", ""]
    for name, b in [*((k, res["by_kind"][k]) for k in KINDS), ("в целом", res["overall"])]:
        m = b["matrix"]
        out += [f"**{name}**", "", "| человек \\ судья | 0 | 1 |", "|---|---|---|",
                f"| 0 | {m[0][0]} | {m[0][1]} |", f"| 1 | {m[1][0]} | {m[1][1]} |", ""]
    f1 = res["f1"]
    out += ["## fact_f1 по ответам: человек против судьи", "",
            "Recall по required — из разметки человека и из вердикта судьи; precision у обоих — судьи "
            "(утверждения ответа человек не размечает), поэтому разница показывает только расхождение по required.",
            ""]
    if f1["n"]:
        out += [f"- Ответов: {f1['n']} (пропущено, не все required размечены или нет precision: {len(f1['skipped'])}).",
                f"- Среднее fact_f1: человек **{f1['human_mean']:.3f}**, судья **{f1['judge_mean']:.3f}**; "
                f"разница (судья − человек) **{f1['diff']:+.3f}**; средний |Δ| {f1['mean_abs_diff']:.3f}; "
                f"ответов с разным f1: {f1['n_differ']}.", ""]
        differ = [i for i in f1["items"] if abs(i["judge_f1"] - i["human_f1"]) > 1e-9]
        if differ:
            out += ["| label_id | question_id | Тип | precision | recall человек | recall судья "
                    "| f1 человек | f1 судья |",
                    "|---|---|---|---|---|---|---|---|"]
            out += [f"| {i['label_id']} | {i['question_id']} | {i['type']} | {i['precision']:.2f} | "
                    f"{i['human_recall']:.2f} | {i['judge_recall']:.2f} | {i['human_f1']:.3f} | {i['judge_f1']:.3f} |"
                    for i in differ]
            out.append("")
    else:
        out += ["- Нет ни одного ответа с полностью размеченными required.", ""]
    out += [f"## Расхождения ({len(res['mismatches'])})", ""]
    if res["mismatches"]:
        out += ["| row_id | label_id | question_id | Тип | Вид | Факт | Человек | Судья "
                "| Ответ системы | Комментарий |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        out += [f"| {r['row_id']} | {r['label_id']} | {r['question_id']} | {r['type']} | {r['fact_kind']} | "
                f"{_cell(r['fact'], 80)} | {r['human']} | {r['judge']} | {_cell(r['system_answer'])} | "
                f"{_cell(r.get('comment') or '', 80)} |" for r in res["mismatches"]]
    else:
        out.append("Нет.")
    return "\n".join(out) + "\n"


def print_summary(res: dict) -> None:
    print(f"размечено строк: {res['filled']} из {res['total']}")
    for name, b in [*((k, res["by_kind"][k]) for k in KINDS), ("в целом", res["overall"])]:
        m = b["matrix"]
        print(f"{name}: {b['same']}/{b['n']} = {pct(b['agreement'])}, kappa {fmt(b['kappa'])}; "
              f"матрица [ч\\с] 0: {m[0]} 1: {m[1]}")
    f1 = res["f1"]
    if f1["n"]:
        print(f"fact_f1 ({f1['n']} ответов): человек {f1['human_mean']:.3f}, судья {f1['judge_mean']:.3f}, "
              f"разница {f1['diff']:+.3f}")
    print(f"расхождений: {len(res['mismatches'])}")
    for r in res["mismatches"]:
        print(f"  {r['row_id']} {r['label_id']} {r['question_id']} {r['fact_kind']}: человек {r['human']}, "
              f"судья {r['judge']} — {_cell(r['fact'], 80)}")


def _rel(p: Path) -> str:
    p = p.resolve()
    return p.relative_to(ROOT).as_posix() if p.is_relative_to(ROOT) else str(p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    ap.add_argument("--key", type=Path, default=DEFAULT_KEY)
    ap.add_argument("--eval-json", type=Path, default=DEFAULT_EVAL_JSON)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    try:
        res = score(read_csv(args.sheet), read_csv(args.key), load_precision(args.eval_json))
    except (ScoreError, OSError, KeyError) as e:
        print(f"ошибка: {e}", file=sys.stderr)
        return 1
    print_summary(res)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(res, sheet_path=_rel(args.sheet), eval_json=_rel(args.eval_json)), encoding="utf-8")
    print(f"записано: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
