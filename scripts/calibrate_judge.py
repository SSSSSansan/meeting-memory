"""РОЛЬ A. Калибровка LLM-судьи (A8): судья оценивает те же 20 ответов, что размечены вручную; считаем согласованность.

    python scripts/calibrate_judge.py --dry-run        # план, число вызовов, оценка стоимости, без модели
    python scripts/calibrate_judge.py --confirm-paid   # реальный прогон (LLM_BACKEND не stub): 20 вызовов модели

Читает results/judge_labeling/labeling_sheet.csv (с колонками human_score, human_fabrication), пишет
  results/judge_labeling/judge_output.csv — label_id, judge_score, judge_fabrication, reason;
  results/judge_labeling/agreement.md     — % точных совпадений, Cohen kappa (обычный и квадратично взвешенный)
                                            по score, совпадение и kappa по fabrication, матрица 3x3, расхождения.
Судья видит только вопрос, эталон, ответ системы и answerable (см. src/eval/judge.py); ручные оценки,
комментарий и cited_meetings ему не передаются. Платный backend: без --confirm-paid прогон не стартует
(как в src.eval.run); лимит расходов держит scripts/cost_guard внутри llm_client.complete.
kappa считается вручную, sklearn не нужен.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # запуск файлом: в sys.path только scripts/
    sys.path.insert(0, str(ROOT))

import src.llm_client  # noqa: E402,F401  — читает .env до того, как смотрим на LLM_BACKEND
from scripts import cost_guard  # noqa: E402
from src.eval import judge as judge_mod  # noqa: E402
from src.eval.estimate import CHARS_PER_TOKEN  # noqa: E402

DEFAULT_DIR = ROOT / "results" / "judge_labeling"
DEFAULT_SHEET = DEFAULT_DIR / "labeling_sheet.csv"
DEFAULT_QUESTIONS = ROOT / "data" / "gold" / "questions.json"
SCORES = (0, 1, 2)
FABS = (0, 1)
OUTPUT_COLUMNS = ["label_id", "judge_score", "judge_fabrication", "reason"]
COMPLETION_TOKENS_PER_CALL = 100  # JSON с reason до ~200 символов
ANSWER_SHORT = 160

_FAB_VALUES = {"да": 1, "yes": 1, "true": 1, "1": 1, "нет": 0, "no": 0, "false": 0, "0": 0}


class SheetError(ValueError):
    """Лист нельзя прочитать вообще (нет файла, нет шапки, нет нужных колонок)."""


# ---------- чтение листа ----------

@dataclass
class SheetRow:
    label_id: str
    question: str
    expected_answer: str
    system_answer: str
    answerable: bool
    human_score: Optional[int]
    human_fabrication: Optional[int]


def parse_score(value: str) -> Optional[int]:
    v = (value or "").strip()
    return int(v) if v in {"0", "1", "2"} else None


def parse_fabrication(value: str) -> Optional[int]:
    return _FAB_VALUES.get((value or "").strip().lower())


def _read_table(path: Path) -> list[dict[str, str]]:
    """CSV из генератора (запятая) и из Excel (точка с запятой, лишняя строка «Column1;Column2…» над шапкой)."""
    try:
        text = Path(path).read_bytes().decode("utf-8-sig")
    except OSError as e:
        raise SheetError(f"не удалось прочитать {path}: {e}") from None
    lines = text.splitlines(keepends=True)
    start = next((i for i, ln in enumerate(lines) if "label_id" in ln), None)
    if start is None:
        raise SheetError(f"в {path} не найдена шапка с колонкой label_id")
    header = lines[start]
    delim = max((";", ",", "\t"), key=header.count)
    reader = csv.DictReader(io.StringIO("".join(lines[start:])), delimiter=delim)
    need = {"label_id", "question", "expected_answer", "system_answer", "human_score", "human_fabrication"}
    missing = need - set(reader.fieldnames or [])
    if missing:
        raise SheetError("в листе нет колонок: " + ", ".join(sorted(missing)))
    return [r for r in reader if (r.get("label_id") or "").strip()]


def load_answerable(questions_path: Path) -> dict[str, bool]:
    """question → answerable из gold; лист вопроса-идентификатора не содержит, связываем по тексту."""
    try:
        raw = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {q["question"].strip(): bool(q["answerable"]) for q in raw}


def load_sheet(path: Path, answerable_by_question: dict[str, bool]) -> tuple[list[SheetRow], list[str]]:
    """→ (строки, замечания). Замечания: некорректные ручные оценки, answerable, выведенный по эталону."""
    rows: list[SheetRow] = []
    notes: list[str] = []
    for r in _read_table(path):
        lid = r["label_id"].strip()
        question = r["question"].strip()
        expected = r["expected_answer"].strip()
        answerable = answerable_by_question.get(question)
        if answerable is None:  # запасной путь: так размечали вручную («в эталоне написано, что информации нет»)
            answerable = not expected.lower().startswith("в данных нет")
            notes.append(f"{lid}: вопроса нет в questions.json, answerable={answerable} выведен по эталону")
        score, fab = parse_score(r["human_score"]), parse_fabrication(r["human_fabrication"])
        if score is None:
            notes.append(f"{lid}: human_score={r['human_score']!r} — допустимо 0, 1, 2")
        if fab is None:
            notes.append(f"{lid}: human_fabrication={r['human_fabrication']!r} — допустимо да/нет или 1/0")
        rows.append(SheetRow(lid, question, expected, r["system_answer"].strip(), answerable, score, fab))
    if not rows:
        raise SheetError(f"в {path} нет строк с label_id")
    return rows, notes


def label_problems(notes: Sequence[str]) -> list[str]:
    return [n for n in notes if "human_" in n]


# ---------- прогон судьи ----------

@dataclass
class JudgeRow:
    label_id: str
    score: Optional[int]
    fabrication: Optional[int]
    reason: str


JudgeFn = Callable[[str, str, str, bool], judge_mod.JudgeResult]


def run_judge(rows: Sequence[SheetRow], judge_fn: JudgeFn) -> list[JudgeRow]:
    """Судье уходят только вопрос, эталон, ответ системы и answerable. Битый формат — строка с ошибкой, не падение;
    BudgetExceeded и прочие сбои не глотаем."""
    out: list[JudgeRow] = []
    for r in rows:
        try:
            v = judge_fn(r.question, r.expected_answer, r.system_answer, r.answerable)
            out.append(JudgeRow(r.label_id, v.score, v.fabrication, v.reason))
        except judge_mod.JudgeFormatError as e:
            out.append(JudgeRow(r.label_id, None, None, f"ОШИБКА ФОРМАТА: {e}"))
    return out


def write_judge_csv(path: Path, rows: Sequence[JudgeRow]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(OUTPUT_COLUMNS)
        for r in rows:
            w.writerow([r.label_id, "" if r.score is None else r.score,
                        "" if r.fabrication is None else r.fabrication, r.reason])


# ---------- согласованность ----------

def confusion_matrix(a: Sequence[int], b: Sequence[int], labels: Sequence[int]) -> list[list[int]]:
    """Строки — оценка человека (a), столбцы — судьи (b)."""
    idx = {lab: i for i, lab in enumerate(labels)}
    m = [[0] * len(labels) for _ in labels]
    for x, y in zip(a, b):
        m[idx[x]][idx[y]] += 1
    return m


def exact_agreement(a: Sequence[int], b: Sequence[int]) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cohen_kappa(a: Sequence[int], b: Sequence[int], labels: Sequence[int], *,
                weights: Optional[str] = None) -> Optional[float]:
    """Cohen's kappa; weights=None — обычный, "quadratic" — квадратично взвешенный (штраф ~ (i-j)²).

    None, если kappa не определён: нет данных или у обоих оценщиков одна и та же единственная метка."""
    n = len(a)
    if n == 0 or n != len(b):
        return None
    k = len(labels)
    obs = confusion_matrix(a, b, labels)
    row = [sum(r) for r in obs]
    col = [sum(obs[i][j] for i in range(k)) for j in range(k)]

    def w(i: int, j: int) -> float:  # вес расхождения: 0 на диагонали
        if weights == "quadratic":
            return (i - j) ** 2 / (k - 1) ** 2
        return 0.0 if i == j else 1.0

    observed = sum(w(i, j) * obs[i][j] for i in range(k) for j in range(k)) / n
    expected = sum(w(i, j) * row[i] * col[j] for i in range(k) for j in range(k)) / (n * n)
    if expected == 0:
        return None
    return 1 - observed / expected


def disagreements(human: Sequence[SheetRow], judged: dict[str, JudgeRow]) -> list[tuple[SheetRow, JudgeRow]]:
    """Расхождение больше чем на 1 балл или по fabrication."""
    out = []
    for h in human:
        j = judged.get(h.label_id)
        if j is None or j.score is None or h.human_score is None or h.human_fabrication is None:
            continue
        if abs(h.human_score - j.score) > 1 or h.human_fabrication != j.fabrication:
            out.append((h, j))
    return out


def _fmt_kappa(k: Optional[float]) -> str:
    return "не определён (у обоих оценщиков одна и та же метка во всех строках)" if k is None else f"{k:.3f}"


def _short(text: str, n: int = ANSWER_SHORT) -> str:
    t = " ".join(text.split())
    return t if len(t) <= n else t[: n - 1] + "…"


def _git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                           timeout=10)
        return r.stdout.strip() or "n/a"
    except Exception:
        return "n/a"


def render_report(human: Sequence[SheetRow], judged: Sequence[JudgeRow], *, date: str, backend: str,
                  model: str) -> str:
    by_id = {j.label_id: j for j in judged}
    pairs = [(h, by_id[h.label_id]) for h in human
             if h.label_id in by_id and by_id[h.label_id].score is not None
             and h.human_score is not None and h.human_fabrication is not None]
    failed = [j for j in judged if j.score is None]
    hs = [h.human_score for h, _ in pairs]
    js = [j.score for _, j in pairs]
    hf = [h.human_fabrication for h, _ in pairs]
    jf = [j.fabrication for _, j in pairs]
    n = len(pairs)

    lines = [f"# Согласованность судьи с ручной разметкой ({date})", "",
             f"- commit: {_git_commit()}, backend: {backend}, model: {model}, "
             f"JUDGE_PROMPT_VERSION: {judge_mod.JUDGE_PROMPT_VERSION}, температура 0",
             f"- строк в листе: {len(human)}, в расчёте: {n}"
             + (f", судья не вернул корректный вердикт: {', '.join(j.label_id for j in failed)}" if failed else ""),
             f"- **Ограничение:** судья и отвечающая система работают на одной модели ({model}), это возможное "
             "смещение в пользу собственных ответов; судья видит только эталон, не транскрипты.",
             f"- **n = {n}**: доверительные интервалы широкие, цифры ориентировочные."]
    if n == 0:
        return "\n".join(lines + ["", "Нет ни одной пары оценок — считать нечего."]) + "\n"

    exact = exact_agreement(hs, js)
    kappa_w = cohen_kappa(hs, js, SCORES, weights="quadratic")
    lines += ["", "## score (0 / 1 / 2)", "",
              f"- точное совпадение: **{exact * 100:.0f}%** ({sum(x == y for x, y in zip(hs, js))} из {n})",
              f"- Cohen kappa: **{_fmt_kappa(cohen_kappa(hs, js, SCORES))}**",
              f"- Cohen kappa, квадратично взвешенный: **{_fmt_kappa(kappa_w)}**",
              f"- расхождение ровно на 1 балл: {sum(abs(x - y) == 1 for x, y in zip(hs, js))}, "
              f"больше чем на 1: {sum(abs(x - y) > 1 for x, y in zip(hs, js))}"]
    lines += ["", "Матрица ошибок: строки — человек, столбцы — судья.", "",
              "| человек \\ судья | 0 | 1 | 2 |", "|---|---|---|---|"]
    for lab, row in zip(SCORES, confusion_matrix(hs, js, SCORES)):
        lines.append(f"| {lab} | " + " | ".join(str(x) for x in row) + " |")

    fab_agree = sum(x == y for x, y in zip(hf, jf))
    lines += ["", "## fabrication (0 / 1)", "",
              f"- совпадение: **{fab_agree / n * 100:.0f}%** ({fab_agree} из {n})",
              f"- Cohen kappa: **{_fmt_kappa(cohen_kappa(hf, jf, FABS))}**",
              f"- fabrication=1: человек {sum(hf)}, судья {sum(jf)} (при малом числе единиц kappa нестабилен)"]

    bad = disagreements(human, by_id)
    lines += ["", "## Расхождения (больше 1 балла или по fabrication)", ""]
    if not bad:
        lines.append("Нет.")
    for h, j in bad:
        lines += [f"- **{h.label_id}**: человек score={h.human_score}, fab={h.human_fabrication}; "
                  f"судья score={j.score}, fab={j.fabrication}",
                  f"  - вопрос: {_short(h.question, 200)}",
                  f"  - ответ системы: {_short(h.system_answer)}",
                  f"  - причина судьи: {j.reason}"]

    lines += ["", "## Все строки", "", "| label_id | answerable | человек score/fab | судья score/fab |",
              "|---|---|---|---|"]
    for h in human:
        j = by_id.get(h.label_id)
        jv = "—" if j is None or j.score is None else f"{j.score}/{j.fabrication}"
        ans = "да" if h.answerable else "нет"
        lines.append(f"| {h.label_id} | {ans} | {h.human_score}/{h.human_fabrication} | {jv} |")
    return "\n".join(lines) + "\n"


# ---------- стоимость ----------

@dataclass
class CostEstimate:
    calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_lo: Optional[float]
    cost_hi: Optional[float]    # если каждый вердикт придётся повторить из-за формата


def estimate_cost(rows: Sequence[SheetRow]) -> CostEstimate:
    chars = sum(len(m["content"]) for r in rows
                for m in judge_mod.build_messages(r.question, r.expected_answer, r.system_answer, r.answerable))
    p_tok = chars // CHARS_PER_TOKEN
    c_tok = len(rows) * COMPLETION_TOKENS_PER_CALL
    p_in, p_out = (float(os.environ.get(k, "0") or 0) for k in ("PRICE_IN_PER_M", "PRICE_OUT_PER_M"))
    if not (p_in or p_out):
        return CostEstimate(len(rows), p_tok, c_tok, None, None)
    lo = (p_tok * p_in + c_tok * p_out) / 1e6
    hi = lo + (p_tok * 1.4 * p_in + c_tok * p_out) / 1e6  # ретрай: тот же вход + ответ + замечание ≈ 2.4× вход
    return CostEstimate(len(rows), p_tok, c_tok, lo, hi)


def _remaining_budget() -> float:
    spent = json.loads(cost_guard.BUDGET_FILE.read_text(encoding="utf-8"))["spent_usd"] \
        if cost_guard.BUDGET_FILE.exists() else 0.0
    return float(os.environ.get("BUDGET_USD", "3.0")) - spent


# ---------- CLI ----------

def main(argv: Optional[list[str]] = None, judge_fn: Optional[JudgeFn] = None) -> int:
    ap = argparse.ArgumentParser(description="Калибровка LLM-судьи на ручной разметке")
    ap.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--dry-run", action="store_true", help="план и оценка стоимости, без вызовов модели")
    ap.add_argument("--confirm-paid", action="store_true", help="разрешить прогон с платным backend")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # консоль Windows (cp1251) не знает казахских букв
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    try:
        rows, notes = load_sheet(a.sheet, load_answerable(a.questions))
    except SheetError as e:
        print(f"Лист не прочитан: {e}", file=sys.stderr)
        return 2
    backend = os.environ.get("LLM_BACKEND", "stub")
    model = os.environ.get("LLM_MODEL", "stub-model")
    print(f"Строк: {len(rows)}, backend: {backend}, model: {model}, JUDGE_PROMPT_VERSION: "
          f"{judge_mod.JUDGE_PROMPT_VERSION}")
    print(cost_guard.status() + f" (лимит BUDGET_USD={os.environ.get('BUDGET_USD', '3.0')}, cost_guard включён: "
          "llm_client.complete проверяет его перед каждым вызовом)")
    est = estimate_cost(rows)
    print(f"Оценка: вызовов модели {est.calls} (+ повтор только при невалидном JSON, максимум {2 * est.calls}), "
          f"~{est.prompt_tokens} токенов на входе и ~{est.completion_tokens} на выходе")
    print("Стоимость: цены не заданы (PRICE_IN_PER_M / PRICE_OUT_PER_M в .env = 0)" if est.cost_lo is None
          else f"Стоимость: ~${est.cost_lo:.4f}, не больше ~${est.cost_hi:.4f} с повторами")
    for n in notes:
        print(f"ВНИМАНИЕ: {n}")
    problems = label_problems(notes)
    if a.dry_run:
        print("--dry-run: модель не вызывалась." + (" Исправьте ручные оценки до запуска." if problems else ""))
        return 0
    if problems:
        print("Ручные оценки некорректны, запуск остановлен до вызовов модели (деньги не потрачены).",
              file=sys.stderr)
        return 2
    if backend == "stub":
        print("LLM_BACKEND=stub: заглушка не оценивает ответы. Для проверки без денег — --dry-run; "
              "для реального прогона поставьте LLM_BACKEND=openai.", file=sys.stderr)
        return 2
    if backend != "stub" and not a.confirm_paid:
        print(f"\nLLM_BACKEND={backend} — прогон платный. Подтвердите числа выше и запустите с --confirm-paid.",
              file=sys.stderr)
        return 3
    if est.cost_hi is not None and est.cost_hi > _remaining_budget():
        print("Оценка стоимости больше остатка бюджета (cost_guard). Остановлено.", file=sys.stderr)
        return 4

    judge_fn = judge_fn or judge_mod.judge
    judged = run_judge(rows, judge_fn)
    out_csv, out_md = a.out_dir / "judge_output.csv", a.out_dir / "agreement.md"
    write_judge_csv(out_csv, judged)
    report = render_report(rows, judged, date=dt.date.today().isoformat(), backend=backend, model=model)
    out_md.write_text(report, encoding="utf-8")
    print(f"\n{report}\nВывод судьи: {out_csv}\nОтчёт: {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
