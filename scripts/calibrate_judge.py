"""РОЛЬ A. Калибровка LLM-судьи (A8): судья оценивает те же 20 ответов, что размечены вручную; считаем согласованность.

    python scripts/calibrate_judge.py --version v2 --dry-run       # план, число вызовов, оценка стоимости, без модели
    python scripts/calibrate_judge.py --version v2 --confirm-paid  # реальный прогон (LLM_BACKEND не stub)
    python scripts/calibrate_judge.py --version v1 ...             # v1 без транскриптов (те же ответы из кэша, $0)

Читает results/judge_labeling/labeling_sheet.csv (с колонками human_score, human_fabrication), пишет
  results/judge_labeling/judge_output_<версия>.csv — label_id, judge_score, judge_fabrication, reason;
  results/judge_labeling/agreement_<версия>.md     — % точных совпадений, Cohen kappa (обычный и квадратично
      взвешенный) по score, совпадение и kappa по fabrication, матрица 3x3, расхождения; для v2 в конце ещё
      сравнение с v1 (v1 берётся из judge_output_v1.csv, а если его нет — из старого judge_output.csv).
Файлы без суффикса версии (первый прогон v1) не перезаписываются.

v1: судья видит только вопрос, эталон, ответ системы и answerable. v2: дополнительно чистые транскрипты встреч —
source_meetings вопроса ∪ cited_meetings ответа, не позже as_of (src/eval/judge.py::select_meetings). Ручные
оценки и комментарий судье не передаются никогда; source_meetings, as_of и cited_meetings служат только для
выбора встреч. Связь label_id → question_id — через key_private.csv (читаются только эти две колонки), вопросы
— data/gold/questions.json. Платный backend: без --confirm-paid прогон не стартует (как в src.eval.run); лимит
расходов держит scripts/cost_guard внутри llm_client.complete. kappa считается вручную, sklearn не нужен.
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
DEFAULT_KEY = DEFAULT_DIR / "key_private.csv"
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
VERSIONS = {"v1": judge_mod.JUDGE_PROMPT_VERSION_V1, "v2": judge_mod.JUDGE_PROMPT_VERSION_V2}
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
    # Только для выбора транскриптов в v2; в промпт эти поля не попадают.
    question_id: str = ""
    as_of: str = ""
    source_meetings: tuple[str, ...] = ()
    cited_meetings: tuple[str, ...] = ()


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


def load_questions(questions_path: Path) -> list[dict]:
    try:
        return list(json.loads(Path(questions_path).read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return []


def load_answerable(questions_path: Path) -> dict[str, bool]:
    """question → answerable из gold (лист идентификатора вопроса не содержит)."""
    return {q["question"].strip(): bool(q["answerable"]) for q in load_questions(questions_path)}


def load_key(key_path: Path) -> dict[str, str]:
    """label_id → question_id из key_private.csv. Тип, split и язык не читаются: они нужны только человеку."""
    try:
        with open(key_path, encoding="utf-8-sig", newline="") as f:
            return {r["label_id"].strip(): r["question_id"].strip() for r in csv.DictReader(f)}
    except (OSError, KeyError):
        return {}


def _split_ids(value: str) -> tuple[str, ...]:
    return tuple(x.strip() for x in (value or "").replace(";", ",").split(",") if x.strip())


def load_sheet(path: Path, answerable_by_question: Optional[dict[str, bool]] = None, *,
               questions: Optional[Sequence[dict]] = None,
               key: Optional[dict[str, str]] = None) -> tuple[list[SheetRow], list[str]]:
    """→ (строки, замечания). Замечания: некорректные ручные оценки (human_…), проблемы связи с вопросом (meta:),
    answerable, выведенный по эталону. questions + key нужны для v2 (as_of, source_meetings)."""
    by_id = {q["id"]: q for q in questions or [] if "id" in q}
    by_text = {q["question"].strip(): q for q in questions or []}
    answerable_by_question = answerable_by_question or {}
    rows: list[SheetRow] = []
    notes: list[str] = []
    for r in _read_table(path):
        lid = r["label_id"].strip()
        question = r["question"].strip()
        expected = r["expected_answer"].strip()
        q = by_id.get((key or {}).get(lid, ""))
        if q is not None and q["question"].strip() != question:
            notes.append(f"{lid}: meta: key_private связывает с {q['id']}, но текст вопроса другой")
            q = None
        if q is None:
            q = by_text.get(question)
        if questions is not None and q is None:
            notes.append(f"{lid}: meta: вопрос не найден в questions.json (нет as_of и source_meetings)")
        answerable = bool(q["answerable"]) if q is not None else answerable_by_question.get(question)
        if answerable is None:  # запасной путь: так размечали вручную («в эталоне написано, что информации нет»)
            answerable = not expected.lower().startswith("в данных нет")
            notes.append(f"{lid}: вопроса нет в questions.json, answerable={answerable} выведен по эталону")
        score, fab = parse_score(r["human_score"]), parse_fabrication(r["human_fabrication"])
        if score is None:
            notes.append(f"{lid}: human_score={r['human_score']!r} — допустимо 0, 1, 2")
        if fab is None:
            notes.append(f"{lid}: human_fabrication={r['human_fabrication']!r} — допустимо да/нет или 1/0")
        rows.append(SheetRow(
            lid, question, expected, r["system_answer"].strip(), answerable, score, fab,
            question_id=q["id"] if q else "", as_of=q["as_of"] if q else "",
            source_meetings=tuple(q["source_meetings"]) if q else (),
            cited_meetings=_split_ids(r.get("cited_meetings", ""))))
    if not rows:
        raise SheetError(f"в {path} нет строк с label_id")
    return rows, notes


def label_problems(notes: Sequence[str]) -> list[str]:
    return [n for n in notes if "human_" in n]


def meta_problems(notes: Sequence[str]) -> list[str]:
    return [n for n in notes if ": meta:" in n]


def select_for_rows(rows: Sequence[SheetRow], raw_dir: Path) -> dict[str, judge_mod.SelectedMeetings]:
    """label_id → встречи для судьи v2: source_meetings ∪ cited_meetings, не позже as_of, чистые тексты."""
    return {r.label_id: judge_mod.select_meetings(r.source_meetings, r.cited_meetings, r.as_of, raw_dir)
            for r in rows}


# ---------- прогон судьи ----------

@dataclass
class JudgeRow:
    label_id: str
    score: Optional[int]
    fabrication: Optional[int]
    reason: str


JudgeFn = Callable[..., judge_mod.JudgeResult]


def run_judge(rows: Sequence[SheetRow], judge_fn: JudgeFn,
              selected: Optional[dict[str, judge_mod.SelectedMeetings]] = None) -> list[JudgeRow]:
    """v1 (selected=None): судье уходят только вопрос, эталон, ответ системы и answerable. v2: ещё тексты встреч.
    Битый формат — строка с ошибкой, не падение; BudgetExceeded и прочие сбои не глотаем."""
    out: list[JudgeRow] = []
    for r in rows:
        try:
            if selected is None:
                v = judge_fn(r.question, r.expected_answer, r.system_answer, r.answerable)
            else:
                v = judge_fn(r.question, r.expected_answer, r.system_answer, r.answerable,
                             with_transcripts=True, meetings=selected[r.label_id].docs)
            out.append(JudgeRow(r.label_id, v.score, v.fabrication, v.reason))
        except judge_mod.JudgeFormatError as e:
            out.append(JudgeRow(r.label_id, None, None, f"ОШИБКА ФОРМАТА: {e}"))
    return out


def load_judge_csv(path: Path) -> list[JudgeRow]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return [JudgeRow(r["label_id"], int(r["judge_score"]) if r["judge_score"] != "" else None,
                         int(r["judge_fabrication"]) if r["judge_fabrication"] != "" else None, r["reason"])
                for r in csv.DictReader(f)]


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
                  model: str, version: str = "v1", answer_model: Optional[str] = None,
                  selected: Optional[dict[str, judge_mod.SelectedMeetings]] = None) -> str:
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

    same_model = answer_model is None or answer_model == model
    if same_model:
        limit = (f"судья и отвечающая система работают на одной модели ({model}), это возможное смещение в "
                 "пользу собственных ответов")
    else:
        limit = f"судья ({model}) и отвечающая система ({answer_model}) — разные модели"
    limit += ("; судья видит транскрипты выбранных встреч, fabrication проверяется только по ним и эталону."
              if selected is not None else "; судья видит только эталон, не транскрипты.")
    lines = [f"# Согласованность судьи с ручной разметкой, {version} ({date})", "",
             f"- commit: {_git_commit()}, backend: {backend}, model судьи: {model}, "
             f"JUDGE_PROMPT_VERSION: {VERSIONS[version]}, температура 0",
             f"- строк в листе: {len(human)}, в расчёте: {n}"
             + (f", судья не вернул корректный вердикт: {', '.join(j.label_id for j in failed)}" if failed else ""),
             f"- **Ограничение:** {limit}",
             f"- **n = {n}**: доверительные интервалы широкие, цифры ориентировочные."]
    if selected is not None:
        lines.append("- Транскрипты: чистые (data/raw), source_meetings вопроса ∪ cited_meetings ответа, "
                     "не позже as_of; на score они не влияют.")
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

    head = "| label_id | answerable | человек score/fab | судья score/fab |" + (" встречи |" if selected else "")
    lines += ["", "## Все строки", "", head, "|---|---|---|---|" + ("---|" if selected else "")]
    for h in human:
        j = by_id.get(h.label_id)
        jv = "—" if j is None or j.score is None else f"{j.score}/{j.fabrication}"
        ans = "да" if h.answerable else "нет"
        tail = f" {', '.join(d.meeting_id for d in selected[h.label_id].docs) or '—'} |" if selected else ""
        lines.append(f"| {h.label_id} | {ans} | {h.human_score}/{h.human_fabrication} | {jv} |" + tail)
    return "\n".join(lines) + "\n"


def _metrics(human: Sequence[SheetRow], judged: Sequence[JudgeRow]) -> Optional[dict]:
    by_id = {j.label_id: j for j in judged}
    pairs = [(h, by_id[h.label_id]) for h in human if h.label_id in by_id and by_id[h.label_id].score is not None
             and h.human_score is not None and h.human_fabrication is not None]
    if not pairs:
        return None
    hs, js = [h.human_score for h, _ in pairs], [j.score for _, j in pairs]
    hf, jf = [h.human_fabrication for h, _ in pairs], [j.fabrication for _, j in pairs]
    return {"n": len(pairs), "exact": exact_agreement(hs, js), "kappa": cohen_kappa(hs, js, SCORES),
            "kappa_w": cohen_kappa(hs, js, SCORES, weights="quadratic"),
            "fab_agree": exact_agreement(hf, jf), "fab_kappa": cohen_kappa(hf, jf, FABS),
            "fab_human": sum(hf), "fab_judge": sum(jf)}


def render_comparison(human: Sequence[SheetRow], old: Sequence[JudgeRow], new: Sequence[JudgeRow], *,
                      old_name: str = "v1", new_name: str = "v2") -> str:
    """Сравнительная таблица двух прогонов судьи и список строк, где вердикт изменился."""
    mo, mn = _metrics(human, old), _metrics(human, new)
    if mo is None or mn is None:
        return f"\n## Сравнение {old_name} и {new_name}\n\nНет пар оценок для сравнения.\n"

    def k(x: Optional[float]) -> str:
        return "н/д" if x is None else f"{x:.3f}"

    rows = [("строк в расчёте", str(mo["n"]), str(mn["n"])),
            ("score: точное совпадение", f"{mo['exact'] * 100:.0f}%", f"{mn['exact'] * 100:.0f}%"),
            ("score: kappa", k(mo["kappa"]), k(mn["kappa"])),
            ("score: kappa квадратично взвешенный", k(mo["kappa_w"]), k(mn["kappa_w"])),
            ("fabrication: совпадение", f"{mo['fab_agree'] * 100:.0f}%", f"{mn['fab_agree'] * 100:.0f}%"),
            ("fabrication: kappa", k(mo["fab_kappa"]), k(mn["fab_kappa"])),
            ("fabrication=1: человек / судья", f"{mo['fab_human']} / {mo['fab_judge']}",
             f"{mn['fab_human']} / {mn['fab_judge']}")]
    lines = ["", f"## Сравнение {old_name} и {new_name}", "", f"| метрика | {old_name} | {new_name} |", "|---|---|---|"]
    lines += [f"| {a} | {b} | {c} |" for a, b, c in rows]
    o_by, n_by = {j.label_id: j for j in old}, {j.label_id: j for j in new}
    changed = [(h, o_by[h.label_id], n_by[h.label_id]) for h in human
               if h.label_id in o_by and h.label_id in n_by
               and (o_by[h.label_id].score, o_by[h.label_id].fabrication)
               != (n_by[h.label_id].score, n_by[h.label_id].fabrication)]
    lines += ["", f"Строки, где вердикт изменился (score/fab): {len(changed)}", ""]
    if changed:
        lines += [f"| label_id | человек | {old_name} | {new_name} | причина {new_name} |", "|---|---|---|---|---|"]
    for h, o, n in changed:
        fmt = lambda j: "—" if j.score is None else f"{j.score}/{j.fabrication}"  # noqa: E731
        lines.append(f"| {h.label_id} | {h.human_score}/{h.human_fabrication} | {fmt(o)} | {fmt(n)} | "
                     f"{_short(n.reason, 220).replace('|', '/')} |")
    return "\n".join(lines) + "\n"


# ---------- стоимость ----------

@dataclass
class CostEstimate:
    calls: int
    prompt_tokens: int
    completion_tokens: int
    cost_lo: Optional[float]
    cost_hi: Optional[float]    # если каждый вердикт придётся повторить из-за формата


    @property
    def avg_prompt_tokens(self) -> int:
        return self.prompt_tokens // self.calls if self.calls else 0


def estimate_cost(rows: Sequence[SheetRow],
                  selected: Optional[dict[str, judge_mod.SelectedMeetings]] = None) -> CostEstimate:
    """Токены по настоящим сообщениям судьи (v2 — вместе с транскриптами), цены — JUDGE_PRICE_* или PRICE_*."""
    chars = sum(len(m["content"]) for r in rows
                for m in judge_mod.build_messages(
                    r.question, r.expected_answer, r.system_answer, r.answerable,
                    meetings=selected[r.label_id].docs if selected is not None else None))
    p_tok = chars // CHARS_PER_TOKEN
    c_tok = len(rows) * COMPLETION_TOKENS_PER_CALL
    p_in, p_out = judge_mod.judge_prices()
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
    ap.add_argument("--version", choices=list(VERSIONS), default="v2",
                    help="v1: без транскриптов; v2: с транскриптами встреч (по умолчанию)")
    ap.add_argument("--sheet", type=Path, default=DEFAULT_SHEET)
    ap.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    ap.add_argument("--key", type=Path, default=DEFAULT_KEY, help="key_private.csv: label_id → question_id")
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="чистые транскрипты для v2")
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_DIR)
    ap.add_argument("--dry-run", action="store_true", help="план и оценка стоимости, без вызовов модели")
    ap.add_argument("--confirm-paid", action="store_true", help="разрешить прогон с платным backend")
    a = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):  # консоль Windows (cp1251) не знает казахских букв
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    try:
        rows, notes = load_sheet(a.sheet, questions=load_questions(a.questions), key=load_key(a.key))
    except SheetError as e:
        print(f"Лист не прочитан: {e}", file=sys.stderr)
        return 2
    v2 = a.version == "v2"
    selected = select_for_rows(rows, a.raw_dir) if v2 else None
    backend = os.environ.get("LLM_BACKEND", "stub")
    model = judge_mod.judge_model()
    answer_model = os.environ.get("LLM_MODEL", "stub-model")
    p_in, p_out = judge_mod.judge_prices()
    print(f"Версия: {a.version} ({VERSIONS[a.version]}), строк: {len(rows)}, backend: {backend}, "
          f"модель судьи: {model} (отвечающая система: {answer_model}), цены судьи за 1M токенов: "
          f"вход ${p_in}, выход ${p_out}")
    print(cost_guard.status() + f" (лимит BUDGET_USD={os.environ.get('BUDGET_USD', '3.0')}, cost_guard включён: "
          "llm_client.complete проверяет его перед каждым вызовом)")
    est = estimate_cost(rows, selected)
    print(f"Оценка: вызовов модели {est.calls} (+ повтор только при невалидном JSON, максимум {2 * est.calls}), "
          f"в среднем ~{est.avg_prompt_tokens} токенов на входе и ~{COMPLETION_TOKENS_PER_CALL} на выходе на вызов; "
          f"всего ~{est.prompt_tokens} + ~{est.completion_tokens}")
    print("Стоимость: цены не заданы (JUDGE_PRICE_* / PRICE_* в .env = 0)" if est.cost_lo is None
          else f"Стоимость: ~${est.cost_lo:.4f}, не больше ~${est.cost_hi:.4f} с повторами")
    if selected is not None:
        print("Встречи, которые получит судья v2 (source_meetings ∪ cited_meetings, не позже as_of):")
        for r in rows:
            sel = selected[r.label_id]
            print(f"  {r.label_id} ({r.question_id or '?'}, as_of {r.as_of or '?'}): "
                  f"{', '.join(d.meeting_id for d in sel.docs) or 'нет'}"
                  + (f"  [{'; '.join(sel.notes)}]" if sel.notes else ""))
    for n in notes:
        print(f"ВНИМАНИЕ: {n}")
    problems = label_problems(notes) + (meta_problems(notes) if v2 else [])
    if a.dry_run:
        print("--dry-run: модель не вызывалась." + (" Исправьте замечания выше до запуска." if problems else ""))
        return 0
    if problems:
        print("Лист или связь с вопросами некорректны, запуск остановлен до вызовов модели (деньги не потрачены).",
              file=sys.stderr)
        return 2
    if backend == "stub":
        print("LLM_BACKEND=stub: заглушка не оценивает ответы. Для проверки без денег — --dry-run; "
              "для реального прогона поставьте LLM_BACKEND=openai.", file=sys.stderr)
        return 2
    if not a.confirm_paid:
        print(f"\nLLM_BACKEND={backend} — прогон платный. Подтвердите числа выше и запустите с --confirm-paid.",
              file=sys.stderr)
        return 3
    if est.cost_hi is not None and est.cost_hi > _remaining_budget():
        print("Оценка стоимости больше остатка бюджета (cost_guard). Остановлено.", file=sys.stderr)
        return 4

    judged = run_judge(rows, judge_fn or judge_mod.judge, selected)
    out_csv, out_md = a.out_dir / f"judge_output_{a.version}.csv", a.out_dir / f"agreement_{a.version}.md"
    write_judge_csv(out_csv, judged)
    report = render_report(rows, judged, date=dt.date.today().isoformat(), backend=backend, model=model,
                           version=a.version, answer_model=answer_model, selected=selected)
    if v2:
        v1_csv = next((p for p in (a.out_dir / "judge_output_v1.csv", a.out_dir / "judge_output.csv")
                       if p.exists()), None)
        report += (render_comparison(rows, load_judge_csv(v1_csv), judged) if v1_csv else
                   "\n## Сравнение v1 и v2\n\nФайл с результатами v1 (judge_output_v1.csv или judge_output.csv) "
                   "не найден.\n")
    out_md.write_text(report, encoding="utf-8")
    print(f"\n{report}\nВывод судьи: {out_csv}\nОтчёт: {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
