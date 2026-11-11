"""РОЛЬ A. Eval-harness (черновик): прогон вопросов через src.api.ask и простые метрики, без судьи.

    python -m src.eval.run --mode baseline|graph|both --split dev|test|all [--as-of m_008] [--repeats 1]
    python -m src.eval.run --mode both --split test --dry-run      # только план и оценка стоимости
    python -m src.eval.run --mode both --split test --judge --dry-run   # то же вместе с судьёй

Вопросы — data/gold/questions.json; в систему уходит только текст вопроса, эталон она не видит.
У каждого вопроса свой as_of: система видит встречи до него включительно (см. src/eval/systems.py).
--as-of m_008 оставляет только вопросы с этим as_of (по умолчанию — все).

Выход: results/eval_<дата>.csv (по вопросам) и results/summary_<дата>.md (по типам, режимы рядом,
dev и test отдельно). Правило из memory.md: чиним по dev, в отчёт идёт test.
--judge (по умолчанию выключен): после каждого ответа судья v2 (src/eval/judge.py, с транскриптами) ставит
judge_score 0/1/2 и judge_fabrication 0/1; в сводке accuracy = среднее judge_score/2, fabrication_rate = доля
ответов с judge_fabrication=1. Судью зовут одинаково для всех режимов: ему уходят вопрос, эталон, ответ, answerable
и тексты встреч, но не режим, тип и split. Одинаковый вход судья не получает дважды (хэш входа; плюс кэш llm_client
на диске). Без --judge accuracy и fabrication_rate = n/a. Повторы (A9) не реализованы.

Если LLM_BACKEND не stub, прогон платный: сначала печатается оценка (ответы + судья), запуск — только с --confirm-paid.
Лимит расходов держит scripts/cost_guard через src/llm_client (complete/embed).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Callable, Optional

import src.llm_client  # noqa: F401  — читает .env до того, как мы смотрим на LLM_BACKEND
from scripts import cost_guard
from scripts.check_questions import Question, load_json
from src.eval import estimate as est_mod
from src.eval import judge as judge_mod
from src.eval import systems
from src.eval.metrics import MeetingTexts, load_gold_events, score_answer
from src.retrieval.baseline import STEP, WINDOW
from src.schemas import Answer

ROOT = Path(__file__).resolve().parent.parent.parent
QUESTIONS = ROOT / "data" / "gold" / "questions.json"
GOLD = ROOT / "data" / "gold" / "decisions.json"
RAW_DIR = ROOT / "data" / "raw"
INDEX_ROOT = ROOT / "runs" / "eval_index"
RESULTS = ROOT / "results"
TYPES = ["T1", "T2", "T3", "T4", "T5", "none"]
MODES = {"baseline": ["baseline"], "graph": ["graph"], "both": ["baseline", "graph"]}


@dataclass
class Row:
    question_id: str
    type: str
    split: str
    lang: str
    as_of: str
    mode: str
    system: str                     # real | stub
    backend: str                    # LLM_BACKEND; при stub цифры качества ничего не значат
    status: str                     # ok | stub | no_answer | invalid_format | error
    error: str = ""
    answerable: bool = True
    answer: str = ""
    cited_meetings: str = ""
    source_meetings: str = ""
    correct_meetings: Optional[bool] = None
    meetings_hit: Optional[float] = None
    cited_after_as_of: Optional[int] = None
    events_found: Optional[int] = None
    events_total: Optional[int] = None
    citations_total: Optional[int] = None
    citations_valid: Optional[int] = None
    citation_valid: Optional[float] = None
    abstained: Optional[bool] = None
    judge_score: Optional[int] = None          # 0 | 1 | 2; None — судья не запускался или не вернул вердикт
    judge_fabrication: Optional[int] = None    # 0 | 1
    judge_reason: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    cost_usd: Optional[float] = None
    latency_ms: Optional[int] = None
    repeat: int = 1


CSV_FIELDS = [f.name for f in fields(Row)]

JudgeFn = Callable[..., judge_mod.JudgeResult]


@dataclass
class JudgeRunner:
    """Судья v2 в harness. Один вызов на уникальный вход; режим, тип и split судье не передаются."""
    raw_dir: Path
    fn: JudgeFn = judge_mod.judge
    memo: dict[str, Any] = field(default_factory=dict)  # хэш входа → JudgeResult | текст ошибки формата
    calls: int = 0      # разных входов с непустым ответом (пустой ответ судья оценивает без модели)
    reused: int = 0     # вердикт взят из этого же прогона
    failed: int = 0     # судья не вернул вердикт в нужном формате

    def apply(self, row: "Row", q: Question) -> None:
        if row.status != "ok":  # система не ответила: это неверный ответ, а не «пропуск»
            row.judge_score, row.judge_fabrication, row.judge_reason = 0, 0, f"нет ответа системы ({row.status})"
            return
        cited = [m for m in row.cited_meetings.split(",") if m]
        meetings = judge_mod.select_meetings(q.source_meetings, cited, q.as_of, self.raw_dir).docs
        key = judge_mod.input_hash(q.question, q.expected_answer, row.answer, q.answerable, meetings=meetings)
        if key in self.memo:
            self.reused += 1
        else:
            if row.answer.strip():
                self.calls += 1
            try:
                self.memo[key] = self.fn(q.question, q.expected_answer, row.answer, q.answerable,
                                         with_transcripts=True, meetings=meetings)
            except judge_mod.JudgeFormatError as e:
                self.memo[key] = str(e)
        verdict = self.memo[key]
        if isinstance(verdict, str):
            self.failed += 1
            row.judge_reason = f"ОШИБКА ФОРМАТА: {verdict}"
        else:
            row.judge_score, row.judge_fabrication, row.judge_reason = verdict


@dataclass
class JudgeInfo:
    version: str
    model: str
    calls: int
    reused: int
    failed: int


def coerce_answer(result: Any) -> tuple[Optional[Answer], str, str]:
    """Что вернул ask() → (Answer | None, статус, пояснение). Мусор не роняет прогон."""
    if result is None:
        return None, "no_answer", "ask() вернул None"
    if isinstance(result, Answer):
        return result, "ok", ""
    if isinstance(result, dict):
        try:
            return Answer.model_validate(result), "ok", ""
        except Exception as e:
            return None, "invalid_format", f"не формат 4.6: {str(e).splitlines()[0]}"
    return None, "invalid_format", f"ожидался Answer, пришёл {type(result).__name__}"


def run_one(q: Question, system: systems.System, *, gold: dict, texts: MeetingTexts, backend: str) -> Row:
    row = Row(question_id=q.id, type=q.type, split=q.split, lang=q.lang, as_of=q.as_of, mode=system.mode,
              system="stub" if system.is_stub else "real", backend=backend, status="ok",
              answerable=q.answerable, source_meetings=",".join(q.source_meetings))
    t0 = time.perf_counter()
    try:
        result = system.ask(q.question)  # только текст вопроса
    except cost_guard.BudgetExceeded:
        raise  # лимит расходов — стоп всего прогона, а не «ошибка вопроса»
    except Exception as e:
        result, row.status, row.error = None, "error", f"{type(e).__name__}: {e}"
    wall_ms = int((time.perf_counter() - t0) * 1000)

    if system.is_stub:
        row.status, row.error, row.answer = "stub", system.note, ""
        return row
    answer = None
    if row.status != "error":
        answer, row.status, row.error = coerce_answer(result)
    row.latency_ms = (answer.latency_ms if answer and answer.latency_ms else wall_ms)
    row.answer = answer.answer if answer else ""
    citations = answer.citations if answer else []
    m = score_answer(answer=row.answer, citations=citations, answerable=q.answerable,
                     source_meetings=q.source_meetings, decision_ids=q.decision_ids, as_of=q.as_of,
                     gold=gold, texts=texts)
    row.cited_meetings = ",".join(m.cited_meetings)
    row.correct_meetings, row.meetings_hit = m.correct_meetings, m.meetings_hit
    row.cited_after_as_of = m.cited_after_as_of
    row.events_found, row.events_total = m.events_found, m.events_total
    row.citations_total, row.citations_valid = m.citations_total, m.citations_valid
    row.citation_valid = round(m.citations_valid / m.citations_total, 4) if m.citations_total else None
    row.abstained = m.abstained
    usage = answer.usage if answer else None
    row.prompt_tokens = usage.prompt_tokens if usage else 0
    row.completion_tokens = usage.completion_tokens if usage else 0
    row.cost_usd = round(usage.cost_usd, 6) if usage else 0.0
    return row


def run_eval(questions: list[Question], modes: list[str], build: Callable[[str, str], systems.System], *,
             gold: dict, texts: MeetingTexts, backend: str, judge: Optional[JudgeRunner] = None) -> list[Row]:
    """build(mode, as_of) → System. Каждая пара (as_of, режим) строится один раз и отвечает на все свои вопросы."""
    by_asof: dict[str, list[Question]] = defaultdict(list)
    for q in questions:
        by_asof[q.as_of].append(q)
    rows: list[Row] = []
    for as_of, qs in sorted(by_asof.items()):
        for mode in modes:
            try:
                system = build(mode, as_of)
            except cost_guard.BudgetExceeded:
                raise
            except Exception as e:
                for q in qs:
                    row = Row(question_id=q.id, type=q.type, split=q.split, lang=q.lang, as_of=as_of,
                              mode=mode, system="real", backend=backend, status="error",
                              error=f"{type(e).__name__}: {e}", answerable=q.answerable,
                              source_meetings=",".join(q.source_meetings))
                    if judge:
                        judge.apply(row, q)
                    rows.append(row)
                continue
            for q in qs:
                row = run_one(q, system, gold=gold, texts=texts, backend=backend)
                if judge and row.status != "stub":
                    judge.apply(row, q)
                rows.append(row)
    return rows


# ---------- сводка ----------

def _mean(xs: list[float]) -> Optional[float]:
    return sum(xs) / len(xs) if xs else None


def _fmt(x: Optional[float], spec: str = ".2f") -> str:
    return "—" if x is None else format(x, spec)


def _agg(rows: list[Row], qtype: str, judge_on: bool = False) -> dict[str, str]:
    """Метрики группы строк одного режима, уже отформатированные."""
    if rows and all(r.status == "stub" for r in rows):
        return {k: "stub" for k in ("correct", "events", "cite", "abst", "acc", "fab", "pt", "ct", "cost", "lat")}
    judged = [r for r in rows if r.judge_score is not None]
    ans = [r for r in rows if r.answerable]
    ev_tot = sum(r.events_total or 0 for r in ans)
    cit_tot = sum(r.citations_total or 0 for r in rows)
    # abstained: для none — доля отказов (хорошо, когда высокая); для T1–T5 — ложные отказы (хорошо, когда низкая)
    abst_rows = rows if qtype == "none" else ans if qtype != "all" else []
    return {
        "correct": _fmt(_mean([float(bool(r.correct_meetings)) for r in ans])),
        "events": _fmt(sum(r.events_found or 0 for r in ans) / ev_tot) if ev_tot else "—",
        "cite": _fmt(sum(r.citations_valid or 0 for r in rows) / cit_tot) if cit_tot else "—",
        "abst": _fmt(_mean([float(bool(r.abstained)) for r in abst_rows])),
        # без судьи — n/a; с судьёй, но без вердиктов в группе — «—»
        "acc": _fmt(_mean([r.judge_score / 2 for r in judged])) if judge_on else "n/a",
        "fab": _fmt(_mean([float(r.judge_fabrication == 1) for r in judged])) if judge_on else "n/a",
        "pt": _fmt(_mean([float(r.prompt_tokens or 0) for r in rows]), ".0f"),
        "ct": _fmt(_mean([float(r.completion_tokens or 0) for r in rows]), ".0f"),
        "cost": _fmt(sum(r.cost_usd or 0 for r in rows), ".4f"),
        "lat": _fmt(_mean([float(r.latency_ms or 0) for r in rows]), ".0f"),
    }


COLUMNS = [("correct", "correct_meetings"), ("events", "events_found"), ("cite", "citation_valid"),
           ("abst", "abstained*"), ("acc", "accuracy"), ("fab", "fabrication_rate"), ("pt", "prompt_tok"),
           ("ct", "compl_tok"), ("cost", "cost_usd Σ"), ("lat", "latency_ms")]


def split_table(rows: list[Row], modes: list[str], judge_on: bool = False) -> list[str]:
    head = ["Тип", "n"] + [title for _, title in COLUMNS]
    out = ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for qtype in TYPES + ["all"]:
        group = [r for r in rows if qtype == "all" or r.type == qtype]
        n_q = len({r.question_id for r in group})
        if not n_q:
            continue
        per_mode = [_agg([r for r in group if r.mode == m], qtype, judge_on) for m in modes]
        out.append("| " + " | ".join([qtype, str(n_q)] + [" / ".join(a[k] for a in per_mode)
                                                          for k, _ in COLUMNS]) + " |")
    return out


def _git_commit() -> str:
    try:
        r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                           timeout=10)
        return r.stdout.strip() or "n/a"
    except Exception:
        return "n/a"


def render_summary(rows: list[Row], modes: list[str], *, date: str, k: int, depth: int, notes: dict[str, str],
                   splits: list[str], judge: Optional[JudgeInfo] = None) -> str:
    backend = os.environ.get("LLM_BACKEND", "stub")
    lines = [f"# Eval summary {date}", "",
             f"- commit: {_git_commit()}, backend: {backend}, model: {os.environ.get('LLM_MODEL', 'stub-model')}, "
             f"embed: {os.environ.get('EMBED_MODEL') or 'n/a'}",
             f"- PROMPT_VERSION: в коде не задан; k={k}, глубина обхода графа={depth}, "
             f"чанк={WINDOW} реплик со сдвигом {STEP}",
             f"- as_of в вопросах: {', '.join(sorted({r.as_of for r in rows})) or '—'}; "
             "повторов: 1 (mean±std появится в A9)",
             f"- режимы (в ячейках через « / » в этом порядке): {' / '.join(modes)}"]
    if judge is None:
        lines.append("- судья: выключен (--judge не задан), accuracy и fabrication_rate = n/a")
    else:
        same = (" — та же модель, что у системы, возможное смещение в пользу своих ответов"
                if judge.model == os.environ.get("LLM_MODEL", "stub-model") else "")
        lines.append(f"- судья: {judge.version} (с транскриптами встреч), модель {judge.model}{same}; "
                     f"вызовов {judge.calls}, повторных входов пропущено {judge.reused}, "
                     f"без вердикта (ошибка формата) {judge.failed}")
    if backend == "stub":
        lines += ["", "> **LLM_BACKEND=stub**: модель и эмбеддинги — заглушки. Таблица проверяет только "
                      "работу harness, цифры качества ничего не значат."]
    for mode in modes:
        if mode in notes:
            lines += ["", f"> **{mode}: stub** — {notes[mode]}. В таблице вместо цифр «stub»."]
    errs = Counter((r.mode, r.status) for r in rows if r.status not in ("ok", "stub"))
    if errs:
        lines += ["", "> Строки без нормального ответа (считаются в n, метрики по ним нулевые): "
                      + ", ".join(f"{m}/{s}: {c}" for (m, s), c in sorted(errs.items()))]
    notes_md = [
        "correct_meetings — доля ответимых вопросов, где цитаты ссылаются только на встречи из source_meetings "
        "(цитата есть); events_found — найденные события эталона / всего (несколько цитат на событие = 1); "
        "citation_valid — цитаты, найденные в тексте встречи, / все цитаты.",
        "abstained*: для none — доля отказов «данных нет» (выше лучше); для T1–T5 — ложные отказы (ниже лучше). "
        "Определяется по фразам, см. ABSTAIN_RE.",
        "accuracy — среднее judge_score / 2 (2 верно или честный отказ, 1 частично, 0 неверно или не ответили); "
        "fabrication_rate — доля ответов, где судья нашёл выдумку (judge_fabrication = 1); без --judge обе n/a. "
        "Токены и латентность — среднее на вопрос, $ — сумма (стоимость судьи в $ не входит, она в runs/calls.jsonl, "
        "purpose=judge)."]
    for split, title in (("test", "test — цифры для отчёта"), ("dev", "dev — для отладки, в отчёт не идут")):
        if split not in splits:
            continue
        part = [r for r in rows if r.split == split]
        lines += ["", f"## {title}", ""]
        lines += split_table(part, modes, judge is not None) if part else ["_нет вопросов_"]
        if split == "test" and part:
            lines += ["", "_На каждый тип приходится 1–2 тестовых вопроса: это иллюстрация, не статистика._"]
    lines += [""] + [f"{i}. {t}" for i, t in enumerate(notes_md, 1)]
    return "\n".join(lines) + "\n"


def write_csv(path: Path, rows: list[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:  # BOM: Excel открывает кириллицу без плясок
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(asdict(r) for r in rows)


# ---------- CLI ----------

def select_questions(path: Path, split: str, as_of: Optional[str], limit: Optional[int]) -> list[Question]:
    raw = load_json(path)
    questions = [Question.model_validate(x) for x in raw]
    questions = [q for q in questions if split == "all" or q.split == split]
    if as_of:
        questions = [q for q in questions if q.as_of == as_of]
    return questions[:limit] if limit else questions


def print_plan(est: est_mod.Estimate, jest: Optional[est_mod.JudgeEstimate], backend: str) -> None:
    """Оценка прогона: ответы, судья, итог и сравнение с остатком бюджета."""
    print("Оценка ответов системы:\n  " + "\n  ".join(est.lines()))
    if jest is None:
        return
    print("Оценка судьи (judge-v2, модель " + judge_mod.judge_model() + "):\n  " + "\n  ".join(jest.lines()))
    total = est_mod.total_cost(est, jest)
    calls = f"{est.chat_calls_min}–{est.chat_calls_max} вызовов ответов + до {jest.calls} вызовов судьи"
    print(f"Итого (ответы + судья): {calls}; " + (
        f"≈${total[0]:.4f} без ретраев, до ≈${total[1]:.4f} с ретраями" if total else "стоимость не оценить (нет цен)"))
    if total and backend != "stub":
        left = est_mod.remaining_budget()
        if total[0] > left:
            print(f"ВНИМАНИЕ: оценка ${total[0]:.4f} больше остатка бюджета ${left:.4f}: даже без ретраев не хватит, "
                  "cost_guard остановит прогон на лимите, и результаты не запишутся.")
        elif total[1] > left:
            print(f"ВНИМАНИЕ: оценка с ретраями ${total[1]:.4f} больше остатка бюджета ${left:.4f}: "
                  "при ретраях может не хватить.")


def main(argv: Optional[list[str]] = None, judge_fn: Optional[JudgeFn] = None) -> int:
    ap = argparse.ArgumentParser(description="Eval-harness: вопросы → api.ask → простые метрики")
    ap.add_argument("--mode", choices=list(MODES), default="both")
    ap.add_argument("--split", choices=["dev", "test", "all"], default="test")
    ap.add_argument("--as-of", dest="as_of", help="только вопросы с этим as_of (m_008); по умолчанию все")
    ap.add_argument("--repeats", type=int, default=1, help="пока только 1")
    ap.add_argument("--limit", type=int, help="первые N вопросов — для отладки")
    ap.add_argument("--k", type=int, default=5, help="чанков в baseline")
    ap.add_argument("--depth", type=int, default=2, help="глубина обхода графа")
    ap.add_argument("--questions", type=Path, default=QUESTIONS)
    ap.add_argument("--gold", type=Path, default=GOLD)
    ap.add_argument("--raw-dir", type=Path, default=RAW_DIR, help="встречи для baseline и проверки цитат")
    ap.add_argument("--index-dir", type=Path, default=INDEX_ROOT)
    ap.add_argument("--out-dir", type=Path, default=RESULTS)
    ap.add_argument("--judge", action="store_true",
                    help="оценить ответы судьёй v2: заполнить accuracy и fabrication_rate (платные вызовы)")
    ap.add_argument("--dry-run", action="store_true", help="план и оценка стоимости, без прогона")
    ap.add_argument("--confirm-paid", action="store_true", help="разрешить прогон с платным backend")
    a = ap.parse_args(argv)
    if a.repeats != 1:
        ap.error("--repeats: пока поддерживается только 1 (повторы и mean±std — задача A9)")

    modes = MODES[a.mode]
    questions = select_questions(a.questions, a.split, a.as_of, a.limit)
    if not questions:
        print("Нет вопросов под выбранные --split/--as-of.", file=sys.stderr)
        return 2
    backend = os.environ.get("LLM_BACKEND", "stub")
    plan: dict[tuple[str, str], list[str]] = defaultdict(list)
    for m in modes:
        for q in questions:
            plan[(m, q.as_of)].append(q.question)

    print(f"Вопросов: {len(questions)} (split={a.split}), режимы: {', '.join(modes)}, backend: {backend}")
    print(cost_guard.status())
    estimate = est_mod.estimate_run(plan, raw_dir=a.raw_dir, k=a.k, depth=a.depth)
    judge_estimate = est_mod.estimate_judge(questions, modes, raw_dir=a.raw_dir) if a.judge else None
    print_plan(estimate, judge_estimate, backend)
    if a.dry_run:
        return 0
    if a.judge and backend == "stub" and judge_fn is None:
        print("--judge при LLM_BACKEND=stub: заглушка не оценивает ответы. Для проверки без денег — --dry-run; "
              "для настоящего прогона поставьте LLM_BACKEND=openai.", file=sys.stderr)
        return 2
    if backend != "stub" and not a.confirm_paid:
        print(f"\nLLM_BACKEND={backend} — прогон платный. Подтвердите числа выше и запустите с --confirm-paid.",
              file=sys.stderr)
        return 3

    def build(mode: str, as_of: str) -> systems.System:
        return systems.build_system(mode, as_of, raw_dir=a.raw_dir, index_root=a.index_dir, k=a.k, depth=a.depth)

    from src import api
    runner = JudgeRunner(raw_dir=a.raw_dir, fn=judge_fn or judge_mod.judge) if a.judge else None
    try:
        rows = run_eval(questions, modes, build, gold=load_gold_events(a.gold), texts=MeetingTexts(a.raw_dir),
                        backend=backend, judge=runner)
    finally:
        api.use_index_dir(api.DEFAULT_INDEX_DIR)  # UI-индекс не должен остаться подменённым

    date = dt.date.today().isoformat()
    notes = {r.mode: r.error for r in rows if r.status == "stub"}
    csv_path, md_path = a.out_dir / f"eval_{date}.csv", a.out_dir / f"summary_{date}.md"
    write_csv(csv_path, rows)
    info = JudgeInfo(judge_mod.JUDGE_PROMPT_VERSION_V2, judge_mod.judge_model(), runner.calls, runner.reused,
                     runner.failed) if runner else None
    summary = render_summary(rows, modes, date=date, k=a.k, depth=a.depth, notes=notes,
                             splits=["dev", "test"] if a.split == "all" else [a.split], judge=info)
    md_path.write_text(summary, encoding="utf-8")
    print(f"\n{summary}\nCSV: {csv_path}\nСводка: {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
