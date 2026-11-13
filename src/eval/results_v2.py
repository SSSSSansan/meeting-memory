"""РОЛЬ A. Json-результат прогона (results/eval_<run_id>.json), агрегаты и промежуточный файл для --resume.

Контракт файла (поля только добавляются):
  run_id, timestamp,
  config {model, embed_model, judge_model, JUDGE_FACTS_PROMPT_VERSION, modes, split, n_questions, as_of,
          judge_repeats, + repeats, backend, commit, judge_facts},
  per_question [{qid, question_id (оба = id вопроса), type, mode, repeat, answer, citations, retrieved_meetings,
                 metrics {citation_recall, citation_precision, retrieval_recall, quote_validity, fact_recall,
                          fact_precision, fact_f1, forbidden_hits, + fact_precision_strict, forbidden_rate,
                          context_tokens},
                 usage {prompt_tokens, completion_tokens, cost_usd}, latency_ms,
                 + status, split, lang, as_of, answerable, abstained, context_tokens_source, …}],
  aggregate {by_mode_type[mode][type][metric] = {mean, ci_low, ci_high, n, note},
             by_mode_lang[mode][lang][metric] = {mean, ci_low, ci_high, n, note},      # lang из вопросов (ru, kk)
             abstention[mode] = {abstain_rate_unanswerable, false_abstain_rate},   # каждое {mean, ci_low, …, n, note}
             paired[type][metric] = {diff_mean, ci_low, ci_high, n, verdict, n_excluded},   # graph − baseline
             paired_pairs[«a-b»][type][metric] — остальные пары режимов, bootstrap {n_boot, seed, …}}.
aggregate считает харнес (этот модуль); дашборд только читает. Значение null — «не измерено / нет оценки»; оно
исключается из средних и интервалов, а не превращается в 0.

Новые метрики (только добавлены):
  forbidden_rate — 1.0, если в ответе forbidden_hits > 0 (при повторах судьи — хотя бы в одном), иначе 0.0; среднее по
                   вопросам = доля ответов с запрещённым фактом; null, если судья по фактам не считал;
  context_tokens — токены, ушедшие в промпт ответа: usage.context_tokens, если он есть, иначе usage.prompt_tokens
                   (сейчас отдельного числа нет, значит prompt_tokens: контекст + системный промпт + вопрос);
  abstention     — отказ = is_abstention (metrics.py); abstain_rate_unanswerable — доля отказов среди answerable=false
                   (выше лучше), false_abstain_rate — среди answerable=true (ниже лучше); считаются только по строкам со
                   статусом ok (ошибка и заглушка ответа не дали, отказом они не являются); нет вопросов группы —
                   n=0, null.

Единица агрегата — вопрос: если у вопроса несколько повторов ответа, берётся их среднее, затем бутстрап по вопросам
(src/eval/bootstrap.py). Тип «all» — все вопросы прогона.

Промежуточный файл results/eval_<run_id>.partial.jsonl: первая строка — {"header": …} (по ней --resume проверяет, что
прогон тот же), дальше по строке {"row": …} на каждый готовый ответ (с оценкой судьи).
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from src.eval import bootstrap as bs
from src.eval.judge import judge_model
from src.eval.judge_facts import JUDGE_FACTS_PROMPT_VERSION

CONTRACT_METRICS = ("citation_recall", "citation_precision", "retrieval_recall", "quote_validity",
                    "fact_recall", "fact_precision", "fact_f1", "forbidden_hits")
EXTRA_METRICS = ("fact_precision_strict", "forbidden_rate", "context_tokens")
ALL_METRICS = CONTRACT_METRICS + EXTRA_METRICS
BOOT_METRICS = ("fact_f1", "fact_recall", "fact_precision", "citation_recall", "retrieval_recall")
FACT_METRICS = ("fact_f1", "fact_recall", "fact_precision")
TYPE_ORDER = ("T1", "T2", "T3", "T4", "T5", "none")
LANG_ORDER = ("ru", "mixed", "kk")
ABSTENTION_METRICS = ("abstain_rate_unanswerable", "false_abstain_rate")
CONTEXT_FIELD = "usage.context_tokens"
CONTEXT_FALLBACK = "usage.prompt_tokens"
ALL = "all"
MAIN_PAIR = ("graph", "baseline")       # paired считается как graph − baseline
NOT_MEASURED = "не измерено"
ROUND = 6


def _r(x: Optional[float]) -> Optional[float]:
    if x is None or (isinstance(x, int) and not isinstance(x, bool)):
        return x    # целые (forbidden_hits, n) остаются целыми
    return round(float(x), ROUND)


# ---------- per_question ----------

def row_metrics(row: Any) -> dict[str, Optional[float]]:
    """Метрики строки в контрактных именах; quote_validity — доля цитат, найденных в тексте встречи."""
    out: dict[str, Optional[float]] = {}
    for k in ALL_METRICS:
        if k == "quote_validity":
            v = row.citation_valid
        elif k == "forbidden_rate":
            hits = getattr(row, "forbidden_hits", None)
            v = None if hits is None else float(hits > 0)
        else:
            v = getattr(row, k, None)
        out[k] = _r(v)
    return out


def per_question_record(row: Any) -> dict:
    return {
        "qid": row.question_id, "question_id": row.question_id, "type": row.type, "mode": row.mode,
        "repeat": row.repeat, "answer": row.answer, "citations": list(getattr(row, "citations", None) or []),
        "retrieved_meetings": getattr(row, "retrieved_meetings", None),
        "metrics": row_metrics(row),
        "usage": {"prompt_tokens": row.prompt_tokens, "completion_tokens": row.completion_tokens,
                  "cost_usd": row.cost_usd},
        "latency_ms": row.latency_ms,
        # дополнительно (контракт только расширяется)
        "status": row.status, "split": row.split, "lang": row.lang, "as_of": row.as_of,
        "from_cache": row.from_cache,
        "answerable": row.answerable, "abstained": row.abstained,
        "context_tokens_source": getattr(row, "context_tokens_source", ""),
        "judge_facts": getattr(row, "facts_detail", None),
        "metrics_std": getattr(row, "metrics_std", None),
        "judge_usage": getattr(row, "judge_usage", None),
    }


# ---------- aggregate ----------

def question_values(records: Iterable[dict], metric: str) -> dict[str, dict[str, Optional[float]]]:
    """{mode: {question_id: среднее по повторам | None}}. None — ни у одного повтора нет значения."""
    acc: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    seen: dict[str, set[str]] = defaultdict(set)
    for r in records:
        seen[r["mode"]].add(r["question_id"])
        v = r["metrics"].get(metric)
        if v is not None:
            acc[r["mode"]][r["question_id"]].append(v)
    return {m: {q: (sum(acc[m][q]) / len(acc[m][q]) if acc[m].get(q) else None) for q in qs}
            for m, qs in seen.items()}


def _cell(iv: bs.Interval) -> dict:
    return {"mean": _r(iv.mean), "ci_low": _r(iv.ci_low), "ci_high": _r(iv.ci_high), "n": iv.n, "note": iv.note}


def _paired_cell(p: bs.Paired) -> dict:
    return {"diff_mean": _r(p.diff.mean), "ci_low": _r(p.diff.ci_low), "ci_high": _r(p.diff.ci_high), "n": p.n,
            "verdict": p.verdict, "n_excluded": p.n_excluded,
            "mean_a": _r(p.a.mean), "mean_b": _r(p.b.mean)}


def compute_paired(records: Sequence[dict], a: str, b: str, metrics: Sequence[str] = ALL_METRICS) -> dict:
    """{type: {metric: ячейка}} для разности режимов a − b по общим вопросам; пары вопросов не рвутся."""
    qtype = {r["question_id"]: r["type"] for r in records}
    out: dict[str, dict] = {}
    for metric in metrics:
        vals = question_values(records, metric)
        if a not in vals or b not in vals:
            continue
        for t in [*TYPE_ORDER, ALL]:
            ids = {q for q, qt in qtype.items() if t == ALL or qt == t}
            if not ids:
                continue
            out.setdefault(t, {})[metric] = _paired_cell(bs.paired_bootstrap(
                {q: v for q, v in vals[a].items() if q in ids}, {q: v for q, v in vals[b].items() if q in ids}))
    return out


def _by_mode_group(records: Sequence[dict], modes: Sequence[str], key: str, groups: Sequence[str],
                   with_all: bool) -> dict:
    """{mode: {группа: {метрика: ячейка}}}, группа — значение поля key вопроса (type | lang)."""
    group_of = {r["question_id"]: r.get(key) for r in records}
    out: dict[str, dict] = {}
    for metric in ALL_METRICS:
        vals = question_values(records, metric)
        for mode in modes:
            for g in [*groups, *([ALL] if with_all else [])]:
                ids = {q for q, qg in group_of.items() if g == ALL or qg == g}
                in_mode = {q for q in vals.get(mode, {}) if q in ids}
                if not ids or not in_mode:
                    continue
                out.setdefault(mode, {}).setdefault(g, {})[metric] = _cell(
                    bs.mean_ci({q: vals[mode][q] for q in in_mode}))
    return out


def abstention_values(records: Sequence[dict], mode: str, answerable: bool) -> dict[str, float]:
    """{question_id: доля повторов с отказом} по строкам режима со статусом ok и нужным answerable. Строка без поля
    abstained (старый файл) пропускается."""
    acc: dict[str, list[float]] = defaultdict(list)
    for r in records:
        if (r["mode"] == mode and r.get("status") == "ok" and r.get("answerable") is answerable
                and r.get("abstained") is not None):
            acc[r["question_id"]].append(float(bool(r["abstained"])))
    return {q: sum(v) / len(v) for q, v in acc.items()}


def compute_abstention(records: Sequence[dict], modes: Sequence[str]) -> dict:
    """aggregate.abstention[mode]: честные отказы на вопросах без ответа (выше лучше) и ложные отказы на вопросах с
    ответом (ниже лучше). Пустая группа — ячейка с n=0 и null, без деления на ноль."""
    return {m: {"abstain_rate_unanswerable": _cell(bs.mean_ci(abstention_values(records, m, False))),
                "false_abstain_rate": _cell(bs.mean_ci(abstention_values(records, m, True)))} for m in modes}


def compute_aggregate(records: Sequence[dict], modes: Sequence[str]) -> dict:
    by_mode_type = _by_mode_group(records, modes, "type", TYPE_ORDER, with_all=True)
    by_mode_lang = _by_mode_group(records, modes, "lang", LANG_ORDER, with_all=False)
    paired = compute_paired(records, *MAIN_PAIR) if set(MAIN_PAIR) <= set(modes) else {}
    pairs: dict[str, dict] = {}
    for a, b in combinations(modes, 2):
        if {a, b} == set(MAIN_PAIR):
            continue
        pairs[f"{a}-{b}"] = compute_paired(records, a, b)
    return {"by_mode_type": by_mode_type, "by_mode_lang": by_mode_lang,
            "abstention": compute_abstention(records, modes), "paired": paired, "paired_pairs": pairs,
            "bootstrap": {"n_boot": bs.N_BOOT, "seed": bs.SEED, "percentiles": list(bs.PERCENTILES),
                          "min_n": bs.MIN_N, "paired": f"{MAIN_PAIR[0]} - {MAIN_PAIR[1]}",
                          "unit": "question", "note": bs.INTERVAL_NOTE}}


# ---------- config и файл ----------

def build_config(*, modes: Sequence[str], split: str, n_questions: int, as_of: str, judge_repeats: int,
                 repeats: int, backend: str, commit: str, judge_facts: bool, fixture: bool = False) -> dict:
    cfg = {"model": os.environ.get("LLM_MODEL", "stub-model"), "embed_model": os.environ.get("EMBED_MODEL") or None,
           "judge_model": judge_model(), "JUDGE_FACTS_PROMPT_VERSION": JUDGE_FACTS_PROMPT_VERSION,
           "modes": list(modes), "split": split, "n_questions": n_questions, "as_of": as_of,
           "judge_repeats": judge_repeats,
           "repeats": repeats, "backend": backend, "commit": commit, "judge_facts": judge_facts}
    if fixture:
        cfg["fixture"] = True
    return cfg


def as_of_label(as_of_values: Iterable[str]) -> str:
    return ",".join(sorted(set(as_of_values)))


def build_result(*, run_id: str, timestamp: str, config: dict, rows: Sequence[Any]) -> dict:
    records = [per_question_record(r) for r in rows]
    return {"run_id": run_id, "timestamp": timestamp, "config": config, "per_question": records,
            "aggregate": compute_aggregate(records, config["modes"])}


def write_result(path: Path, result: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)       # файл либо старый, либо целиком новый


# ---------- промежуточный файл ----------

def partial_path(out_dir: Path, run_id: str) -> Path:
    return Path(out_dir) / f"eval_{run_id}.partial.jsonl"


def result_path(out_dir: Path, run_id: str) -> Path:
    return Path(out_dir) / f"eval_{run_id}.json"


def append_jsonl(path: Path, obj: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        os.fsync(f.fileno())    # падение процесса не должно терять уже записанные вопросы


def read_partial(path: Path) -> tuple[Optional[dict], list[dict]]:
    """(header, строки row). Оборванная последняя строка (падение посреди записи) пропускается."""
    header, rows = None, []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "header" in obj:
            header = obj["header"]
        elif "row" in obj:
            rows.append(obj["row"])
    return header, rows


# ---------- сводка ----------

def _fmt(c: Optional[dict]) -> str:
    if not c or c.get("mean") is None:
        return "—"
    return f"{c['mean']:.2f} [{c['ci_low']:.2f}; {c['ci_high']:.2f}] (n={c['n']})"


def _fmt_diff(c: dict) -> str:
    if c.get("diff_mean") is None:
        return "—"
    return f"{c['diff_mean']:+.2f} [{c['ci_low']:+.2f}; {c['ci_high']:+.2f}]"


def render_v2_section(aggregate: dict, records: Sequence[dict], modes: Sequence[str], *, judge_facts: bool,
                      judge_repeats: int = 1, split: str = "test",
                      header_lines: Sequence[str] = ()) -> list[str]:
    """Строки сводки (md и консоль): метрики v2 по типам с бутстрап-интервалами, разность graph − baseline."""
    lines = ["## Метрики v2: полнота, цитаты, факты", ""]
    lines.append(f"> {bs.INTERVAL_NOTE[0].upper()}{bs.INTERVAL_NOTE[1:]}. "
                 f"Бутстрап по вопросам: n_boot = {bs.N_BOOT}, seed = {bs.SEED}, перцентили 2,5 и 97,5.")
    lines.extend(header_lines)
    if split == "all":
        lines.append("> split=all: dev и test смешаны в одном агрегате, в отчёт он не идёт.")
    ok_rows = [r for r in records if r["status"] != "stub"]
    if not judge_facts:
        lines.append("- fact_recall, fact_precision, fact_f1: " + NOT_MEASURED + " (нет --judge-facts)")
    if not any(r["metrics"].get("retrieval_recall") is not None for r in ok_rows if r["metrics"] is not None):
        lines.append("- retrieval_recall: " + NOT_MEASURED + " (api.ask() не отдаёт retrieved_meetings, "
                     "нужно от роли C: список встреч, попавших в контекст до генерации)")
    if judge_facts and judge_repeats > 1:
        stds = [r["metrics_std"] for r in ok_rows if r.get("metrics_std")]
        if stds:
            def avg(k: str) -> str:
                xs = [s[k] for s in stds if s.get(k) is not None]
                return f"{sum(xs) / len(xs):.3f}" if xs else "—"
            lines.append(f"- повторы судьи: {judge_repeats}; std по повторам судьи (среднее по ответам): "
                         f"fact_f1 {avg('fact_f1')}, fact_recall {avg('fact_recall')}, "
                         f"fact_precision {avg('fact_precision')}")
    paired = aggregate.get("paired") or {}
    has_pair = bool(paired)
    head = ["Тип", "Метрика"] + [f"{m} (среднее [95% интервал])" for m in modes]
    if has_pair:
        head += [f"{MAIN_PAIR[0]} − {MAIN_PAIR[1]} [интервал]", "n пар", "исключено", "вывод"]
    lines += ["", "| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    by = aggregate.get("by_mode_type") or {}
    n_rows = 0
    for t in [*TYPE_ORDER, ALL]:
        for metric in BOOT_METRICS:
            cells = [by.get(m, {}).get(t, {}).get(metric) for m in modes]
            if all(c is None or c["n"] == 0 for c in cells):
                continue
            row = [t, metric] + [_fmt(c) for c in cells]
            if has_pair:
                p = paired.get(t, {}).get(metric)
                row += ([_fmt_diff(p), str(p["n"]), str(p["n_excluded"]), p["verdict"]] if p else ["—"] * 4)
            lines.append("| " + " | ".join(row) + " |")
            n_rows += 1
    if not n_rows:
        lines.append("| _нет метрик с оценкой_ |" + " |" * (len(head) - 1))
    if has_pair:
        excl = {(t, k): p["n_excluded"] for t, d in paired.items() for k, p in d.items() if p["n_excluded"]}
        if excl:
            lines += ["", "Исключены из обоих режимов (у одного из них нет оценки): "
                      + ", ".join(f"{t}/{k}: {n}" for (t, k), n in sorted(excl.items()))]
        else:
            lines += ["", "Исключённых вопросов (без оценки хотя бы в одном режиме) нет."]
    lines += ["", f"_Вердикты: «{bs.VERDICT_NOT_PROVEN}» — интервал разности содержит 0; «{bs.VERDICT_SMALL_N}» — "
                  f"меньше {bs.MIN_N} вопросов с оценкой в паре._"]
    lines += _group_table("### Срез по языку вопроса", "Язык", aggregate.get("by_mode_lang") or {}, modes,
                          LANG_ORDER, BOOT_METRICS)
    lines += _abstention_table(aggregate.get("abstention") or {}, modes)
    lines += _group_table("### Запрещённые факты и контекст в промпте", "Тип", aggregate.get("by_mode_type") or {},
                          modes, [*TYPE_ORDER, ALL], ("forbidden_rate", "context_tokens"))
    lines += _context_note(ok_rows, judge_facts)
    return lines


def _fmt_cell(c: Optional[dict], spec: str = ".2f") -> str:
    if not c or c.get("mean") is None:
        return "—"
    return f"{format(c['mean'], spec)} [{format(c['ci_low'], spec)}; {format(c['ci_high'], spec)}] (n={c['n']})"


def _group_table(title: str, first_col: str, by: dict, modes: Sequence[str], groups: Sequence[str],
                 metrics: Sequence[str]) -> list[str]:
    """Таблица группа × метрика, режимы в колонках; группа меньше 5 вопросов получает пометку в последней колонке."""
    head = [first_col, "Метрика"] + [f"{m} (среднее [95% интервал])" for m in modes] + ["Примечание"]
    rows = []
    for g in groups:
        for metric in metrics:
            cells = [by.get(m, {}).get(g, {}).get(metric) for m in modes]
            if all(c is None or c["n"] == 0 for c in cells):
                continue
            spec = ".0f" if metric == "context_tokens" else ".2f"
            notes = sorted({c["note"] for c in cells if c and c.get("note")})
            row = [g, metric] + [_fmt_cell(c, spec) for c in cells] + ["; ".join(notes)]
            rows.append("| " + " | ".join(row) + " |")
    lines = ["", title, ""]
    if not rows:
        return lines + ["_нет метрик с оценкой_"]
    return lines + ["| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"] + rows


def _abstention_table(ab: dict, modes: Sequence[str]) -> list[str]:
    titles = {"abstain_rate_unanswerable":
              "abstain_rate_unanswerable (честные отказы при answerable=false, выше лучше)",
              "false_abstain_rate": "false_abstain_rate (отказы при answerable=true, ниже лучше)"}
    head = ["Метрика"] + [f"{m} (среднее [95% интервал])" for m in modes] + ["Примечание"]
    lines = ["", "### Отказы", "", "| " + " | ".join(head) + " |", "|" + "|".join(["---"] * len(head)) + "|"]
    for metric in ABSTENTION_METRICS:
        cells = [(ab.get(m) or {}).get(metric) for m in modes]
        notes = sorted({c["note"] for c in cells if c and c.get("note")})
        lines.append("| " + " | ".join([titles[metric]] + [_fmt_cell(c) for c in cells] + ["; ".join(notes)]) + " |")
    return lines + ["", "_Отказ — по фразам (is_abstention); считаются только ответы со статусом ok; «—» — нет "
                        "вопросов этой группы или режим не дал ответов._"]


def _context_note(ok_rows: Sequence[dict], judge_facts: bool) -> list[str]:
    own = any(r.get("context_tokens_source") == CONTEXT_FIELD for r in ok_rows)
    out = ["", ("_context_tokens — usage.context_tokens (отдельное число для контекста)._" if own else
                "_context_tokens = usage.prompt_tokens: отдельного числа для контекста в usage нет, поэтому в него "
                "входят контекст, системный промпт и вопрос; режимы сравнивать можно, размер контекста — нет._")]
    if not judge_facts:
        out.append("_forbidden_rate: " + NOT_MEASURED + " (нет --judge-facts)._")
    return out
