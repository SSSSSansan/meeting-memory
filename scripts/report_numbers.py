"""РОЛЬ A. Цифры для REPORT.md из файлов репозитория, без вызовов модели.

    python scripts/report_numbers.py            # -> results/report_numbers.md

Источники: results/eval_20261114_0935.json (baseline, dev), results/eval_20261114_1053.json (bm25, dev),
results/eval_20261114_1100.json (longctx, dev, 5 вопросов), results/language_share.md, data/raw,
data/gold/{world.yaml,decisions.json,questions.json}, PRICE_IN_PER_M из .env (только этот ключ).
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Optional

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.bootstrap import mean_ci, paired_bootstrap  # noqa: E402

BASELINE = "results/eval_20261114_0935.json"
BM25 = "results/eval_20261114_1053.json"
LONGCTX = "results/eval_20261114_1100.json"
LANG_SHARE = "results/language_share.md"
WORDS_PER_SEC = 2.33        # «очищенная» скорость речи настоящего образца, см. memory.md (A2, 2026-11-05)
CONTEXT_WINDOW = 128_000
SCALE_MEETINGS = (14, 100, 1000, 6000)
TYPES = ("T1", "T2", "T3", "T4", "T5", "none")


def load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def env_price(path: Path, key: str = "PRICE_IN_PER_M") -> Optional[float]:
    """Только один ключ; остальное в .env (ключи API) не читаем в память дальше строки."""
    if not path.exists():
        return None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(rf"\s*{key}\s*=\s*([0-9.]+)\s*$", line)
        if m:
            v = float(m.group(1))
            return v if v > 0 else None
    return None


def f(x: Optional[float], nd: int = 3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def ci(cell: Optional[dict], nd: int = 3) -> str:
    if not cell or cell.get("mean") is None:
        return "—"
    s = f"{cell['mean']:.{nd}f} [{cell['ci_low']:.{nd}f}; {cell['ci_high']:.{nd}f}], n={cell['n']}"
    return s + (" *" if cell.get("note") else "")


def interval_cell(iv) -> dict:
    return {"mean": iv.mean, "ci_low": iv.ci_low, "ci_high": iv.ci_high, "n": iv.n, "note": iv.note}


def by_qid(run: dict) -> dict[str, dict]:
    return {q["question_id"]: q for q in run["per_question"]}


# ---------- 1. long-context против baseline ----------

def section_longctx(base: dict, lc: dict) -> tuple[list[str], dict]:
    b, lq = by_qid(base), by_qid(lc)
    qids = [q["question_id"] for q in lc["per_question"]]
    missing = [q for q in qids if q not in b]
    qids = [q for q in qids if q in b]
    out = ["## 1. Long-context против baseline на тех же вопросах", "",
           f"Источники: `{LONGCTX}` (longctx), `{BASELINE}` (baseline). Вопросы: {', '.join(qids)}"
           + (f"; нет в baseline: {', '.join(missing)}" if missing else "") + ".", "",
           "| Вопрос | Тип | fact_f1 base | fact_f1 longctx | prompt_tokens base | prompt_tokens longctx "
           "| cost_usd base | cost_usd longctx | latency_ms base | latency_ms longctx | base из кэша |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|"]
    cols = {"f1": [], "pt": [], "cost": [], "lat": []}
    for q in qids:
        qb, ql = b[q], lq[q]
        row = [(qb["metrics"]["fact_f1"], ql["metrics"]["fact_f1"]),
               (qb["usage"]["prompt_tokens"], ql["usage"]["prompt_tokens"]),
               (qb["usage"]["cost_usd"], ql["usage"]["cost_usd"]),
               (qb["latency_ms"], ql["latency_ms"])]
        for k, v in zip(cols, row):
            cols[k].append(v)
        out.append(f"| {q} | {ql['type']} | {f(row[0][0])} | {f(row[0][1])} | {row[1][0]} | {row[1][1]} "
                   f"| {f(row[2][0], 6)} | {f(row[2][1], 6)} | {row[3][0]} | {row[3][1]} "
                   f"| {'да' if qb.get('from_cache') else 'нет'} |")
    out += ["", "| Метрика | baseline, среднее | longctx, среднее | longctx / baseline |", "|---|---:|---:|---:|"]
    summary = {}
    for k, name, nd in (("f1", "fact_f1", 3), ("pt", "prompt_tokens", 1), ("cost", "cost_usd", 6),
                        ("lat", "latency_ms", 1)):
        mb, ml = mean(v[0] for v in cols[k]), mean(v[1] for v in cols[k])
        summary[name] = (mb, ml)
        out.append(f"| {name} | {mb:.{nd}f} | {ml:.{nd}f} | {ml / mb:.2f} |")
    n_cached = sum(1 for q in qids if b[q].get("from_cache"))
    out += ["", f"- Типы вопросов: {dict(Counter(lq[q]['type'] for q in qids))}; n={len(qids)}, интервалы не "
            "считаются (n<5 на тип), это иллюстрация, а не сравнение.",
            f"- У baseline {n_cached} из {len(qids)} ответов взяты из кэша (`from_cache`), поэтому их latency_ms — "
            "время чтения кэша, а не вызова модели; отношение задержек не информативно. "
            "Токены и стоимость в кэше сохранены от исходного вызова.", ""]
    return out, summary


def baseline_live_latency(base: dict) -> Optional[float]:
    xs = [q["latency_ms"] for q in base["per_question"] if not q.get("from_cache")]
    return mean(xs) if xs else None


# ---------- 2. масштабирование long-context ----------

def section_scaling(lc: dict, price_in: Optional[float]) -> list[str]:
    pq = lc["per_question"]
    n_meet = len(pq[0]["retrieved_meetings"])
    mean_all = mean(q["metrics"]["context_tokens"] for q in pq)
    single = [q for q in pq if q["metrics"]["context_tokens"] < 1.5 * min(x["metrics"]["context_tokens"] for x in pq)]
    mean_single = mean(q["metrics"]["context_tokens"] for q in single)
    outliers = [q["question_id"] for q in pq if q not in single]
    out = ["## 2. Масштабирование long-context", "",
           f"Источник: `{LONGCTX}` (context_tokens = usage.prompt_tokens, в контекст идут все {n_meet} встреч до "
           f"as_of). Окно: {CONTEXT_WINDOW} токенов. Цена входа: "
           + (f"PRICE_IN_PER_M = {price_in} $ за 1M токенов (`.env`)." if price_in
              else "**TODO** (PRICE_IN_PER_M не задан).")
           + " Выходные токены не учитываются; токены вопроса и системного промпта отнесены к встречам "
             "(оценка сверху).",
           ""]
    variants = [("все 5 вопросов", mean_all)]
    if outliers:
        variants.append((f"без {', '.join(outliers)} (prompt_tokens ≈ вдвое больше остальных)", mean_single))
    for label, m in variants:
        per = m / n_meet
        fit = math.floor(CONTEXT_WINDOW / per)
        out += [f"### Средний контекст: {label}", "",
                f"- средний контекст: {m:.0f} токенов на {n_meet} встреч → **{per:.0f} токенов на встречу**;",
                f"- в окно {CONTEXT_WINDOW} помещается **{fit} встреч**.", "",
                "| Встреч | Токенов промпта на вопрос | Помещается в окно | Стоимость входа на вопрос, $ |",
                "|---:|---:|:---:|---:|"]
        for n in SCALE_MEETINGS:
            tok = per * n
            cost = f"{tok * price_in / 1e6:.2f}" if price_in else "TODO"
            out.append(f"| {n} | {tok:,.0f} | {'да' if tok <= CONTEXT_WINDOW else 'нет'} | {cost} |".replace(",", " "))
        out.append("")
    return out


# ---------- 3. baseline и bm25 на dev ----------

def abstain_by_type(run: dict) -> dict[str, dict]:
    res = {}
    for t in TYPES + ("all",):
        vals = {q["question_id"]: float(bool(q["abstained"])) for q in run["per_question"]
                if q["answerable"] and (t == "all" or q["type"] == t)}
        if vals:
            res[t] = interval_cell(mean_ci(vals))
    return res


def section_dev(base: dict, bm: dict) -> list[str]:
    ab, am = base["aggregate"]["by_mode_type"]["baseline"], bm["aggregate"]["by_mode_type"]["bm25"]
    fb, fm = abstain_by_type(base), abstain_by_type(bm)
    out = ["## 3. Baseline и BM25 на dev", "",
           f"Источники: `{BASELINE}` (baseline, commit {base['config']['commit']}), `{BM25}` (bm25, commit "
           f"{bm['config']['commit']}); модель {base['config']['model']}, судья {base['config']['judge_model']} "
           f"({base['config']['JUDGE_FACTS_PROMPT_VERSION']}), CONTEXT_K={base['config']['context_k']}, "
           f"as_of {base['config']['as_of']}, вопросов {base['config']['n_questions']}. "
           "Среднее [95% бутстрап-интервал], n. fact_f1 — из `aggregate.by_mode_type`; false_abstain_rate по типам "
           "посчитан здесь из `per_question` тем же `src/eval/bootstrap.mean_ci` (10000, seed 42), в целом совпадает "
           "с `aggregate.abstention`. * — n<5, интервал не информативен.", "",
           "| Тип | fact_f1 baseline | fact_f1 bm25 | false_abstain baseline | false_abstain bm25 |",
           "|---|---|---|---|---|"]
    for t in TYPES + ("all",):
        out.append(f"| {t} | {ci(ab.get(t, {}).get('fact_f1'))} | {ci(am.get(t, {}).get('fact_f1'))} "
                   f"| {ci(fb.get(t))} | {ci(fm.get(t))} |")
    agg_b = base["aggregate"]["abstention"]["baseline"]
    agg_m = bm["aggregate"]["abstention"]["bm25"]
    out += ["", "Отказы из `aggregate.abstention`: "
            f"baseline false_abstain_rate {ci(agg_b['false_abstain_rate'])}, честные отказы при answerable=false "
            f"{ci(agg_b['abstain_rate_unanswerable'])}; bm25 {ci(agg_m['false_abstain_rate'])}, "
            f"{ci(agg_m['abstain_rate_unanswerable'])}.", "",
            "### Парная разность fact_f1: bm25 − baseline (те же вопросы, `paired_bootstrap`, 10000, seed 42)", "",
            "| Тип | n | bm25 − baseline | Вывод |", "|---|---:|---|---|"]
    bq, mq = by_qid(base), by_qid(bm)
    for t in TYPES + ("all",):
        a = {q: v["metrics"]["fact_f1"] for q, v in mq.items() if t == "all" or v["type"] == t}
        b = {q: v["metrics"]["fact_f1"] for q, v in bq.items() if t == "all" or v["type"] == t}
        p = paired_bootstrap(a, b)
        if p.n:
            out.append(f"| {t} | {p.n} | {ci(interval_cell(p.diff))} | {p.verdict} |")
    out += ["", "### Стоимость и токены на вопрос (dev, средние по `per_question`)", "",
            "| Режим | Вопросов | prompt_tokens | cost_usd ответа | cost_usd судьи | latency_ms (без кэша) |",
            "|---|---:|---:|---:|---:|---:|"]
    for name, run in (("baseline", base), ("bm25", bm)):
        pq = run["per_question"]
        live = [q["latency_ms"] for q in pq if not q.get("from_cache")]
        out.append(f"| {name} | {len(pq)} | {mean(q['usage']['prompt_tokens'] for q in pq):.0f} "
                   f"| {mean(q['usage']['cost_usd'] for q in pq):.6f} "
                   f"| {mean(q['judge_usage']['cost_usd'] for q in pq if q.get('judge_usage')):.6f} "
                   f"| {mean(live):.0f} (n={len(live)}) |")
    out.append("")
    return out


# ---------- 4. состав данных ----------

def section_data() -> list[str]:
    world = yaml.safe_load((ROOT / "data/gold/world.yaml").read_text(encoding="utf-8"))
    meetings = world["meetings"]
    rows, tot_w, tot_s, tot_plan, tot_seg = [], 0, 0.0, 0, 0
    for m in meetings:
        mid = m["id"]
        tr = json.loads((ROOT / f"data/raw/{mid}.transcript.json").read_text(encoding="utf-8"))
        meta = json.loads((ROOT / f"data/raw/{mid}.meeting.json").read_text(encoding="utf-8"))
        words = sum(len(s["text"].split()) for s in tr["segments"])
        mins = words / WORDS_PER_SEC / 60
        tot_w, tot_s, tot_plan, tot_seg = (tot_w + words, tot_s + meta["actual_duration_seconds"],
                                           tot_plan + m["target_minutes"], tot_seg + len(tr["segments"]))
        rows.append(f"| {mid} | {m['date']} | {m['language_profile']} | {', '.join(m['projects_discussed'])} "
                    f"| {len(tr['segments'])} | {words} | {mins:.1f} | {m['target_minutes']} "
                    f"| {meta['actual_duration_seconds'] / 60:.1f} |")
    ids = [m["id"] for m in meetings]
    events = json.loads((ROOT / "data/gold/decisions.json").read_text(encoding="utf-8"))
    qs = json.loads((ROOT / "data/gold/questions.json").read_text(encoding="utf-8"))
    qs = qs if isinstance(qs, list) else qs["questions"]
    by_ts = Counter((q["type"], q["split"]) for q in qs)
    lang_split = Counter((q["lang"], q["split"]) for q in qs)
    share = re.findall(r"^\|\s*([\d,]+%)\s*\|\s*(\d+ из \d+)\s*\|\s*([\d,]+%)\s*\|$",
                       (ROOT / LANG_SHARE).read_text(encoding="utf-8"), flags=re.M)
    profiles = Counter(m["language_profile"] for m in meetings)
    out = ["## 4. Состав данных", "",
           "Источники: `data/gold/world.yaml` (meetings), `data/raw/*.transcript.json` и `*.meeting.json`, "
           f"`data/gold/decisions.json`, `data/gold/questions.json`, `{LANG_SHARE}`. Минуты — слова / "
           f"{WORDS_PER_SEC} слова/с (скорость настоящего образца, memory.md); actual_duration_seconds — из "
           "синтетических метаданных.", "",
           f"- Встреч: **{len(meetings)}** ({ids[0]}–{ids[-2]} и {ids[-1]}; m_014 исключена, номера не менялись); "
           f"даты {meetings[0]['date']} – {meetings[-1]['date']}; проектов {len(world['projects'])}; "
           f"профили языка {dict(profiles)}.",
           f"- Реплик {tot_seg}, слов {tot_w}; по {WORDS_PER_SEC} слова/с **{tot_w / WORDS_PER_SEC / 60:.1f} мин** "
           f"(в среднем {tot_w / WORDS_PER_SEC / 60 / len(meetings):.1f} мин на встречу) при плане "
           f"{tot_plan} мин (target_minutes); по actual_duration_seconds {tot_s / 60:.1f} мин.",
           f"- Событий эталона (`decisions.json`): **{len(events)}**; в world.yaml связей {len(world['links'])}, "
           f"закрытий {len(world['closures'])}.",
           "- Доля встреч с казахской речью (`language_share.md`, порог по доле реплик с казахскими буквами или "
           "казахских слов): " + "; ".join(f"порог {p}: {n} = {s}" for p, n, s in share) + ".",
           f"- Вопросов: **{len(qs)}**; dev {sum(v for (t, s), v in by_ts.items() if s == 'dev')}, "
           f"test {sum(v for (t, s), v in by_ts.items() if s == 'test')}; казахских "
           f"{sum(v for (lg, s), v in lang_split.items() if lg == 'kk')} (dev {lang_split[('kk', 'dev')]}, "
           f"test {lang_split[('kk', 'test')]}).", "",
           "| Тип | dev | test | всего |", "|---|---:|---:|---:|"]
    for t in TYPES:
        out.append(f"| {t} | {by_ts[(t, 'dev')]} | {by_ts[(t, 'test')]} | {by_ts[(t, 'dev')] + by_ts[(t, 'test')]} |")
    out += ["", "| Встреча | Дата | Профиль | Проекты | Реплик | Слов | Мин (2,33 сл/с) | План, мин "
            "| actual_duration, мин |", "|---|---|---|---|---:|---:|---:|---:|---:|", *rows, ""]
    return out


def build(price_in: Optional[float]) -> str:
    base, bm, lc = load(BASELINE), load(BM25), load(LONGCTX)
    lc_lines, _ = section_longctx(base, lc)
    live = baseline_live_latency(base)
    lc_lines.insert(-1, f"- Для сравнения: средняя latency_ms baseline по ответам не из кэша на dev — {live:.0f}.")
    parts = ["# Цифры для REPORT.md", "",
             "Сгенерировано `python scripts/report_numbers.py`; модель не вызывалась. Все цифры — из файлов, "
             "указанных в каждом разделе.", "",
             *lc_lines, *section_scaling(lc, price_in), *section_dev(base, bm), *section_data()]
    return "\n".join(parts).rstrip() + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/report_numbers.md")
    ap.add_argument("--env", default=".env")
    args = ap.parse_args(argv)
    text = build(env_price(ROOT / args.env))
    (ROOT / args.out).write_text(text, encoding="utf-8")
    print(f"written {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
