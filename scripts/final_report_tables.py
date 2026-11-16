"""РОЛЬ A. Итоговые таблицы, парные разности и проверка гипотезы по сохранённым прогонам. Модель не вызывается.

    python scripts/final_report_tables.py        # -> results/final_report.md

Читает только json: test-прогон results/eval_20261116_1102.json и dev-прогон results/eval_20261116_1052.json,
плюс data/gold/questions.json (тексты вопросов) и docs/hypothesis_2026-11-14.md (формулировки пунктов).
Интервалы и парные разности — src/eval/bootstrap.py (10000 повторов, seed 42).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval.bootstrap import MIN_N, mean_ci, paired_bootstrap  # noqa: E402

TEST_RUN = "results/eval_20261116_1102.json"
DEV_RUN = "results/eval_20261116_1052.json"
HYPOTHESIS = "docs/hypothesis_2026-11-14.md"
QUESTIONS = "data/gold/questions.json"
OUT = "results/final_report.md"

MODES = ["baseline", "graph", "hybrid", "bm25"]
PAIRS = [("graph", "baseline"), ("hybrid", "baseline"), ("hybrid", "graph"), ("bm25", "baseline")]
GROUPS = ("all", "T1", "T2-T5")
GROUP_TITLES = {"all": "все", "T1": "T1", "T2-T5": "T2-T5"}
TYPES = ["T1", "T2", "T3", "T4", "T5"]
MULTI = {"T2", "T3", "T4", "T5"}
SMALL = f"n<{MIN_N}, интервал не информативен"


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def in_group(qtype: str, group: str) -> bool:
    return group == "all" or (group == "T1" and qtype == "T1") or (group == "T2-T5" and qtype in MULTI)


def per_mode(run: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {m: [] for m in MODES}
    for r in run["per_question"]:
        out.setdefault(r["mode"], []).append(r)
    return out


def by_qid(rows: list[dict], metric: str = "fact_f1", group: str = "all") -> dict[str, Optional[float]]:
    """{qid: значение}; при нескольких повторах — среднее по повторам вопроса."""
    acc: dict[str, list[float]] = {}
    for r in rows:
        if not in_group(r["type"], group):
            continue
        v = r["metrics"].get(metric)
        acc.setdefault(r["qid"], [])
        if v is not None:
            acc[r["qid"]].append(v)
    return {q: (mean(v) if v else None) for q, v in acc.items()}


def fmt(x: Optional[float], nd: int = 3) -> str:
    return "—" if x is None else f"{x:.{nd}f}"


def cell(iv, with_ci: bool = True) -> str:
    if iv.n == 0:
        return "—"
    s = f"{iv.mean:.3f}"
    if with_ci:
        s += f" [{iv.ci_low:.3f}; {iv.ci_high:.3f}]"
    return s + (f" (n={iv.n}{', n<5' if iv.n < MIN_N else ''})")


def md_table(head: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(head) + " |", "|" + "|".join("---" for _ in head) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def abstention(rows: list[dict]) -> tuple[int, int, int, int]:
    """(отказов при answerable=false, всего таких, ложных отказов при answerable=true, всего таких)."""
    un = [r for r in rows if not r["answerable"]]
    an = [r for r in rows if r["answerable"]]
    return sum(bool(r["abstained"]) for r in un), len(un), sum(bool(r["abstained"]) for r in an), len(an)


def mean_of(rows: list[dict], getter) -> Optional[float]:
    xs = [getter(r) for r in rows]
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def main_table(pm: dict[str, list[dict]]) -> str:
    head = ["режим", "fact_f1 [95%], все 23", "fact_f1 T1", "fact_f1 T2-T5", "честные отказы (answerable=false)",
            "ложные отказы (answerable=true)", "forbidden_rate", "стоимость ответа на вопрос, $",
            "задержка, мс", "токены промпта"]
    rows = []
    for m in MODES:
        r = pm[m]
        a_ok, a_n, f_ok, f_n = abstention(r)
        rows.append([
            m,
            cell(mean_ci(by_qid(r))),
            cell(mean_ci(by_qid(r, group="T1")), with_ci=False),
            cell(mean_ci(by_qid(r, group="T2-T5")), with_ci=False),
            f"{a_ok}/{a_n}", f"{f_ok}/{f_n} ({f_ok / f_n:.2f})" if f_n else "—",
            fmt(mean_of(r, lambda x: x["metrics"].get("forbidden_rate"))),
            f"{mean_of(r, lambda x: x['usage']['cost_usd']):.4f}",
            f"{mean_of(r, lambda x: x['latency_ms']):.0f}",
            f"{mean_of(r, lambda x: x['usage']['prompt_tokens']):.0f}",
        ])
    return md_table(head, rows)


def paired_table(pm: dict[str, list[dict]]) -> tuple[str, dict]:
    res = {}
    rows = []
    for a, b in PAIRS:
        for g in GROUPS:
            p = paired_bootstrap(by_qid(pm[a], group=g), by_qid(pm[b], group=g))
            res[(a, b, g)] = p
            d = p.diff
            if d.n == 0:
                rows.append([f"{a} − {b}", GROUP_TITLES[g], "0", "—", "—", p.verdict])
                continue
            verdict = p.verdict + ("; " + SMALL if d.n < MIN_N and SMALL not in p.verdict else "")
            rows.append([f"{a} − {b}", GROUP_TITLES[g], str(p.n), f"{d.mean:+.3f}",
                         f"[{d.ci_low:+.3f}; {d.ci_high:+.3f}]", verdict])
    return md_table(["пара", "группа", "n", "разность fact_f1", "95% интервал", "вердикт"], rows), res


def type_table(pm: dict[str, list[dict]]) -> str:
    rows = []
    for t in TYPES:
        n = len({r["qid"] for r in pm["baseline"] if r["type"] == t})
        rows.append([t, str(n)] + [fmt(mean_of([r for r in pm[m] if r["type"] == t],
                                               lambda x: x["metrics"].get("fact_f1"))) for m in MODES])
    return md_table(["тип", "n"] + MODES, rows)


def dev_table(dev: dict) -> str:
    pm = per_mode(dev)
    rows = []
    for m in MODES:
        row = [m]
        for g in GROUPS:
            iv = mean_ci(by_qid(pm[m], group=g))
            row.append(cell(iv, with_ci=False) if g != "all" else cell(iv))
        a_ok, a_n, f_ok, f_n = abstention(pm[m])
        row.append(f"{a_ok}/{a_n}")
        row.append(f"{f_ok}/{f_n}")
        rows.append(row)
    return md_table(["режим", "fact_f1 [95%], все", "fact_f1 T1", "fact_f1 T2-T5", "честные отказы",
                     "ложные отказы"], rows)


def hypothesis_items() -> list[str]:
    text = (ROOT / HYPOTHESIS).read_text(encoding="utf-8")
    return [m.group(2).strip() for m in re.finditer(r"^(\d)\. (.+)$", text, re.M)]


def proven_worse(p) -> bool:
    return p.diff.n > 0 and p.diff.ci_high < 0


def proven_better(p) -> bool:
    return p.diff.n > 0 and p.diff.ci_low > 0


def dstr(p) -> str:
    d = p.diff
    if d.n == 0:
        return "нет данных"
    return f"{d.mean:+.3f} [{d.ci_low:+.3f}; {d.ci_high:+.3f}], n={d.n}" + (" (n<5)" if d.n < MIN_N else "")


def check_hypothesis(pm: dict[str, list[dict]], pr: dict) -> list[tuple[str, str, str]]:
    items = hypothesis_items()
    out = []

    # 1
    g, h = pr[("graph", "baseline", "T2-T5")], pr[("hybrid", "baseline", "T2-T5")]
    v = "подтвердилось" if proven_better(g) and proven_better(h) else "не подтвердилось"
    out.append((items[0], f"T2-T5: graph − baseline {dstr(g)}; hybrid − baseline {dstr(h)}.", v))

    # 2
    g1 = pr[("graph", "baseline", "T1")]
    v = "подтвердилось" if g1.diff.n and g1.diff.ci_low <= 0 <= g1.diff.ci_high else "не подтвердилось"
    out.append((items[1], f"T1: graph − baseline {dstr(g1)}. Интервал при n=4 малоинформативен.", v))

    # 3
    cmp3 = {(a, b, g): pr[(a, b, g)] for a, b in (("hybrid", "graph"), ("hybrid", "baseline"))
            for g in ("T1", "T2-T5")}
    nums = "; ".join(f"{a} − {b} на {g}: {dstr(p)}" for (a, b, g), p in cmp3.items())
    worse = [k for k, p in cmp3.items() if proven_worse(p)]
    v = "не подтвердилось" if worse else "подтвердилось"
    out.append((items[2], nums + ". «Хуже» понято как интервал разности целиком ниже 0; "
                "ниже 0 только среднее при интервале с 0 — не хуже по доказанному.", v))

    # 4
    def bm_vs_base(t: str):
        return paired_bootstrap(*({q: v for q, v in by_qid(pm[m]).items() if _type(pm[m], q) == t}
                                  for m in ("bm25", "baseline")))
    t1, t3 = bm_vs_base("T1"), bm_vs_base("T3")
    ok = (not proven_worse(t1)) and proven_worse(t3)
    out.append((items[3], f"bm25 − baseline на T1: {dstr(t1)}; на T3: {dstr(t3)}. На T3 всего 3 вопроса.",
                "подтвердилось" if ok else "не подтвердилось"))

    # 5
    t5 = {m: mean_of([r for r in pm[m] if r["type"] == "T5"], lambda x: x["metrics"].get("fact_f1")) for m in MODES}
    others = {m: {t: mean_of([r for r in pm[m] if r["type"] == t], lambda x: x["metrics"].get("fact_f1"))
                  for t in ("T1", "T2", "T3", "T4")} for m in MODES}
    info = "; ".join(f"{m}: T5 {fmt(t5[m])} (прочие типы {fmt(min(others[m].values()))}–{fmt(max(others[m].values()))})"
                     for m in MODES)
    out.append((items[4], "Условие «если в графе нет связи закрытия» не наступило: связь добавлена ролью B. "
                f"Для справки, fact_f1 T5 и диапазон по T1–T4: {info}.", "условие не наступило"))

    # 6
    tok = {m: mean_of(pm[m], lambda x: x["usage"]["prompt_tokens"]) for m in MODES}
    cost = {m: mean_of(pm[m], lambda x: x["usage"]["cost_usd"]) for m in MODES}
    pt = {m: {r["qid"]: r["usage"]["prompt_tokens"] for r in pm[m]} for m in MODES}
    diffs = {m: paired_bootstrap(pt[m], pt["baseline"]) for m in ("graph", "hybrid")}
    nums = f"baseline: {tok['baseline']:.0f} токенов, ${cost['baseline']:.4f} за вопрос; " + "; ".join(
        f"{m}: {tok[m]:.0f} токенов, ${cost[m]:.4f} за вопрос (разность токенов с baseline "
        f"{diffs[m].diff.mean:+.0f} [{diffs[m].diff.ci_low:+.0f}; {diffs[m].diff.ci_high:+.0f}])"
        for m in ("graph", "hybrid"))
    v = "подтвердилось" if all(tok[m] > tok["baseline"] and cost[m] > cost["baseline"]
                               for m in ("graph", "hybrid")) else "не подтвердилось"
    unproven = [m for m in ("graph", "hybrid") if diffs[m].diff.ci_low <= 0]
    if v == "подтвердилось" and unproven:
        v += (f" по средним; для {', '.join(unproven)} интервал разности токенов содержит 0, "
              "то есть превышение токенов статистически не доказано")
    out.append((items[5], nums + ".", v))

    # 7
    ab = {m: abstention(pm[m]) for m in MODES}
    nums = "; ".join(f"{m}: {ab[m][0]}/{ab[m][1]}" for m in MODES)
    v = "подтвердилось" if all(a[1] and a[0] / a[1] >= 0.75 for a in ab.values()) else "не подтвердилось"
    out.append((items[6], f"Честные отказы на answerable=false: {nums} (порог 3 из 4).", v))
    return out


def _type(rows: list[dict], qid: str) -> str:
    return next(r["type"] for r in rows if r["qid"] == qid)


def empty_graph(pm: dict[str, list[dict]]) -> str:
    qs = {q["id"]: q for q in json.loads((ROOT / QUESTIONS).read_text(encoding="utf-8"))}
    rows = [[r["qid"], r["type"], qs.get(r["qid"], {}).get("question", "—")]
            for r in pm["graph"] if not r["retrieved_meetings"]]
    if not rows:
        return "Таких ответов нет."
    return md_table(["question_id", "тип", "текст вопроса"], rows)


def build() -> str:
    test, dev = load(TEST_RUN), load(DEV_RUN)
    pm = per_mode(test)
    cfg = test["config"]
    ptab, pr = paired_table(pm)
    hyp = check_hypothesis(pm, pr)
    n_cached = sum(bool(r.get("from_cache")) for r in test["per_question"])
    out = [
        "# Итоговые таблицы: test-прогон 2026-11-16",
        "",
        f"Источник: `{TEST_RUN}` (split={cfg['split']}, вопросов {cfg['n_questions']}, повторов {cfg['repeats']}, "
        f"модель {cfg['model']}, судья {cfg['judge_model']}, {cfg['JUDGE_FACTS_PROMPT_VERSION']}, "
        f"commit {cfg['commit']}). "
        "Модель не вызывалась, всё посчитано из json (`scripts/final_report_tables.py`).",
        "Интервалы: парный бутстрап по вопросам, 10000 повторов, seed 42 (`src/eval/bootstrap.py`). "
        "Интервал показывает зависимость от набора вопросов, а не разброс повторов. "
        "Группа T1 — 4 вопроса, T3 — 3: меньше 5, интервалы не информативны. "
        "«Все 23» включают 4 вопроса без ответа в данных (тип none); T1 и T2-T5 — 19 отвечаемых.",
        f"Ответов из кэша в прогоне: {n_cached}. Стоимость — только ответ модели, без судьи.",
        "",
        "## 1. Главная таблица (test)",
        "",
        main_table(pm),
        "",
        "## 2. Парные разности fact_f1 (test)",
        "",
        ptab,
        "",
        "## 3. По типам вопросов (test, разведочная, n по 3–4)",
        "",
        type_table(pm),
        "",
        "## 4. Dev (dev, разведочное; обход графа подбирали на dev)",
        "",
        f"Источник: `{DEV_RUN}` (вопросов {dev['config']['n_questions']}).",
        "",
        dev_table(dev),
        "",
        "## 5. Проверка гипотезы 2026-11-14",
        "",
        "Формулировки из `docs/hypothesis_2026-11-14.md` без изменений.",
        "",
    ]
    for i, (text, nums, verdict) in enumerate(hyp, 1):
        out += [f"**{i}. {text}**", "", f"- Числа: {nums}", f"- Вердикт: **{verdict}**", ""]
    out += [
        "## 6. Ответы graph с пустым retrieved_meetings (test)",
        "",
        empty_graph(pm),
        "",
    ]
    return "\n".join(out)


if __name__ == "__main__":
    (ROOT / OUT).write_text(build(), encoding="utf-8")
    print(f"written {OUT}")
