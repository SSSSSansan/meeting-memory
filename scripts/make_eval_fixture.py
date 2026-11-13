"""РОЛЬ A. Фикстура tests/fixtures/eval_fixture.json: json-результат eval на ВЫДУМАННЫХ числах (stub) — чтобы роль C
сделала вкладку метрик, не дожидаясь платного прогона.

    python scripts/make_eval_fixture.py            # пересоздать tests/fixtures/eval_fixture.json

Формат — ровно тот, что пишет src/eval/run.py (контракт в src/eval/results_v2.py); aggregate считается тем же кодом
по выдуманным per_question. В config стоит fixture: true. Ни модель, ни сеть не используются; вопросы берутся из
data/gold/questions_dev.json (только id, тип и as_of; тексты вопросов, эталон и test не читаются). Числа случайные
(зерно 42) и ничего не говорят о системе; retrieval_recall заполнен у ~90% строк, чтобы вкладка видела и число, и null,
хотя в настоящих прогонах он null, пока api.ask() не отдаёт retrieved_meetings.
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # запуск файлом: в sys.path только scripts/
    sys.path.insert(0, str(ROOT))

from src.eval import results_v2  # noqa: E402
from src.eval.judge_facts import JUDGE_FACTS_PROMPT_VERSION, harmonic  # noqa: E402
from src.eval.run import Row  # noqa: E402

DEV_QUESTIONS = ROOT / "data" / "gold" / "questions_dev.json"
FIXTURE = ROOT / "tests" / "fixtures" / "eval_fixture.json"
SEED = 42
MODES = ["baseline", "graph"]
TIMESTAMP = "2026-01-01T00:00:00+05:00"
NOTE = ("fixture: числа выдуманы (stub), не результат запуска системы; нужен только для вёрстки вкладки метрик. "
        "В настоящих прогонах retrieval_recall = null, пока api.ask() не отдаёт retrieved_meetings.")


def _pick(rng: random.Random, options: list[float], weights: list[float]) -> float:
    return rng.choices(options, weights=weights)[0]


def fake_rows(questions: list[dict], seed: int = SEED) -> list[Row]:
    rng = random.Random(seed)
    rows: list[Row] = []
    for q in questions:
        answerable = bool(q["answerable"])
        meetings = list(q["source_meetings"])
        for mode in MODES:
            # graph чуть лучше на вопросах про эволюцию и конфликты — чтобы в вёрстке были и «доказано», и «не доказано»
            boost = 0.25 if mode == "graph" and q["type"] in ("T2", "T4") else 0.0
            good = min(0.95, 0.55 + boost)
            r = Row(question_id=q["id"], type=q["type"], split=q["split"], lang=q["lang"], as_of=q["as_of"], mode=mode,
                    system="real", backend="fixture", status="ok", answerable=answerable,
                    answer=f"[fixture] выдуманный ответ на {q['id']} ({mode})",
                    source_meetings=",".join(meetings), repeat=1, from_cache=False)
            if answerable:
                cited = [m for m in meetings if rng.random() < good]
                if rng.random() < 0.2:
                    cited.append("m_%03d" % rng.randint(1, 15))
                cited = sorted(set(cited))
                r.citations = [{"meeting_id": m, "ts": "00:01:05", "quote": "[fixture] цитата"} for m in cited]
                r.cited_meetings = ",".join(cited)
                hit = len(set(cited) & set(meetings))
                r.citation_recall = round(hit / len(meetings), 4)
                r.citation_precision = round(hit / len(cited), 4) if cited else 0.0
                r.citations_total = len(cited)
                r.citations_valid = len(cited)
                r.citation_valid = 1.0 if cited else None
                if rng.random() < 0.9:
                    ret = sorted(set(meetings) | {m for m in ["m_%03d" % rng.randint(1, 15)] if rng.random() < 0.5})
                    ret = [m for m in ret if m not in meetings or rng.random() < min(1.0, good + 0.3)]
                    r.retrieved_meetings = ret
                    r.retrieval_recall = round(len(set(ret) & set(meetings)) / len(meetings), 4)
                n_req = rng.randint(1, 3)
                present = sum(rng.random() < good for _ in range(n_req))
                claims = rng.randint(2, 6)
                wrong = sum(rng.random() < (0.12 if mode == "graph" else 0.18) for _ in range(claims))
                sup = min(present + rng.randint(0, 1), claims - wrong)
                r.fact_recall = round(present / n_req, 4)
                r.fact_precision = round((claims - wrong) / claims, 4)
                r.fact_precision_strict = round(sup / claims, 4)
                r.forbidden_hits = wrong if rng.random() < 0.5 else 0
                r.abstained = rng.random() < (0.05 if mode == "graph" else 0.1)    # ложный отказ
                if r.abstained:     # отказ на вопрос с ответом: факты не найдены, утверждений нет
                    r.fact_recall = r.fact_precision = r.fact_precision_strict = 0.0
                    r.forbidden_hits = 0
            else:
                refuse = rng.random() < (0.85 if mode == "graph" else 0.7)
                r.fact_recall = 1.0 if refuse else 0.0
                r.fact_precision = r.fact_precision_strict = 1.0 if refuse else _pick(rng, [0.0, 0.33], [1, 1])
                r.forbidden_hits = 0 if refuse else 1
                r.abstained = refuse
            r.fact_f1 = round(harmonic(r.fact_recall, r.fact_precision), 4)
            r.prompt_tokens = rng.randint(1500, 4500) + (900 if mode == "baseline" else 0)
            r.context_tokens, r.context_tokens_source = r.prompt_tokens, "usage.prompt_tokens"
            r.completion_tokens = rng.randint(80, 300)
            r.cost_usd = round(r.prompt_tokens * 2.5e-6 + r.completion_tokens * 1e-5, 6)
            r.latency_ms = rng.randint(1800, 6500) + (700 if mode == "graph" else 0)
            r.judge_usage = {"prompt_tokens": 900, "completion_tokens": 250, "cost_usd": 0.0048}
            rows.append(r)
    return rows


def build_fixture(questions: list[dict]) -> dict:
    rows = fake_rows(questions)
    as_of = results_v2.as_of_label(q["as_of"] for q in questions)
    config = results_v2.build_config(
        modes=MODES, split="dev", n_questions=len(questions), as_of=as_of, judge_repeats=1,
        repeats=1, backend="fixture", commit="fixture", judge_facts=True, fixture=True)
    config.update(model="fixture-model", embed_model="fixture-embed", judge_model="fixture-judge",
                  JUDGE_FACTS_PROMPT_VERSION=JUDGE_FACTS_PROMPT_VERSION, fixture_note=NOTE)
    return results_v2.build_result(run_id="fixture", timestamp=TIMESTAMP, config=config, rows=rows)


def main() -> int:
    questions = json.loads(DEV_QUESTIONS.read_text(encoding="utf-8"))
    keep = ("id", "type", "split", "lang", "as_of", "answerable", "source_meetings")
    results_v2.write_result(FIXTURE, build_fixture([{k: q[k] for k in keep} for q in questions]))
    print(f"Фикстура записана: {FIXTURE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
