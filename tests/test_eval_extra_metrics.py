"""Срез по языку, отказы, forbidden_rate, context_tokens. Заглушки, без сети и модели."""
import csv
import json

import pytest

import src.answer.answerer as answerer
import src.llm_client as llm_client
from src import api
from src.eval import judge_facts as jf
from src.eval import run as ev
from src.eval import results_v2 as r2
from src.eval import systems
from src.eval.judge_facts import Claim, FactsVerdict
from src.schemas import Answer, Usage
from tests.test_eval_v2_results import FIXTURE, AnswerWithRetrieval, cli, load_result, q, run, system

REFUSAL = "В контексте нет информации по этому вопросу."


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_BACKEND", "stub")
    monkeypatch.setenv("EMBED_MODEL", "")
    monkeypatch.setenv("LLM_MODEL", "gpt-answer")
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    monkeypatch.setattr(answerer, "REPORT", tmp_path / "answer_report.jsonl")
    monkeypatch.setattr(ev, "RESULTS", tmp_path / "results")
    monkeypatch.setattr(llm_client, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(llm_client, "RUNS", tmp_path / "runs")
    yield
    api.use_index_dir(api.DEFAULT_INDEX_DIR)


@pytest.fixture
def raw(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    segs = [{"start": 65.0, "speaker": "SPEAKER_00", "text": "Договорились сдвинуть релиз Zhol на 4 сентября."}]
    (raw_dir / "m_001.transcript.json").write_text(json.dumps({"segments": segs}, ensure_ascii=False),
                                                   encoding="utf-8")
    gold = {"ev_m001_01": {"id": "ev_m001_01", "meeting": "m_001", "start_s": 65.0, "quote": segs[0]["text"]}}
    return raw_dir, gold


def rec(qid, mode="baseline", qtype="T1", lang="ru", answerable=True, abstained=False, status="ok", **metrics):
    """Строка per_question: только то, что читает агрегат."""
    return {"qid": qid, "question_id": qid, "type": qtype, "mode": mode, "lang": lang, "status": status,
            "answerable": answerable, "abstained": abstained, "metrics": metrics,
            "context_tokens_source": "usage.prompt_tokens"}


def cell(agg_part):
    return agg_part["mean"], agg_part["n"], agg_part["note"]


# ---------- 1. срез по языку ----------

def test_language_slice_known_example_and_small_group_note():
    recs = ([rec(f"q_{i}", lang="ru", fact_f1=v) for i, v in enumerate([1, 0, 1, 1, 0, 1])]
            + [rec("k_1", lang="kk", fact_f1=1.0), rec("k_2", lang="kk", fact_f1=0.0)])
    agg = r2.compute_aggregate(recs, ["baseline"])
    ru, kk = agg["by_mode_lang"]["baseline"]["ru"]["fact_f1"], agg["by_mode_lang"]["baseline"]["kk"]["fact_f1"]
    assert ru["mean"] == pytest.approx(4 / 6) and ru["n"] == 6 and ru["note"] is None
    assert ru["ci_low"] <= ru["mean"] <= ru["ci_high"]
    assert kk["mean"] == 0.5 and kk["n"] == 2 and kk["note"] == "n<5, интервал не информативен"
    assert {"mean", "ci_low", "ci_high", "n"} <= set(kk)
    md = "\n".join(r2.render_v2_section(agg, recs, ["baseline"], judge_facts=True))
    assert "### Срез по языку вопроса" in md
    kk_row = next(line for line in md.splitlines() if line.startswith("| kk | fact_f1"))
    assert "n<5, интервал не информативен" in kk_row
    ru_row = next(line for line in md.splitlines() if line.startswith("| ru | fact_f1"))
    assert "n<5" not in ru_row


def test_language_slice_skips_languages_that_have_no_questions_and_modes_without_answers():
    recs = [rec(f"q_{i}", lang="ru", fact_f1=1.0) for i in range(6)]
    recs += [rec(f"q_{i}", mode="graph", lang="ru", status="stub") for i in range(6)]   # режим без ответов
    agg = r2.compute_aggregate(recs, ["baseline", "graph"])
    assert set(agg["by_mode_lang"]["baseline"]) == {"ru"}                  # kk нет — группы нет, а не нулевая
    g = agg["by_mode_lang"]["graph"]["ru"]["fact_f1"]
    assert g["n"] == 0 and g["mean"] is None and g["ci_low"] is None       # пусто, но не 0 и не ошибка
    md = "\n".join(r2.render_v2_section(agg, recs, ["baseline", "graph"], judge_facts=True))
    assert any(line.startswith("| ru | fact_f1") and "—" in line for line in md.splitlines())


# ---------- 2. отказы ----------

def test_abstention_rates_known_example():
    unans = [rec(f"u_{i}", qtype="none", answerable=False, abstained=(i < 3)) for i in range(4)]   # 3 из 4 отказали
    ans = [rec(f"a_{i}", abstained=(i == 0)) for i in range(5)]                                     # 1 из 5 ложный
    ab = r2.compute_aggregate(unans + ans, ["baseline"])["abstention"]["baseline"]
    assert cell(ab["abstain_rate_unanswerable"]) == (0.75, 4, "n<5, интервал не информативен")
    assert cell(ab["false_abstain_rate"]) == (pytest.approx(0.2), 5, None)


def test_abstention_excludes_error_and_stub_rows_and_averages_repeats_per_question():
    recs = [rec("a_1", abstained=True), rec("a_1", abstained=False),          # два повтора одного вопроса → 0.5
            rec("a_2", abstained=False),
            rec("a_3", status="error", abstained=False),                       # ошибка: отказом не считается
            rec("u_1", answerable=False, status="stub", abstained=None)]       # заглушка
    ab = r2.compute_aggregate(recs, ["baseline"])["abstention"]["baseline"]
    assert ab["false_abstain_rate"]["n"] == 2 and ab["false_abstain_rate"]["mean"] == pytest.approx(0.25)
    assert ab["abstain_rate_unanswerable"]["n"] == 0


def test_abstention_without_questions_of_a_group_or_without_answers_has_no_division_by_zero():
    ab = r2.compute_aggregate([rec("a_1")], ["baseline", "graph"])["abstention"]       # графа вообще нет в записях
    assert ab["baseline"]["abstain_rate_unanswerable"] == {"mean": None, "ci_low": None, "ci_high": None, "n": 0,
                                                           "note": None}
    assert ab["graph"]["false_abstain_rate"]["mean"] is None and ab["graph"]["false_abstain_rate"]["n"] == 0
    empty = r2.compute_aggregate([], ["baseline"])
    assert empty["abstention"]["baseline"]["false_abstain_rate"]["n"] == 0 and empty["by_mode_lang"] == {}
    md = "\n".join(r2.render_v2_section(empty, [], ["baseline"], judge_facts=False))
    assert "### Отказы" in md and "false_abstain_rate" in md


def test_abstention_through_the_harness_uses_is_abstention_unchanged(raw):
    answers = {"вопрос q_01": REFUSAL, "вопрос q_02": "Бюджет составил 5 млн.", "вопрос q_03": "Релиз 4 сентября.",
               "вопрос q_04": "Нет данных об этом."}                  # ложный отказ

    def ask(question):
        return Answer(answer=answers[question], mode="baseline", citations=[])
    qs = [q("q_01", "none", answerable=False), q("q_02", "none", answerable=False), q("q_03"), q("q_04")]
    rows = run(qs, {"baseline": systems.System("baseline", ask)}, raw)
    assert [r.abstained for r in rows] == [True, False, False, True]
    recs = [r2.per_question_record(r) for r in rows]
    assert [(x["answerable"], x["abstained"]) for x in recs] == [(False, True), (False, False), (True, False),
                                                                 (True, True)]
    ab = r2.compute_aggregate(recs, ["baseline"])["abstention"]["baseline"]
    assert ab["abstain_rate_unanswerable"]["mean"] == 0.5 and ab["false_abstain_rate"]["mean"] == 0.5


# ---------- 3. forbidden_rate ----------

def verdict_with_hits(hits):
    return FactsVerdict([("срок 4 сентября", True)], [(f"запрет {i}", True) for i in range(hits)] or
                        [("запрет", False)], [Claim("х", "supported")])


def test_forbidden_rate_is_share_of_answers_with_hits_by_mode_and_type(raw):
    hits = {"вопрос q_01": 0, "вопрос q_02": 2, "вопрос q_03": 1, "вопрос q_04": 0}
    fn = lambda question, *a: verdict_with_hits(hits[question])  # noqa: E731
    qs = [q("q_01", "T2"), q("q_02", "T2"), q("q_03", "T2"), q("q_04", "T4")]
    rows = run(qs, {"baseline": system("baseline")}, raw, facts=jf.JudgeFactsRunner(fn=fn))
    assert [r.forbidden_hits for r in rows] == [0, 2, 1, 0]
    agg = r2.compute_aggregate([r2.per_question_record(r) for r in rows], ["baseline"])["by_mode_type"]["baseline"]
    assert cell(agg["T2"]["forbidden_rate"])[:2] == (pytest.approx(2 / 3), 3)     # 2 из 3 ответов с запретным фактом
    assert agg["T4"]["forbidden_rate"]["mean"] == 0.0
    assert agg["all"]["forbidden_rate"]["mean"] == 0.5 and agg["all"]["forbidden_hits"]["mean"] == 0.75


def test_forbidden_rate_counts_fractional_hits_of_judge_repeats_and_is_null_without_the_judge(raw):
    (row,) = run([q("q_01")], {"baseline": system("baseline")}, raw)           # без --judge-facts
    assert r2.row_metrics(row)["forbidden_rate"] is None
    row.forbidden_hits = 0.5        # судья по фактам нашёл запретное в одном из двух повторов
    assert r2.row_metrics(row)["forbidden_rate"] == 1.0
    row.forbidden_hits = 0
    assert r2.row_metrics(row)["forbidden_rate"] == 0.0
    agg = r2.compute_aggregate([r2.per_question_record(row)], ["baseline"])
    assert agg["by_mode_type"]["baseline"]["T1"]["forbidden_rate"]["n"] == 1


# ---------- 4. context_tokens ----------

class UsageWithContext(Usage):
    context_tokens: int | None = None


def test_context_tokens_is_prompt_tokens_when_usage_has_no_separate_number(raw):
    def ask(question):
        return Answer(answer="ответ", mode="baseline", usage=Usage(prompt_tokens={"вопрос q_01": 10,
                                                                                  "вопрос q_02": 30}[question]))
    rows = run([q("q_01"), q("q_02")], {"baseline": systems.System("baseline", ask)}, raw)
    assert [r.context_tokens for r in rows] == [10, 30]
    assert {r.context_tokens_source for r in rows} == {"usage.prompt_tokens"}
    recs = [r2.per_question_record(r) for r in rows]
    assert recs[0]["metrics"]["context_tokens"] == 10 and recs[0]["usage"]["prompt_tokens"] == 10
    agg = r2.compute_aggregate(recs, ["baseline"])["by_mode_type"]["baseline"]
    assert agg["T1"]["context_tokens"]["mean"] == 20 and agg["T1"]["context_tokens"]["n"] == 2
    md = "\n".join(r2.render_v2_section(r2.compute_aggregate(recs, ["baseline"]), recs, ["baseline"],
                                        judge_facts=False))
    assert "context_tokens = usage.prompt_tokens" in md and "отдельного числа для контекста в usage нет" in md


def test_context_tokens_prefers_a_separate_context_number_when_usage_has_it(raw):
    def ask(question):
        return AnswerWithRetrieval(answer="ответ", mode="baseline", usage=UsageWithContext(prompt_tokens=100,
                                                                                           context_tokens=70))
    (row,) = run([q()], {"baseline": systems.System("baseline", ask)}, raw)
    assert row.context_tokens == 70 and row.context_tokens_source == "usage.context_tokens"
    recs = [r2.per_question_record(row)]
    md = "\n".join(r2.render_v2_section(r2.compute_aggregate(recs, ["baseline"]), recs, ["baseline"],
                                        judge_facts=False))
    assert "usage.context_tokens (отдельное число для контекста)" in md and "= usage.prompt_tokens" not in md


def test_context_tokens_is_none_for_stub_and_failed_answers_not_zero(raw):
    def boom(question):
        raise RuntimeError("сбой")
    rows = run([q("q_01"), q("q_02")], {"baseline": systems.System("baseline", boom),
                                         "graph": systems.stub_system("graph", "нет графа")}, raw)
    assert all(r.context_tokens is None for r in rows)
    agg = r2.compute_aggregate([r2.per_question_record(r) for r in rows], ["baseline", "graph"])
    cell_ = agg["by_mode_type"]["baseline"]["T1"]["context_tokens"]
    assert cell_["n"] == 0 and cell_["mean"] is None


# ---------- всё вместе: json, csv, сводка, фикстура ----------

def test_cli_json_csv_and_summary_contain_the_new_metrics_and_keep_the_old_ones(tmp_path):
    assert cli(tmp_path, "--run-id", "x1") == 0
    d = load_result(tmp_path, "x1")
    agg = d["aggregate"]
    assert set(agg) >= {"by_mode_type", "by_mode_lang", "abstention", "paired", "paired_pairs", "bootstrap"}
    assert set(agg["by_mode_lang"]["baseline"]) == {"ru", "kk"}               # в тестовом файле вопросов ru и kk
    assert agg["by_mode_lang"]["baseline"]["kk"]["citation_recall"]["n"] == 0      # kk-вопрос без ответа: не считается
    assert set(agg["abstention"]["baseline"]) == {"abstain_rate_unanswerable", "false_abstain_rate"}
    rec0 = d["per_question"][0]
    assert {"forbidden_rate", "context_tokens"} <= set(rec0["metrics"]) and "abstained" in rec0
    out = tmp_path / "out"
    (csv_path,) = out.glob("eval_*.csv")
    cols = csv.DictReader(csv_path.open(encoding="utf-8-sig")).fieldnames
    assert "context_tokens" in cols and "correct_meetings" in cols and "judge_score" in cols
    md = next(out.glob("summary_*.md")).read_text(encoding="utf-8")
    for part in ("### Срез по языку вопроса", "### Отказы", "### Запрещённые факты и контекст в промпте",
                 "n<5, интервал не информативен", "## test"):
        assert part in md, part


def test_fixture_has_the_new_aggregates_and_metrics_and_stays_marked_as_fixture():
    d = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert d["config"]["fixture"] is True
    agg = d["aggregate"]
    assert set(agg["by_mode_lang"]["graph"]) == {"ru", "kk"}
    for mode in ("baseline", "graph"):
        assert set(agg["abstention"][mode]) == {"abstain_rate_unanswerable", "false_abstain_rate"}
        assert {"forbidden_rate", "context_tokens"} <= set(agg["by_mode_type"][mode]["all"])
    assert all({"forbidden_rate", "context_tokens"} <= set(r["metrics"]) and "abstained" in r
               for r in d["per_question"])
    assert any(r["context_tokens_source"] == "usage.prompt_tokens" for r in d["per_question"])
    assert json.loads(json.dumps(r2.compute_aggregate(d["per_question"], d["config"]["modes"]))) == agg
    assert any(c["note"] for m in agg["abstention"].values() for c in m.values())      # есть группа n<5
