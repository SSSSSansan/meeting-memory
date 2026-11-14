"""Json-результат eval v2, фикстура для дашборда, --judge-facts, --resume, --dry-run. Заглушки, без сети и модели."""
import csv
import json
from pathlib import Path

import pytest

import src.answer.answerer as answerer
import src.llm_client as llm_client
from src import api
from src.eval import bootstrap as bs
from src.eval import estimate as est_mod
from src.eval import judge_facts as jf
from src.eval import results_v2
from src.eval import run as ev
from src.eval import systems
from src.eval.judge import JudgeResult
from src.eval.judge_facts import Claim, FactsVerdict
from src.eval.metrics import MeetingTexts
from src.schemas import Answer, Citation
from tests.eval_helpers import build_index, graph_as_stub

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "fixtures" / "eval_fixture.json"
CITE = Citation(meeting_id="m_001", ts="00:01:05", quote="сдвинуть релиз Zhol на 4 сентября")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    graph_as_stub(monkeypatch)
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


class AnswerWithRetrieval(Answer):
    """Как будет выглядеть Answer, когда роль C добавит retrieved_meetings (аддитивное поле)."""
    retrieved_meetings: list[str] | None = None


def q(qid="q_01", qtype="T1", answerable=True, **kw):
    data = dict(id=qid, type=qtype, split="test", lang="ru", question=f"вопрос {qid}", as_of="m_001",
                expected_answer="эталон", source_meetings=["m_001", "m_002"] if answerable else [],
                decision_ids=["ev_m001_01"] if answerable else [], answerable=answerable,
                required_facts=["срок 4 сентября"] if answerable else ["в данных этого нет"],
                forbidden_facts=["срок 15 сентября"])
    return ev.Question.model_validate({**data, **kw})


@pytest.fixture
def raw(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    segs = [{"start": 65.0, "speaker": "SPEAKER_00", "text": "Договорились сдвинуть релиз Zhol на 4 сентября."}]
    (raw_dir / "m_001.transcript.json").write_text(json.dumps({"segments": segs}, ensure_ascii=False),
                                                   encoding="utf-8")
    gold = {"ev_m001_01": {"id": "ev_m001_01", "meeting": "m_001", "start_s": 65.0, "quote": segs[0]["text"]}}
    return raw_dir, gold


def system(mode, retrieved=None, text="Релиз 4 сентября."):
    def ask(question):
        return AnswerWithRetrieval(answer=text, mode=mode, citations=[CITE], latency_ms=5, retrieved_meetings=retrieved,
                                   usage={"prompt_tokens": 10, "completion_tokens": 2, "cost_usd": 0.01})
    return systems.System(mode=mode, ask=ask)


def facts_fn_factory(per_question=None, calls=None):
    """Судья по фактам-заглушка: per_question {текст вопроса: FactsVerdict}; по умолчанию всё найдено."""
    def fn(question, answer, required, forbidden, answerable):
        if calls is not None:
            calls.append(question)
        if per_question and question in per_question:
            return per_question[question]
        return FactsVerdict([(f, True) for f in required], [(f, False) for f in forbidden],
                            [Claim("утверждение", "supported")], 10, 5, 0.001)
    return fn


def run(questions, builders, raw, *, facts=None, repeats=1, **kw):
    raw_dir, gold = raw
    return ev.run_eval(questions, list(builders), lambda mode, as_of: builders[mode], gold=gold,
                       texts=MeetingTexts(raw_dir), backend="stub", repeats=repeats, facts=facts, **kw)


# ---------- retrieval_recall: не измерено, пока ask() не отдаёт retrieved_meetings ----------

def test_retrieval_recall_is_null_when_ask_does_not_return_retrieved_meetings(raw):
    plain = systems.System("baseline", lambda question: Answer(answer="Релиз 4 сентября.", mode="baseline",
                                                               citations=[CITE]))
    (row,) = run([q()], {"baseline": plain}, raw)
    assert row.retrieved_meetings is None and row.retrieval_recall is None
    assert row.citation_recall == 0.5 and row.citation_precision == 1.0
    rec = results_v2.per_question_record(row)
    assert rec["retrieved_meetings"] is None and rec["metrics"]["retrieval_recall"] is None


def test_retrieval_recall_is_computed_as_soon_as_ask_returns_the_list(raw):
    (row,) = run([q()], {"baseline": system("baseline", retrieved=["m_001", "m_009"])}, raw)
    assert row.retrieved_meetings == ["m_001", "m_009"] and row.retrieval_recall == 0.5
    (row,) = run([q("q_02", "none", answerable=False)], {"baseline": system("baseline", retrieved=["m_001"])}, raw)
    assert row.retrieval_recall is None and row.citation_recall is None   # answerable=false не считается


# ---------- структура json ----------

KEYS = {"run_id", "timestamp", "config", "per_question", "aggregate"}
CONFIG_KEYS = {"model", "embed_model", "judge_model", "JUDGE_FACTS_PROMPT_VERSION", "modes", "split", "n_questions",
               "as_of", "judge_repeats"}
RECORD_KEYS = {"qid", "question_id", "type", "mode", "repeat", "answer", "citations", "retrieved_meetings", "metrics",
               "usage", "latency_ms"}
METRIC_KEYS = {"citation_recall", "citation_precision", "retrieval_recall", "quote_validity", "fact_recall",
               "fact_precision", "fact_f1", "forbidden_hits"}


def assert_contract(d):
    assert KEYS <= set(d)
    assert CONFIG_KEYS <= set(d["config"]) and d["config"]["JUDGE_FACTS_PROMPT_VERSION"] == "judge-facts-v1"
    assert d["per_question"], "нет per_question"
    for rec in d["per_question"]:
        assert RECORD_KEYS <= set(rec) and METRIC_KEYS <= set(rec["metrics"])
        assert rec["qid"] == rec["question_id"]
        assert {"prompt_tokens", "completion_tokens", "cost_usd"} <= set(rec["usage"])
    by = d["aggregate"]["by_mode_type"]
    for mode, types in by.items():
        for qtype, metrics in types.items():
            for metric, cell in metrics.items():
                assert {"mean", "ci_low", "ci_high", "n"} <= set(cell), (mode, qtype, metric)
    for qtype, metrics in d["aggregate"]["paired"].items():
        for metric, cell in metrics.items():
            assert {"diff_mean", "ci_low", "ci_high", "n", "verdict"} <= set(cell), (qtype, metric)


def test_build_result_has_the_contract_and_aggregate_is_computed_by_the_harness(raw):
    qs = [q(f"q_{i:02d}", "T2") for i in range(1, 7)]
    facts = jf.JudgeFactsRunner(fn=facts_fn_factory())
    rows = run(qs, {"baseline": system("baseline", ["m_001"]), "graph": system("graph", ["m_001", "m_002"])}, raw,
               facts=facts)
    cfg = results_v2.build_config(modes=["baseline", "graph"], split="test", n_questions=6, as_of="m_001",
                                  judge_repeats=1, repeats=1, backend="stub", commit="abc", judge_facts=True)
    d = results_v2.build_result(run_id="20260101_1200", timestamp="2026-01-01T12:00:00+05:00", config=cfg, rows=rows)
    assert_contract(d)
    assert len(d["per_question"]) == 12
    cell = d["aggregate"]["by_mode_type"]["graph"]["T2"]["retrieval_recall"]
    assert cell["mean"] == 1.0 and cell["n"] == 6 and cell["note"] is None
    paired = d["aggregate"]["paired"]["T2"]["retrieval_recall"]
    assert paired["diff_mean"] == pytest.approx(0.5) and paired["n"] == 6      # graph − baseline
    assert d["aggregate"]["paired"]["T2"]["fact_f1"]["verdict"] == "различие не доказано"   # одинаковые режимы
    json.dumps(d, ensure_ascii=False)            # сериализуется целиком


def test_aggregate_takes_mean_over_answer_repeats_before_bootstrap_over_questions(raw):
    answers = iter(["Релиз 4 сентября.", "Релиз 15 сентября."] * 6)
    sys_ = systems.System("baseline", lambda question: Answer(answer=next(answers), mode="baseline", citations=[CITE]))
    qs = [q(f"q_{i:02d}") for i in range(1, 4)]
    per_answer = {"Релиз 4 сентября.": 1.0, "Релиз 15 сентября.": 0.0}

    def fn(question, answer, required, forbidden, answerable):
        return FactsVerdict([(required[0], bool(per_answer[answer]))], [], [Claim("х", "supported")])
    rows = run(qs, {"baseline": sys_}, raw, repeats=2, facts=jf.JudgeFactsRunner(fn=fn))
    recs = [results_v2.per_question_record(r) for r in rows]
    assert len(recs) == 6                                      # повторы остаются в per_question
    cell = results_v2.compute_aggregate(recs, ["baseline"])["by_mode_type"]["baseline"]["all"]["fact_recall"]
    assert cell["n"] == 3 and cell["mean"] == 0.5              # n — вопросы, а не вопросы × повторы


def test_stub_mode_rows_have_null_metrics_and_are_reported_as_excluded(raw):
    qs = [q(f"q_{i:02d}") for i in range(1, 7)]
    stub = systems.stub_system("graph", "нет графа")
    rows = run(qs, {"baseline": system("baseline", ["m_001"]), "graph": stub}, raw)
    recs = [results_v2.per_question_record(r) for r in rows]
    agg = results_v2.compute_aggregate(recs, ["baseline", "graph"])
    cell = agg["paired"]["all"]["citation_recall"]
    assert cell["n"] == 0 and cell["n_excluded"] == 6 and cell["verdict"] == bs.VERDICT_NO_DATA
    g = agg["by_mode_type"]["graph"]["all"]["citation_recall"]      # у заглушки ячейка есть, но пустая: не «0»
    assert g["n"] == 0 and g["mean"] is None and g["ci_low"] is None
    md = "\n".join(results_v2.render_v2_section(agg, recs, ["baseline", "graph"], judge_facts=False))
    assert "Исключены из обоих режимов" in md and "citation_recall: 6" in md


def test_unequal_missing_values_are_excluded_from_both_modes_in_the_summary(raw):
    qs = [q(f"q_{i:02d}") for i in range(1, 8)]
    # у graph по q_01 судья не вернул вердикт → вопрос выпадает и из baseline
    calls = {"n": 0}

    def fn(question, answer, required, forbidden, answerable):
        calls["n"] += 1
        if question == "вопрос q_01" and answer == "ответ graph":
            raise jf.FactsJudgeFormatError("плохой формат")
        return FactsVerdict([(f, True) for f in required], [], [Claim("х", "supported")])
    builders = {"baseline": system("baseline", text="ответ baseline"), "graph": system("graph", text="ответ graph")}
    rows = run(qs, builders, raw, facts=jf.JudgeFactsRunner(fn=fn))
    recs = [results_v2.per_question_record(r) for r in rows]
    agg = results_v2.compute_aggregate(recs, ["baseline", "graph"])
    p = agg["paired"]["all"]["fact_f1"]
    assert p["n"] == 6 and p["n_excluded"] == 1
    assert agg["by_mode_type"]["baseline"]["all"]["fact_f1"]["n"] == 7     # без пары режим считает всё, что есть
    md = "\n".join(results_v2.render_v2_section(agg, recs, ["baseline", "graph"], judge_facts=True))
    assert "fact_f1: 1" in md


def test_summary_section_states_interval_meaning_not_measured_and_verdicts(raw):
    qs = [q(f"q_{i:02d}") for i in range(1, 4)]       # три вопроса < 5
    rows = run(qs, {"baseline": system("baseline"), "graph": system("graph")}, raw)
    recs = [results_v2.per_question_record(r) for r in rows]
    agg = results_v2.compute_aggregate(recs, ["baseline", "graph"])
    md = "\n".join(results_v2.render_v2_section(agg, recs, ["baseline", "graph"], judge_facts=False))
    assert "зависимость от набора вопросов, а не разброс повторов" in md and "температуре 0 повторы ответов" in md
    assert "retrieval_recall: не измерено" in md and "роли C" in md
    assert "fact_recall, fact_precision, fact_f1: не измерено (нет --judge-facts)" in md
    assert "n<5, интервал не информативен" in md


# ---------- фикстура для вкладки метрик ----------

def test_fixture_follows_the_contract_is_marked_fixture_and_its_aggregate_is_consistent():
    d = json.loads(FIXTURE.read_text(encoding="utf-8"))
    assert_contract(d)
    assert d["config"]["fixture"] is True and "fixture" in d["run_id"] and "выдуманы" in d["config"]["fixture_note"]
    assert d["config"]["split"] == "dev"                      # test в фикстуре не используется
    assert {r["split"] for r in d["per_question"]} == {"dev"}
    # aggregate в файле — ровно то, что считает харнес по этим per_question
    again = results_v2.compute_aggregate(d["per_question"], d["config"]["modes"])
    assert json.loads(json.dumps(again)) == d["aggregate"]
    modes = {r["mode"] for r in d["per_question"]}
    assert modes == {"baseline", "graph"}
    metrics = {k for r in d["per_question"] for k, v in r["metrics"].items() if v is None}
    assert "retrieval_recall" in metrics                      # и число, и null — вкладка видит оба случая
    verdicts = {c["verdict"] for t in d["aggregate"]["paired"].values() for c in t.values()}
    assert {bs.VERDICT_NOT_PROVEN, bs.VERDICT_SMALL_N} <= verdicts


def test_fixture_generator_is_deterministic():
    from scripts import make_eval_fixture as mk
    questions = json.loads(mk.DEV_QUESTIONS.read_text(encoding="utf-8"))
    keep = ("id", "type", "split", "lang", "as_of", "answerable", "source_meetings")
    qs = [{k: x[k] for k in keep} for x in questions]
    assert mk.build_fixture(qs) == mk.build_fixture(qs)


# ---------- CLI: json, --judge-facts, старый путь ----------

def questions_file(tmp_path, n_extra=0):
    items = [{"id": "q_01", "type": "T1", "split": "test", "lang": "ru", "question": "Когда релиз Zhol?",
              "as_of": "m_003", "expected_answer": "x", "source_meetings": ["m_001"], "decision_ids": ["ev_m001_01"],
              "answerable": True, "required_facts": ["срок 4 сентября"], "forbidden_facts": ["срок 15 сентября"]},
             {"id": "q_02", "type": "none", "split": "test", "lang": "kk", "question": "Какой бюджет?",
              "as_of": "m_003", "expected_answer": "нет", "source_meetings": [], "decision_ids": [],
              "answerable": False, "required_facts": ["в данных этого нет"], "forbidden_facts": ["бюджет 5 млн"]}]
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return path


def cli(tmp_path, *extra, judge_fn=None, facts_fn=None):
    build_index(tmp_path / "idx")
    return ev.main(["--mode", "both", "--split", "test", "--as-of", "m_003", "--questions-file",
                    str(questions_file(tmp_path)), "--out-dir", str(tmp_path / "out"),
                    "--index-dir", str(tmp_path / "idx"), "--raw-dir", str(ROOT / "data" / "raw"), *extra],
                   judge_fn=judge_fn, facts_fn=facts_fn)


def load_result(tmp_path, run_id):
    return json.loads((tmp_path / "out" / f"eval_{run_id}.json").read_text(encoding="utf-8"))


def test_cli_writes_json_without_judge_facts_with_null_fact_metrics_and_keeps_csv_and_summary(tmp_path):
    assert cli(tmp_path, "--run-id", "t1") == 0
    d = load_result(tmp_path, "t1")
    assert_contract(d)
    assert d["run_id"] == "t1" and d["config"]["judge_facts"] is False and d["config"]["judge_repeats"] == 1
    base = [r for r in d["per_question"] if r["mode"] == "baseline"]
    assert len(base) == 2 and all(r["metrics"]["fact_f1"] is None for r in base)
    # С PR #42 api.ask() отдаёт Answer.retrieved_meetings: полнота поиска у настоящей системы измерена.
    # У вопроса без ответа в данных (answerable=false) эталонных встреч нет — там null.
    assert all(isinstance(r["retrieved_meetings"], list) for r in base)
    assert all((r["metrics"]["retrieval_recall"] is not None) == r["answerable"] for r in base)
    out = tmp_path / "out"
    assert not list(out.glob("*.partial.jsonl"))                      # итог записан — промежуточный файл удалён
    (csv_path,) = out.glob("eval_*.csv")
    cols = csv.DictReader(csv_path.open(encoding="utf-8-sig")).fieldnames
    assert {"question_id", "correct_meetings", "events_found", "judge_score", "judge_fabrication",
            "repeat"} <= set(cols)
    assert {"citation_recall", "fact_f1", "forbidden_hits"} <= set(cols)       # новые колонки только добавлены
    assert not {"citations", "facts_detail"} & set(cols)
    (md_path,) = out.glob("summary_*.md")
    md = md_path.read_text(encoding="utf-8")
    assert "## Метрики v2" in md and "retrieval_recall: не измерено" not in md and "## test" in md


def test_cli_judge_facts_fills_metrics_and_old_judge_v2_still_works_alongside(tmp_path):
    seen = []
    verdict = FactsVerdict([("срок 4 сентября", True)], [("срок 15 сентября", False)], [Claim("релиз", "supported")])
    assert cli(tmp_path, "--run-id", "t2", "--judge", "--judge-facts", judge_fn=lambda *a, **k: JudgeResult(2, 0, "ок"),
               facts_fn=facts_fn_factory({"Когда релиз Zhol?": verdict}, seen)) == 0
    d = load_result(tmp_path, "t2")
    assert_contract(d)
    first = next(r for r in d["per_question"] if r["qid"] == "q_01" and r["mode"] == "baseline")
    assert first["metrics"]["fact_recall"] == 1.0 and first["metrics"]["fact_f1"] == 1.0
    assert first["metrics"]["forbidden_hits"] == 0 and first["judge_facts"]["claims"][0]["label"] == "supported"
    assert d["config"]["judge_facts"] is True and d["aggregate"]["by_mode_type"]["baseline"]["T1"]["fact_f1"]["n"] == 1
    assert seen == ["Когда релиз Zhol?", "Какой бюджет?"]            # graph — заглушка, судью не зовёт
    out = tmp_path / "out"
    (csv_path,) = out.glob("eval_*.csv")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    assert {r["judge_score"] for r in rows if r["mode"] == "baseline"} == {"2"}     # старый --judge работает
    assert {r["fact_f1"] for r in rows if r["mode"] == "baseline" and r["question_id"] == "q_01"} == {"1.0"}


def test_judge_facts_flag_is_off_by_default_and_never_touches_the_judge(tmp_path):
    assert cli(tmp_path, "--run-id", "t3", facts_fn=lambda *a: pytest.fail("судья по фактам не должен вызываться")) == 0


def test_judge_receives_neither_mode_nor_type_in_the_harness(raw):
    seen = []

    def fn(*args, **kwargs):
        seen.append((args, kwargs))
        return FactsVerdict([(f, True) for f in args[2]], [], [])
    run([q("q_01", "T4")], {"baseline": system("baseline"), "graph": system("graph")}, raw,
        facts=jf.JudgeFactsRunner(fn=fn))
    assert len(seen) == 1                                       # одинаковый ответ двух режимов — один вызов
    (args, kwargs), = seen
    assert kwargs == {} and len(args) == 5                      # вопрос, ответ, required, forbidden, answerable
    assert args == ("вопрос q_01", "Релиз 4 сентября.", ["срок 4 сентября"],
                    ["срок 15 сентября"], True)
    assert "T4" not in repr(args) and "baseline" not in repr(args) and "graph" not in repr(args)


def test_cli_stub_backend_refuses_judge_facts_without_dry_run(tmp_path, capsys):
    assert cli(tmp_path, "--judge-facts") == 2
    assert "заглушка не оценивает" in capsys.readouterr().err


@pytest.mark.parametrize("args", [["--judge-repeats", "2"], ["--judge-facts", "--judge-repeats", "6"],
                                  ["--judge-facts", "--judge-repeats", "0"], ["--resume"],
                                  ["--run-id", "a/b"]])
def test_cli_rejects_bad_judge_repeats_resume_and_run_id(tmp_path, args):
    with pytest.raises(SystemExit):
        cli(tmp_path, *args)


# ---------- --dry-run ----------

def paid_env(monkeypatch, budget_left):
    monkeypatch.setenv("LLM_BACKEND", "openai")
    for key, value in {"PRICE_IN_PER_M": "2.5", "PRICE_OUT_PER_M": "10", "PRICE_EMBED_PER_M": "0.02"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(est_mod, "remaining_budget", lambda: budget_left)
    monkeypatch.setattr(llm_client, "complete", lambda *a, **k: pytest.fail("dry-run не вызывает модель"))
    monkeypatch.setattr(llm_client, "embed", lambda *a, **k: pytest.fail("dry-run не вызывает эмбеддинги"))


def test_dry_run_reports_judge_calls_tokens_cost_separately_and_calls_nothing(monkeypatch, tmp_path, capsys):
    paid_env(monkeypatch, 100.0)
    assert cli(tmp_path, "--judge-facts", "--dry-run") == 0
    one = capsys.readouterr().out
    assert cli(tmp_path, "--judge-facts", "--judge-repeats", "3", "--repeats", "2", "--dry-run") == 0
    many = capsys.readouterr().out
    assert "Оценка ответов системы" in one and "Оценка судьи по фактам (judge-facts-v1" in one
    assert "вызовов судьи по фактам: до 2 " in one            # baseline × 2 вопроса; graph — заглушка
    assert "вызовов судьи по фактам: до 12 " in many           # 2 вопроса × 2 повтора ответов × 3 повтора судьи
    assert "токенов судьи по фактам" in one and "стоимость судьи по фактам: ≈$" in one
    assert "Итого (ответы + судья)" in one and "ВНИМАНИЕ" not in one
    assert not (tmp_path / "out").exists()                     # dry-run ничего не пишет


def test_dry_run_warns_when_estimate_exceeds_remaining_budget(monkeypatch, tmp_path, capsys):
    paid_env(monkeypatch, 0.0001)
    assert cli(tmp_path, "--judge-facts", "--judge-repeats", "3", "--dry-run") == 0
    assert "ВНИМАНИЕ" in capsys.readouterr().out


def test_dry_run_with_both_judges_sums_them(monkeypatch, tmp_path, capsys):
    paid_env(monkeypatch, 100.0)
    assert cli(tmp_path, "--judge", "--judge-facts", "--dry-run") == 0
    out = capsys.readouterr().out
    assert "Оценка судьи (judge-v2" in out and "Оценка судьи по фактам" in out and "вызовов судьи: до 2" in out


def test_paid_backend_needs_confirm_paid_even_with_judge_facts(monkeypatch, tmp_path, capsys):
    paid_env(monkeypatch, 100.0)
    assert cli(tmp_path, "--judge-facts") == 3
    assert "--confirm-paid" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


# ---------- --resume ----------

def crash_on(n):
    """Судья по фактам, который падает на n-м вызове: имитация сбоя посреди прогона."""
    state = {"n": 0}
    inner = facts_fn_factory()

    def fn(*a):
        state["n"] += 1
        if state["n"] == n:
            raise RuntimeError("сбой посреди прогона")
        return inner(*a)
    return fn


def test_crash_midway_keeps_finished_rows_and_resume_continues_without_repeating_them(tmp_path):
    with pytest.raises(RuntimeError):
        cli(tmp_path, "--run-id", "r1", "--judge-facts", facts_fn=crash_on(2))
    out = tmp_path / "out"
    part = out / "eval_r1.partial.jsonl"
    assert part.exists() and not (out / "eval_r1.json").exists()
    header, rows = results_v2.read_partial(part)
    assert header["run_id"] == "r1" and [r["question_id"] for r in rows] == ["q_01"]   # q_01 не потерян

    seen = []
    assert cli(tmp_path, "--run-id", "r1", "--judge-facts", "--resume", facts_fn=facts_fn_factory(calls=seen)) == 0
    assert seen == ["Какой бюджет?"]                           # q_01 заново не оценивался
    d = load_result(tmp_path, "r1")
    assert_contract(d)
    assert [(r["qid"], r["mode"]) for r in d["per_question"]] == [
        ("q_01", "baseline"), ("q_02", "baseline"), ("q_01", "graph"), ("q_02", "graph")]
    assert d["per_question"][0]["metrics"]["fact_f1"] == 1.0
    assert not part.exists()


def test_resume_on_finished_run_and_existing_partial_without_resume(tmp_path, capsys):
    assert cli(tmp_path, "--run-id", "r2") == 0
    assert cli(tmp_path, "--run-id", "r2", "--resume") == 0                    # уже завершён
    assert "уже завершён" in capsys.readouterr().out
    assert cli(tmp_path, "--run-id", "r2") == 2                                # не затираем готовый результат
    assert "уже есть" in capsys.readouterr().err
    with pytest.raises(RuntimeError):
        cli(tmp_path, "--run-id", "r3", "--judge-facts", facts_fn=crash_on(1))
    assert cli(tmp_path, "--run-id", "r3", "--judge-facts", facts_fn=facts_fn_factory()) == 2   # нужен --resume
    assert "--resume" in capsys.readouterr().err


def test_resume_refuses_when_run_parameters_changed(tmp_path, capsys):
    with pytest.raises(RuntimeError):
        cli(tmp_path, "--run-id", "r4", "--judge-facts", facts_fn=crash_on(2))
    assert cli(tmp_path, "--run-id", "r4", "--judge-facts", "--judge-repeats", "2", "--resume",
               facts_fn=facts_fn_factory()) == 2
    err = capsys.readouterr().err
    assert "параметры прогона изменились" in err and "judge_repeats" in err


def test_run_eval_resume_does_not_ask_done_questions_and_skips_building_finished_groups(raw):
    asked = []

    def counting(mode):
        return systems.System(mode, lambda question: (asked.append((mode, question)), Answer(
            answer="ответ", mode=mode, citations=[CITE]))[1])

    qs = [q("q_01"), q("q_02")]
    builds = []

    def build(mode, as_of):
        builds.append(mode)
        return counting(mode)
    raw_dir, gold = raw
    first = ev.run_eval(qs, ["baseline", "graph"], build, gold=gold, texts=MeetingTexts(raw_dir), backend="stub")
    done = {ev.row_key(r): r for r in first}
    asked.clear(), builds.clear()
    again = ev.run_eval(qs, ["baseline", "graph"], build, gold=gold, texts=MeetingTexts(raw_dir), backend="stub",
                        done=done)
    assert asked == [] and builds == [] and [ev.row_key(r) for r in again] == [ev.row_key(r) for r in first]


def test_partial_file_skips_a_torn_last_line(tmp_path):
    path = tmp_path / "p.jsonl"
    results_v2.append_jsonl(path, {"header": {"run_id": "x"}})
    results_v2.append_jsonl(path, {"row": {"question_id": "q_01"}})
    with path.open("a", encoding="utf-8") as f:
        f.write('{"row": {"question_id": "q_0')          # процесс упал посреди записи
    header, rows = results_v2.read_partial(path)
    assert header == {"run_id": "x"} and rows == [{"question_id": "q_01"}]
