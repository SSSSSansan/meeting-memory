import csv
import json
from pathlib import Path

import pytest

import src.answer.answerer as answerer
from scripts.cost_guard import BudgetExceeded
from src import api
from src.eval import run as ev
from src.eval import systems
from src.eval.metrics import MeetingTexts
from src.schemas import Answer, Citation

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_BACKEND", "stub")
    monkeypatch.setenv("EMBED_MODEL", "")
    monkeypatch.setattr(answerer, "REPORT", tmp_path / "answer_report.jsonl")
    monkeypatch.setattr(ev, "RESULTS", tmp_path / "results")  # не писать в настоящий results/
    yield
    api.use_index_dir(api.DEFAULT_INDEX_DIR)


def q(qid="q_01", qtype="T1", split="test", answerable=True, as_of="m_001", **kw):
    data = dict(id=qid, type=qtype, split=split, lang="ru", question=f"вопрос {qid}", as_of=as_of,
                expected_answer="эталон", source_meetings=["m_001"] if answerable else [],
                decision_ids=["ev_m001_01"] if answerable else [], answerable=answerable)
    return ev.Question.model_validate({**data, **kw})


@pytest.fixture
def world(tmp_path):
    segs = [{"start": 65.0, "text": "Договорились сдвинуть релиз Zhol на 4 сентября."}]
    (tmp_path / "m_001.transcript.json").write_text(json.dumps({"segments": segs}, ensure_ascii=False),
                                                    encoding="utf-8")
    gold = {"ev_m001_01": {"id": "ev_m001_01", "meeting": "m_001", "start_s": 65.0, "quote": segs[0]["text"]}}
    return gold, MeetingTexts(tmp_path)


def system_returning(result, mode="baseline"):
    seen = []

    def ask(question):
        seen.append(question)
        if isinstance(result, Exception):
            raise result
        return result
    return systems.System(mode=mode, ask=ask), seen


def run_with(system, questions, world):
    gold, texts = world
    return ev.run_eval(questions, [system.mode], lambda mode, as_of: system, gold=gold, texts=texts, backend="stub")


def good_answer():
    return Answer(answer="Релиз сдвинули.", mode="baseline", latency_ms=12,
                  citations=[Citation(meeting_id="m_001", ts="00:01:05", quote="сдвинуть релиз Zhol на 4 сентября")],
                  usage={"prompt_tokens": 100, "completion_tokens": 20, "cost_usd": 0.01})


def test_metrics_and_only_question_text_reaches_system(world):
    system, seen = system_returning(good_answer())
    (row,) = run_with(system, [q()], world)
    assert seen == ["вопрос q_01"]  # ни эталона, ни id, ни source_meetings
    assert row.status == "ok" and row.correct_meetings is True and row.events_found == 1
    assert row.citation_valid == 1.0 and row.abstained is False and row.judge_score is None
    assert (row.prompt_tokens, row.completion_tokens, row.cost_usd, row.latency_ms) == (100, 20, 0.01, 12)


@pytest.mark.parametrize("result,status", [
    (None, "no_answer"),
    ("просто строка", "invalid_format"),
    ({"answer": 5}, "invalid_format"),
    (RuntimeError("упало"), "error"),
])
def test_bad_replies_do_not_crash_the_run(world, result, status):
    system, _ = system_returning(result)
    rows = run_with(system, [q("q_01"), q("q_02")], world)
    assert [r.status for r in rows] == [status, status]
    assert all(r.correct_meetings is False and r.citations_total == 0 for r in rows)


def test_dict_in_contract_format_is_accepted(world):
    system, _ = system_returning({"answer": "ок", "citations": [], "mode": "baseline"})
    (row,) = run_with(system, [q()], world)
    assert row.status == "ok"


def test_budget_exceeded_stops_the_run(world):
    system, _ = system_returning(BudgetExceeded("лимит"))
    with pytest.raises(BudgetExceeded):
        run_with(system, [q()], world)


def test_stub_system_gets_no_metrics_and_is_marked(world):
    (row,) = run_with(systems.stub_system("graph", "нет графа"), [q()], world)
    assert row.status == "stub" and row.system == "stub" and row.error == "нет графа"
    assert row.correct_meetings is None and row.latency_ms is None


def test_build_failure_becomes_error_rows(world):
    gold, texts = world

    def build(mode, as_of):
        raise RuntimeError("индекс сломан")
    rows = ev.run_eval([q()], ["baseline"], build, gold=gold, texts=texts, backend="stub")
    assert rows[0].status == "error" and "индекс сломан" in rows[0].error


def test_summary_splits_dev_test_and_marks_stub(world):
    gold, texts = world
    good, _ = system_returning(good_answer())

    def build(mode, as_of):
        return good if mode == "baseline" else systems.stub_system("graph", "нет графа")
    questions = [q("q_01", "T1", "test"), q("q_02", "T2", "dev"), q("q_03", "none", "test", answerable=False)]
    rows = ev.run_eval(questions, ["baseline", "graph"], build, gold=gold, texts=texts, backend="stub")
    md = ev.render_summary(rows, ["baseline", "graph"], date="2026-01-01", k=5, depth=2,
                           notes={"graph": "нет графа"}, splits=["dev", "test"])
    test_part, dev_part = md.split("## dev")[0], md.split("## dev")[1]
    assert "## test" in test_part and "| T1 | 1 |" in test_part and "| T2 |" not in test_part
    assert "| T2 | 1 |" in dev_part
    assert "1.00 / stub" in test_part and "graph: stub" in md and "n/a / stub" in test_part


@pytest.mark.parametrize("value", ["0", "6", "-1", "abc"])
def test_repeats_outside_1_to_5_rejected(value, capsys):
    with pytest.raises(SystemExit):
        ev.main(["--repeats", value])
    assert "--repeats" in capsys.readouterr().err


def test_paid_backend_needs_confirmation(monkeypatch, tmp_path, capsys):
    monkeypatch.setenv("LLM_BACKEND", "openai")
    code = ev.main(["--mode", "baseline", "--limit", "1", "--out-dir", str(tmp_path / "out"),
                    "--index-dir", str(tmp_path / "idx")])
    assert code == 3 and not (tmp_path / "out").exists()
    assert "--confirm-paid" in capsys.readouterr().err


def test_end_to_end_on_stub_backend_respects_as_of(tmp_path):
    """Реальный api.ask на заглушке LLM: без сети, индекс только по m_001–m_003, граф — stub с причиной."""
    questions = [{"id": "q_01", "type": "T1", "split": "test", "lang": "ru", "question": "Когда релиз Zhol?",
                  "as_of": "m_003", "expected_answer": "x", "source_meetings": ["m_001"],
                  "decision_ids": ["ev_m001_01"], "answerable": True},
                 {"id": "q_02", "type": "none", "split": "dev", "lang": "ru", "question": "Какой бюджет?",
                  "as_of": "m_003", "expected_answer": "нет", "source_meetings": [], "decision_ids": [],
                  "answerable": False}]
    qfile = tmp_path / "questions.json"
    qfile.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "out"
    # индекс готовится заранее (harness его не строит): m_001–m_005, будущее отрезает as_of_date
    raw = ROOT / "data" / "raw"
    api.use_index_dir(tmp_path / "idx")
    for _ in api.index_batches(systems.load_transcripts(raw, "m_005")):
        pass
    code = ev.main(["--mode", "both", "--split", "all", "--as-of", "m_003", "--questions", str(qfile),
                    "--out-dir", str(out), "--index-dir", str(tmp_path / "idx"), "--raw-dir", str(raw)])
    assert code == 0
    (csv_path,) = out.glob("eval_*.csv")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    assert {(r["mode"], r["status"]) for r in rows} == {("baseline", "ok"), ("graph", "ok")}
    assert all(r["cited_after_as_of"] in ("0", "") for r in rows)
    assert all(r["as_of_date"] == "2026-08-10" for r in rows)     # дата встречи m_003: в вопросе поля нет
    assert all(set(filter(None, r["retrieved_meetings"].split(","))) <= {"m_001", "m_002", "m_003"} for r in rows)
    (md_path,) = out.glob("summary_*.md")
    assert "## test" in md_path.read_text(encoding="utf-8")


def test_graph_without_real_graph_is_stub_with_reason(monkeypatch):
    monkeypatch.setattr(api, "graph_info", lambda: {"is_sample": True, "path": "data/sample/graph.json",
                                                    "meetings": [], "aliases": 0})
    system = systems.build_system("graph", raw_dir=ROOT / "data" / "raw", index_dir=Path("unused"),
                                  as_of_dates=["2026-08-10"])
    assert system.is_stub and "заглушку" in system.note


def test_raw_meetings_up_to_as_of_by_date_inclusive():
    raw = ROOT / "data" / "raw"
    assert systems.raw_meeting_ids(raw, "m_003") == ["m_001", "m_002", "m_003"]
    assert systems.raw_meeting_ids(raw, "2026-08-10") == ["m_001", "m_002", "m_003"]   # день встречи включён
    assert systems.raw_meeting_ids(raw, "2026-08-09") == ["m_001", "m_002"]
