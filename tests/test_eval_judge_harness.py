"""Судья в harness (--judge): подсчёт метрик, одинаковый вызов для режимов, кэш, оценка стоимости. Только заглушки."""
import csv
import json
from pathlib import Path

import pytest

import src.answer.answerer as answerer
import src.llm_client as llm_client
from src import api
from src.eval import estimate as est_mod
from src.eval import judge as judge_mod
from src.eval import run as ev
from src.eval import systems
from src.eval.judge import JudgeFormatError, JudgeResult
from src.eval.metrics import MeetingTexts
from src.schemas import Answer, Citation
from tests.eval_helpers import build_index, graph_as_stub

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    graph_as_stub(monkeypatch)
    monkeypatch.setenv("LLM_BACKEND", "stub")
    monkeypatch.setenv("EMBED_MODEL", "")
    monkeypatch.setenv("LLM_MODEL", "gpt-answer")
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
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
def raw(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    segs = [{"start": 65.0, "speaker": "SPEAKER_00", "text": "Договорились сдвинуть релиз Zhol на 4 сентября."}]
    (raw_dir / "m_001.transcript.json").write_text(json.dumps({"segments": segs}, ensure_ascii=False),
                                                   encoding="utf-8")
    gold = {"ev_m001_01": {"id": "ev_m001_01", "meeting": "m_001", "start_s": 65.0, "quote": segs[0]["text"]}}
    return raw_dir, gold


def answering(text="Релиз сдвинули.", mode="baseline", cite=True):
    cits = [Citation(meeting_id="m_001", ts="00:01:05", quote="сдвинуть релиз Zhol на 4 сентября")] if cite else []
    ans = Answer(answer=text, mode=mode, latency_ms=5, citations=cits,
                 usage={"prompt_tokens": 10, "completion_tokens": 2, "cost_usd": 0.0})
    return systems.System(mode=mode, ask=lambda question: ans)


class SpyJudge:
    """Подставной судья: записывает, что ему передали, и отвечает по таблице {текст ответа: (score, fab)}."""

    def __init__(self, verdicts=None, default=(2, 0)):
        self.calls: list[tuple[tuple, dict]] = []
        self.verdicts, self.default = verdicts or {}, default

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        score, fab = self.verdicts.get(args[2], self.default)
        return JudgeResult(score, fab, f"оценка {score}")


def run(questions, builders, raw, judge=None):
    raw_dir, gold = raw
    runner = ev.JudgeRunner(raw_dir=raw_dir, fn=judge) if judge else None
    rows = ev.run_eval(questions, list(builders), lambda mode, as_of: builders[mode], gold=gold,
                       texts=MeetingTexts(raw_dir), backend="stub", judge=runner)
    return rows, runner


# ---------- метрики ----------

def test_judge_off_by_default_leaves_na(raw):
    rows, _ = run([q()], {"baseline": answering()}, raw)
    assert rows[0].judge_score is None and rows[0].judge_fabrication is None and rows[0].judge_reason == ""
    md = ev.render_summary(rows, ["baseline"], date="2026-01-01", k=5, depth=2, notes={}, splits=["test"])
    assert "судья: выключен" in md and "| n/a | n/a |" in md and "fabrication_rate" in md


def test_accuracy_and_fabrication_rate_by_type_and_overall(raw):
    spy = SpyJudge({"a": (2, 0), "b": (1, 0), "c": (0, 1), "отказ": (2, 0)})
    sys_by_q = {"вопрос q_01": "a", "вопрос q_02": "b", "вопрос q_03": "c", "вопрос q_04": "отказ"}
    system = systems.System(mode="baseline", ask=lambda text: Answer(answer=sys_by_q[text], mode="baseline"))
    questions = [q("q_01", "T1"), q("q_02", "T1"), q("q_03", "T2"), q("q_04", "none", answerable=False)]
    rows, runner = run(questions, {"baseline": system}, raw, spy)
    assert [r.judge_score for r in rows] == [2, 1, 0, 2] and [r.judge_fabrication for r in rows] == [0, 0, 1, 0]
    info = ev.JudgeInfo("judge-v2", "gpt-answer", runner.calls, runner.reused, runner.failed)
    md = ev.render_summary(rows, ["baseline"], date="2026-01-01", k=5, depth=2, notes={}, splits=["test"], judge=info)
    head = next(line for line in md.splitlines() if line.startswith("| Тип"))
    cols = [c.strip() for c in head.strip("|").split("|")]
    acc, fab = cols.index("accuracy"), cols.index("fabrication_rate")

    def cell(t, i):
        return [c.strip() for c in next(line for line in md.splitlines() if line.startswith(f"| {t} |")).strip("|")
                .split("|")][i]
    assert (cell("T1", acc), cell("T1", fab)) == ("0.75", "0.00")      # (1 + 0.5) / 2
    assert (cell("T2", acc), cell("T2", fab)) == ("0.00", "1.00")
    assert (cell("none", acc), cell("none", fab)) == ("1.00", "0.00")  # честный отказ = 2 = accuracy 1
    assert (cell("all", acc), cell("all", fab)) == ("0.62", "0.25")    # (1 + .5 + 0 + 1) / 4 = 0.625
    assert "судья: judge-v2" in md and "gpt-answer" in md and "та же модель" in md


def test_judge_model_line_says_when_models_differ(raw, monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "gpt-judge")
    rows, runner = run([q()], {"baseline": answering()}, raw, SpyJudge())
    info = ev.JudgeInfo("judge-v2", judge_mod.judge_model(), runner.calls, runner.reused, runner.failed)
    md = ev.render_summary(rows, ["baseline"], date="d", k=5, depth=2, notes={}, splits=["test"], judge=info)
    assert "модель gpt-judge;" in md and "та же модель" not in md


# ---------- что получает судья ----------

def test_judge_gets_same_call_for_both_modes_and_never_mode_or_type(raw):
    spy = SpyJudge()
    builders = {"baseline": answering("Ответ baseline", "baseline"), "graph": answering("Ответ graph", "graph")}
    run([q("q_01", "T4", "dev")], builders, raw, spy)
    assert len(spy.calls) == 2
    (a1, k1), (a2, k2) = spy.calls
    assert set(k1) == set(k2) == {"with_transcripts", "meetings"} and k1["with_transcripts"] is True
    assert len(a1) == len(a2) == 4
    seen = json.dumps([a1, a2], ensure_ascii=False)
    assert "T4" not in seen and "dev" not in seen and "mode" not in seen
    assert [a1[0], a1[1], a1[3]] == [a2[0], a2[1], a2[3]] == ["вопрос q_01", "эталон", True]
    assert [d.meeting_id for d in k1["meetings"]] == ["m_001"]   # source ∪ cited, чистый транскрипт


def test_identical_input_is_judged_once(raw):
    spy = SpyJudge()
    same = {"baseline": answering("Одинаковый ответ", "baseline"), "graph": answering("Одинаковый ответ", "graph")}
    rows, runner = run([q()], same, raw, spy)
    assert len(spy.calls) == 1 and (runner.calls, runner.reused) == (1, 1)
    assert [r.judge_score for r in rows] == [2, 2]


def test_judge_verdict_is_cached_by_llm_client_on_disk(monkeypatch, tmp_path):
    """Повторный запуск с тем же входом не идёт в модель: кэш llm_client по хэшу сообщений."""
    monkeypatch.setattr(llm_client, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(llm_client, "RUNS", tmp_path / "runs")
    seen = []

    def fake_stub(messages, json_schema):
        seen.append(messages)
        return llm_client.LLMResponse(text='{"reason": "ок", "score": 2, "fabrication": 0}')
    monkeypatch.setattr(llm_client, "_complete_stub", fake_stub)
    args = ("вопрос", "эталон", "ответ")
    assert judge_mod.judge(*args, True, with_transcripts=True, meetings=[]).score == 2
    assert judge_mod.judge(*args, True, with_transcripts=True, meetings=[]).score == 2
    assert len(seen) == 1
    judge_mod.judge(*args, True, with_transcripts=True, meetings=[judge_mod.MeetingDoc("m_9", "d", "t", "текст")])
    assert len(seen) == 2  # другой вход — другой вызов


def test_input_hash_depends_on_answer_question_meetings_and_model(monkeypatch):
    doc = judge_mod.MeetingDoc("m_001", "2026-08-03", "Zhol", "SPEAKER_00: текст")
    base = judge_mod.input_hash("в", "э", "о", True, meetings=[doc])
    assert base == judge_mod.input_hash("в", "э", "о", True, meetings=[doc])
    assert base != judge_mod.input_hash("в", "э", "о2", True, meetings=[doc])
    assert base != judge_mod.input_hash("в", "э", "о", True, meetings=[])
    assert base != judge_mod.input_hash("в", "э", "о", True)
    monkeypatch.setenv("JUDGE_MODEL", "другая-модель")
    assert base != judge_mod.input_hash("в", "э", "о", True, meetings=[doc])


# ---------- сбои ----------

def test_rows_without_answer_score_zero_without_calling_judge(raw):
    spy = SpyJudge()

    def boom(question):
        raise RuntimeError("упало")
    rows, _ = run([q()], {"baseline": systems.System(mode="baseline", ask=boom)}, raw, spy)
    assert spy.calls == [] and (rows[0].judge_score, rows[0].judge_fabrication) == (0, 0)
    assert "нет ответа системы (error)" in rows[0].judge_reason


def test_stub_system_is_not_judged(raw):
    spy = SpyJudge()
    rows, _ = run([q()], {"graph": systems.stub_system("graph", "нет графа")}, raw, spy)
    assert spy.calls == [] and rows[0].judge_score is None


def test_format_error_gives_no_verdict_and_is_counted(raw):
    def broken(*args, **kwargs):
        raise JudgeFormatError("не JSON", "мусор")
    rows, runner = run([q()], {"baseline": answering()}, raw, broken)
    assert rows[0].judge_score is None and rows[0].judge_reason.startswith("ОШИБКА ФОРМАТА")
    assert runner.failed == 1
    info = ev.JudgeInfo("judge-v2", "m", runner.calls, runner.reused, runner.failed)
    md = ev.render_summary(rows, ["baseline"], date="d", k=5, depth=2, notes={}, splits=["test"], judge=info)
    assert "без вердикта (ошибка формата) 1" in md and "| — | — |" in md


# ---------- CSV и CLI ----------

def questions_file(tmp_path):
    items = [{"id": "q_01", "type": "T1", "split": "test", "lang": "ru", "question": "Когда релиз Zhol?",
              "as_of": "m_003", "expected_answer": "x", "source_meetings": ["m_001"],
              "decision_ids": ["ev_m001_01"], "answerable": True},
             {"id": "q_02", "type": "none", "split": "test", "lang": "ru", "question": "Какой бюджет?",
              "as_of": "m_003", "expected_answer": "нет", "source_meetings": [], "decision_ids": [],
              "answerable": False}]
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return path


def cli(tmp_path, *extra, judge_fn=None):
    build_index(tmp_path / "idx")
    return ev.main(["--mode", "both", "--split", "test", "--unlock-test", "--as-of", "m_003", "--questions",
                    str(questions_file(tmp_path)), "--out-dir", str(tmp_path / "out"),
                    "--index-dir", str(tmp_path / "idx"), "--raw-dir", str(ROOT / "data" / "raw"), *extra],
                   judge_fn=judge_fn)


def test_cli_with_judge_writes_columns_and_summary(tmp_path):
    spy = SpyJudge(default=(1, 1))
    assert cli(tmp_path, "--judge", judge_fn=spy) == 0
    (csv_path,) = (tmp_path / "out").glob("eval_*.csv")
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    assert {"judge_score", "judge_fabrication", "judge_reason"} <= set(rows[0]) and "accuracy" not in rows[0]
    by_mode = {(r["mode"], r["question_id"]): r for r in rows}
    assert by_mode[("baseline", "q_01")]["judge_score"] == "1" and by_mode[("baseline", "q_01")]["judge_reason"]
    assert by_mode[("graph", "q_01")]["judge_score"] == ""            # граф для m_003 — stub, не оценивается
    assert len(spy.calls) == 2                                         # только baseline: 2 вопроса
    (md_path,) = (tmp_path / "out").glob("summary_*.md")
    md = md_path.read_text(encoding="utf-8")
    assert "судья: judge-v2" in md and "0.50 / stub" in md and "1.00 / stub" in md


def test_cli_judge_on_stub_backend_is_refused_without_calls(tmp_path, capsys):
    assert cli(tmp_path, "--judge") == 2
    assert "--dry-run" in capsys.readouterr().err and not (tmp_path / "out").exists()


def test_cli_without_judge_does_not_call_judge(tmp_path):
    spy = SpyJudge()
    assert cli(tmp_path, judge_fn=spy) == 0 and spy.calls == []


def paid_env(monkeypatch, tmp_path, budget_left):
    monkeypatch.setenv("LLM_BACKEND", "openai")
    for key, value in {"PRICE_IN_PER_M": "2.5", "PRICE_OUT_PER_M": "10", "PRICE_EMBED_PER_M": "0.02"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(est_mod, "remaining_budget", lambda: budget_left)


def test_dry_run_prints_judge_calls_and_total_cost(monkeypatch, tmp_path, capsys):
    paid_env(monkeypatch, tmp_path, budget_left=100.0)
    assert cli(tmp_path, "--judge", "--dry-run") == 0
    out = capsys.readouterr().out
    assert "вызовов судьи: до 2" in out and "стоимость судьи" in out
    assert "Итого (ответы + судья)" in out and "≈$" in out and "ВНИМАНИЕ" not in out
    assert not (tmp_path / "out").exists()


def test_warns_when_estimate_exceeds_remaining_budget(monkeypatch, tmp_path, capsys):
    paid_env(monkeypatch, tmp_path, budget_left=0.0001)
    assert cli(tmp_path, "--judge", "--dry-run") == 0
    out = capsys.readouterr().out
    assert "ВНИМАНИЕ" in out and "больше остатка бюджета" in out


def test_paid_judge_run_needs_confirmation_and_calls_nothing(monkeypatch, tmp_path, capsys):
    paid_env(monkeypatch, tmp_path, budget_left=100.0)
    spy = SpyJudge()
    assert cli(tmp_path, "--judge", judge_fn=spy) == 3
    assert "--confirm-paid" in capsys.readouterr().err and spy.calls == [] and not (tmp_path / "out").exists()


def test_judge_estimate_counts_only_real_answers():
    questions = [q("q_01", as_of="m_003"), q("q_02", "none", answerable=False, as_of="m_003")]
    raw_dir = ROOT / "data" / "raw"
    both = est_mod.estimate_judge(questions, ["baseline", "graph"], raw_dir=raw_dir)
    base = est_mod.estimate_judge(questions, ["baseline"], raw_dir=raw_dir)
    assert base.calls == 2 and both.calls == 2        # граф для m_003 — заглушка, судья на него не тратится
    assert base.prompt_tokens > 3 * 500 and base.completion_tokens == 2 * est_mod.JUDGE_COMPLETION_TOKENS
