"""A7: harness зовёт api.ask(question, mode, as_of=<as_of_date вопроса>), k — CONTEXT_K из конфига C, режим hybrid,
проверка симметрии контекста, retrieved_meetings в csv/json, «нужна переиндексация». Всё на заглушке LLM, без сети."""
import csv
import inspect
import json
from pathlib import Path

import pytest

import src.answer.answerer as answerer
from src import api
from src.eval import results_v2, systems
from src.eval import run as ev
from src.eval.metrics import MeetingTexts
from src.retrieval import hybrid as hybrid_mod
from src.retrieval.baseline import CHUNK_FORMAT, CONTEXT_K
from src.schemas import Answer

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_BACKEND", "stub")
    monkeypatch.setenv("EMBED_MODEL", "")
    monkeypatch.setattr(answerer, "REPORT", tmp_path / "answer_report.jsonl")
    monkeypatch.setattr(ev, "RESULTS", tmp_path / "results")
    yield
    api.use_index_dir(api.DEFAULT_INDEX_DIR)


@pytest.fixture
def index_dir(tmp_path):
    """Готовый индекс по m_001–m_005 (его собирает не harness, а UI / api.index_batches)."""
    path = tmp_path / "idx"
    api.use_index_dir(path)
    for _ in api.index_batches(systems.load_transcripts(RAW, "m_005")):
        pass
    api.use_index_dir(api.DEFAULT_INDEX_DIR)
    return path


def q(qid="q_01", as_of="m_015", as_of_date="2026-08-05", **kw):
    data = dict(id=qid, type="T1", split="dev", lang="ru", question=f"Когда релиз Zhol? {qid}", as_of=as_of,
                as_of_date=as_of_date, expected_answer="x", source_meetings=["m_001"], decision_ids=["ev_m001_01"],
                answerable=True)
    return ev.Question.model_validate({**data, **kw})


def row(mode="baseline", **kw):
    base = dict(question_id="q_01", type="T1", split="dev", lang="ru", as_of="m_015", mode=mode, system="real",
                backend="stub", status="ok", answer="Релиз 4 сентября.")
    return ev.Row(**{**base, **kw})


# ---------- будущее не попадает в контекст ----------

@pytest.mark.parametrize("mode", ["baseline", "graph", "hybrid"])
def test_future_meeting_is_not_in_context(index_dir, mode):
    system = systems.build_system(mode, raw_dir=RAW, index_dir=index_dir, as_of_dates=["2026-08-05"])
    assert not system.is_stub and system.uses_as_of
    ans = system.ask("Что решили по срокам релиза Zhol и по Qamqor?", as_of="2026-08-05")
    assert ans.retrieved_meetings is not None
    assert set(ans.retrieved_meetings) <= {"m_001", "m_002"}       # m_003 (2026-08-10) и позже — будущее
    assert {c.meeting_id for c in ans.citations} <= {"m_001", "m_002"}


def test_meeting_on_as_of_date_is_included(index_dir):
    system = systems.build_system("baseline", raw_dir=RAW, index_dir=index_dir, as_of_dates=["2026-08-10"])
    ans = system.ask("Что обсуждали со Squad на встрече с клиентом?", as_of="2026-08-10")
    assert "m_003" in ans.retrieved_meetings and "m_004" not in ans.retrieved_meetings


@pytest.mark.parametrize("bad", [None, "", "m_002"])
def test_not_iso_as_of_is_an_error_not_a_silent_leak(index_dir, bad):
    # «2026-…» <= «m_002» как строка — всегда да: фильтр будущего отключился бы молча
    system = systems.build_system("baseline", raw_dir=RAW, index_dir=index_dir, as_of_dates=["2026-08-05"])
    with pytest.raises(ValueError, match="as_of"):
        system.ask("Когда релиз?", as_of=bad)


# ---------- as_of_date берётся из вопроса ----------

def test_as_of_date_comes_from_question():
    seen = []

    def ask(question, as_of=None):
        seen.append((question, as_of))
        return Answer(answer="ок", mode="baseline", retrieved_meetings=["m_001"])
    system = systems.System("baseline", ask, uses_as_of=True)
    (r,) = ev.run_eval([q(as_of="m_015", as_of_date="2026-08-05")], ["baseline"], lambda m, a: system, gold={},
                       texts=MeetingTexts(RAW), backend="stub")
    assert seen == [("Когда релиз Zhol? q_01", "2026-08-05")]      # не дата m_015 и не сам meeting_id
    assert r.as_of_date == "2026-08-05"


def test_as_of_date_fallback_for_historical_questions_without_field():
    assert systems.question_as_of_date(q(as_of="m_008", as_of_date=None), RAW) == "2026-09-02"
    assert systems.question_as_of_date(q(as_of="m_008", as_of_date="2026-09-01"), RAW) == "2026-09-01"
    with pytest.raises(ValueError):
        systems.question_as_of_date(q(as_of="m_014", as_of_date=None), RAW)   # m_014 нет в data/raw


def test_dev_questions_have_as_of_date():
    qs = ev.select_questions(ROOT / "data" / "gold" / "questions_dev.json", "dev", None, None)
    assert qs and all(systems.question_as_of_date(x, RAW) == x.as_of_date for x in qs)


# ---------- CONTEXT_K из конфига C ----------

def test_harness_does_not_pass_its_own_k(index_dir, monkeypatch):
    calls = []
    assert inspect.signature(api.ask).parameters["k"].default == CONTEXT_K == systems.CONTEXT_K
    monkeypatch.setattr(api, "ask", lambda *a, **kw: (calls.append(kw), Answer(answer="ок", mode="baseline"))[1])
    system = systems.build_system("baseline", raw_dir=RAW, index_dir=index_dir, as_of_dates=["2026-08-05"])
    system.ask("Когда релиз?", as_of="2026-08-05")
    assert calls and "k" not in calls[0] and calls[0]["as_of"] == "2026-08-05"


def test_cli_has_no_k_and_header_uses_context_k(capsys):
    with pytest.raises(SystemExit):
        ev.main(["--k", "5"])
    assert "--k" in capsys.readouterr().err
    ns = type("A", (), dict(mode="both", split="dev", as_of=None, limit=None, questions=Path("q.json"), repeats=1,
                            judge=False, judge_facts=False, judge_repeats=1, depth=2))()
    assert ev.run_header(ns, "r", [])["k"] == CONTEXT_K


# ---------- hybrid ----------

def test_hybrid_in_modes_and_marked_while_graph_part_is_stub(index_dir, monkeypatch):
    assert "hybrid" in systems.MODES and ev.MODES["hybrid"] == ["hybrid"] and "hybrid" in ev.MODES["all"]
    assert ev.MODES["both"] == ["baseline", "graph"]
    system = systems.build_system("hybrid", raw_dir=RAW, index_dir=index_dir, as_of_dates=["2026-08-05"])
    assert system.flag == systems.HYBRID_GRAPH_STUB == "graph part: stub"
    # когда B подставит свой обход, пометка уходит сама
    monkeypatch.setattr(hybrid_mod, "graph_part", lambda question, depth, G, as_of: {"items": []})
    assert systems.build_system("hybrid", raw_dir=RAW, index_dir=index_dir, as_of_dates=["2026-08-05"]).flag == ""


def test_hybrid_flag_in_summary_json_and_not_in_final_table():
    rows = [row("baseline", split="test"), row("graph", split="test"),
            row("hybrid", split="test", mode_flag=systems.HYBRID_GRAPH_STUB)]
    modes = ev.MODES["all"]
    md = ev.render_summary(rows, modes, date="d", k=CONTEXT_K, depth=2, notes={}, splits=["test"])
    assert "hybrid [graph part: stub]" in md and "**hybrid: graph part: stub**" in md
    final = ev.render_final_table(rows, modes, date="d", k=CONTEXT_K, depth=2, notes={})
    assert ": hybrid" not in final and "hybrid**: в таблицу не входит (graph part: stub" in final
    cfg = results_v2.build_config(modes=modes, split="test", n_questions=1, as_of="m_015", judge_repeats=1,
                                  repeats=1, backend="stub", commit="x", judge_facts=False)
    result = ev.annotate_result(results_v2.build_result(run_id="r", timestamp="t", config=cfg, rows=rows), rows, modes)
    assert result["config"]["mode_flags"] == {"hybrid": "graph part: stub"}
    assert result["config"]["context_k"] == CONTEXT_K and result["config"]["final_table_modes"] == ["baseline", "graph"]
    assert [r["mode_flag"] for r in result["per_question"]] == ["", "", "graph part: stub"]


# ---------- симметрия контекста ----------

def test_context_elements_are_counted_per_mode(index_dir):
    for mode in ("baseline", "graph", "hybrid"):
        system = systems.build_system(mode, raw_dir=RAW, index_dir=index_dir, as_of_dates=["2026-08-05"])
        (r,) = ev.run_eval([q()], [mode], lambda m, a: system, gold={}, texts=MeetingTexts(RAW), backend="stub")
        assert r.context_elements is not None
        if mode == "baseline":
            assert r.context_graph_items is None and 0 < r.context_chunks <= CONTEXT_K
        if mode == "graph":
            assert r.context_chunks is None and r.context_graph_items is not None
    # обёртки сняты: answerer работает с исходными функциями
    assert answerer.format_context.__module__ == "src.answer.answerer"
    assert answerer.retrieve_hybrid.__module__ == "src.retrieval.hybrid"


def test_symmetry_warning_when_context_exceeds_context_k():
    rows = [row("baseline", context_elements=CONTEXT_K), row("baseline", context_elements=CONTEXT_K - 2),
            row("hybrid", context_elements=CONTEXT_K + 5), row("hybrid", context_elements=CONTEXT_K)]
    stats = ev.symmetry_stats(rows, ["baseline", "hybrid"])
    assert stats["baseline"] == {"max": CONTEXT_K, "mean": CONTEXT_K - 1, "n": 2, "over_k": 0, "context_k": CONTEXT_K}
    assert stats["hybrid"]["max"] == CONTEXT_K + 5 and stats["hybrid"]["over_k"] == 1
    text = "\n".join(ev.symmetry_lines(rows, ["baseline", "hybrid"]))
    assert "Предупреждение" in text and f"hybrid (max {CONTEXT_K + 5}" in text and "baseline (max" not in text
    ok = "\n".join(ev.symmetry_lines(rows[:2], ["baseline"]))
    assert "Предупреждение" not in ok and f"| baseline | {CONTEXT_K} |" in ok


# ---------- retrieved_meetings ----------

def test_retrieved_meetings_warning_when_empty_with_answer():
    rows = [row(retrieved_meetings=[], answer="Релиз 4 сентября."), row(retrieved_meetings=["m_001"]),
            row(retrieved_meetings=[], answer="", status="ok"), row(retrieved_meetings=None)]
    text = "\n".join(ev.retrieved_lines(rows))
    assert "retrieved_meetings пуст при непустом ответе — 1 из 4" in text
    assert "не отдал retrieved_meetings в 1" in text
    assert ev.retrieved_lines([row(retrieved_meetings=["m_001"])]) == []
    md = ev.render_summary(rows, ["baseline"], date="d", k=CONTEXT_K, depth=2, notes={}, splits=["dev"])
    assert "retrieved_meetings пуст при непустом ответе" in md


def test_retrieved_meetings_written_to_csv_and_json(tmp_path):
    answers = iter([Answer(answer="ок", mode="baseline", retrieved_meetings=["m_002", "m_001"]),
                    Answer(answer="ок", mode="baseline", retrieved_meetings=[])])
    system = systems.System("baseline", lambda question, as_of=None: next(answers), uses_as_of=True)
    rows = ev.run_eval([q("q_01"), q("q_02")], ["baseline"], lambda m, a: system, gold={}, texts=MeetingTexts(RAW),
                       backend="stub")
    ev.write_csv(tmp_path / "e.csv", rows)
    got = list(csv.DictReader((tmp_path / "e.csv").open(encoding="utf-8-sig")))
    assert [(r["retrieved_meetings"], r["retrieved_count"]) for r in got] == [("m_002,m_001", "2"), ("", "0")]
    cfg = results_v2.build_config(modes=["baseline"], split="dev", n_questions=2, as_of="m_015", judge_repeats=1,
                                  repeats=1, backend="stub", commit="x", judge_facts=False)
    result = ev.annotate_result(results_v2.build_result(run_id="r", timestamp="t", config=cfg, rows=rows), rows,
                                ["baseline"])
    assert [r["retrieved_meetings"] for r in result["per_question"]] == [["m_002", "m_001"], []]
    assert [r["as_of_date"] for r in result["per_question"]] == ["2026-08-05", "2026-08-05"]


# ---------- индекс: «нужна переиндексация» ----------

def test_missing_index_needs_reindex_and_is_not_built(tmp_path):
    missing = tmp_path / "nope"
    system = systems.build_system("baseline", raw_dir=RAW, index_dir=missing, as_of_dates=["2026-08-05"])
    assert system.is_stub and systems.REINDEX in system.note
    assert not missing.exists()                      # harness ничего не пересобрал
    hybrid = systems.build_system("hybrid", raw_dir=RAW, index_dir=missing, as_of_dates=["2026-08-05"])
    assert hybrid.is_stub and systems.REINDEX in hybrid.note


def test_old_chunk_format_needs_reindex(index_dir):
    path = index_dir / "chunks.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    before = path.read_text(encoding="utf-8")
    data["meta"]["chunk_format"] = "header-v1"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    problem = systems.index_problem(index_dir, RAW, ["2026-08-05"])
    assert problem.startswith(systems.REINDEX) and "header-v1" in problem and CHUNK_FORMAT in problem
    del data["meta"]["chunk_format"]                 # индекс до C1 — поля нет вовсе
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert "старого формата" in systems.index_problem(index_dir, RAW, ["2026-08-05"])
    assert before != path.read_text(encoding="utf-8")  # (проверка, что тест правда подменил файл)


def test_index_without_needed_meetings_needs_reindex(index_dir):
    assert systems.index_problem(index_dir, RAW, ["2026-08-19"]) == ""          # m_005 — последняя в индексе
    problem = systems.index_problem(index_dir, RAW, ["2026-08-24"])
    assert systems.REINDEX in problem and "m_006" in problem


def test_dry_run_prints_reindex_and_hybrid_stub(tmp_path, capsys):
    qfile = tmp_path / "q.json"
    qfile.write_text(json.dumps([q().model_dump()], ensure_ascii=False), encoding="utf-8")
    code = ev.main(["--mode", "all", "--split", "dev", "--questions", str(qfile), "--index-dir", str(tmp_path / "x"),
                    "--dry-run", "--judge-facts"])
    out = capsys.readouterr().out
    assert code == 0 and "нужна переиндексация" in out and "hybrid: graph part: stub" in out
    assert f"CONTEXT_K={CONTEXT_K}" in out and "as_of_date: 2026-08-05" in out
    assert not (tmp_path / "x").exists()
