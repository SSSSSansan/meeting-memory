"""Повторы, mean ± std, срез по языку, секция m_008, задержка без кэша, итоговая таблица. Только заглушки, без сети."""
import json
from pathlib import Path

import pytest

import src.answer.answerer as answerer
import src.llm_client as llm_client
from src import api
from src.eval import estimate as est_mod
from src.eval import run as ev
from src.eval import systems
from src.eval.judge import JudgeResult
from src.eval.metrics import MeetingTexts
from src.schemas import Answer, Citation
from tests.eval_helpers import build_index, graph_as_stub

ROOT = Path(__file__).resolve().parent.parent
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
    monkeypatch.setattr(llm_client, "CACHE", tmp_path / "cache")   # ни кэш, ни журнал вызовов не из настоящего runs/
    monkeypatch.setattr(llm_client, "RUNS", tmp_path / "runs")
    yield
    api.use_index_dir(api.DEFAULT_INDEX_DIR)


def q(qid="q_01", qtype="T1", split="test", answerable=True, as_of="m_001", lang="ru"):
    return ev.Question.model_validate(dict(
        id=qid, type=qtype, split=split, lang=lang, question=f"вопрос {qid}", as_of=as_of,
        expected_answer="эталон", source_meetings=["m_001"] if answerable else [],
        decision_ids=["ev_m001_01"] if answerable else [], answerable=answerable))


@pytest.fixture
def raw(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    segs = [{"start": 65.0, "speaker": "SPEAKER_00", "text": "Договорились сдвинуть релиз Zhol на 4 сентября."}]
    (raw_dir / "m_001.transcript.json").write_text(json.dumps({"segments": segs}, ensure_ascii=False),
                                                   encoding="utf-8")
    gold = {"ev_m001_01": {"id": "ev_m001_01", "meeting": "m_001", "start_s": 65.0, "quote": segs[0]["text"]}}
    return raw_dir, gold


def sequence_system(answers, mode="baseline", cite=True, latency=40):
    """Система отвечает по очереди из списка (по кругу): так задаём разный ответ на каждом повторе."""
    state = {"i": 0, "asked": []}

    def ask(question):
        state["asked"].append(question)
        text = answers[state["i"] % len(answers)]
        state["i"] += 1
        return Answer(answer=text, mode=mode, latency_ms=latency, citations=[CITE] if cite else [],
                      usage={"prompt_tokens": 10, "completion_tokens": 2, "cost_usd": 0.01})
    return systems.System(mode=mode, ask=ask), state


def run(questions, builders, raw, *, repeats=1, judge=None, repeat_cache=None, call_log=None):
    raw_dir, gold = raw
    runner = ev.JudgeRunner(raw_dir=raw_dir, fn=judge) if judge else None
    rows = ev.run_eval(questions, list(builders), lambda mode, as_of: builders[mode], gold=gold,
                       texts=MeetingTexts(raw_dir), backend="stub", judge=runner, repeats=repeats,
                       repeat_cache=repeat_cache, call_log=call_log)
    return rows, runner


def summary(rows, modes=("baseline",), judge=None, splits=("test",)):
    return ev.render_summary(rows, list(modes), date="2026-01-01", k=5, depth=2, notes={}, splits=list(splits),
                             judge=judge)


def table_cells(md, first_col, header_start="| Тип"):
    lines = md.splitlines()
    head = next(i for i, line in enumerate(lines) if line.startswith(header_start))
    cols = [c.strip() for c in lines[head].strip("|").split("|")]
    row = next(line for line in lines[head:] if line.startswith(f"| {first_col} |"))
    return dict(zip(cols, [c.strip() for c in row.strip("|").split("|")]))


# ---------- повторы ----------

def test_repeats_ask_each_question_n_times_and_number_the_rows(raw):
    system, state = sequence_system(["ответ"])
    rows, _ = run([q("q_01"), q("q_02")], {"baseline": system}, raw, repeats=3)
    assert len(rows) == 6 and len(state["asked"]) == 6
    assert sorted((r.question_id, r.repeat) for r in rows) == [(f"q_0{i}", rep) for i in (1, 2) for rep in (1, 2, 3)]
    assert set(state["asked"]) == {"вопрос q_01", "вопрос q_02"}   # в систему по-прежнему только текст вопроса


def test_stub_system_is_not_repeated(raw):
    rows, _ = run([q()], {"graph": systems.stub_system("graph", "нет графа")}, raw, repeats=3)
    assert [(r.status, r.repeat) for r in rows] == [("stub", 1)]


def test_build_failure_gives_error_rows_for_every_repeat(raw):
    raw_dir, gold = raw

    def build(mode, as_of):
        raise RuntimeError("индекс сломан")
    rows = ev.run_eval([q()], ["baseline"], build, gold=gold, texts=MeetingTexts(raw_dir), backend="stub", repeats=2)
    assert [(r.status, r.repeat) for r in rows] == [("error", 1), ("error", 2)]


@pytest.mark.parametrize("value", ["0", "6", "-2", "три"])
def test_repeats_value_is_checked(value, capsys):
    with pytest.raises(SystemExit):
        ev.main(["--repeats", value])
    assert "--repeats" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["1", "5"])
def test_repeats_bounds_are_accepted(value):
    assert ev.repeats_arg(value) == int(value)


# ---------- кэш ----------

def cached_probe(raw):
    """Система, которая ходит в настоящий llm_client.complete (stub backend) с одним и тем же сообщением."""
    def ask(question):
        llm_client.complete([{"role": "user", "content": "один и тот же запрос"}], purpose="answer_baseline")
        return Answer(answer="ответ", mode="baseline", latency_ms=7, citations=[CITE])
    return systems.System(mode="baseline", ask=ask)


def test_without_isolation_repeats_come_from_cache(raw):
    rows, _ = run([q()], {"baseline": cached_probe(raw)}, raw, repeats=3, call_log=ev.CallLog())
    assert [r.from_cache for r in sorted(rows, key=lambda r: r.repeat)] == [False, True, True]


def test_isolated_cache_per_repeat_keeps_repeats_out_of_cache(raw, tmp_path):
    seen = []

    def repeat_cache(rep):
        seen.append((rep, llm_client.CACHE))
        return ev.isolated_cache(tmp_path / "eval_repeats" / f"rep{rep}")
    rows, _ = run([q()], {"baseline": cached_probe(raw)}, raw, repeats=3, repeat_cache=repeat_cache,
                  call_log=ev.CallLog())
    assert [r.from_cache for r in sorted(rows, key=lambda r: r.repeat)] == [False, False, False]
    assert [rep for rep, _ in seen] == [1, 2, 3]
    assert llm_client.CACHE == tmp_path / "cache"                    # после прогона кэш вернулся на место
    assert {p.parent.name for p in (tmp_path / "eval_repeats").rglob("*.json")} == {"rep1", "rep2", "rep3"}


def test_isolated_cache_is_restored_after_error(tmp_path):
    before = llm_client.CACHE
    with pytest.raises(RuntimeError):
        with ev.isolated_cache(tmp_path / "x"):
            assert llm_client.CACHE == tmp_path / "x"
            raise RuntimeError("упало")
    assert llm_client.CACHE == before


# ---------- mean ± std ----------

def test_mean_and_sample_std_over_repeats(raw):
    # повтор 1 — верный ответ (accuracy 1), повтор 2 — неверный (0): mean 0.50, выборочное std = 0.71
    system, _ = sequence_system(["верно", "неверно"])
    judge = lambda *a, **k: JudgeResult(2, 0, "ок") if a[2] == "верно" else JudgeResult(0, 1, "нет")  # noqa: E731
    rows, runner = run([q()], {"baseline": system}, raw, repeats=2, judge=judge)
    info = ev.JudgeInfo("judge-v2", "gpt-answer", runner.calls, runner.reused, runner.failed)
    md = summary(rows, judge=info)
    cells = table_cells(md, "T1")
    assert cells["accuracy"] == "0.50 ± 0.71" and cells["fabrication_rate"] == "0.50 ± 0.71"
    assert cells["prompt_tok"] == "10 ± 0" and cells["cost_usd/вопрос"] == "0.0100 ± 0.0000"
    assert "повторов: 2" in md and "различаются в 1 из 1" in md and "Предупреждение" not in md


def test_std_is_per_repeat_aggregate_not_per_question(raw):
    # два вопроса, на повторе 1 оба верны, на повторе 2 один: accuracy повторов 1.0 и 0.5
    system, _ = sequence_system(["верно", "верно", "верно", "неверно"])
    judge = lambda *a, **k: JudgeResult(2, 0, "ок") if a[2] == "верно" else JudgeResult(0, 0, "нет")  # noqa: E731
    rows, _ = run([q("q_01"), q("q_02")], {"baseline": system}, raw, repeats=2, judge=judge)
    # порядок ответов: повтор 1 — q_01, q_02; повтор 2 — q_01, q_02
    assert table_cells(summary(rows, judge=ev.JudgeInfo("judge-v2", "m", 0, 0, 0)), "T1")["accuracy"] == "0.75 ± 0.35"


def test_identical_answers_give_std_zero_and_a_warning_not_simulated_spread(raw):
    system, _ = sequence_system(["один и тот же ответ"])
    rows, _ = run([q("q_01"), q("q_02")], {"baseline": system}, raw, repeats=3)
    md = summary(rows)
    assert table_cells(md, "T1")["correct_meetings"] == "1.00 ± 0.00"
    assert "Предупреждение: ответы всех повторов совпали дословно (2 пар" in md
    assert "разброс не имитируется" in md and "температура отвечающей системы 0.0" in md


def test_single_repeat_prints_plain_mean_without_std_or_warning(raw):
    system, _ = sequence_system(["ответ"])
    rows, _ = run([q()], {"baseline": system}, raw, repeats=1)
    md = summary(rows)
    assert table_cells(md, "T1")["correct_meetings"] == "1.00" and "±" not in md and "Предупреждение" not in md


def test_judge_scores_each_repeat_but_same_input_is_one_call(raw):
    system, _ = sequence_system(["А", "А", "Б"])
    calls = []

    def judge(*a, **k):
        calls.append(a[2])
        return JudgeResult(2, 0, "ок")
    rows, runner = run([q()], {"baseline": system}, raw, repeats=3, judge=judge)
    assert [r.judge_score for r in sorted(rows, key=lambda r: r.repeat)] == [2, 2, 2]   # вердикт у каждого повтора
    assert sorted(calls) == ["А", "Б"] and runner.calls == 2 and runner.reused == 1   # одинаковый вход — один вызов


def test_cached_repeat_rows_trigger_a_warning(raw):
    rows, _ = run([q()], {"baseline": cached_probe(raw)}, raw, repeats=2, call_log=ev.CallLog())
    assert "часть ответов повторов взята из кэша" in summary(rows)


# ---------- задержка без кэша ----------

def test_latency_counts_only_calls_without_cache(raw, tmp_path):
    log = tmp_path / "calls.jsonl"
    plan = {"вопрос q_01": (False, 100), "вопрос q_02": (True, 9000), "вопрос q_03": (False, 300)}

    def ask(question):
        from_cache, latency = plan[question]
        with log.open("a", encoding="utf-8") as f:
            for purpose, hit in (("answer_baseline", from_cache), ("judge", True)):  # судья в счёт не идёт
                f.write(json.dumps({"kind": "chat", "purpose": purpose, "from_cache": hit}) + "\n")
        return Answer(answer="ответ", mode="baseline", latency_ms=latency, citations=[CITE])
    rows, _ = run([q("q_01"), q("q_02"), q("q_03")], {"baseline": systems.System(mode="baseline", ask=ask)}, raw,
                  call_log=ev.CallLog(log))
    assert [r.from_cache for r in rows] == [False, True, False]
    md = summary(rows)
    assert table_cells(md, "T1")["latency_ms (без кэша)"] == "200"     # (100 + 300) / 2, ответ из кэша не входит
    assert "исключено из-за кэша: 1 из 3 строк" in md


def test_latency_column_says_may_include_cache_when_cache_hit_is_unknown(raw, tmp_path):
    system, _ = sequence_system(["ответ"])
    rows, _ = run([q()], {"baseline": system}, raw, call_log=ev.CallLog(tmp_path / "нет" / "calls.jsonl"))
    assert rows[0].from_cache is False                      # журнал читается, вызовов модели не было

    class Broken(ev.CallLog):
        def mark(self):
            return None                                      # журнал недоступен: позицию не запомнить
    rows, _ = run([q()], {"baseline": system}, raw, call_log=Broken())
    assert rows[0].from_cache is None
    md = summary(rows)
    assert "latency_ms (может включать кэш)" in md and "латентность" not in md
    rows, _ = run([q()], {"baseline": system}, raw)          # без call_log кэш определить нечем
    assert "latency_ms (может включать кэш)" in summary(rows)


def test_call_log_ignores_judge_and_embed_rows_and_survives_garbage(tmp_path):
    log = tmp_path / "calls.jsonl"
    cl = ev.CallLog(log)
    mark = cl.mark()
    assert mark == 0
    log.write_text(json.dumps({"kind": "embed", "purpose": "embed_query", "from_cache": False}) + "\n"
                   + json.dumps({"kind": "chat", "purpose": "judge", "from_cache": True}) + "\n"
                   + json.dumps({"kind": "chat", "purpose": "answer_graph_retry", "from_cache": False}) + "\n",
                   encoding="utf-8")
    assert [c["purpose"] for c in cl.answer_calls_since(mark)] == ["answer_graph_retry"]
    log.write_text("не json\n", encoding="utf-8")
    assert cl.answer_calls_since(0) is None


# ---------- срез по языку ----------

def test_language_slice_next_to_type_slice_and_says_mixed_is_absent(raw):
    system, _ = sequence_system(["ответ"])
    questions = [q("q_01", "T1", lang="ru"), q("q_02", "T2", lang="ru"), q("q_03", "T3", lang="kk")]
    rows, _ = run(questions, {"baseline": system}, raw)
    md = summary(rows)
    assert "### По типу вопроса" in md and "### По языку вопроса" in md
    assert md.index("### По типу вопроса") < md.index("### По языку вопроса")
    assert table_cells(md, "ru", "| Язык")["n"] == "2" and table_cells(md, "kk", "| Язык")["n"] == "1"
    assert "| mixed |" not in md and "Значения mixed в вопросах нет" in md
    assert "меньше 3 вопросов" in md and "В срезе kk," in md


def test_language_slice_shows_mixed_row_when_questions_exist(raw):
    system, _ = sequence_system(["ответ"])
    rows, _ = run([q("q_01"), q("q_02")], {"baseline": system}, raw)
    rows[0].lang = "mixed"      # в схеме вопросов lang только ru | kk, поэтому строку правим после прогона
    md = summary(rows)
    assert table_cells(md, "mixed", "| Язык")["n"] == "1" and "Значения mixed в вопросах нет" not in md


def test_language_slice_is_for_test_only(raw):
    system, _ = sequence_system(["ответ"])
    rows, _ = run([q("q_01", split="test"), q("q_02", split="dev", lang="kk")], {"baseline": system}, raw)
    md = summary(rows, splits=("dev", "test"))
    test_part, dev_part = md.split("## dev")
    assert "### По языку вопроса" in test_part and "| kk |" not in test_part and "По языку" not in dev_part


# ---------- m_008 отдельно ----------

def test_m008_questions_are_a_separate_section_not_mixed_into_the_main_table(raw):
    system, _ = sequence_system(["ответ"])
    questions = [q("q_01", "T1", as_of="m_001"), q("q_16", "T5", split="test", as_of="m_008"),
                 q("q_17", "T5", split="dev", as_of="m_008")]
    rows, _ = run(questions, {"baseline": system}, raw)
    md = summary(rows, splits=("dev", "test"))
    main_part, m008_part = md.split("## as_of m_008")
    assert "| T5 |" not in main_part and "| T1 | 1 |" in main_part
    assert "| T5 | 1 |" in m008_part and "split=test" in m008_part and "split=dev" in m008_part
    assert table_cells(main_part, "all")["n"] == "1"                              # в основной таблице только q_01


def test_questions_file_runs_only_m008_and_main_section_is_empty(tmp_path, capsys):
    items = [{"id": f"q_{i:02d}", "type": "T1", "split": "test", "lang": "ru", "question": f"Когда релиз Zhol {i}?",
              "as_of": "m_008", "expected_answer": "x", "source_meetings": ["m_001"],
              "decision_ids": ["ev_m001_01"], "answerable": True} for i in (1, 2)]
    qfile = tmp_path / "questions_m008.json"
    qfile.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "out"
    build_index(tmp_path / "idx")
    code = ev.main(["--mode", "baseline", "--unlock-test", "--questions-file", str(qfile), "--out-dir", str(out),
                    "--index-dir", str(tmp_path / "idx"), "--raw-dir", str(ROOT / "data" / "raw")])
    assert code == 0
    (md_path,) = out.glob("summary_*.md")
    assert md_path.name.endswith("_questions_m008.md")             # не затирает сводку основного прогона
    md = md_path.read_text(encoding="utf-8")
    assert "## as_of m_008" in md and "_нет вопросов_" in md
    assert not list(out.glob("final_table_*")) and "Итоговая таблица не записана" in capsys.readouterr().out


def test_old_questions_flag_still_works(tmp_path):
    assert ev.out_suffix(ev.QUESTIONS, None) == "" and ev.out_suffix(ev.QUESTIONS, "m_015") == "_m_015"
    assert ev.out_suffix(tmp_path / "questions_m008.json", "m_008") == "_questions_m008_m_008"


# ---------- итоговая таблица ----------

def final(rows, modes=("baseline", "graph"), notes=None, judge=None):
    return ev.render_final_table(rows, list(modes), date="2026-01-01", k=5, depth=2, notes=notes or {}, judge=judge)


def test_final_table_has_only_test_rows_t1_t5_and_none_with_modes_side_by_side(raw):
    system, _ = sequence_system(["ответ"])
    questions = [q("q_01", "T1"), q("q_02", "none", answerable=False), q("q_03", "T2", split="dev"),
                 q("q_04", "T3", as_of="m_008")]
    rows, _ = run(questions, {"baseline": system, "graph": systems.stub_system("graph", "нет графа")}, raw)
    md = final(rows, notes={"graph": "нет графа"})
    head = next(line for line in md.splitlines() if line.startswith("| Тип"))
    cols = [c.strip() for c in head.strip("|").split("|")]
    assert cols[:4] == ["Тип", "n", "accuracy: baseline", "accuracy: graph"]
    for metric in ("fabrication_rate", "correct_meetings", "events_found", "cost_usd/вопрос"):
        assert cols.index(f"{metric}: graph") == cols.index(f"{metric}: baseline") + 1
    assert any(c.startswith("latency_ms") and c.endswith(": baseline") for c in cols)
    body = [line for line in md.splitlines() if line.startswith("| T") or line.startswith("| none")]
    assert [line.split("|")[1].strip() for line in body] == ["T1", "none"]   # dev и m_008 не входят, строки all нет
    assert "| all |" not in md


def test_final_table_marks_stub_graph_and_invents_no_numbers(raw):
    system, _ = sequence_system(["ответ"])
    rows, _ = run([q("q_01", "T1")], {"baseline": system, "graph": systems.stub_system("graph", "нет графа")}, raw)
    md = final(rows, notes={"graph": "нет графа"})
    cells = table_cells(md, "T1")
    graph_cells = [v for k, v in cells.items() if k.endswith(": graph")]
    assert graph_cells and set(graph_cells) == {"stub"}
    assert "graph: stub" in md and "нет графа" in md
    assert all(v != "stub" for k, v in cells.items() if k.endswith(": baseline"))


def test_final_table_says_when_a_mode_was_not_run_and_uses_mean_std(raw):
    system, _ = sequence_system(["верно", "неверно"])
    judge = lambda *a, **k: JudgeResult(2, 0, "ок") if a[2] == "верно" else JudgeResult(0, 0, "нет")  # noqa: E731
    rows, runner = run([q("q_01", "T1")], {"baseline": system}, raw, repeats=2, judge=judge)
    info = ev.JudgeInfo("judge-v2", "gpt-answer", runner.calls, runner.reused, runner.failed)
    md = final(rows, modes=("baseline",), judge=info)
    cells = table_cells(md, "T1")
    assert cells["accuracy: baseline"] == "0.50 ± 0.71" and cells["accuracy: graph"] == "не запускался"
    assert "graph**: в этом прогоне не запускался" in md


def test_final_table_without_judge_has_na_accuracy(raw):
    system, _ = sequence_system(["ответ"])
    rows, _ = run([q()], {"baseline": system}, raw)
    cells = table_cells(final(rows, modes=("baseline",)), "T1")
    assert cells["accuracy: baseline"] == "n/a" and cells["fabrication_rate: baseline"] == "n/a"


def questions_file(tmp_path):
    items = [{"id": "q_01", "type": "T1", "split": "test", "lang": "ru", "question": "Когда релиз Zhol?",
              "as_of": "m_003", "expected_answer": "x", "source_meetings": ["m_001"],
              "decision_ids": ["ev_m001_01"], "answerable": True},
             {"id": "q_02", "type": "none", "split": "test", "lang": "kk", "question": "Какой бюджет?",
              "as_of": "m_003", "expected_answer": "нет", "source_meetings": [], "decision_ids": [],
              "answerable": False}]
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(items, ensure_ascii=False), encoding="utf-8")
    return path


def cli(tmp_path, *extra, judge_fn=None):
    build_index(tmp_path / "idx")
    return ev.main(["--mode", "both", "--split", "test", "--unlock-test", "--as-of", "m_003", "--questions-file",
                    str(questions_file(tmp_path)), "--out-dir", str(tmp_path / "out"),
                    "--index-dir", str(tmp_path / "idx"), "--raw-dir", str(ROOT / "data" / "raw"), *extra],
                   judge_fn=judge_fn)


def test_cli_with_repeats_writes_csv_summary_and_final_table_on_stub_backend(tmp_path):
    assert cli(tmp_path, "--repeats", "3") == 0
    out = tmp_path / "out"
    (csv_path,) = out.glob("eval_*.csv")
    import csv
    rows = list(csv.DictReader(csv_path.open(encoding="utf-8-sig")))
    base = [r for r in rows if r["mode"] == "baseline"]
    assert sorted({r["repeat"] for r in base}) == ["1", "2", "3"] and len(base) == 6
    assert [r["repeat"] for r in rows if r["mode"] == "graph"] == ["1", "1"]            # заглушка — один раз
    assert all(r["from_cache"] in ("True", "False") for r in base)
    assert not any(r["from_cache"] == "True" for r in base if r["repeat"] != "1")      # повторы мимо кэша
    (md_path,) = out.glob("summary_*.md")
    md = md_path.read_text(encoding="utf-8")
    assert "повторов: 3" in md and "± 0.00" in md and "Предупреждение: ответы всех повторов совпали" in md
    (final_path,) = out.glob("final_table_*.md")
    assert final_path.name.endswith("_stub.md")                                         # не затирает платный результат
    text = final_path.read_text(encoding="utf-8")
    assert "stub" in text and "не запускался" not in text and "| none | 1 |" in text
    assert any((tmp_path / "cache" / "eval_repeats").rglob("*.json"))


def test_cli_limit_does_not_write_final_table(tmp_path):
    assert cli(tmp_path, "--limit", "1") == 0
    assert not list((tmp_path / "out").glob("final_table_*"))


# ---------- dry-run ----------

def paid_env(monkeypatch, budget_left):
    monkeypatch.setenv("LLM_BACKEND", "openai")
    for key, value in {"PRICE_IN_PER_M": "2.5", "PRICE_OUT_PER_M": "10", "PRICE_EMBED_PER_M": "0.02"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(est_mod, "remaining_budget", lambda: budget_left)


def test_dry_run_with_repeats_triples_calls_and_cost(monkeypatch, tmp_path, capsys):
    paid_env(monkeypatch, 100.0)
    assert cli(tmp_path, "--judge", "--dry-run") == 0
    one = capsys.readouterr().out
    assert cli(tmp_path, "--judge", "--dry-run", "--repeats", "3") == 0
    three = capsys.readouterr().out
    assert "вызовов модели (chat): 2–4" in one and "вызовов модели (chat): 6–12" in three
    assert "вызовов судьи: до 2" in one and "вызовов судьи: до 6" in three
    assert "повторов: 3" in three and "мимо кэша" in three and "ВНИМАНИЕ" not in three
    assert not (tmp_path / "out").exists()                                              # dry-run ничего не пишет

    def total(text):
        return float(text.split("Итого (ответы + судья)")[1].split("≈$")[1].split()[0])
    assert total(three) > 2.5 * total(one)                                               # индекс считается один раз


def test_dry_run_with_repeats_warns_when_over_budget(monkeypatch, tmp_path, capsys):
    paid_env(monkeypatch, 0.0001)
    assert cli(tmp_path, "--judge", "--dry-run", "--repeats", "3") == 0
    out = capsys.readouterr().out
    assert "ВНИМАНИЕ" in out and "больше остатка бюджета" in out


def test_dry_run_warns_even_on_stub_backend(monkeypatch, tmp_path, capsys):
    """Dry-run — план для платного прогона: предупреждение нужно и когда сейчас стоит заглушка."""
    monkeypatch.setenv("LLM_BACKEND", "stub")
    for key, value in {"PRICE_IN_PER_M": "2.5", "PRICE_OUT_PER_M": "10", "PRICE_EMBED_PER_M": "0.02"}.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(est_mod, "remaining_budget", lambda: 0.0001)
    assert cli(tmp_path, "--judge", "--dry-run", "--repeats", "3") == 0
    assert "ВНИМАНИЕ" in capsys.readouterr().out


def test_paid_run_with_repeats_still_needs_confirmation(monkeypatch, tmp_path, capsys):
    paid_env(monkeypatch, 100.0)
    assert cli(tmp_path, "--repeats", "3") == 3
    assert "--confirm-paid" in capsys.readouterr().err and not (tmp_path / "out").exists()


def test_answer_temperature_constant_matches_the_answerer():
    """Сводка пишет «температура 0.0»; если answerer начнёт звать модель с другой, цифра в заголовке станет ложью."""
    import re
    src = Path(answerer.__file__).read_text(encoding="utf-8")
    assert re.findall(r"temperature=([0-9.]+)", src) == [str(ev.ANSWER_TEMPERATURE)]
