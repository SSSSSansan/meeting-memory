"""C5: вкладка «Метрики» только читает results/eval_*.json роли A (контракт src/eval/results_v2.py)."""
import copy
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import src.answer.answerer as answerer
from src import api

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")
FIXTURE = json.loads(api.EVAL_FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_BACKEND", "stub")
    monkeypatch.setenv("EMBED_MODEL", "")
    monkeypatch.setenv("MEETING_GRAPH", "data/sample/graph.json")
    monkeypatch.setattr(answerer, "REPORT", tmp_path / "answer_report.jsonl")
    monkeypatch.setattr(api, "RESULTS_DIR", tmp_path / "results")
    (tmp_path / "results").mkdir()
    api.use_index_dir(tmp_path / "index")
    api.use_graph(None)
    yield
    api.use_graph(None)
    api.use_index_dir(api.DEFAULT_INDEX_DIR)


def _real_run(run_id: str, timestamp: str) -> dict:
    data = copy.deepcopy(FIXTURE)
    data["run_id"], data["timestamp"] = run_id, timestamp
    data["config"].pop("fixture", None)
    return data


def _write(results: Path, name: str, data) -> None:
    (results / name).write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_without_real_runs_shows_fixture_marked_as_fake():
    runs = api.eval_runs()
    assert len(runs) == 1 and runs[0]["fixture"] and "ФИКСТУРА" in runs[0]["label"]


def test_real_runs_newest_first_broken_files_skipped():
    _write(api.RESULTS_DIR, "eval_20261114_2210.json", _real_run("20261114_2210", "2026-11-14T22:10:00+05:00"))
    _write(api.RESULTS_DIR, "eval_20261115_1100.json", _real_run("20261115_1100", "2026-11-15T11:00:00+05:00"))
    _write(api.RESULTS_DIR, "eval_old.json", {"rows": []})           # старый формат без aggregate
    (api.RESULTS_DIR / "eval_broken.json").write_text("{", encoding="utf-8")
    runs = api.eval_runs()
    assert [r["run_id"] for r in runs] == ["20261115_1100", "20261114_2210"]
    assert not any(r["fixture"] for r in runs)
    with pytest.raises(ValueError):
        api.load_eval(api.RESULTS_DIR / "eval_old.json")


def test_quality_table_reads_aggregate_without_recomputing():
    t = api.quality_table(FIXTURE, "fact_f1")
    assert [r["режим"] for r in t["rows"]] == FIXTURE["config"]["modes"]
    cell = FIXTURE["aggregate"]["by_mode_type"]["graph"]["T2"]["fact_f1"]
    graph = next(r for r in t["rows"] if r["режим"] == "graph")
    assert graph["T2"] == f"{cell['mean']:.2f} [{cell['ci_low']:.2f}; {cell['ci_high']:.2f}] n={cell['n']}"
    assert list(graph)[1:] == ["T1", "T2", "T3", "T4", "T5", "все типы"]
    assert t["measured"] and t["paired"]
    p = FIXTURE["aggregate"]["paired"]["all"]["fact_f1"]
    row = next(r for r in t["paired"] if r["тип"] == "все типы")
    assert row["graph − baseline"].startswith(f"{p['diff_mean']:+.2f}") and row["вывод"] == p["verdict"]


def test_small_groups_are_flagged_and_null_is_not_zero():
    t = api.quality_table(FIXTURE, "fact_f1")
    assert any(v.endswith("*") for r in t["rows"] for v in r.values()) and t["notes"]
    data = copy.deepcopy(FIXTURE)
    for groups in data["aggregate"]["by_mode_type"].values():
        for cells in groups.values():
            cells["fact_f1"] = {"mean": None, "ci_low": None, "ci_high": None, "n": 0, "note": None}
    t = api.quality_table(data, "fact_f1")
    assert not t["measured"] and {v for r in t["rows"] for k, v in r.items() if k != "режим"} == {"—"}


def test_fmt_interval():
    assert api.fmt_interval(None) == "—"
    assert api.fmt_interval({"mean": None}) == "—"
    assert api.fmt_interval({"mean": 0.5, "ci_low": 0.25, "ci_high": 0.75, "n": 4, "note": "n<5"}) == \
        "0.50 [0.25; 0.75] n=4 *"


def test_cost_latency_prefers_harness_aggregate():
    plain = {r["режим"]: r for r in api.cost_latency(FIXTURE)}
    ok = [r for r in FIXTURE["per_question"] if r["mode"] == "graph" and r["status"] == "ok"]
    assert plain["graph"]["cost_usd"] == pytest.approx(sum(r["usage"]["cost_usd"] for r in ok) / len(ok))
    assert plain["graph"]["n"] == len(ok)

    data = copy.deepcopy(FIXTURE)
    data["aggregate"]["cost_latency"] = {"graph": {"cost_usd": {"mean": 0.5, "n": 3}, "latency_ms": {"mean": 900}}}
    given = {r["режим"]: r for r in api.cost_latency(data)}
    assert (given["graph"]["cost_usd"], given["graph"]["latency_ms"], given["graph"]["n"]) == (0.5, 900, 3)


def test_lang_and_abstention_tables():
    langs = api.lang_table(FIXTURE)
    assert {r["язык"] for r in langs} == set(FIXTURE["aggregate"]["by_mode_lang"]["graph"])
    assert len(api.abstention_table(FIXTURE)) == 2


def test_cards_say_whose_file_is_missing_then_show_values():
    cards = api.metric_cards()
    assert all(c["value"] is None for c in cards)
    assert "роль B" in cards[0]["hint"] and "роль A" in cards[2]["hint"]
    _write(api.RESULTS_DIR, api.EXTRACTION_QUALITY, {"hallucination_rate": 0.0, "n_meetings": 14})
    _write(api.RESULTS_DIR, api.INDEXING_COST, {"full_corpus": {"meetings": 14, "cost_usd": 0.035}})
    _write(api.RESULTS_DIR, api.JUDGE_AGREEMENT, {"accuracy": 0.9, "kappa": 0.81, "n": 90})
    assert [c["value"] for c in api.metric_cards()] == ["0.0%", "$0.035", "90%"]


def _metrics_text(at: AppTest) -> str:
    return " ".join([w.value for w in at.warning] + [i.value for i in at.info] + [s.value for s in at.subheader])


def test_tab_renders_fixture_with_warning():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    text = _metrics_text(at)
    assert "Фикстура роли A" in text and "Качество ответа по типам вопросов" in text
    assert "Split dev" in text


def test_tab_renders_real_run_and_switches_metric():
    _write(api.RESULTS_DIR, "eval_20261114_2210.json", _real_run("20261114_2210", "2026-11-14T22:10:00+05:00"))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception and "Фикстура роли A" not in _metrics_text(at)
    radio = next(r for r in at.radio if r.key == "metrics_metric")
    radio.set_value("retrieval_recall").run()
    assert not at.exception
