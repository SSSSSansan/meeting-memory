import json
from pathlib import Path

import pytest

import src.answer.answerer as answerer
from src import api
from src.llm_client import LLMResponse

ROOT = Path(__file__).resolve().parent.parent
FIX = ROOT / "tests" / "fixtures"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_BACKEND", "stub")
    monkeypatch.setenv("EMBED_MODEL", "")
    monkeypatch.setattr(answerer, "REPORT", tmp_path / "answer_report.jsonl")
    api.use_index_dir(tmp_path / "index")
    yield
    api.use_index_dir(api.DEFAULT_INDEX_DIR)


def _fixture_pair(mid: str) -> dict[str, bytes]:
    return {f"{mid}.transcript.json": (FIX / "scribal_transcript.json").read_bytes(),
            f"{mid}.meeting.json": (FIX / "scribal_meeting.json").read_bytes()}


def test_pair_scribal_files_reports_missing_and_bad_names():
    files = {**_fixture_pair("m_101"), "m_102.transcript.json": b"{}", "notes.json": b"{}"}
    pairs, errors = api.pair_scribal_files(files)
    assert [p[0] for p in pairs] == ["m_101"]
    assert any("m_102.meeting.json" in e for e in errors) and any("notes.json" in e for e in errors)


def test_load_scribal_uses_adapter():
    (mid, tr, meta), = api.pair_scribal_files(_fixture_pair("m_101"))[0]
    t = api.load_scribal(mid, tr, meta)
    assert t.meeting_id == "m_101" and t.date == "2026-10-21"
    assert [s.speaker_raw for s in t.segments] == ["SPEAKER_00", "SPEAKER_01"]  # пустой сегмент выброшен


def _demo_transcripts():
    pairs, errors = api.pair_scribal_files(api.demo_files())
    assert not errors
    return [api.load_scribal(*p) for p in pairs]


def test_index_batches_progress_persistence_and_idempotency(monkeypatch):
    ts = _demo_transcripts()
    many = [t.model_copy(update={"meeting_id": f"m_{i:03d}"}) for i in range(1, 8) for t in ts[:1]]
    assert list(api.index_batches(many, batch_size=5)) == [(5, 7), (7, 7)]
    assert [m["meeting_id"] for m in api.indexed_meetings()] == [f"m_{i:03d}" for i in range(1, 8)]

    n_chunks = len(api.get_index().chunks)
    list(api.index_batches(many))                 # повтор ничего не дублирует
    assert len(api.get_index().chunks) == n_chunks

    api.use_index_dir(api._index_dir)             # «перезапуск»: индекс читается с диска
    assert len(api.get_index().chunks) == n_chunks


def test_reset_index():
    list(api.index_batches(_demo_transcripts()))
    api.reset_index()
    assert api.indexed_meetings() == []
    api.use_index_dir(api._index_dir)
    assert api.indexed_meetings() == []


def test_ask_uses_api_index(monkeypatch):
    list(api.index_batches(_demo_transcripts()))
    seen = {}

    def fake(messages, **kw):
        seen["user"] = messages[1]["content"]
        return LLMResponse(text=json.dumps({"answer": "ok", "citations": []}))

    monkeypatch.setattr(answerer, "complete", fake)
    a = api.ask("Когда релиз Zhol?")
    assert a.answer == "ok" and a.mode == "baseline"
    assert "=== Встреча m_00" in seen["user"]


def test_ask_graph_mode_reads_graph(monkeypatch):
    seen = {}

    def fake(messages, **kw):
        seen["user"] = messages[1]["content"]
        return LLMResponse(text=json.dumps({"answer": "ok", "citations": []}))

    monkeypatch.setattr(answerer, "complete", fake)
    assert api.ask("Как менялся срок релиза Атласа?", mode="graph").mode == "graph"
    assert "=== Проект Атлас ===" in seen["user"]


def test_graph_info_points_to_sample_until_role_b_builds_graph():
    info = api.graph_info()
    assert info["meetings"] and info["path"].replace("\\", "/").startswith("data/")


def test_route_is_exposed_for_ui():
    r = api.route("Что по Атласу ещё не закрыто?")
    assert (r.mode, r.qtype) == ("graph", "T5")
