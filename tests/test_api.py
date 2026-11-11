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
    # тесты UI и api — на заглушке: граф роли B меняется с каждым батчем встреч
    monkeypatch.setenv("MEETING_GRAPH", "data/sample/graph.json")
    monkeypatch.setattr(answerer, "REPORT", tmp_path / "answer_report.jsonl")
    api.use_index_dir(tmp_path / "index")
    api.use_graph(None)  # выбор графа в UI живёт в модуле — между тестами сбрасываем
    yield
    api.use_graph(None)
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


def test_list_projects():
    assert api.list_projects() == ["Атлас", "Горизонт"]


def test_timeline_is_chronological_with_statuses_and_links():
    t = api.get_timeline("Атлас")
    assert [i["id"] for i in t] == ["d_m001_01", "t_m001_01", "d_m002_01", "t_m002_01"]
    first = t[0]
    assert (first["kind"], first["status"], first["meeting_title"]) == \
        ("Decision", "superseded", "Еженедельный статус — Атлас")
    assert first["links"] == [{"type": "SUPERSEDES", "direction": "in", "other": "d_m002_01",
                               "rationale": first["links"][0]["rationale"]}]
    assert t[1]["owner"] == "Айгерим Нурланова" and t[1]["due"] == "2026-02-10"


@pytest.mark.parametrize("name", ["атлас", "Атласу", " Атлас "])
def test_timeline_accepts_name_variants(name):
    assert [i["id"] for i in api.get_timeline(name)][:1] == ["d_m001_01"]


def test_timeline_includes_items_from_meetings_about_other_project():
    t = api.get_timeline("Горизонт")
    assert {i["id"] for i in t} == {"q_m001_01", "d_m002_02"}
    assert all("Атлас" in i["meeting_title"] for i in t)


def test_unknown_project_gives_empty_results():
    assert api.get_timeline("Марс") == [] and api.get_graph("Марс") == {"nodes": [], "edges": []}


def test_project_graph_for_drawing():
    g = api.get_graph("Атлас")
    ids = {n["id"] for n in g["nodes"]}
    assert {"project:Атлас", "d_m001_01", "d_m002_01", "meeting:m_001", "person:Айгерим Нурланова"} <= ids
    assert "project:Горизонт" not in ids
    types = {e["type"] for e in g["edges"]}
    assert "SUPERSEDES" in types and "PARTICIPATED_IN" not in types
    assert all(e["source"] in ids and e["target"] in ids for e in g["edges"])


def test_whole_graph_skips_participation_edges():
    g = api.get_graph()
    assert any(n["type"] == "Person" for n in g["nodes"])
    assert all(e["type"] != "PARTICIPATED_IN" for e in g["edges"])


# ---------- C7: выбор графа, конфликты и замены, источник цитаты, рисунок ----------

def test_available_graphs_and_switching():
    paths = [g["path"] for g in api.available_graphs()]
    assert api.SAMPLE_GRAPH_PATH in paths
    assert api.demo_default_graph() == api.SAMPLE_GRAPH_PATH          # MEETING_GRAPH задана в фикстуре
    api.use_graph(api.SAMPLE_GRAPH_PATH)
    assert api.graph_info()["is_sample"]
    if api.B_GRAPH_PATH.exists():
        api.use_graph(api.B_GRAPH_PATH)
        assert api.graph_info()["path"] == "data/graph_raw/graph.json"
        assert "Zhol" in api.list_projects()
    api.use_graph(None)
    assert api.graph_info()["is_sample"]


def test_demo_default_graph_prefers_full_graph_without_env(monkeypatch):
    monkeypatch.delenv("MEETING_GRAPH")
    assert api.demo_default_graph() == api.available_graphs()[0]["path"]


def test_changes_on_sample_graph():
    ch = api.get_changes()
    assert ch["conflicts"] == []
    (r,) = ch["replacements"]
    assert (r["new"]["id"], r["old"]["id"]) == ("d_m002_01", "d_m001_01")
    assert r["old"]["status"] == "superseded" and r["rationale"]
    assert api.get_changes("Горизонт")["replacements"] == [] and api.get_changes("Марс")["conflicts"] == []


def test_source_highlights_quote_in_replica():
    files = {k: v for k, v in api.demo_files().items() if k.startswith("m_001.")}
    (mid, tr, meta), = api.pair_scribal_files(files)[0]
    t = api.load_scribal(mid, tr, meta)
    seg = next(s for s in t.segments if len(s.text.split()) >= 8)
    quote = " ".join(seg.text.split()[2:7]).upper()                    # регистр и пробелы не важны
    src = api.get_source("m_001", seg.ts, quote)
    a, b = src["highlight"]
    assert src["ts"] == seg.ts and src["text"][a:b].upper().split() == quote.split()
    assert api.get_source("m_001", "99:99:99", quote)["ts"] == seg.ts   # неверный таймкод — ищем по тексту
    assert api.get_source("m_404", seg.ts, quote) is None


def test_graph_dot_marks_superseded_and_conflicts():
    dot = api.graph_dot("Атлас")
    assert dot.startswith("digraph G {") and dot.rstrip().endswith("}")
    assert "[заменено]" in dot and 'label="заменяет"' in dot
    assert '"d_m002_01" -> "d_m001_01" [color="#5F5E5A" style=bold label="заменяет"];' in dot
    assert '"project:Горизонт"' not in dot
