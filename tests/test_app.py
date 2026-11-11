import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import src.answer.answerer as answerer
from src import api
from src.llm_client import LLMResponse

APP = str(Path(__file__).resolve().parent.parent / "app" / "streamlit_app.py")


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


def _button(at: AppTest, label: str):
    return next(b for b in at.button if b.label == label)


def test_app_opens_with_empty_index():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("stub" in w.value for w in at.warning)
    assert any("В индексе: 0" in s.value for s in at.sidebar.subheader)


def test_demo_indexing_then_question(monkeypatch):
    cit = {"meeting_id": "m_001", "ts": "00:00:00", "quote": "заведомо выдуманная цитата"}
    monkeypatch.setattr(answerer, "complete", lambda messages, **kw: LLMResponse(
        text=json.dumps({"answer": "Ответ модели", "citations": [cit]}, ensure_ascii=False),
        prompt_tokens=10, completion_tokens=5))

    at = AppTest.from_file(APP, default_timeout=60).run()
    _button(at, "Демо-встречи").click().run()
    assert not at.exception
    demo_meetings = len(api.pair_scribal_files(api.demo_files())[0])   # роль A добавляет встречи в data/raw/
    assert demo_meetings >= 2
    assert any(f"В индексе: {demo_meetings}" in s.value for s in at.sidebar.subheader)

    at.text_input[0].input("Когда релиз Zhol?").run()
    _button(at, "Спросить").click().run()
    assert not at.exception
    shown = " ".join(m.value for m in at.markdown)
    assert "Ответ модели" in shown and "Удалено неподтверждённых цитат: 1" in shown
    assert any("Токены: 20 + 10" in c.value for c in at.caption)


def test_graph_mode_answers_without_index(monkeypatch):
    cit = {"meeting_id": "m_002", "ts": "00:01:05", "quote": "Переносим релиз Атласа на пятое мая"}
    monkeypatch.setattr(answerer, "complete", lambda messages, **kw: LLMResponse(
        text=json.dumps({"answer": "Срок — 5 мая", "citations": [cit]}, ensure_ascii=False)))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.radio[0].set_value("graph").run()
    assert any("Граф: data" in c.value for c in at.caption)
    at.text_input[0].input("Как менялся срок релиза Атласа?").run()
    _button(at, "Спросить").click().run()
    assert not at.exception
    shown = " ".join(m.value for m in at.markdown)
    assert "Срок — 5 мая" in shown and "m_002 · 00:01:05" in shown


def test_auto_mode_shows_route_decision(monkeypatch):
    monkeypatch.setattr(answerer, "complete", lambda messages, **kw: LLMResponse(
        text=json.dumps({"answer": "ok", "citations": []})))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert at.radio[0].value == "auto"
    at.text_input[0].input("Как менялся срок релиза Атласа?").run()
    _button(at, "Спросить").click().run()
    assert not at.exception
    assert any(c.value.startswith("auto → graph (T2)") for c in at.caption)


def test_all_tabs_render_on_sample_graph():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("Граф: data/sample/graph.json" in c.value for c in at.sidebar.caption)
    shown = " ".join(m.value for m in at.markdown)
    assert "~~Релиз Атласа переносится на 20 апреля~~" in shown          # таймлайн: заменённое зачёркнуто
    assert "Замены решений: 1" in [s.value for s in at.subheader]
    assert at.get("graphviz_chart")


def test_compare_mode_shows_both_branches(monkeypatch):
    monkeypatch.setattr(answerer, "complete", lambda messages, **kw: LLMResponse(
        text=json.dumps({"answer": "ok", "citations": []})))
    at = AppTest.from_file(APP, default_timeout=60).run()
    _button(at, "Демо-встречи").click().run()
    at.radio[0].set_value("сравнить").run()
    at.text_input[0].input("Как менялся срок релиза Атласа?").run()
    _button(at, "Спросить").click().run()
    assert not at.exception
    assert {"baseline", "graph"} <= {s.value for s in at.subheader}


def test_sources_are_highlighted_in_transcript(monkeypatch):
    from src.schemas import Answer, Citation
    files = {k: v for k, v in api.demo_files().items() if k.startswith("m_001.")}
    (mid, tr, meta), = api.pair_scribal_files(files)[0]
    seg = next(s for s in api.load_scribal(mid, tr, meta).segments if len(s.text.split()) >= 8)
    quote = " ".join(seg.text.split()[1:6])
    monkeypatch.setattr(api, "ask", lambda q, mode="baseline", **kw: Answer(
        answer="ответ", mode="graph", citations=[Citation(meeting_id="m_001", ts=seg.ts, quote=quote)]))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.radio[0].set_value("graph").run()
    at.text_input[0].input("Что решили?").run()
    _button(at, "Спросить").click().run()
    assert not at.exception
    assert any("<mark>" in m.value for m in at.markdown)
