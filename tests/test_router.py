import json
from pathlib import Path

import networkx as nx
import pytest

import src.answer.answerer as ans
from src.graph.queries import load_graph
from src.llm_client import LLMResponse
from src.retrieval.baseline import BaselineIndex
from src.retrieval.router import route
from src.schemas import Segment, Transcript

ROOT = Path(__file__).resolve().parent.parent
G = load_graph(ROOT / "data/sample/graph.json")


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_BACKEND", "stub")
    monkeypatch.setenv("EMBED_MODEL", "")
    monkeypatch.setattr(ans, "REPORT", tmp_path / "answer_report.jsonl")


@pytest.mark.parametrize("question, mode, qtype", [
    ("Кто отвечает за миграцию базы данных?", "baseline", "T1"),
    ("Что сказала Дана про скрипт миграции?", "baseline", "T1"),
    ("Долго ли обсуждали Атлас?", "baseline", "T1"),
    ("Что решили по Атласу на последней встрече?", "baseline", "T1"),
    ("Как менялся срок релиза Атласа?", "graph", "T2"),
    ("Когда и на сколько переносили релиз?", "graph", "T2"),
    ("Что решили по Атласу за последние три встречи?", "graph", "T2"),
    ("Что решали по Горизонту на встречах про Атлас?", "graph", "T3"),
    ("Есть ли конфликты по загрузке людей?", "graph", "T4"),
    ("Не противоречат ли друг другу решения по Горизонту?", "graph", "T4"),
    ("Что по Атласу ещё не закрыто?", "graph", "T5"),
    ("Какие задачи просрочены?", "graph", "T5"),
    ("Атлас бойынша мерзім өзгерді ме?", "graph", "T2"),
    ("Қандай тапсырмалар жабылмаған?", "graph", "T5"),
])
def test_route(question, mode, qtype):
    r = route(question, G)
    assert (r.mode, r.qtype) == (mode, qtype), r.reason
    assert r.reason


def _fake_llm(monkeypatch):
    calls = []

    def fake(messages, **kw):
        calls.append(kw["purpose"])
        return LLMResponse(text=json.dumps({"answer": "ok", "citations": []}))

    monkeypatch.setattr(ans, "complete", fake)
    return calls


def _index():
    t = Transcript(meeting_id="m_001", title="Атлас", date="2026-02-03", participants=[],
                   segments=[Segment(ts="00:00:01", speaker="Дана", text="Миграцию делает Айгерим.")])
    index = BaselineIndex()
    index.add([t])
    return index


def test_auto_sends_multi_meeting_question_to_graph(monkeypatch):
    calls = _fake_llm(monkeypatch)
    stats = ans.AnswerStats(mode="auto")
    a = ans.answer("Как менялся срок релиза Атласа?", mode="auto", graph=G, index=_index(), stats=stats)
    assert a.mode == "graph" and calls == ["answer_graph"]
    assert (stats.requested_mode, stats.mode, stats.route_qtype) == ("auto", "graph", "T2")


def test_auto_sends_simple_fact_to_baseline(monkeypatch):
    calls = _fake_llm(monkeypatch)
    a = ans.answer("Кто делает миграцию?", mode="auto", graph=G, index=_index())
    assert a.mode == "baseline" and calls == ["answer_baseline"]


def test_auto_falls_back_to_baseline_when_graph_has_nothing(monkeypatch):
    calls = _fake_llm(monkeypatch)
    stats = ans.AnswerStats(mode="auto")
    a = ans.answer("Как менялся срок релиза?", mode="auto", graph=nx.MultiDiGraph(), index=_index(), stats=stats)
    assert a.mode == "baseline" and calls == ["answer_baseline"]
    assert stats.route_qtype == "T2" and "отвечает baseline" in stats.route_reason


def test_explicit_modes_are_not_routed(monkeypatch):
    _fake_llm(monkeypatch)
    stats = ans.AnswerStats(mode="baseline")
    ans.answer("Как менялся срок релиза Атласа?", mode="baseline", graph=G, index=_index(), stats=stats)
    assert stats.mode == "baseline" and stats.requested_mode == "baseline" and stats.route_qtype == ""
