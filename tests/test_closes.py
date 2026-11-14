"""Связь CLOSES: отбор кандидатов, проверка цитаты, запись в граф. Модель подменена, сеть не нужна."""
from types import SimpleNamespace

import networkx as nx
import pytest

from src.graph import closes as cl

CANDS = [{"meeting_id": "m_004", "date": "2026-01-20", "ts": "00:00:21", "speaker": "Абду",
          "text": "Это готово. Миграция базы данных выполнена. Скрипты все отработали.", "before": "", "after": ""}]


def _graph() -> nx.MultiDiGraph:
    G = nx.MultiDiGraph()
    G.add_node("project:Zhol", type="Project", label="Zhol")
    G.add_node("t_m001_01", type="Task", statement="Сделать миграцию БД", project="Zhol", meeting_id="m_001",
               date="2026-01-06", ts="00:01:00", status="open", owner="Абду")
    G.add_node("d_m004_02", type="Decision", statement="Миграция БД выполнена", project="Zhol", meeting_id="m_004",
               date="2026-01-20", ts="00:00:25")
    return G


def test_parse_answer_strips_code_fence():
    assert cl.parse_answer('```json\n{"closed": false}\n```') == {"closed": False}
    with pytest.raises(ValueError):
        cl.parse_answer('{"closed": "yes"}')


def test_validate_requires_verbatim_quote():
    ok = {"closed": True, "candidate": 1, "quote": "миграция базы данных выполнена", "confidence": 0.9}
    assert cl.validate(ok, CANDS)[0] is CANDS[0]
    bad = dict(ok, quote="миграция успешно завершена")
    assert cl.validate(bad, CANDS)[0] is None
    assert "цитата" in cl.validate(bad, CANDS)[1]


def test_validate_rejects_low_confidence_and_bad_index():
    base = {"closed": True, "candidate": 1, "quote": "Миграция базы данных выполнена", "confidence": 0.3}
    assert cl.validate(base, CANDS)[0] is None
    assert cl.validate(dict(base, confidence=0.9, candidate=7), CANDS)[0] is None
    assert cl.validate({"closed": False}, CANDS)[0] is None


def test_closing_source_picks_nearby_item_of_same_meeting():
    G = _graph()
    assert cl.closing_source(G, "t_m001_01", CANDS[0]) == "d_m004_02"
    assert cl.closing_source(G, "t_m001_01", dict(CANDS[0], ts="00:09:00")) is None


def test_apply_closes_marks_done_and_is_idempotent():
    G = _graph()
    rec = {"from": "d_m004_02", "to": "t_m001_01", "confidence": 0.9, "rationale": "готово", "quote": "x",
           "meeting_id": "m_004", "date": "2026-01-20", "ts": "00:00:21"}
    assert cl.apply_closes(G, [rec]) == 1
    assert cl.apply_closes(G, [rec]) == 0
    assert G.nodes["t_m001_01"]["status"] == "done"
    assert G.nodes["t_m001_01"]["closed_at"] == "2026-01-20"
    assert G.has_edge("d_m004_02", "t_m001_01", key="CLOSES")


def test_apply_closes_creates_node_when_no_source_item():
    G = _graph()
    rec = {"from": "t_m004_c01", "to": "t_m001_01", "confidence": 0.8, "rationale": "Миграция выполнена",
           "quote": "Миграция базы данных выполнена", "meeting_id": "m_004", "date": "2026-01-20", "ts": "00:00:21"}
    cl.apply_closes(G, [rec])
    new = G.nodes["t_m004_c01"]
    assert new["type"] == "Closure" and new["origin"] == "closes"      # не Task: чужие счётчики его не видят
    assert new["project"] == "Zhol"


def test_low_confidence_not_applied():
    G = _graph()
    rec = {"from": "d_m004_02", "to": "t_m001_01", "confidence": 0.4, "meeting_id": "m_004", "date": "2026-01-20"}
    assert cl.apply_closes(G, [rec]) == 0


def test_classify_with_stub_model(monkeypatch):
    G = _graph()
    monkeypatch.setattr(cl, "candidates", lambda *a, **k: CANDS)
    reply = ('{"closed": true, "candidate": 1, "quote": "Миграция базы данных выполнена", '
             '"confidence": 0.95, "rationale": "Сказано, что миграция выполнена."}')
    monkeypatch.setattr("src.llm_client.complete", lambda *a, **k: SimpleNamespace(text=reply, cost_usd=0.0001))
    r = cl.classify(G, "t_m001_01")
    assert r["record"]["from"] == "d_m004_02" and r["record"]["meeting_id"] == "m_004"


def test_save_results_overwrites_not_duplicates(tmp_path):
    G = _graph()
    rec = {"from": None, "to": "t_m001_01", "confidence": 0.9, "meeting_id": "m_004", "date": "2026-01-20",
           "ts": "00:00:21", "quote": "q", "rationale": "r"}
    res = [{"item": "t_m001_01", "record": rec, "reason": "", "answer": {}, "cands": []}]
    assert cl.save_results(res, G, tmp_path) == (1, 0)
    assert cl.save_results(res, G, tmp_path) == (1, 0)
    import json
    saved = json.loads((tmp_path / "closes.json").read_text(encoding="utf-8"))
    assert saved[0]["from"] == "t_m004_c01" and saved[0]["new_node"] is True


def test_open_items_skips_already_closed():
    G = _graph()
    assert cl.open_items(G) == ["t_m001_01"]
    G.add_edge("d_m004_02", "t_m001_01", key="CLOSES", type="CLOSES")
    assert cl.open_items(G) == []


def test_closed_by_reads_quote_from_edge_and_respects_as_of():
    from src.graph.status import closed_by
    G = _graph()
    rec = {"from": "t_m004_c01", "to": "t_m001_01", "confidence": 0.8, "rationale": "r", "quote": "миграция выполнена",
           "meeting_id": "m_004", "date": "2026-01-20", "ts": "00:00:21"}
    cl.apply_closes(G, [rec])
    cb = closed_by(G, "t_m001_01", "2026-12-01")
    assert cb["by"] == "t_m004_c01" and cb["quote"] == "миграция выполнена" and cb["ts"] == "00:00:21"
    assert closed_by(G, "t_m001_01", "2026-01-10") is None             # на дату раньше закрытия задача ещё открыта


def test_closure_node_is_not_a_traverse_item():
    from src.graph.traverse import _items
    G = _graph()
    cl.apply_closes(G, [{"from": "t_m004_c01", "to": "t_m001_01", "confidence": 0.8, "quote": "q",
                         "meeting_id": "m_004", "date": "2026-01-20", "ts": "00:00:21"}])
    assert "t_m004_c01" not in _items(G, None)


def test_done_status_set_by_closing_is_not_visible_before_closing_date():
    from src.graph.status import task_state
    G = _graph()
    G.add_node("meeting:m_004", type="Meeting", label="m_004", date="2026-01-20")
    G.nodes["t_m001_01"]["due"] = "2026-01-10"
    cl.apply_closes(G, [{"from": "t_m004_c01", "to": "t_m001_01", "confidence": 0.9, "quote": "q",
                         "meeting_id": "m_004", "date": "2026-01-20", "ts": "00:00:21"}])
    assert task_state(G, "t_m001_01", "2026-01-15")["state"].startswith("overdue")   # закрытие ещё в будущем
    assert task_state(G, "t_m001_01", "2026-01-25")["state"] == "closed"
