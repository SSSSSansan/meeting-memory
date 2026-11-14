"""Тесты статусов задач (src/graph/status.py, роль B). Без LLM и без сети; транскрипты подменены."""
from __future__ import annotations

from types import SimpleNamespace

import networkx as nx
import pytest

from src.graph import status as st


def _g() -> nx.MultiDiGraph:
    g = nx.MultiDiGraph()
    g.add_node("project:Alpha", type="Project", label="Alpha")
    for mid, d in (("m_001", "2026-08-01"), ("m_002", "2026-08-20"), ("m_003", "2026-09-10")):
        g.add_node(f"meeting:{mid}", type="Meeting", meeting_id=mid, date=d, label=mid)
        g.add_edge(f"meeting:{mid}", "project:Alpha", type="DISCUSSES")
    g.add_node("t_m001_01", type="Task", date="2026-08-01", ts="00:01:00", meeting_id="m_001", project="Alpha",
               statement="Показать прототип бота", quote="x", due="2026-08-15", status="open", owner="Амина")
    return g


@pytest.fixture()
def transcripts(monkeypatch):
    """Подмена загрузки встречи: {meeting_id: [реплики]}."""
    data: dict[str, list[str]] = {}

    def fake(mid, base, mtimes):
        if mid not in data:
            return None
        return SimpleNamespace(segments=[SimpleNamespace(ts=f"00:00:{i + 10:02d}", text=t)
                                         for i, t in enumerate(data[mid])])
    monkeypatch.setattr(st, "_load_meeting", fake)
    return data


def test_action_verbs_and_names_are_not_keywords():
    g = _g()
    assert st._keywords(g, "t_m001_01") == ["прототип", "бота"]  # без «показать»; Амина и Alpha — имена, не суть


def test_open_before_due(transcripts):
    s = st.task_state(_g(), "t_m001_01", as_of="2026-08-10")
    assert s["state"] == "open" and s["days_overdue"] is None


def test_overdue_and_never_mentioned_is_forgotten(transcripts):
    transcripts["m_002"] = ["Обсудили погоду"]
    s = st.task_state(_g(), "t_m001_01", as_of="2026-09-01")
    assert s["state"] == "overdue_forgotten" and s["days_overdue"] == 17 and s["mentions"] == []


def test_overdue_with_later_mention_is_not_forgotten(transcripts):
    transcripts["m_002"] = ["Прототип бота почти готов"]
    s = st.task_state(_g(), "t_m001_01", as_of="2026-09-01")
    assert s["state"] == "overdue_mentioned" and s["mentions"][0]["meeting_id"] == "m_002"


def test_mention_after_as_of_does_not_count(transcripts):
    transcripts["m_003"] = ["Прототип бота готов"]
    assert st.task_state(_g(), "t_m001_01", as_of="2026-09-01")["state"] == "overdue_forgotten"
    assert st.task_state(_g(), "t_m001_01", as_of="2026-09-30")["state"] == "overdue_mentioned"


def test_mention_in_issuing_meeting_does_not_count(transcripts):
    transcripts["m_001"] = ["Прототип бота покажем к пятнице"]
    assert st.task_state(_g(), "t_m001_01", as_of="2026-09-01")["state"] == "overdue_forgotten"


def test_closes_edge_closes_task_only_from_its_date(transcripts):
    g = _g()
    g.add_node("t_m003_c1", type="Task", date="2026-09-10", ts="00:00:05", meeting_id="m_003", project="Alpha",
               statement="Прототип готов", quote="Прототип готов", status="done")
    g.add_edge("t_m003_c1", "t_m001_01", type="CLOSES")
    assert st.task_state(g, "t_m001_01", as_of="2026-09-30")["state"] == "closed"
    assert st.task_state(g, "t_m001_01", as_of="2026-09-30")["closed_by"] == "t_m003_c1"
    # на вопрос о более ранней дате закрытия ещё не было
    assert st.task_state(g, "t_m001_01", as_of="2026-09-01")["state"] == "overdue_forgotten"


def test_done_in_extraction_is_closed(transcripts):
    g = _g()
    g.nodes["t_m001_01"]["status"] = "done"
    assert st.task_state(g, "t_m001_01", as_of="2026-09-01")["state"] == "closed"


def test_open_question_has_no_due_and_closes_by_edge(transcripts):
    g = _g()
    g.add_node("q_m001_01", type="OpenQuestion", date="2026-08-01", ts="00:02:00", meeting_id="m_001",
               project="Alpha", statement="Кто делает интерфейс?", quote="x")
    assert st.task_state(g, "q_m001_01", as_of="2026-09-01")["state"] == "open"
    g.add_node("d_m002_01", type="Decision", date="2026-08-20", ts="00:00:09", meeting_id="m_002", project="Alpha",
               statement="Интерфейс делает Абду", quote="x", status="active")
    g.add_edge("d_m002_01", "q_m001_01", type="CLOSES")
    assert st.task_state(g, "q_m001_01", as_of="2026-09-01")["state"] == "closed"


def test_task_states_order_forgotten_first(transcripts):
    g = _g()
    g.add_node("t_m001_02", type="Task", date="2026-08-01", ts="00:03:00", meeting_id="m_001", project="Alpha",
               statement="Подготовить отчёт для клиента", quote="x", due=None, status="open")
    transcripts["m_002"] = ["Обсудили погоду"]
    order = [s["state"] for s in st.task_states(g, as_of="2026-09-01", project="alpha")]
    assert order == ["overdue_forgotten", "open"]


def test_real_graph_t5_question_is_not_empty():
    from src.graph.traverse import traverse
    r = traverse("Что ещё не закрыто по Zhol?", as_of="2026-09-10", with_text=False)
    if "t_m006_01" not in r["details"]:
        pytest.skip("нет графа по 14 встречам")
    assert r["items"][0] == "t_m006_01"
    assert r["details"]["t_m006_01"]["task_state"]["state"] == "overdue_forgotten"
