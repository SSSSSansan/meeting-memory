import copy
import json
from pathlib import Path

import pytest

from scripts.check_questions import DEFAULT_QUESTIONS, DEFAULT_WORLD, check, load_json, load_world, main


@pytest.fixture
def world() -> dict:
    return load_world(DEFAULT_WORLD)


@pytest.fixture
def questions() -> list[dict]:
    return copy.deepcopy(load_json(DEFAULT_QUESTIONS))


def _by_id(questions: list[dict], qid: str) -> dict:
    return next(q for q in questions if q["id"] == qid)


def _run(tmp_path: Path, questions: list[dict], capsys) -> tuple[int, str]:
    path = tmp_path / "questions.json"
    path.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
    code = main([str(path), str(DEFAULT_WORLD)])
    return code, capsys.readouterr().err


def test_real_questions_pass(capsys):
    assert main([]) == 0
    assert "Все проверки пройдены" in capsys.readouterr().out


def test_real_questions_have_no_errors(questions, world):
    assert check(questions, world) == []


def test_real_questions_shape(questions):
    assert len(questions) == 18
    assert {q["split"] for q in questions} == {"dev", "test"}
    assert {q["as_of"] for q in questions} == {"m_008"}
    assert sum(q["lang"] == "kk" for q in questions) >= 3


def test_broken_event_id_fails(questions, world):
    _by_id(questions, "q_01")["event_ids"] = ["ev_m006_99"]
    errors = check(questions, world)
    assert any("q_01" in e and "ev_m006_99" in e and "нет в world.yaml" in e for e in errors)


def test_broken_source_meeting_fails(questions, world):
    _by_id(questions, "q_01")["source_meetings"] = ["m_006", "m_099"]
    errors = check(questions, world)
    assert any("q_01" in e and "m_099" in e for e in errors)


def test_event_meeting_not_in_source_meetings_fails(questions, world):
    _by_id(questions, "q_01")["source_meetings"] = ["m_003"]
    errors = check(questions, world)
    assert any("q_01" in e and "не входит в source_meetings" in e for e in errors)


def test_event_after_as_of_fails(questions, world):
    q = _by_id(questions, "q_05")
    q["event_ids"] = ["ev_m001_01", "ev_m011_01"]
    q["source_meetings"] = ["m_001", "m_011"]
    errors = check(questions, world)
    assert any("q_05" in e and "позже as_of" in e for e in errors)


def test_source_meeting_after_as_of_fails(questions, world):
    _by_id(questions, "q_05")["source_meetings"].append("m_010")
    errors = check(questions, world)
    assert any("q_05" in e and "m_010" in e and "позже as_of" in e for e in errors)


def test_unknown_as_of_fails(questions, world):
    _by_id(questions, "q_01")["as_of"] = "m_099"
    errors = check(questions, world)
    assert any("q_01" in e and "as_of" in e for e in errors)


def test_unanswerable_with_events_fails(questions, world):
    _by_id(questions, "q_18")["event_ids"] = ["ev_m004_01"]
    errors = check(questions, world)
    assert any("q_18" in e and "answerable=false" in e for e in errors)


def test_unanswerable_with_meetings_fails(questions, world):
    _by_id(questions, "q_18")["source_meetings"] = ["m_004"]
    errors = check(questions, world)
    assert any("q_18" in e and "answerable=false" in e for e in errors)


def test_answerable_without_sources_fails(questions, world):
    q = _by_id(questions, "q_01")
    q["source_meetings"] = []
    q["event_ids"] = []
    errors = check(questions, world)
    assert any("q_01" in e and "answerable=true" in e for e in errors)


def test_type_none_requires_unanswerable(questions, world):
    _by_id(questions, "q_01")["type"] = "none"
    errors = check(questions, world)
    assert any("q_01" in e and "type none" in e for e in errors)


def test_duplicate_id_fails(questions, world):
    questions[1]["id"] = questions[0]["id"]
    errors = check(questions, world)
    assert any("дубль id вопроса: q_01" in e for e in errors)


def test_missing_field_fails(questions, world):
    del _by_id(questions, "q_02")["as_of"]
    errors = check(questions, world)
    assert any("q_02" in e and "as_of" in e for e in errors)


def test_extra_field_fails(questions, world):
    _by_id(questions, "q_02")["decision_ids"] = ["ev_m003_02"]
    errors = check(questions, world)
    assert any("q_02" in e and "decision_ids" in e for e in errors)


@pytest.mark.parametrize(
    "field,value",
    [("type", "T9"), ("split", "val"), ("lang", "en"), ("id", "question1"), ("as_of", "008"), ("question", "  ")],
)
def test_bad_field_value_fails(questions, world, field, value):
    _by_id(questions, "q_03")[field] = value
    errors = check(questions, world)
    assert any("q_03" in e or "question1" in e for e in errors)


def test_too_few_kk_questions_fails(questions, world):
    for q in questions:
        if q["lang"] == "kk" and q["id"] != "q_04":
            q["lang"] = "ru"
    errors = check(questions, world)
    assert any("казахском" in e for e in errors)


def test_not_a_list_fails(world):
    assert check({"id": "q_01"}, world) == ["questions.json: ожидался список записей"]


def test_record_not_an_object_fails(questions, world):
    questions.append("q_99")
    errors = check(questions, world)
    assert any("ожидался объект" in e for e in errors)


def test_main_returns_1_on_broken_reference(tmp_path, questions, capsys):
    _by_id(questions, "q_01")["event_ids"] = ["ev_m006_99"]
    code, err = _run(tmp_path, questions, capsys)
    assert code == 1
    assert "ev_m006_99" in err


def test_main_returns_1_on_missing_file(tmp_path, capsys):
    assert main([str(tmp_path / "nope.json"), str(DEFAULT_WORLD)]) == 1
    assert "Не удалось прочитать файл" in capsys.readouterr().err
