import copy
import json
from pathlib import Path

import pytest

from scripts.check_questions import (
    DEFAULT_QUESTIONS,
    DEFAULT_WORLD,
    check,
    load_json,
    load_world,
    main,
    warnings,
)


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
    assert len(questions) == 58
    assert len({q["id"] for q in questions}) == 58
    assert [q["id"] for q in questions] == [f"q_{i:02d}" for i in range(1, 59)]
    assert {q["split"] for q in questions} == {"dev", "test"}
    assert sum(q["split"] == "test" for q in questions) == 23
    assert sum(q["split"] == "dev" for q in questions) == 35
    assert {q["as_of"] for q in questions} == {"m_008", "m_015"}
    # q_16 привязан к дате 2 сентября и остаётся на m_008; остальные — на m_015
    assert {q["id"] for q in questions if q["as_of"] == "m_008"} == {"q_16"}
    for split in ("dev", "test"):
        assert sum(q["lang"] == "kk" for q in questions if q["split"] == split) >= 3


def test_old_questions_are_dev_new_are_test(questions):
    # старые q_01..q_35 запускали до заморозки, поэтому все dev; q_36..q_58 система не запускала
    for q in questions:
        number = int(q["id"][2:])
        assert q["split"] == ("dev" if number <= 35 else "test"), q["id"]
        assert q["as_of_date"] == ("2026-09-02" if q["as_of"] == "m_008" else "2026-10-26")


def test_old_questions_have_empty_facts_new_have_required(questions):
    for q in questions:
        number = int(q["id"][2:])
        if number <= 35:
            assert q["required_facts"] == [] and q["forbidden_facts"] == [], q["id"]
        else:
            assert q["required_facts"] and q["forbidden_facts"], q["id"]
            if not q["answerable"]:
                assert q["source_meetings"] == [] and q["decision_ids"] == [], q["id"]


def test_test_split_has_each_type_at_least_three(questions):
    from collections import Counter

    by_type = Counter(q["type"] for q in questions if q["split"] == "test")
    assert {t: by_type[t] for t in ("T1", "T2", "T3", "T4", "T5", "none")} == {
        "T1": 4, "T2": 4, "T3": 3, "T4": 4, "T5": 4, "none": 4,
    }


def test_questions_use_only_existing_meetings(questions, world):
    # m_014 исключена из набора: ни as_of, ни источники на неё не ссылаются
    ids = {m["id"] for m in world["meetings"]}
    assert "m_014" not in ids
    for q in questions:
        assert {q["as_of"], *q["source_meetings"]} <= ids


def test_m008_copy_is_valid_for_incremental_check(world):
    # копия вопросов на состояние m_008: все as_of m_008, ответы без встреч после неё
    old = load_json(DEFAULT_QUESTIONS.with_name("questions_m008.json"))
    assert len(old) == 18 and {q["as_of"] for q in old} == {"m_008"}
    # в копии нет новых полей и нет требований к полному набору: strict=False
    assert check(old, world, strict=False) == []
    assert any("нет поля as_of_date" in e for e in check(old, world))


def test_broken_event_id_fails(questions, world):
    _by_id(questions, "q_01")["decision_ids"] = ["ev_m006_99"]
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
    q["as_of"] = "m_008"
    q["decision_ids"] = ["ev_m001_01", "ev_m011_01"]
    q["source_meetings"] = ["m_001", "m_011"]
    errors = check(questions, world)
    assert any("q_05" in e and "позже as_of" in e for e in errors)


def test_source_meeting_after_as_of_fails(questions, world):
    q = _by_id(questions, "q_05")
    q["as_of"] = "m_008"
    q["source_meetings"].append("m_010")
    errors = check(questions, world)
    assert any("q_05" in e and "m_010" in e and "позже as_of" in e for e in errors)


def test_as_of_compared_by_date_not_number(questions, world):
    # номер m_010 больше, чем у m_008, но по дате она раньше: источник допустим
    for m in world["meetings"]:
        if m["id"] == "m_010":
            m["date"] = world["meetings"][0]["date"]
    q = _by_id(questions, "q_01")
    q["as_of"] = "m_008"
    q["as_of_date"] = "2026-09-02"
    q["source_meetings"] = ["m_006", "m_010"]
    assert check(questions, world) == []
    # и наоборот: m_007 раньше m_008 по номеру, но по дате позже
    for m in world["meetings"]:
        if m["id"] == "m_007":
            m["date"] = world["meetings"][-1]["date"]
    q["source_meetings"] = ["m_006", "m_007"]
    assert any("m_007" in e and "позже as_of" in e for e in check(questions, world))


def test_as_of_of_excluded_meeting_fails(questions, world):
    _by_id(questions, "q_01")["as_of"] = "m_014"
    errors = check(questions, world)
    assert any("q_01" in e and "m_014" in e and "нет в world.yaml" in e for e in errors)


def test_duplicate_decision_ids_fail(questions, world):
    q = _by_id(questions, "q_05")
    q["decision_ids"] = [*q["decision_ids"], q["decision_ids"][0]]
    errors = check(questions, world)
    assert any("q_05" in e and "больше одного раза" in e for e in errors)


def test_unknown_as_of_fails(questions, world):
    _by_id(questions, "q_01")["as_of"] = "m_099"
    errors = check(questions, world)
    assert any("q_01" in e and "as_of" in e for e in errors)


def test_unanswerable_with_events_fails(questions, world):
    _by_id(questions, "q_18")["decision_ids"] = ["ev_m004_01"]
    errors = check(questions, world)
    assert any("q_18" in e and "answerable=false" in e for e in errors)


def test_unanswerable_with_meetings_fails(questions, world):
    _by_id(questions, "q_18")["source_meetings"] = ["m_004"]
    errors = check(questions, world)
    assert any("q_18" in e and "answerable=false" in e for e in errors)


def test_answerable_without_sources_fails(questions, world):
    q = _by_id(questions, "q_01")
    q["source_meetings"] = []
    q["decision_ids"] = []
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
    _by_id(questions, "q_02")["answer_ids"] = ["ev_m003_02"]
    errors = check(questions, world)
    assert any("q_02" in e and "answer_ids" in e for e in errors)


@pytest.mark.parametrize(
    "field,value",
    [("type", "T9"), ("split", "val"), ("lang", "en"), ("id", "question1"), ("as_of", "008"), ("question", "  ")],
)
def test_bad_field_value_fails(questions, world, field, value):
    _by_id(questions, "q_03")[field] = value
    errors = check(questions, world)
    assert any("q_03" in e or "question1" in e for e in errors)


def test_too_few_kk_in_dev_fails(questions, world):
    dev_kk = [q for q in questions if q["split"] == "dev" and q["lang"] == "kk"]
    for q in dev_kk[2:]:
        q["lang"] = "ru"
    errors = check(questions, world)
    assert any("казахском в dev 2" in e for e in errors)
    assert not any("казахском в test" in e for e in errors)


def test_too_few_kk_in_test_fails(questions, world):
    test_kk = [q for q in questions if q["split"] == "test" and q["lang"] == "kk"]
    for q in test_kk[2:]:
        q["lang"] = "ru"
    errors = check(questions, world)
    assert any("казахском в test 2" in e for e in errors)
    assert not any("казахском в dev" in e for e in errors)


def test_wrong_split_counts_fail(questions, world):
    _by_id(questions, "q_36")["split"] = "dev"
    errors = check(questions, world)
    assert any("выборке test 22, ожидалось 23" in e for e in errors)
    assert any("выборке dev 36, ожидалось 35" in e for e in errors)


def test_wrong_total_fails(questions, world):
    questions.pop()
    errors = check(questions, world)
    assert any("всего 57, ожидалось 58" in e for e in errors)


def test_test_needs_three_questions_of_each_type(questions, world):
    # q_37..q_39 — все T3 в test; один переводим в T2 (answerable не трогаем)
    _by_id(questions, "q_39")["type"] = "T2"
    errors = check(questions, world)
    assert any("типа T3 в выборке test 2" in e for e in errors)
    assert not any("типа T2" in e for e in errors)


def test_test_question_without_required_facts_fails(questions, world):
    _by_id(questions, "q_41")["required_facts"] = []
    errors = check(questions, world)
    assert any("q_41" in e and "required_facts пусты" in e for e in errors)


def test_unanswerable_test_question_needs_required_facts(questions, world):
    q = _by_id(questions, "q_43")
    assert q["answerable"] is False and q["split"] == "test"
    q["required_facts"] = []
    errors = check(questions, world)
    assert any("q_43" in e and "required_facts пусты" in e for e in errors)


def test_dev_question_with_empty_required_facts_is_allowed(questions, world):
    q = _by_id(questions, "q_01")
    assert q["split"] == "dev" and q["required_facts"] == []
    assert check(questions, world) == []


def test_as_of_date_must_match_world(questions, world):
    _by_id(questions, "q_36")["as_of_date"] = "2026-10-27"
    errors = check(questions, world)
    assert any("q_36" in e and "as_of_date 2026-10-27" in e and "2026-10-26" in e for e in errors)


def test_as_of_date_for_m008_question_uses_its_own_meeting(questions, world):
    assert _by_id(questions, "q_16")["as_of_date"] == "2026-09-02"
    _by_id(questions, "q_16")["as_of_date"] = "2026-10-26"
    errors = check(questions, world)
    assert any("q_16" in e and "2026-09-02" in e for e in errors)


@pytest.mark.parametrize("value", ["26.10.2026", "2026-13-01", "m_015", ""])
def test_as_of_date_bad_format_fails(questions, world, value):
    _by_id(questions, "q_36")["as_of_date"] = value
    errors = check(questions, world)
    assert any("q_36" in e and "as_of_date" in e for e in errors)


@pytest.mark.parametrize("field", ["as_of_date", "required_facts", "forbidden_facts"])
def test_new_field_is_required(questions, world, field):
    del _by_id(questions, "q_01")[field]
    errors = check(questions, world)
    assert any("q_01" in e and f"нет поля {field}" in e for e in errors)


def test_blank_fact_fails(questions, world):
    _by_id(questions, "q_36")["required_facts"] = ["срок", "  "]
    errors = check(questions, world)
    assert any("q_36" in e and "required_facts" in e for e in errors)


def test_facts_must_be_a_list_of_strings(questions, world):
    _by_id(questions, "q_36")["forbidden_facts"] = "один факт строкой"
    errors = check(questions, world)
    assert any("q_36" in e and "forbidden_facts" in e for e in errors)


def test_warnings_count_dev_questions_without_required_facts(questions):
    assert warnings(questions) == ["у 35 вопросов dev пустые required_facts (допустимо для dev, нужно дописать)"]
    _by_id(questions, "q_01")["required_facts"] = ["Абдурахман"]
    assert "у 34 вопросов dev" in warnings(questions)[0]
    # пустые required_facts в test — не предупреждение, а ошибка check()
    _by_id(questions, "q_41")["required_facts"] = []
    assert "у 34 вопросов dev" in warnings(questions)[0]
    for q in questions:
        q["required_facts"] = q["required_facts"] or ["x"]
    assert warnings(questions) == []
    assert warnings({"not": "a list"}) == []


def test_main_prints_dev_warning_with_count(capsys):
    assert main([]) == 0
    out = capsys.readouterr().out
    assert "ПРЕДУПРЕЖДЕНИЕ" in out and "35 вопросов dev" in out


def test_not_a_list_fails(world):
    assert check({"id": "q_01"}, world) == ["questions.json: ожидался список записей"]


def test_record_not_an_object_fails(questions, world):
    questions.append("q_99")
    errors = check(questions, world)
    assert any("ожидался объект" in e for e in errors)


def test_main_returns_1_on_broken_reference(tmp_path, questions, capsys):
    _by_id(questions, "q_01")["decision_ids"] = ["ev_m006_99"]
    code, err = _run(tmp_path, questions, capsys)
    assert code == 1
    assert "ev_m006_99" in err


def test_main_returns_1_on_missing_file(tmp_path, capsys):
    assert main([str(tmp_path / "nope.json"), str(DEFAULT_WORLD)]) == 1
    assert "Не удалось прочитать файл" in capsys.readouterr().err
