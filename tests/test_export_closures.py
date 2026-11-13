import copy
import json

import pytest
import yaml

from scripts.export_closures import DEFAULT_OUT, DEFAULT_WORLD, build_closures, dumps_closures, main

FIELDS = ["task_event_id", "closing_event_id", "task_statement", "closing_statement", "task_meeting", "closing_meeting"]


@pytest.fixture
def world() -> dict:
    return yaml.safe_load(DEFAULT_WORLD.read_text(encoding="utf-8"))


def test_one_record_per_closure_with_exact_fields(world):
    items = build_closures(world)
    assert len(items) == len(world["closures"]) == 6
    assert all(list(i) == FIELDS for i in items)


def test_record_content_matches_world(world):
    items = {i["task_event_id"]: i for i in build_closures(world)}
    rec = items["ev_m001_03"]
    assert rec["closing_event_id"] == "ev_m004_03"
    assert rec["task_statement"] == "Сделать миграцию БД"
    assert rec["closing_statement"] == "Миграция БД выполнена"
    assert (rec["task_meeting"], rec["closing_meeting"]) == ("m_001", "m_004")
    last = items["ev_m012_02"]
    assert (last["closing_event_id"], last["task_meeting"], last["closing_meeting"]) == ("ev_m015_05", "m_012", "m_015")


def test_closures_do_not_include_links_or_unclosed_tasks(world):
    closed = {i["task_event_id"] for i in build_closures(world)}
    # документация API Squad и блок метрик продаж в Boost не закрыты: в выгрузке их быть не должно
    assert not {"ev_m003_03", "ev_m015_02", "ev_m010_02"} & closed


def test_committed_file_matches_world(world):
    text = DEFAULT_OUT.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert text == dumps_closures(build_closures(world)), "closures.json устарел: запустите export_closures.py"


def test_unknown_event_is_an_error(world):
    broken = copy.deepcopy(world)
    broken["closures"][0]["closes"] = "ev_m001_99"
    with pytest.raises(ValueError, match="ev_m001_99"):
        build_closures(broken)


def test_closing_before_task_is_an_error(world):
    broken = copy.deepcopy(world)
    broken["closures"][0] = {"from": "ev_m001_03", "closes": "ev_m004_03"}
    with pytest.raises(ValueError, match="раньше"):
        build_closures(broken)


def test_main_is_deterministic_and_writes_lf(tmp_path):
    out = tmp_path / "closures.json"
    assert main([str(DEFAULT_WORLD), str(out)]) == 0
    first = out.read_bytes()
    assert main([str(DEFAULT_WORLD), str(out)]) == 0
    assert out.read_bytes() == first and b"\r\n" not in first
    assert len(json.loads(first)) == 6


def test_main_fails_without_writing_on_error(tmp_path, capsys):
    out = tmp_path / "closures.json"
    assert main([str(tmp_path / "nope.yaml"), str(out)]) == 1
    assert not out.exists()
    assert "Не удалось выгрузить" in capsys.readouterr().err
    assert main(["a", "b", "c"]) == 2
