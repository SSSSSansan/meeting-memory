import copy
from pathlib import Path

import pytest
import yaml

from scripts.check_world import DEFAULT_PATH, check, load_world, main


@pytest.fixture
def world() -> dict:
    return load_world(DEFAULT_PATH)


def _run_on_copy(tmp_path: Path, world: dict, capsys) -> tuple[int, str]:
    path = tmp_path / "world.yaml"
    path.write_text(yaml.safe_dump(world, allow_unicode=True, sort_keys=False), encoding="utf-8")
    code = main([str(path)])
    return code, capsys.readouterr().err


def test_real_world_passes(capsys):
    assert main([str(DEFAULT_PATH)]) == 0
    out = capsys.readouterr().out
    assert "Не-русских (mixed + kk)" in out


def test_real_world_has_no_check_errors(world):
    assert check(world) == []


def test_duplicate_event_id_fails(world, tmp_path, capsys):
    w = copy.deepcopy(world)
    w["events"][1]["id"] = w["events"][0]["id"]
    code, err = _run_on_copy(tmp_path, w, capsys)
    assert code == 1
    assert "дубль id событий" in err


def test_forward_link_fails(world, tmp_path, capsys):
    w = copy.deepcopy(world)
    w["links"][0]["from"], w["links"][0]["to"] = w["links"][0]["to"], w["links"][0]["from"]
    code, err = _run_on_copy(tmp_path, w, capsys)
    assert code == 1
    assert "[5]" in err


def test_unknown_participant_fails(world, tmp_path, capsys):
    w = copy.deepcopy(world)
    w["meetings"][0]["participants"].append("ghost")
    code, err = _run_on_copy(tmp_path, w, capsys)
    assert code == 1
    assert "'ghost'" in err


def test_link_to_non_decision_fails(world):
    w = copy.deepcopy(world)
    w["links"][0]["to"] = "ev_m001_03"  # task
    assert any("[6]" in e for e in check(w))


def test_event_project_outside_discussed_fails(world):
    w = copy.deepcopy(world)
    w["events"][0]["project"] = "boost"  # m_001 обсуждает только zhol
    assert any("[4]" in e for e in check(w))


def test_unknown_actor_fails(world):
    w = copy.deepcopy(world)
    w["events"][0]["made_by"] = "ghost"
    assert any("[7]" in e for e in check(w))


def test_unquoted_comma_in_note_fails(tmp_path, capsys):
    text = DEFAULT_PATH.read_text(encoding="utf-8")
    quoted = 'note: "предложила Сандуғаш, Амина согласилась"'
    assert quoted in text
    path = tmp_path / "world.yaml"
    path.write_text(text.replace(quoted, quoted.replace('"', "")), encoding="utf-8")
    assert main([str(path)]) == 1
    err = capsys.readouterr().err
    assert "[8]" in err
    assert "ev_m002_01" in err


def test_dates_out_of_order_fail(world):
    w = copy.deepcopy(world)
    w["meetings"][1]["date"] = w["meetings"][0]["date"].replace(year=2025)
    assert any("[3]" in e for e in check(w))
