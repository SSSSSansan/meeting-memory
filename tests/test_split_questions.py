import json
from pathlib import Path

import pytest

from scripts.split_questions import DEFAULT_QUESTIONS, dumps_questions, main, split_questions


def _read(path: Path) -> str:
    # в Windows файл из git может быть с CRLF; на содержимое это не влияет
    return path.read_text(encoding="utf-8").replace("\r\n", "\n")


def _items(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def _q(qid: str, split: str) -> dict:
    return {"id": qid, "split": split, "answerable": True, "required_facts": ["a"], "source_meetings": ["m_001"]}


def test_split_keeps_order_and_records():
    items = [_q("q_01", "test"), _q("q_02", "dev"), _q("q_03", "test"), _q("q_04", "dev")]
    parts = split_questions(items)
    assert [q["id"] for q in parts["dev"]] == ["q_02", "q_04"]
    assert [q["id"] for q in parts["test"]] == ["q_01", "q_03"]
    assert all(q in items for q in parts["dev"] + parts["test"])


def test_unknown_split_is_an_error_not_a_skip():
    with pytest.raises(ValueError, match="q_09"):
        split_questions([_q("q_09", "val")])


def test_main_writes_files_next_to_source_and_is_deterministic(tmp_path, capsys):
    src = tmp_path / "questions.json"
    src.write_text(dumps_questions([_q("q_01", "test"), _q("q_02", "dev")]), encoding="utf-8", newline="")
    assert main([str(src)]) == 0
    first = {n: (tmp_path / n).read_bytes() for n in ("questions_dev.json", "questions_test.json")}
    assert main([str(src)]) == 0
    assert first == {n: (tmp_path / n).read_bytes() for n in first}
    assert b"\r\n" not in b"".join(first.values())
    assert [q["id"] for q in _items(tmp_path / "questions_dev.json")] == ["q_02"]
    assert [q["id"] for q in _items(tmp_path / "questions_test.json")] == ["q_01"]
    assert "questions_test.json: 1" in capsys.readouterr().out


def test_main_does_not_touch_source(tmp_path):
    src = tmp_path / "questions.json"
    src.write_text(dumps_questions([_q("q_01", "test")]), encoding="utf-8", newline="")
    before = src.read_bytes()
    main([str(src)])
    assert src.read_bytes() == before


def test_main_fails_on_bad_input(tmp_path, capsys):
    assert main([str(tmp_path / "nope.json")]) == 1
    bad = tmp_path / "questions.json"
    bad.write_text('{"id": "q_01"}', encoding="utf-8")
    assert main([str(bad)]) == 1
    assert "Не удалось разделить" in capsys.readouterr().err
    assert main(["a", "b"]) == 2


def test_dumps_roundtrip_and_format():
    items = [{"id": "q_01", "source_meetings": ["m_001", "m_002"], "decision_ids": [],
              "required_facts": ["раз", "два"], "forbidden_facts": [], "answerable": True}]
    text = dumps_questions(items)
    assert json.loads(text) == items
    assert '"source_meetings": ["m_001", "m_002"]' in text      # короткие списки в одну строку
    assert '"decision_ids": []' in text
    assert '"required_facts": [\n      "раз",\n      "два"\n    ]' in text  # факты — по строке на элемент
    assert text.endswith("]\n")


def test_real_questions_file_is_in_canonical_format():
    # questions.json пишется тем же форматом, поэтому повторная запись не меняет файл
    text = _read(DEFAULT_QUESTIONS)
    assert dumps_questions(json.loads(text)) == text


def test_committed_dev_and_test_match_questions_json():
    # после правки questions.json нужно перезапустить scripts/split_questions.py
    items = json.loads(_read(DEFAULT_QUESTIONS))
    parts = split_questions(items)
    for split in ("dev", "test"):
        path = DEFAULT_QUESTIONS.with_name(f"questions_{split}.json")
        assert _read(path) == dumps_questions(parts[split]), f"{path.name} устарел: запустите split_questions.py"


def test_dev_and_test_do_not_overlap():
    base = DEFAULT_QUESTIONS
    dev = {q["id"] for q in _items(base.with_name("questions_dev.json"))}
    test = {q["id"] for q in _items(base.with_name("questions_test.json"))}
    assert dev and test and not dev & test
    assert dev | test == {q["id"] for q in _items(base)}
