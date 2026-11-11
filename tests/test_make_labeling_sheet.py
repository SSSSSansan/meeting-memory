import csv
import json
from pathlib import Path

import pytest

from scripts import make_labeling_sheet as mls

FORBIDDEN = {
    "question_id", "type", "split", "lang", "as_of", "answerable", "events_found", "events_total",
    "correct_meetings", "meetings_hit", "citations_total", "citations_valid", "citation_valid",
    "abstained", "accuracy", "source_meetings", "cited_after_as_of",
}
EVAL_COLUMNS = [
    "question_id", "type", "split", "lang", "as_of", "mode", "system", "backend", "status", "error",
    "answerable", "answer", "cited_meetings", "source_meetings", "correct_meetings", "meetings_hit",
    "cited_after_as_of", "events_found", "events_total", "citations_total", "citations_valid",
    "citation_valid", "abstained", "accuracy", "repeat",
]


def _make_eval_data(tmp_path: Path) -> tuple[Path, Path]:
    """Синтетический набор: по 8 вопросов каждого типа (none — 6), часть kk/test/проблемных."""
    questions, rows = [], []
    n = 0
    for t, count in {"none": 6, "T1": 8, "T2": 8, "T3": 8, "T4": 8, "T5": 8}.items():
        for i in range(count):
            n += 1
            qid = f"q_{n:02d}"
            answerable = t != "none"
            bad = i % 3 == 0
            lang = "kk" if i % 4 == 1 else "ru"
            split = "test" if i % 2 == 0 else "dev"
            questions.append({"id": qid, "question": f"Вопрос №{n}?", "expected_answer": f"Эталон №{n}",
                              "type": t, "split": split, "lang": lang, "answerable": answerable})
            rows.append({
                "question_id": qid, "type": t, "split": split, "lang": lang, "as_of": "m_015",
                "mode": "baseline", "system": "real", "backend": "openai", "status": "ok", "error": "",
                "answerable": str(answerable), "answer": f"Ответ №{n}", "cited_meetings": "m_001",
                "source_meetings": "m_001", "correct_meetings": "False" if bad and answerable else "True",
                "meetings_hit": "1.0", "cited_after_as_of": "0", "events_found": "1", "events_total": "1",
                "citations_total": "1", "citations_valid": "1", "citation_valid": "1.0",
                "abstained": str(bad), "accuracy": "n/a", "repeat": "1",
            })
    # Строки другого режима и со статусом error не должны попадать в выборку.
    rows.append({**rows[0], "mode": "graph"})
    rows.append({**rows[1], "status": "error"})
    tmp_path.mkdir(parents=True, exist_ok=True)
    eval_csv, qjson = tmp_path / "eval.csv", tmp_path / "questions.json"
    with open(eval_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=EVAL_COLUMNS)
        w.writeheader()
        w.writerows(rows)
    qjson.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
    return eval_csv, qjson


def _run(tmp_path: Path, name: str = "out") -> tuple[Path, list[dict], list[dict]]:
    eval_csv, qjson = _make_eval_data(tmp_path)
    out = tmp_path / name
    assert mls.main(["--eval-csv", str(eval_csv), "--questions", str(qjson), "--out-dir", str(out)]) == 0
    return out, _read(out / "labeling_sheet.csv"), _read(out / "key_private.csv")


def _read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


@pytest.fixture
def patched(monkeypatch):
    monkeypatch.setattr(mls, "MANDATORY", ("q_07", "q_20", "q_35", "q_41"))


def test_twenty_rows_and_quotas(tmp_path, patched):
    out, sheet, key = _run(tmp_path)
    assert len(sheet) == len(key) == 20
    assert [r["label_id"] for r in sheet] == [f"L{i:02d}" for i in range(1, 21)] or \
        sorted(r["label_id"] for r in sheet) == [f"L{i:02d}" for i in range(1, 21)]
    counts = {t: sum(r["type"] == t for r in key) for t in mls.QUOTAS}
    assert counts == {"none": 3, "T1": 3, "T2": 4, "T3": 3, "T4": 3, "T5": 4}
    assert sum(r["lang"] == "kk" for r in key) >= 3
    assert sum(r["split"] == "test" for r in key) >= 6
    assert {"q_07", "q_20", "q_35", "q_41"} <= {r["question_id"] for r in key}
    assert len({r["question_id"] for r in key}) == 20


def test_sheet_has_no_forbidden_columns(tmp_path, patched):
    out, sheet, key = _run(tmp_path)
    assert list(sheet[0]) == mls.SHEET_COLUMNS
    assert not FORBIDDEN & set(sheet[0])
    assert all(r["human_score"] == r["human_fabrication"] == r["comment"] == "" for r in sheet)
    assert list(key[0]) == mls.KEY_COLUMNS
    assert {r["label_id"] for r in sheet} == {r["label_id"] for r in key}
    # В тексте листа не должно быть id вопросов.
    assert "q_" not in (out / "labeling_sheet.csv").read_text(encoding="utf-8-sig")


def test_encoding_is_utf8_sig(tmp_path, patched):
    out, _, _ = _run(tmp_path)
    for name in ("labeling_sheet.csv", "key_private.csv"):
        assert (out / name).read_bytes().startswith(b"\xef\xbb\xbf")


def test_deterministic(tmp_path, patched):
    a, _, _ = _run(tmp_path / "a")
    b, _, _ = _run(tmp_path / "b")
    for name in ("labeling_sheet.csv", "key_private.csv"):
        assert (a / name).read_bytes() == (b / name).read_bytes()


def test_sheet_rows_match_key(tmp_path, patched):
    _, sheet, key = _run(tmp_path)
    by_label = {r["label_id"]: r["question_id"] for r in key}
    for r in sheet:
        num = int(by_label[r["label_id"]][2:])
        assert r["system_answer"] == f"Ответ №{num}"
        assert r["expected_answer"] == f"Эталон №{num}"


def test_missing_mandatory_question_fails(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(mls, "MANDATORY", ("q_999",))
    eval_csv, qjson = _make_eval_data(tmp_path)
    assert mls.main(["--eval-csv", str(eval_csv), "--questions", str(qjson), "--out-dir", str(tmp_path / "o")]) == 1
    assert "обязательных" in capsys.readouterr().err


def _baseline_question_count() -> int:
    """0, если результатов прогона нет; иначе число разных вопросов baseline в eval-CSV."""
    if not list(mls.DEFAULT_EVAL_CSV.parent.glob("eval_*.csv")) or not mls.DEFAULT_EVAL_CSV.exists():
        return 0
    return len({r["question_id"] for r in _read(mls.DEFAULT_EVAL_CSV) if r["mode"] == "baseline"})


def test_real_eval_file(tmp_path):
    n = _baseline_question_count()
    if n < 34:
        pytest.skip(f"нужен полный прогон: results/eval_*.csv (не в git) с 34 вопросами baseline, найдено {n}")
    assert mls.main(["--out-dir", str(tmp_path)]) == 0
    key = _read(tmp_path / "key_private.csv")
    assert len(key) == 20
    assert {"q_20", "q_31", "q_34", "q_12"} <= {r["question_id"] for r in key}
    assert sum(r["lang"] == "kk" for r in key) >= 3
    assert sum(r["split"] == "test" for r in key) >= 6
