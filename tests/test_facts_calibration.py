"""Лист разметки фактов и подсчёт согласия с судьёй (A8) на заглушках, без модели."""
import csv
import json
from pathlib import Path

import pytest

from scripts import make_facts_sheet as mfs
from scripts import score_facts_calibration as sfc

TYPES = ["T1", "T2", "T3", "T4", "T5"]


def _fixture(tmp_path: Path, n_answerable: int = 31, n_none: int = 3) -> tuple[Path, Path]:
    questions, per_q = [], []
    for i in range(n_answerable + n_none):
        qid = f"q_{i + 1:02d}"
        answerable = i >= n_none
        t = TYPES[i % len(TYPES)] if answerable else "none"
        req = [f"факт {i}-{k}" for k in range(1 + i % 3)] if answerable else ["в данных нет ответа"]
        forb = [f"запрет {i}-{k}" for k in range(2)]
        questions.append({"id": qid, "type": t, "question": f"Вопрос номер {i}?", "answerable": answerable,
                          "required_facts": req, "forbidden_facts": forb})
        verdict = {"required": [{"fact": f, "present": k % 2 == 0} for k, f in enumerate(req)],
                   "forbidden": [{"fact": f, "asserted": k == 1 and i % 4 == 0} for k, f in enumerate(forb)],
                   "claims": []}
        base = {"question_id": qid, "type": t, "repeat": 1, "status": "ok", "answerable": answerable,
                "answer": f"Ответ номер {i}", "judge_facts": verdict, "metrics": {"fact_precision": 0.5}}
        per_q.append({**base, "mode": "baseline"})
        per_q.append({**base, "mode": "graph", "answer": "другой режим"})
    per_q.append({**per_q[-2], "repeat": 2, "answer": "второй повтор"})
    ev, qs = tmp_path / "eval.json", tmp_path / "questions.json"
    ev.write_text(json.dumps({"per_question": per_q}, ensure_ascii=False), encoding="utf-8")
    qs.write_text(json.dumps(questions, ensure_ascii=False), encoding="utf-8")
    return ev, qs


def _read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _make(tmp_path: Path, name: str = "out") -> Path:
    ev, qs = _fixture(tmp_path)
    out = tmp_path / name
    assert mfs.main(["--eval-json", str(ev), "--questions", str(qs), "--out-dir", str(out)]) == 0
    return out


def test_sheet_selection_and_columns(tmp_path):
    out = _make(tmp_path)
    sheet, key = _read(out / "sheet.csv"), _read(out / "key_private.csv")
    assert list(sheet[0]) == mfs.SHEET_COLUMNS
    assert list(key[0]) == mfs.KEY_COLUMNS
    assert (out / "sheet.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    labels = {r["label_id"]: r for r in key}
    assert len(labels) == 30
    qids = {r["question_id"] for r in key}
    assert {"q_01", "q_02", "q_03"} <= qids  # все answerable=false
    types = [r["type"] for r in labels.values()]
    assert {t: types.count(t) for t in TYPES} == {t: 5 if t != "T5" else 6 for t in TYPES} or \
        max(types.count(t) for t in TYPES) - min(types.count(t) for t in TYPES) <= 1
    assert all(r["human"] == "" and r["comment"] == "" for r in sheet)
    assert {r["fact_kind"] for r in sheet} == {"required", "forbidden"}
    assert all("другой режим" not in r["system_answer"] and "второй повтор" not in r["system_answer"] for r in sheet)
    assert [r["row_id"] for r in sheet] == [r["row_id"] for r in key]
    assert {r["judge_value"] for r in key} == {"0", "1"}


def test_sheet_hides_judge_type_and_qid(tmp_path):
    out = _make(tmp_path)
    text = (out / "sheet.csv").read_text(encoding="utf-8-sig")
    for leak in ("judge", "question_id", "q_0", "T1", "none", "present", "asserted"):
        assert leak not in text


def test_sheet_deterministic(tmp_path):
    a, b = _make(tmp_path, "a"), _make(tmp_path, "b")
    assert (a / "sheet.csv").read_bytes() == (b / "sheet.csv").read_bytes()
    assert (a / "key_private.csv").read_bytes() == (b / "key_private.csv").read_bytes()


def test_sheet_rejects_facts_changed_after_run(tmp_path):
    ev, qs = _fixture(tmp_path)
    data = json.loads(qs.read_text(encoding="utf-8"))
    for q in data:
        q["required_facts"] = q["required_facts"] + ["новый факт"]
    qs.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    assert mfs.main(["--eval-json", str(ev), "--questions", str(qs), "--out-dir", str(tmp_path / "o")]) == 1


def test_sheet_too_few_answers(tmp_path):
    ev, qs = _fixture(tmp_path, n_answerable=10)
    assert mfs.main(["--eval-json", str(ev), "--questions", str(qs), "--out-dir", str(tmp_path / "o")]) == 1


def _fill(out: Path, flip: set[int], delimiter: str = ",", blank: set[int] = frozenset()) -> Path:
    sheet = _read(out / "sheet.csv")
    key = {r["row_id"]: r for r in _read(out / "key_private.csv")}
    for i, r in enumerate(sheet):
        v = int(key[r["row_id"]]["judge_value"])
        r["human"] = "" if i in blank else str(1 - v if i in flip else v)
    path = out / "filled.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mfs.SHEET_COLUMNS, delimiter=delimiter)
        w.writeheader()
        w.writerows(sheet)
    return path


def _score(tmp_path: Path, out: Path, sheet: Path) -> int:
    ev = tmp_path / "eval.json"
    return sfc.main(["--sheet", str(sheet), "--key", str(out / "key_private.csv"), "--eval-json", str(ev),
                     "--out", str(out / "agreement.md")])


def test_score_perfect_agreement(tmp_path):
    out = _make(tmp_path)
    assert _score(tmp_path, out, _fill(out, set())) == 0
    md = (out / "agreement.md").read_text(encoding="utf-8")
    assert "100.0%" in md and "1.000" in md
    assert "Расхождения (0)" in md
    assert "разница (судья − человек) **+0.000**" in md


def test_score_flips_counted(tmp_path):
    out = _make(tmp_path)
    sheet = _fill(out, {0, 1, 2}, delimiter=";")
    res = sfc.score(sfc.read_csv(sheet), sfc.read_csv(out / "key_private.csv"),
                    sfc.load_precision(tmp_path / "eval.json"))
    assert res["overall"]["same"] == res["filled"] - 3
    assert len(res["mismatches"]) == 3
    assert sum(sum(row) for row in res["overall"]["matrix"]) == res["filled"]
    assert res["by_kind"]["required"]["n"] + res["by_kind"]["forbidden"]["n"] == res["filled"]
    assert res["f1"]["n"] == 30 and res["f1"]["n_differ"] >= 1
    assert res["overall"]["kappa"] is not None and res["overall"]["kappa"] < 1


def test_score_f1_uses_judge_precision(tmp_path):
    out = _make(tmp_path)
    res = sfc.score(sfc.read_csv(_fill(out, set())), sfc.read_csv(out / "key_private.csv"), {})
    assert res["f1"]["n"] == 0  # без precision из прогона f1 не считается
    res = sfc.score(sfc.read_csv(_fill(out, set())), sfc.read_csv(out / "key_private.csv"),
                    sfc.load_precision(tmp_path / "eval.json"))
    item = res["f1"]["items"][0]
    assert item["precision"] == 0.5
    assert item["judge_f1"] == pytest.approx(sfc.harmonic(item["judge_recall"], 0.5))


def test_score_partial_and_errors(tmp_path):
    out = _make(tmp_path)
    blank = set(range(10, 200))
    res = sfc.score(sfc.read_csv(_fill(out, set(), blank=blank)), sfc.read_csv(out / "key_private.csv"),
                    sfc.load_precision(tmp_path / "eval.json"))
    assert res["filled"] == 10 and res["f1"]["skipped"]
    assert _score(tmp_path, out, out / "sheet.csv") == 1  # ничего не размечено
    bad = _read(out / "sheet.csv")
    bad[0]["human"] = "да"
    path = out / "bad.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=mfs.SHEET_COLUMNS)
        w.writeheader()
        w.writerows(bad)
    assert _score(tmp_path, out, path) == 1


def test_kappa_undefined_single_label():
    rows = [{"human": 0, "judge": 0}] * 3
    assert sfc.agreement_block(rows)["kappa"] is None
    assert sfc.agreement_block(rows)["agreement"] == 1.0
