import csv
import json
from pathlib import Path

import pytest
import yaml

from scripts import make_second_review_sheet as msr
from scripts import score_second_review as ssr

TYPE_CYCLE = list(msr.TYPES)


def _make_inputs(tmp_path: Path, n: int = 37) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    records, events = [], []
    for i in range(n):
        eid = f"ev_m{i // 3 + 1:03d}_{i % 3 + 1:02d}"
        t = TYPE_CYCLE[i % len(TYPE_CYCLE)]
        records.append({"id": eid, "type": t, "project": "zhol", "meeting": f"m_{i // 3 + 1:03d}",
                        "quote": f"цитата №{i} қазір", "start_s": float(i)})
        events.append({"id": eid, "meeting": f"m_{i // 3 + 1:03d}", "type": t, "statement": f"Формулировка №{i}"})
    dec, world = tmp_path / "decisions.json", tmp_path / "world.yaml"
    dec.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")
    world.write_text(yaml.safe_dump({"events": events}, allow_unicode=True), encoding="utf-8")
    return dec, world


def _make(tmp_path: Path, name: str = "out", n: int = 37, extra: list[str] | None = None) -> Path:
    dec, world = _make_inputs(tmp_path, n)
    out = tmp_path / name
    assert msr.main(["--decisions", str(dec), "--world", str(world), "--out-dir", str(out), *(extra or [])]) == 0
    return out


def _read(path: Path) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _fill(out: Path, wrong: set[str] = frozenset(), delimiter: str = ",") -> Path:
    """Рецензент заполняет лист вслепую; на review_id из wrong ставит другой тип, чем в эталоне."""
    sheet, key = _read(out / "sheet.csv"), {r["review_id"]: r for r in _read(out / "key_private.csv")}
    for r in sheet:
        gold = key[r["review_id"]]["type"]
        if r["review_id"] in wrong:
            r["reviewer_type"] = "non_decision" if gold != "non_decision" else "decision"
            r["reviewer_comment"] = "не уверен"
        else:
            r["reviewer_type"] = gold
    path = out / "filled.csv"
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=msr.SHEET_COLUMNS, delimiter=delimiter)
        w.writeheader()
        w.writerows(sheet)
    return path


@pytest.mark.parametrize("n,expected", [(37, 4), (20, 4), (100, 10), (101, 11), (3, 3), (0, 0)])
def test_sample_size(n, expected):
    assert msr.sample_size(n) == expected


def test_sheet_format(tmp_path):
    out = _make(tmp_path)
    assert (out / "sheet.csv").read_bytes().startswith(b"\xef\xbb\xbf")
    sheet = _read(out / "sheet.csv")
    assert len(sheet) == 4 and list(sheet[0]) == msr.SHEET_COLUMNS
    assert [r["review_id"] for r in sheet] == ["R01", "R02", "R03", "R04"]
    assert msr.SHEET_COLUMNS == ["review_id", "meeting", "quote", "reviewer_type", "reviewer_comment"]
    assert all(r["reviewer_type"] == r["reviewer_comment"] == "" for r in sheet)
    assert all("қазір" in r["quote"] for r in sheet)
    key = _read(out / "key_private.csv")
    assert list(key[0]) == msr.KEY_COLUMNS and len({r["event_id"] for r in key}) == 4
    assert all("Формулировка №" in r["statement"] and r["type"] in msr.TYPES for r in key)


def test_sheet_is_blind(tmp_path):
    """Ни типа эталона, ни формулировки, ни событий в листе нет: ни в колонках, ни в тексте файла."""
    out = _make(tmp_path)
    assert not {"type", "statement", "reviewer_agrees", "event_id"} & set(_read(out / "sheet.csv")[0])
    text = (out / "sheet.csv").read_text(encoding="utf-8-sig")
    assert "Формулировка" not in text and "ev_m" not in text
    assert "non_decision" not in text and "status_update" not in text


def test_readme_instruction(tmp_path):
    out = _make(tmp_path)
    text = (out / "README.md").read_text(encoding="utf-8")
    assert "what_is_a_decision.md" in text and "reviewer_type" in text
    for t in msr.TYPES:
        assert t in text


def test_deterministic_and_seed_dependent(tmp_path):
    a, b = _make(tmp_path / "a"), _make(tmp_path / "b")
    assert (a / "sheet.csv").read_bytes() == (b / "sheet.csv").read_bytes()
    other = _make(tmp_path / "c", extra=["--seed", "7"])
    assert (a / "key_private.csv").read_bytes() != (other / "key_private.csv").read_bytes()


def test_missing_statement_fails(tmp_path, capsys):
    dec, world = _make_inputs(tmp_path)
    data = yaml.safe_load(world.read_text(encoding="utf-8"))
    data["events"].pop()
    world.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    assert msr.main(["--decisions", str(dec), "--world", str(world), "--out-dir", str(tmp_path / "o")]) == 1
    assert "нет формулировки" in capsys.readouterr().err


def test_score_full_agreement(tmp_path, capsys):
    out = _make(tmp_path)
    rep = out / "agreement.md"
    assert ssr.main(["--sheet", str(_fill(out)), "--key", str(out / "key_private.csv"), "--out", str(rep)]) == 0
    assert "совпадение: 4 из 4 = 100,0%" in capsys.readouterr().out
    assert "Расхождений: 0" in rep.read_text(encoding="utf-8")


def test_score_lists_disagreements(tmp_path, capsys):
    out = _make(tmp_path)
    rep = out / "agreement.md"
    assert ssr.main(["--sheet", str(_fill(out, wrong={"R02"})), "--key", str(out / "key_private.csv"),
                     "--out", str(rep)]) == 0
    assert "совпадение: 3 из 4 = 75,0%" in capsys.readouterr().out
    text = rep.read_text(encoding="utf-8")
    key = {r["review_id"]: r for r in _read(out / "key_private.csv")}
    assert "## Расхождения" in text and "| R02 |" in text and "Формулировка №" in text
    row = next(line for line in text.splitlines() if line.startswith("| R02 |"))
    assert f"| {key['R02']['type']} | non_decision |" in row or f"| {key['R02']['type']} | decision |" in row
    assert "| R01 |" not in text and "не уверен" in text


def test_score_semicolon_delimiter_from_excel(tmp_path, capsys):
    out = _make(tmp_path)
    sheet = _fill(out, delimiter=";")
    assert ssr.main(["--sheet", str(sheet), "--key", str(out / "key_private.csv"), "--out", str(out / "a.md")]) == 0
    assert "4 из 4" in capsys.readouterr().out


def test_score_skips_unfilled_rows_but_fails_when_all_empty(tmp_path, capsys):
    out = _make(tmp_path)
    assert ssr.main(["--sheet", str(out / "sheet.csv"), "--key", str(out / "key_private.csv"),
                     "--out", str(out / "a.md")]) == 1
    assert "нет заполненных" in capsys.readouterr().err
    rows = _read(_fill(out))
    rows[0]["reviewer_type"] = ""
    res = ssr.score(rows, _read(out / "key_private.csv"))
    assert res["filled"] == 3 and res["empty"] == ["R01"] and res["agreement"] == 1.0


def test_score_rejects_bad_values(tmp_path):
    out = _make(tmp_path)
    key = _read(out / "key_private.csv")
    rows = _read(_fill(out))
    rows[0]["reviewer_type"] = "решение"
    with pytest.raises(ssr.ScoreError, match="reviewer_type"):
        ssr.score(rows, key)
    rows = _read(_fill(out))
    rows[2]["review_id"] = "R99"
    with pytest.raises(ssr.ScoreError, match="R99"):
        ssr.score(rows, key)


def test_score_ignores_case_and_spaces(tmp_path):
    out = _make(tmp_path)
    rows = _read(_fill(out))
    rows[0]["reviewer_type"] = f"  {rows[0]['reviewer_type'].upper()} "
    assert ssr.score(rows, _read(out / "key_private.csv"))["agreement"] == 1.0


def test_real_gold_sheet(tmp_path):
    """На реальном эталоне: 37 записей → 4 строки, у каждой есть формулировка из world.yaml."""
    assert msr.main(["--out-dir", str(tmp_path)]) == 0
    sheet = _read(tmp_path / "sheet.csv")
    n = len(json.loads(msr.DEFAULT_DECISIONS.read_text(encoding="utf-8")))
    assert len(sheet) == msr.sample_size(n) >= 4
    assert all(r["quote"] for r in sheet)
    assert all(r["statement"] and r["type"] in msr.TYPES for r in _read(tmp_path / "key_private.csv"))
