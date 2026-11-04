import json
from pathlib import Path

import pytest
import yaml

from scripts.build_gold import build, normalize_quote, write_templates

SEGMENTS = [
    {"start": 0.5, "end": 3.0, "speaker": "SPEAKER_00", "text": "Всем привет. [шум] Релиз назначаем"},
    {"start": 4.0, "end": 8.0, "speaker": "SPEAKER_01", "text": "на пятнадцатое сентября."},
    {"start": 9.0, "end": 12.0, "speaker": "SPEAKER_02", "text": "Нұралы сделает миграцию"},
]

WORLD = {
    "events": [
        {"id": "ev_m001_01", "meeting": "m_001", "type": "decision", "project": "zhol",
         "statement": "Релиз Zhol 15 сентября"},
        {"id": "ev_m001_02", "meeting": "m_001", "type": "task", "project": "zhol",
         "statement": "Миграция", "owner": "nuraly", "due": "2026-08-14"},
        {"id": "ev_m002_01", "meeting": "m_002", "type": "decision", "project": "qamqor",
         "statement": "Готовая модель"},
    ]
}


@pytest.fixture
def dirs(tmp_path: Path) -> tuple[Path, Path]:
    raw, quotes = tmp_path / "raw", tmp_path / "quotes"
    raw.mkdir()
    quotes.mkdir()
    (raw / "m_001.transcript.json").write_text(
        json.dumps({"segments": SEGMENTS}, ensure_ascii=False), encoding="utf-8"
    )
    return raw, quotes


def _write_quotes(quotes: Path, mid: str, mapping: dict[str, str]) -> None:
    doc = {"meeting": mid, "events": [{"id": k, "type": "x", "quote": v} for k, v in mapping.items()]}
    (quotes / f"{mid}.yaml").write_text(
        yaml.safe_dump(doc, allow_unicode=True), encoding="utf-8"
    )


def test_normalize_case_spaces_and_noise():
    assert normalize_quote("  Всем   ПРИВЕТ\n[шум] [Смех]  ребята ") == "всем привет ребята"


def test_normalize_kazakh_letters():
    assert normalize_quote("Қамқор Нұралы Сандуғаш Әнуар Өмір Үй Һәм Құрмет Ақпарат Қазақ") == (
        "камкор нуралы сандугаш ануар омир уй хам курмет акпарат казак"
    )
    assert normalize_quote("Айгерім") == normalize_quote("Айгерим")


def test_normalize_nfc():
    decomposed = "й"  # и + combining breve
    assert normalize_quote(decomposed) == "й"


def test_empty_quote_is_error(dirs):
    raw, quotes = dirs
    _write_quotes(quotes, "m_001", {"ev_m001_01": "  ", "ev_m001_02": "Нуралы сделает"})
    records, problems, _ = build(WORLD, quotes, raw)
    assert [r["id"] for r in records] == ["ev_m001_02"]
    assert len(problems) == 1
    assert "ev_m001_01" in problems[0] and "m_001" in problems[0] and "пустая" in problems[0]


def test_invented_quote_is_error(dirs):
    raw, quotes = dirs
    _write_quotes(
        quotes, "m_001",
        {"ev_m001_01": "Релиз назначаем на первое октября", "ev_m001_02": "нуралы сделает"},
    )
    _, problems, _ = build(WORLD, quotes, raw)
    assert len(problems) == 1
    assert "ev_m001_01" in problems[0] and "не найдена" in problems[0]


def test_valid_quotes_are_built(dirs):
    raw, quotes = dirs
    _write_quotes(
        quotes, "m_001",
        {
            "ev_m001_01": "всем привет. релиз назначаем на пятнадцатое сентября",  # через два сегмента
            "ev_m001_02": "Нуралы СДЕЛАЕТ миграцию",  # казахская буква, регистр
        },
    )
    records, problems, warns = build(WORLD, quotes, raw)
    assert problems == []
    first, second = records
    assert (first["id"], first["start_s"], first["project"], first["meeting"]) == (
        "ev_m001_01", 0.5, "zhol", "m_001",
    )
    assert "owner" not in first and "due" not in first
    assert (second["start_s"], second["owner"], second["due"]) == (9.0, "nuraly", "2026-08-14")
    # цитата, начавшаяся во втором сегменте, получает время второго сегмента
    _write_quotes(quotes, "m_001", {"ev_m001_01": "на пятнадцатое сентября", "ev_m001_02": "миграцию"})
    records, _, _ = build(WORLD, quotes, raw)
    assert records[0]["start_s"] == 4.0
    assert warns == ["ev_m002_01 (встреча m_002): нет файла цитат"]


def test_template_lists_events_and_keeps_existing(tmp_path: Path):
    quotes = tmp_path / "quotes"
    created, notes = write_templates(WORLD, ["m_001"], quotes)
    assert len(created) == 1 and notes == []
    doc = yaml.safe_load((quotes / "m_001.yaml").read_text(encoding="utf-8"))
    assert [e["id"] for e in doc["events"]] == ["ev_m001_01", "ev_m001_02"]
    assert all(e["quote"] == "" for e in doc["events"])

    (quotes / "m_001.yaml").write_text("keep", encoding="utf-8")
    created, notes = write_templates(WORLD, ["m_001"], quotes)
    assert created == [] and "не перезаписан" in notes[0]
    assert (quotes / "m_001.yaml").read_text(encoding="utf-8") == "keep"
