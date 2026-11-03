import json
import re
import uuid
from datetime import datetime
from pathlib import Path

import pytest

from scripts.build_scribal_files import (
    OUT_DIR,
    SPEAKER_MAPS_PATH,
    main,
    parse_draft,
)
from src.ingest.scribal_adapter import convert_files
from src.schemas import ScribalMeeting, ScribalTranscript, Transcript

# CRLF, пустые строки, пробел в начале и в конце, таб и двойной пробел внутри:
# всё, что скрипт обязан нормализовать, и реплика с фрагментом для предупреждения
DRAFT = (
    "SPEAKER_00: Начинаем, всем привет.\r\n"
    "\r\n"
    "SPEAKER_01: Привет.\n"
    "SPEAKER_02: Угу\n"
    "SPEAKER_03:  Релиз назначили на пятнадцатое. \n"
    "SPEAKER_00: Это  надо\tобсудить подробнее, потому что срок короткий.\n"
    "   \n"
    "SPEAKER_01: Мы же договорились про миграцию, Нұралы қайта қарайық.\n"
    "SPEAKER_02: Да\n"
    "SPEAKER_03: Хорошо, тогда я возьму интеграцию и тесты на себя до пятницы.\n"
    "SPEAKER_00: Ок.\n"
)


def build(tmp_path: Path, draft: str = DRAFT, name: str = "m_001.txt",
          maps: Path = SPEAKER_MAPS_PATH, out: str = "out") -> int:
    drafts = tmp_path / "drafts"
    drafts.mkdir(exist_ok=True)
    (drafts / name).write_bytes(draft.encode("utf-8"))
    return main([str(drafts), "--out", str(tmp_path / out), "--speaker-maps", str(maps)])


def expected_texts(draft: str) -> list[str]:
    texts = []
    for line in draft.split("\n"):
        line = line.removesuffix("\r")
        if not line.strip():
            continue
        text = line.split(": ", 1)[1]
        texts.append(re.sub(r"[ \t]+", " ", text).strip(" "))
    return texts


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_deterministic_two_runs_same_bytes(tmp_path):
    assert build(tmp_path, out="a") == 0
    assert build(tmp_path, out="b") == 0
    assert build(tmp_path, out="a") == 0  # повтор в ту же папку
    for suffix in ("transcript", "meeting"):
        first = (tmp_path / "a" / f"m_001.{suffix}.json").read_bytes()
        assert first == (tmp_path / "b" / f"m_001.{suffix}.json").read_bytes()
    assert b"\r" not in first


def test_segments_ordered_and_not_overlapping(tmp_path):
    assert build(tmp_path) == 0
    segs = load(tmp_path / "out" / "m_001.transcript.json")["segments"]
    prev_end = 0.0
    for s in segs:
        assert s["start"] >= prev_end
        assert s["end"] > s["start"]
        assert s["end"] - s["start"] >= 0.59
        assert round(s["start"], 2) == s["start"] and round(s["end"], 2) == s["end"]
        prev_end = s["end"]


def test_joined_text_matches_draft_after_space_normalization(tmp_path):
    assert build(tmp_path) == 0
    raw = load(tmp_path / "out" / "m_001.transcript.json")
    expected = expected_texts(DRAFT)
    assert [s["text"] for s in raw["segments"]] == expected
    assert raw["text"] == " ".join(expected)
    # казахские буквы не тронуты
    assert "Нұралы қайта қарайық" in raw["text"]


def test_all_labels_are_in_speaker_map(tmp_path):
    assert build(tmp_path) == 0
    mapping = load(SPEAKER_MAPS_PATH)["m_001"]
    segs = load(tmp_path / "out" / "m_001.transcript.json")["segments"]
    assert {s["speaker"] for s in segs} <= set(mapping)


def test_output_has_no_extra_fields(tmp_path):
    assert build(tmp_path) == 0
    raw = load(tmp_path / "out" / "m_001.transcript.json")
    assert set(raw) == {"text", "language", "detected_language", "segments"}
    assert all(set(s) == {"start", "end", "speaker", "text"} for s in raw["segments"])
    assert "language_profile" not in (tmp_path / "out" / "m_001.meeting.json").read_text("utf-8")


def test_meeting_json_fields(tmp_path):
    assert build(tmp_path) == 0
    raw = load(tmp_path / "out" / "m_001.transcript.json")
    meta = load(tmp_path / "out" / "m_001.meeting.json")
    assert meta["public_id"] == str(uuid.uuid5(uuid.NAMESPACE_URL, "m_001"))
    assert meta["title"] == "Zhol: старт спринта"
    assert (meta["language"], raw["language"], raw["detected_language"]) == ("ru", "ru", "rus")
    assert meta["transcription_provider"] == "synthetic"
    assert meta["is_private"] is False
    assert meta["speaker_names"] is None
    assert meta["status"] == "READY"
    assert meta["has_transcript"] is True
    assert meta["summary_count"] == 0
    assert meta["departments"] == []
    assert meta["actual_duration_seconds"] == raw["segments"][-1]["end"]
    started = datetime.fromisoformat(meta["started_at"])
    ended = datetime.fromisoformat(meta["ended_at"])
    assert meta["started_at"].endswith("+05:00") and str(started.date()) == "2026-08-03"
    assert 9 <= started.hour <= 17
    assert abs((ended - started).total_seconds() - meta["actual_duration_seconds"]) <= 0.5
    names = [p["user"]["full_name"] for p in meta["participants"]]
    assert names == ["Амина", "Нұралы", "Николай", "Абдурахман"]
    assert meta["owner"]["user"]["full_name"] == "Амина"
    assert {p["role"] for p in meta["participants"]} == {"employee"}
    assert meta["participants"][0]["user"]["email"] == "amina@example.com"


def test_kk_meeting_gets_kaz_language(tmp_path):
    # m_005 — kk в world.yaml; метки берём из временной копии соответствия
    maps = tmp_path / "maps.json"
    maps.write_text(json.dumps({"m_005": {"SPEAKER_00": "amina", "SPEAKER_01": "nuraly"}}))
    assert build(tmp_path, "SPEAKER_00: Сәлем.\nSPEAKER_01: Сәлем!\n", "m_005.txt", maps) == 0
    raw = load(tmp_path / "out" / "m_005.transcript.json")
    assert (raw["language"], raw["detected_language"]) == ("kk", "kaz")


def test_pydantic_and_scribal_adapter_accept_output(tmp_path):
    assert build(tmp_path) == 0
    t_path = tmp_path / "out" / "m_001.transcript.json"
    m_path = tmp_path / "out" / "m_001.meeting.json"
    ScribalTranscript.model_validate(load(t_path))
    ScribalMeeting.model_validate(load(m_path))

    result = convert_files(t_path, m_path, "m_001", source="synthetic")

    assert isinstance(result, Transcript)
    assert result.meeting_id == "m_001"
    assert result.date == "2026-08-03"
    assert result.participants == ["Амина", "Нұралы", "Николай", "Абдурахман"]
    assert len(result.segments) == len(expected_texts(DRAFT))
    # speaker_names=null: имена не подставляются, остаются исходные метки
    assert [s.speaker_raw for s in result.segments][:2] == ["SPEAKER_00", "SPEAKER_01"]
    assert result.segments[0].speaker == "SPEAKER_00"


@pytest.mark.parametrize("meeting_id", ["m_001", "m_002"])
def test_committed_raw_files_accepted_by_adapter(meeting_id):
    t_path = OUT_DIR / f"{meeting_id}.transcript.json"
    m_path = OUT_DIR / f"{meeting_id}.meeting.json"
    if not (t_path.exists() and m_path.exists()):
        pytest.skip("data/raw/ не заполнен")
    result = convert_files(t_path, m_path, meeting_id, source="synthetic")
    raw = load(t_path)
    assert len(result.segments) == len(raw["segments"])
    assert {s["speaker"] for s in raw["segments"]} <= set(load(SPEAKER_MAPS_PATH)[meeting_id])


def test_whitespace_changes_are_reported(tmp_path, capsys):
    assert build(tmp_path) == 0
    out = capsys.readouterr().out
    assert "m_001: строка 5: пробел в начале, пробел в конце" in out
    assert "m_001: строка 6: двойной пробел или таб внутри" in out


def test_phrase_warning_is_printed_not_fatal(tmp_path, capsys):
    assert build(tmp_path) == 0
    out = capsys.readouterr().out
    assert "ПРЕДУПРЕЖДЕНИЕ: m_001:8 [договорились]" in out
    assert "Реплик короче трёх слов: 4 из 9" in out


def test_parse_draft_collapses_only_spaces_and_tabs():
    draft = parse_draft("m_001", "SPEAKER_00:  a \t b  \nSPEAKER_01: x y\n")
    assert [r.text for r in draft.replies] == ["a b", "x y"]  # nbsp не трогаем
    assert draft.errors == []


def test_unknown_label_fails_and_writes_nothing(tmp_path, capsys):
    code = build(tmp_path, DRAFT + "SPEAKER_09: Кто это?\n")
    assert code == 1
    assert "SPEAKER_09" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_bad_line_format_reports_line_number(tmp_path, capsys):
    code = build(tmp_path, "SPEAKER_00: Привет.\n\nСпикер ноль: привет\n")
    assert code == 1
    err = capsys.readouterr().err
    assert "m_001:3" in err and "Спикер ноль: привет" in err
    assert not (tmp_path / "out").exists()


def test_label_without_space_after_colon_fails(tmp_path, capsys):
    assert build(tmp_path, "SPEAKER_00:Привет.\n") == 1
    assert "m_001:1" in capsys.readouterr().err


def test_person_who_is_not_participant_fails(tmp_path, capsys):
    maps = tmp_path / "maps.json"
    mapping = load(SPEAKER_MAPS_PATH)
    mapping["m_001"]["SPEAKER_00"] = "daniyar"  # на m_001 его нет
    maps.write_text(json.dumps(mapping), encoding="utf-8")
    assert build(tmp_path, maps=maps) == 1
    assert "daniyar" in capsys.readouterr().err


def test_meeting_missing_in_world_fails(tmp_path, capsys):
    assert build(tmp_path, name="m_099.txt") == 1
    assert "m_099" in capsys.readouterr().err


def test_non_draft_files_are_skipped_unread(tmp_path, capsys):
    drafts = tmp_path / "drafts"
    drafts.mkdir()
    (drafts / "real_sample.txt").write_bytes(b"\xff\xfe not utf-8 at all")
    assert build(tmp_path) == 0  # не падает на нечитаемом постороннем файле
    assert "real_sample.txt" in capsys.readouterr().out
