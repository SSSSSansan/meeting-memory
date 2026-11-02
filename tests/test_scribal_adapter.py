import json
from pathlib import Path

from src.ingest.scribal_adapter import convert_files, seconds_to_ts
from src.schemas import ScribalMeeting, ScribalTranscript

FX = Path(__file__).parent / "fixtures"


def _convert(names=None):
    meta = json.loads((FX / "scribal_meeting.json").read_text(encoding="utf-8"))
    if names:
        meta["speaker_names"] = names
    tmp = FX / "_tmp_meeting.json"
    tmp.write_text(json.dumps(meta), encoding="utf-8")
    try:
        return convert_files(FX / "scribal_transcript.json", tmp, "m_900")
    finally:
        tmp.unlink()


def test_seconds_to_ts():
    assert seconds_to_ts(0.42) == "00:00:00"
    assert seconds_to_ts(3731.5) == "01:02:11"


def test_basic_fields_and_empty_segment_skipped():
    t = _convert()
    assert t.date == "2026-10-21" and t.language == "ru" and t.source == "scribal"
    assert [s.ts for s in t.segments] == ["00:00:00", "01:02:11"]
    assert t.participants == ["Демо Пользователь"]


def test_speaker_names_null_keeps_raw_label():
    t = _convert()
    assert t.segments[0].speaker == "SPEAKER_00" == t.segments[0].speaker_raw


def test_speaker_names_mapping_applied():
    t = _convert({"SPEAKER_00": "Руслан Ким"})
    assert t.segments[0].speaker == "Руслан Ким" and t.segments[0].speaker_raw == "SPEAKER_00"
    assert t.segments[1].speaker == "SPEAKER_01"


def test_raw_models_ignore_unknown_fields():
    ScribalMeeting.model_validate(json.loads((FX / "scribal_meeting.json").read_text(encoding="utf-8")))
    ScribalTranscript.model_validate(json.loads((FX / "scribal_transcript.json").read_text(encoding="utf-8")))
