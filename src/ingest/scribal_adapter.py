"""Адаптер: выгрузка Scribal (transcript.json + ответ API о встрече) → наш Transcript (контракт 4.1).

    python -m src.ingest.scribal_adapter transcript.json meeting.json m_003 [--source own_asr]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from src.schemas import ScribalMeeting, ScribalTranscript, Segment, Transcript

ROOT = Path(__file__).resolve().parent.parent.parent


def seconds_to_ts(sec: float) -> str:
    s = int(sec)
    return f"{s // 3600:02d}:{s % 3600 // 60:02d}:{s % 60:02d}"


def convert(raw: ScribalTranscript, meta: ScribalMeeting, meeting_id: str,
            source: str = "scribal") -> Transcript:
    names = meta.speaker_names or {}
    segments = []
    for seg in raw.segments:
        text = seg.text.strip()
        if not text:
            continue
        label = seg.speaker or "SPEAKER_UNKNOWN"
        segments.append(Segment(
            ts=seconds_to_ts(seg.start), speaker=names.get(label, label), text=text,
            start_s=seg.start, end_s=seg.end, speaker_raw=label))
    return Transcript(
        meeting_id=meeting_id, title=meta.title, date=meta.started_at[:10],
        participants=[p.user.full_name for p in meta.participants if p.user.full_name],
        segments=segments, language=meta.language or raw.language,
        source=source, started_at=meta.started_at)


def convert_files(transcript_path: Path, meeting_path: Path, meeting_id: str,
                  source: str = "scribal") -> Transcript:
    raw = ScribalTranscript.model_validate(json.loads(transcript_path.read_text(encoding="utf-8")))
    meta = ScribalMeeting.model_validate(json.loads(meeting_path.read_text(encoding="utf-8")))
    return convert(raw, meta, meeting_id, source)


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcript")
    ap.add_argument("meeting")
    ap.add_argument("meeting_id")
    ap.add_argument("--source", default="scribal", choices=["synthetic", "own_asr", "scribal"])
    ap.add_argument("--out", default="data/transcripts")
    a = ap.parse_args(argv)
    t = convert_files(Path(a.transcript), Path(a.meeting), a.meeting_id, a.source)
    out = ROOT / a.out / f"{a.meeting_id}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(t.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")
    print(f"{a.meeting_id}: сегментов={len(t.segments)} → {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
