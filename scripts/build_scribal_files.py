"""Черновики встреч (SPEAKER_NN: текст) → файлы формата Scribal в data/raw/.

    python scripts/build_scribal_files.py C:\\Projects\\koz-meeting-memory\\drafts
    python scripts/build_scribal_files.py DRAFTS_DIR [--out data/raw]

Для каждого DRAFTS_DIR/m_NNN.txt, встреча которого есть в data/gold/world.yaml,
создаёт data/raw/m_NNN.transcript.json (сырой транскрипт Scribal) и
data/raw/m_NNN.meeting.json (ответ GET /api/v1/meetings/{public_id}).

Соответствие меток людям — data/gold/speaker_maps.json (лежит в gold: системе
извлечения оно не должно быть видно). В файлы не попадают ни language_profile,
ни имена говорящих, ни поле words.

Текст реплики меняется только так: подряд идущие пробелы и табы схлопываются
в один пробел, пробелы по краям обрезаются. Остальное символ в символ.
Тайминги и время начала встречи детерминированы (зерно от id встречи): повторный
запуск даёт побайтово те же файлы. Файлы не пишутся, если есть хотя бы одна ошибка.

Предупреждения (эвристика поиска лишних поручений и итогов) ничего не исправляют.
Код возврата 1, если есть ошибки.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import unicodedata
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
WORLD_PATH = ROOT / "data" / "gold" / "world.yaml"
SPEAKER_MAPS_PATH = ROOT / "data" / "gold" / "speaker_maps.json"
OUT_DIR = ROOT / "data" / "raw"

LINE_RE = re.compile(r"^(SPEAKER_\d{2}): (.*)$")
DRAFT_NAME_RE = re.compile(r"^(m_\d{3})\.txt$")
WS_RUN = re.compile(r"[ \t]+")
INNER_WS = re.compile(r"\t|  ")

# профиль языка из world.yaml → (language, detected_language) в Scribal
LANGUAGES = {"ru": ("ru", "rus"), "mixed": ("ru", "rus"), "kk": ("kk", "kaz")}

WORDS_PER_SECOND = 2.5
SPEED_SPREAD = 0.2
MIN_REPLY_SECONDS = 0.6
PAUSE = (0.2, 1.5)
SHORT_PAUSE = (0.0, 0.5)
SHORT_REPLY_WORDS = 2  # «угу», «да»: пауза перед такой репликой короткая
TZ = timezone(timedelta(hours=5))

# эвристика утечек лишних поручений и итогов; поиск без учёта регистра
WARN_FRAGMENTS = [
    "я посмотрю", "я напишу", "я скажу", "давай сделаем", "надо будет", "надо бы",
    "решили", "договорились", "договорённост", "итого", "резюмир",
]


@dataclass
class Reply:
    line_no: int
    label: str
    text: str


@dataclass
class Draft:
    meeting_id: str
    replies: list[Reply] = field(default_factory=list)
    changes: list[tuple[int, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass
class Built:
    meeting_id: str
    draft: Draft
    transcript: dict
    meeting: dict
    duration: float


def normalize_spaces(text: str) -> str:
    return WS_RUN.sub(" ", text).strip(" ")


def describe_change(raw: str) -> str:
    kinds = []
    if raw[:1] in (" ", "\t"):
        kinds.append("пробел в начале")
    if raw[-1:] in (" ", "\t"):
        kinds.append("пробел в конце")
    if INNER_WS.search(raw.strip(" \t")):
        kinds.append("двойной пробел или таб внутри")
    return ", ".join(kinds)


def parse_draft(meeting_id: str, raw: str) -> Draft:
    draft = Draft(meeting_id)
    # split по \n, а не splitlines(): в тексте могут быть другие разделители строк
    for line_no, line in enumerate(raw.split("\n"), 1):
        line = line.removesuffix("\r")
        if not line.strip():
            continue
        m = LINE_RE.match(line)
        if not m:
            draft.errors.append(
                f"{meeting_id}:{line_no}: строка не по формату 'SPEAKER_NN: текст': {line!r}"
            )
            continue
        text = normalize_spaces(m.group(2))
        if not text:
            draft.errors.append(f"{meeting_id}:{line_no}: пустая реплика у {m.group(1)}")
            continue
        if text != m.group(2):
            draft.changes.append((line_no, describe_change(m.group(2))))
        draft.replies.append(Reply(line_no, m.group(1), text))
    return draft


def validate_speakers(draft: Draft, world: dict, maps: dict) -> list[str]:
    mid = draft.meeting_id
    mapping = maps.get(mid)
    if mapping is None:
        return [f"{mid}: нет соответствия меток в speaker_maps.json"]
    errors = []
    meeting = next((m for m in world["meetings"] if m["id"] == mid), None)
    participants = set(meeting["participants"])
    people = {p["id"] for p in world["people"]}
    for label in sorted({r.label for r in draft.replies} - set(mapping)):
        lines = [str(r.line_no) for r in draft.replies if r.label == label]
        errors.append(
            f"{mid}: метки {label} нет в speaker_maps.json (строки {', '.join(lines[:5])}"
            f"{'…' if len(lines) > 5 else ''})"
        )
    for label, person in mapping.items():
        if person not in people:
            errors.append(f"{mid}: {label} → {person!r}: такого человека нет в world.yaml")
        elif person not in participants:
            errors.append(f"{mid}: {label} → {person!r} не участник встречи по world.yaml")
    return errors


def _seed(kind: str, meeting_id: str) -> bytes:
    return hashlib.sha256(f"{kind}:{meeting_id}".encode()).digest()


def make_segments(meeting_id: str, replies: list[Reply]) -> list[dict]:
    rng = random.Random(int.from_bytes(_seed("timing", meeting_id)[:8], "big"))
    t = 0.0
    segments = []
    for r in replies:
        n_words = len(r.text.split())
        pause = rng.uniform(*(SHORT_PAUSE if n_words <= SHORT_REPLY_WORDS else PAUSE))
        speed = WORDS_PER_SECOND * rng.uniform(1 - SPEED_SPREAD, 1 + SPEED_SPREAD)
        start = t + pause
        end = start + max(MIN_REPLY_SECONDS, n_words / speed)
        # округление монотонно: сегменты после него по-прежнему не пересекаются
        segments.append(
            {"start": round(start, 2), "end": round(end, 2), "speaker": r.label, "text": r.text}
        )
        t = end
    return segments


def make_started_at(meeting_id: str, day: date) -> datetime:
    h = _seed("start", meeting_id)
    return datetime(
        day.year, day.month, day.day, 9 + h[0] % 9, h[1] % 60, h[2] % 60, tzinfo=TZ
    )


def make_user(person: dict) -> dict:
    if not person.get("name"):
        raise ValueError(f"у человека {person['id']} нет имени в world.yaml")
    return {
        "full_name": person["name"],
        "email": f"{person['id']}@example.com",
        "position": person["role"],
    }


def make_meeting(world: dict, meeting_id: str, duration: float) -> dict:
    m = next(x for x in world["meetings"] if x["id"] == meeting_id)
    people = {p["id"]: p for p in world["people"]}
    language, _ = LANGUAGES[m["language_profile"]]
    started = make_started_at(meeting_id, m["date"])
    ended = started + timedelta(seconds=round(duration))
    started_iso = started.isoformat()
    users = [make_user(people[p]) for p in m["participants"]]
    return {
        "public_id": str(uuid.uuid5(uuid.NAMESPACE_URL, meeting_id)),
        "title": m["title"],
        "language": language,
        "started_at": started_iso,
        "ended_at": ended.isoformat(),
        "actual_duration_seconds": duration,
        "transcription_provider": "synthetic",
        "is_private": False,
        "speaker_names": None,
        "status": "READY",
        "has_transcript": True,
        "summary_count": 0,
        "owner": {"user": users[0]},
        "departments": [],
        "participants": [
            {"user": u, "role": "employee", "added_at": started_iso} for u in users
        ],
    }


def make_transcript(world: dict, meeting_id: str, segments: list[dict]) -> dict:
    m = next(x for x in world["meetings"] if x["id"] == meeting_id)
    language, detected = LANGUAGES[m["language_profile"]]
    return {
        "text": " ".join(s["text"] for s in segments),
        "language": language,
        "detected_language": detected,
        "segments": segments,
    }


def dump_json(obj: dict) -> bytes:
    # LF и UTF-8 независимо от ОС: файлы должны совпадать побайтово
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def find_warnings(draft: Draft) -> list[str]:
    out = []
    for r in draft.replies:
        low = r.text.lower()
        hits = [f for f in WARN_FRAGMENTS if f in low]
        if hits:
            out.append(
                f"{draft.meeting_id}:{r.line_no} [{', '.join(hits)}] {r.label}: {r.text}"
            )
    return out


def nfc_warnings(draft: Draft) -> list[str]:
    return [
        f"{draft.meeting_id}:{r.line_no} реплика не в NFC (текст оставлен как есть)"
        for r in draft.replies
        if not unicodedata.is_normalized("NFC", r.text)
    ]


def stats(world: dict, maps: dict, b: Built) -> str:
    m = next(x for x in world["meetings"] if x["id"] == b.meeting_id)
    replies = b.draft.replies
    words = [len(r.text.split()) for r in replies]
    per_speaker: dict[str, int] = {}
    for r in replies:
        per_speaker[r.label] = per_speaker.get(r.label, 0) + 1
    minutes = b.duration / 60
    target = m["target_minutes"]
    deviation = 100 * (minutes - target) / target
    short = sum(1 for w in words if w < 3)
    who = ", ".join(
        f"{label} ({maps[b.meeting_id][label]}): {n}" for label, n in sorted(per_speaker.items())
    )
    return "\n".join([
        f"== {b.meeting_id}: {m['title']} (language_profile {m['language_profile']}) ==",
        f"Реплик: {len(replies)}, слов: {sum(words)}",
        f"Сегментов по говорящим: {who}",
        f"Длительность: {minutes:.1f} мин; target_minutes {target}; отклонение {deviation:+.1f}%",
        f"Реплик короче трёх слов: {short} из {len(replies)} = {100 * short / len(replies):.1f}%",
    ])


def run(drafts_dir: Path, world_path: Path, maps_path: Path, out_dir: Path) -> int:
    world = yaml.safe_load(world_path.read_text(encoding="utf-8"))
    maps = json.loads(maps_path.read_text(encoding="utf-8"))
    world_ids = {m["id"] for m in world["meetings"]}

    errors: list[str] = []
    drafts: list[Draft] = []
    for path in sorted(drafts_dir.iterdir()):
        match = DRAFT_NAME_RE.match(path.name)
        if not match:
            print(f"Пропущен {path.name}: имя не вида m_NNN.txt (содержимое не читалось)")
            continue
        mid = match.group(1)
        if mid not in world_ids:
            errors.append(f"{path.name}: встречи {mid} нет в world.yaml")
            continue
        draft = parse_draft(mid, path.read_text(encoding="utf-8-sig"))
        errors += draft.errors
        if not draft.errors:
            errors += validate_speakers(draft, world, maps)
        drafts.append(draft)
    if not drafts and not errors:
        errors.append(f"в {drafts_dir} нет черновиков m_NNN.txt")
    if errors:
        for e in errors:
            print(f"ОШИБКА: {e}", file=sys.stderr)
        print(f"Файлы не созданы: ошибок {len(errors)}", file=sys.stderr)
        return 1

    built: list[Built] = []
    for draft in drafts:
        segments = make_segments(draft.meeting_id, draft.replies)
        duration = segments[-1]["end"]
        built.append(Built(
            draft.meeting_id, draft,
            make_transcript(world, draft.meeting_id, segments),
            make_meeting(world, draft.meeting_id, duration), duration,
        ))

    out_dir.mkdir(parents=True, exist_ok=True)
    for b in built:
        for suffix, obj in (("transcript", b.transcript), ("meeting", b.meeting)):
            path = out_dir / f"{b.meeting_id}.{suffix}.json"
            path.write_bytes(dump_json(obj))
            print(f"Записано: {path}")

    print("\n== Изменённые пробелы (схлопнуты пробелы и табы, обрезаны края) ==")
    for b in built:
        for line_no, kind in b.draft.changes:
            print(f"{b.meeting_id}: строка {line_no}: {kind}")
        if not b.draft.changes:
            print(f"{b.meeting_id}: изменений нет")

    print("\n== Предупреждения (эвристика поручений и итогов, ничего не исправлено) ==")
    warned = 0
    for b in built:
        for w in find_warnings(b.draft) + nfc_warnings(b.draft):
            print(f"ПРЕДУПРЕЖДЕНИЕ: {w}")
            warned += 1
    if not warned:
        print("нет")

    print()
    print("\n\n".join(stats(world, maps, b) for b in built))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("drafts_dir", type=Path, help="папка с m_NNN.txt")
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--world", type=Path, default=WORLD_PATH)
    ap.add_argument("--speaker-maps", type=Path, default=SPEAKER_MAPS_PATH)
    a = ap.parse_args(argv)
    if not a.drafts_dir.is_dir():
        print(f"ОШИБКА: {a.drafts_dir} — не папка", file=sys.stderr)
        return 1
    return run(a.drafts_dir, a.world, a.speaker_maps, a.out)


if __name__ == "__main__":
    sys.exit(main())
