"""Сборка эталона решений data/gold/decisions.json из world.yaml и цитат.

    python scripts/build_gold.py --template m_001 m_002   # заготовки цитат
    python scripts/build_gold.py --build                  # data/gold/decisions.json

--template: для каждой встречи создаёт data/gold/quotes/<id>.yaml со списком всех
событий встречи из world.yaml и пустым полем quote. Текст встречи в файл не
копируется, существующие файлы не перезаписываются. Цитату вписывает человек,
слушая запись.

--build: читает data/gold/quotes/*.yaml и проверяет, что каждая цитата найдена
в тексте встречи (сегменты data/raw/<id>.transcript.json, склеенные через пробел).
Обе стороны перед сравнением проходят normalize_quote. Если у какого-то события
цитата пустая или не найдена, печатает список проблем, код возврата 1, ничего
не записывается. События из world.yaml, у которых нет файла цитат, — только
предупреждение.

Эталон лежит только в data/gold и на вход системы не попадает.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from bisect import bisect_right
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
WORLD_PATH = ROOT / "data" / "gold" / "world.yaml"
QUOTES_DIR = ROOT / "data" / "gold" / "quotes"
RAW_DIR = ROOT / "data" / "raw"
OUT_PATH = ROOT / "data" / "gold" / "decisions.json"

NOISE_RE = re.compile(r"\[[^\]]*\]")
SPACES_RE = re.compile(r"\s+")
# Казахские буквы сводим к русским/латинским: ASR и человек пишут их по-разному.
KK_FOLD = {
    ord("қ"): "к", ord("ғ"): "г", ord("ә"): "а", ord("ң"): "н", ord("ө"): "о",
    ord("ұ"): "у", ord("ү"): "у", ord("һ"): "х", ord("і"): "и",
}


def normalize_quote(text: str) -> str:
    """NFC, нижний регистр, без [шума], казахские буквы сведены, пробелы схлопнуты."""
    text = unicodedata.normalize("NFC", text).lower()
    text = NOISE_RE.sub(" ", text)
    text = text.translate(KK_FOLD)
    return SPACES_RE.sub(" ", text).strip()


def load_world(path: Path = WORLD_PATH) -> dict:
    with path.open(encoding="utf-8") as f:
        world = yaml.safe_load(f)
    if not isinstance(world, dict):
        raise TypeError(f"{path}: верхний уровень должен быть словарём")
    return world


def meeting_number(meeting_id: str) -> int:
    match = re.fullmatch(r"m_(\d+)", meeting_id)
    return int(match.group(1)) if match else 10**9


def load_segments(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return json.load(f).get("segments") or []


class MeetingText:
    """Нормализованный текст встречи с привязкой позиций к началу сегмента."""

    def __init__(self, segments: list[dict]) -> None:
        parts: list[str] = []
        self._starts_at: list[int] = []
        self._segment_start_s: list[float] = []
        pos = 0
        for seg in segments:
            norm = normalize_quote(str(seg.get("text") or ""))
            if not norm:
                continue
            self._starts_at.append(pos)
            self._segment_start_s.append(float(seg.get("start") or 0.0))
            parts.append(norm)
            pos += len(norm) + 1
        self.text = " ".join(parts)

    def find(self, quote: str) -> float | None:
        """Время начала сегмента, где цитата начинается; None, если не найдена."""
        needle = normalize_quote(quote)
        if not needle:
            return None
        idx = self.text.find(needle)
        if idx < 0:
            return None
        return self._segment_start_s[bisect_right(self._starts_at, idx) - 1]


def events_by_meeting(world: dict) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for e in world.get("events") or []:
        out.setdefault(str(e.get("meeting")), []).append(e)
    return out


def write_templates(
    world: dict, meeting_ids: list[str], quotes_dir: Path = QUOTES_DIR
) -> tuple[list[str], list[str]]:
    """Возвращает (созданные файлы, сообщения о пропусках и ошибках)."""
    by_meeting = events_by_meeting(world)
    created: list[str] = []
    notes: list[str] = []
    for mid in meeting_ids:
        events = by_meeting.get(mid)
        if not events:
            notes.append(f"{mid}: в world.yaml нет событий, файл не создан")
            continue
        path = quotes_dir / f"{mid}.yaml"
        if path.exists():
            notes.append(f"{mid}: {path.name} уже существует, не перезаписан")
            continue
        doc = {
            "meeting": mid,
            "events": [
                {
                    "id": e["id"],
                    "type": e["type"],
                    "description": e.get("statement"),
                    "quote": "",
                }
                for e in events
            ],
        }
        header = (
            f"# Цитаты эталона для {mid}: дословный фрагмент из транскрипта в поле quote.\n"
            "# Проверка и сборка: python scripts/build_gold.py --build\n"
        )
        quotes_dir.mkdir(parents=True, exist_ok=True)
        body = yaml.safe_dump(doc, allow_unicode=True, sort_keys=False, width=100)
        path.write_text(header + body, encoding="utf-8")
        created.append(str(path))
    return created, notes


def _load_quote_file(path: Path) -> tuple[dict[str, str], list[str]]:
    """Возвращает ({event_id: quote}, ошибки формата файла)."""
    try:
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as e:
        return {}, [f"{path.name}: не удалось прочитать файл: {e}"]
    items = doc.get("events") if isinstance(doc, dict) else None
    if not isinstance(items, list):
        return {}, [f"{path.name}: нет списка events"]
    quotes: dict[str, str] = {}
    errors: list[str] = []
    for item in items:
        if not isinstance(item, dict) or "id" not in item:
            errors.append(f"{path.name}: запись без id: {item!r}")
            continue
        quotes[str(item["id"])] = str(item.get("quote") or "")
    return quotes, errors


def build(
    world: dict,
    quotes_dir: Path = QUOTES_DIR,
    raw_dir: Path = RAW_DIR,
) -> tuple[list[dict], list[str], list[str]]:
    """Возвращает (записи эталона, проблемы, предупреждения)."""
    by_meeting = events_by_meeting(world)
    files = sorted(quotes_dir.glob("*.yaml"), key=lambda p: meeting_number(p.stem))
    problems: list[str] = []
    records: list[dict] = []
    if not files:
        problems.append(f"нет файлов цитат в {quotes_dir}")

    covered: set[str] = set()
    for path in files:
        mid = path.stem
        covered.add(mid)
        quotes, errors = _load_quote_file(path)
        problems.extend(errors)
        events = by_meeting.get(mid)
        if not events:
            problems.append(f"встреча {mid}: в world.yaml нет событий")
            continue
        transcript = raw_dir / f"{mid}.transcript.json"
        if not transcript.exists():
            problems.append(f"встреча {mid}: нет транскрипта {transcript}")
            continue
        text = MeetingText(load_segments(transcript))

        known = {e["id"] for e in events}
        for eid in sorted(set(quotes) - known):
            problems.append(f"{eid}, встреча {mid}: события нет в world.yaml")
        for e in events:
            eid = e["id"]
            if eid not in quotes:
                problems.append(f"{eid}, встреча {mid}: нет записи в файле цитат")
                continue
            quote = quotes[eid]
            if not normalize_quote(quote):
                problems.append(f"{eid}, встреча {mid}: цитата пустая")
                continue
            start_s = text.find(quote)
            if start_s is None:
                problems.append(f"{eid}, встреча {mid}: цитата не найдена в тексте встречи")
                continue
            rec = {"id": eid, "type": e["type"], "project": e.get("project"), "meeting": mid}
            if e.get("owner") is not None:
                rec["owner"] = e["owner"]
            if e.get("due") is not None:
                rec["due"] = str(e["due"])
            rec["quote"] = quote.strip()
            rec["start_s"] = start_s
            records.append(rec)

    records.sort(key=lambda r: meeting_number(r["meeting"]))
    warns = [
        f"{e['id']} (встреча {mid}): нет файла цитат"
        for mid in sorted(by_meeting, key=meeting_number)
        if mid not in covered
        for e in by_meeting[mid]
    ]
    return records, problems, warns


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--template", nargs="+", metavar="ID", help="создать заготовки цитат")
    mode.add_argument("--build", action="store_true", help="собрать decisions.json")
    args = parser.parse_args(argv)

    try:
        world = load_world()
    except (OSError, TypeError, yaml.YAMLError) as e:
        print(f"ОШИБКА: не удалось прочитать {WORLD_PATH}: {e}", file=sys.stderr)
        return 1

    if args.template:
        created, notes = write_templates(world, args.template)
        for path in created:
            print(f"создан {path}")
        for note in notes:
            print(f"ПРЕДУПРЕЖДЕНИЕ: {note}", file=sys.stderr)
        return 0 if created or not notes else 1

    records, problems, warns = build(world)
    for w in warns:
        print(f"ПРЕДУПРЕЖДЕНИЕ: {w}", file=sys.stderr)
    if problems:
        for p in problems:
            print(f"ОШИБКА: {p}", file=sys.stderr)
        print(f"Сборка не выполнена: проблем {len(problems)}", file=sys.stderr)
        return 1
    OUT_PATH.write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"{OUT_PATH}: записей {len(records)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
