"""Проверка результатов извлечения (раздел 4.5 контекста).

    python -m scripts.validate_extraction                      # data/extracted, транскрипты data/transcripts
    python -m scripts.validate_extraction data/extracted_llm
    python -m scripts.validate_extraction data/extracted_llm_raw --raw   # транскрипты из data/raw (адаптер+speakers)

Проверяет для каждого <folder>/*.json:
  1. JSON соответствует схеме (pydantic).
  2. Каждая source.quote — дословная подстрока транскрипта своей встречи.
     Провал = выдуманная запись → идёт в hallucination rate.
  3. Все project из decisions/tasks/questions есть в projects_discussed.
  4. Нет дублей id.

Код возврата 1, если есть ошибки схемы или дубли. Галлюцинации выводятся отдельно
как метрика и не валят скрипт: их доля — это результат, а не поломка.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError

from src.schemas import Extraction, Transcript
from src.utils.text import normalize as _norm

ROOT = Path(__file__).resolve().parent.parent
TRANSCRIPTS = ROOT / "data" / "transcripts"
EXTRACTED = ROOT / "data" / "extracted"


def quote_in_transcript(quote: str, transcript: Transcript) -> bool:
    return _norm(quote) in _norm(transcript.full_text())


def _load_raw_transcript(meeting_id: str) -> Transcript | None:
    """Строит Transcript из data/raw тем же путём, что src/extract/run.py --raw."""
    from src.ingest.scribal_adapter import convert_files
    from src.ingest.speakers import apply_speakers

    tr_path = ROOT / "data" / "raw" / f"{meeting_id}.transcript.json"
    meeting_path = ROOT / "data" / "raw" / f"{meeting_id}.meeting.json"
    if not (tr_path.exists() and meeting_path.exists()):
        return None
    return apply_speakers(convert_files(tr_path, meeting_path, meeting_id))


def validate_one(path: Path, raw: bool = False) -> tuple[list[str], int, int]:
    """Возвращает (ошибки, всего записей с цитатой, записей с несуществующей цитатой)."""
    errors: list[str] = []
    try:
        ext = Extraction.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as e:
        return [f"{path.name}: схема — {e.errors()[0]['msg']} @ {e.errors()[0]['loc']}"], 0, 0

    if raw:
        transcript = _load_raw_transcript(ext.meeting_id)
        if transcript is None:
            return [f"{path.name}: нет data/raw/{ext.meeting_id}.transcript.json"], 0, 0
    else:
        tr_path = TRANSCRIPTS / f"{ext.meeting_id}.json"
        if not tr_path.exists():
            return [f"{path.name}: нет транскрипта {tr_path.name}"], 0, 0
        transcript = Transcript.model_validate_json(tr_path.read_text(encoding="utf-8"))

    items = [*ext.decisions, *ext.tasks, *ext.open_questions]
    ids = [i.id for i in items]
    for dup in {i for i in ids if ids.count(i) > 1}:
        errors.append(f"{path.name}: дубль id {dup}")

    projects = set(ext.projects_discussed)
    bad_quotes = 0
    for it in items:
        if it.project not in projects:
            errors.append(f"{path.name}: {it.id} — проект {it.project!r} нет в projects_discussed")
        if not quote_in_transcript(it.source.quote, transcript):
            bad_quotes += 1
            print(f"  ГАЛЛЮЦИНАЦИЯ {it.id}: цитата не найдена в транскрипте: {it.source.quote!r}")
    return errors, len(items), bad_quotes


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default=str(EXTRACTED))
    ap.add_argument("--raw", action="store_true",
                     help="сравнивать с транскриптами из data/raw (адаптер+speakers), а не data/transcripts")
    args = ap.parse_args()
    folder = Path(args.folder)
    files = sorted(folder.glob("*.json"))
    if not files:
        print(f"Нет файлов в {folder}")
        return 1
    all_errors: list[str] = []
    total = bad = 0
    for f in files:
        errs, n, b = validate_one(f, raw=args.raw)
        all_errors += errs
        total += n
        bad += b
        print(f"{f.name}: записей={n}, выдуманных цитат={b}, ошибок={len(errs)}")
    rate = (bad / total * 100) if total else 0.0
    print(f"\nВсего записей: {total}. Hallucination rate: {rate:.1f}% ({bad}/{total})")
    for e in all_errors:
        print("ОШИБКА:", e)
    return 1 if all_errors else 0


if __name__ == "__main__":
    sys.exit(main())
