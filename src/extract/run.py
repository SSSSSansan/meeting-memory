"""Запуск извлечения.

    python -m src.extract.run                 # все транскрипты → data/extracted_llm/
    python -m src.extract.run m_001           # только одна встреча
    python -m src.extract.run --out data/extracted
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from src.extract.extractor import ExtractStats, ExtractionError, extract
from src.schemas import Transcript

ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("meetings", nargs="*", help="id встреч, по умолчанию все")
    ap.add_argument("--out", default="data/extracted_llm")
    args = ap.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted((ROOT / "data" / "transcripts").glob("*.json"))
    if args.meetings:
        files = [f for f in files if f.stem in args.meetings]
    if not files:
        print("Нет транскриптов для обработки.")
        return 1

    total_cost = 0.0
    for f in files:
        t = Transcript.model_validate_json(f.read_text(encoding="utf-8"))
        stats = ExtractStats(meeting_id=t.meeting_id)
        try:
            ext = extract(t, stats=stats)
        except ExtractionError as e:
            print(f"{t.meeting_id}: ОШИБКА — {e}")
            continue
        (out_dir / f"{t.meeting_id}.json").write_text(
            ext.model_dump_json(indent=2, exclude_none=False), encoding="utf-8")
        total_cost += stats.cost_usd
        print(f"{t.meeting_id}: решений={len(ext.decisions)} задач={len(ext.tasks)} "
              f"вопросов={len(ext.open_questions)} | выдуманных цитат с 1-й попытки="
              f"{stats.bad_quotes_first_attempt}/{stats.items_raw}, отброшено={stats.dropped_final} | "
              f"токены {stats.prompt_tokens}+{stats.completion_tokens}, ${stats.cost_usd:.5f}")
    print(f"\nСтоимость прогона: ${total_cost:.5f}. Результат: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
