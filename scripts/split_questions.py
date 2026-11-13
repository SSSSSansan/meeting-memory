"""Разделение data/gold/questions.json на выборки dev и test.

    python scripts/split_questions.py
    python scripts/split_questions.py path/to/questions.json

Читает полный questions.json (в нём поле split, его читает код) и пишет рядом
questions_dev.json и questions_test.json: только вопросы с нужным split, в исходном порядке,
без перемешивания и без изменения записей. Повторный запуск даёт те же файлы побайтово
(перевод строки всегда LF). questions.json скрипт не меняет.

Здесь же формат записи вопросов (`dumps_questions`): им пишется и сам questions.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS = ROOT / "data" / "gold" / "questions.json"
SPLITS = ("dev", "test")
# длинные строки: по одной на строку файла, иначе diff нечитаем
FACT_KEYS = ("required_facts", "forbidden_facts")


def _dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


def _format_value(key: str, value: Any) -> str:
    if isinstance(value, list):
        if not value:
            return "[]"
        if key in FACT_KEYS:
            return "[\n" + ",\n".join(f"      {_dump(v)}" for v in value) + "\n    ]"
        return "[" + ", ".join(_dump(v) for v in value) + "]"
    return _dump(value)


def dumps_questions(questions: list[dict]) -> str:
    """JSON-список вопросов в формате репозитория: отступ 2, короткие списки в одну строку, LF."""
    records = []
    for q in questions:
        fields = ",\n".join(f"    {_dump(k)}: {_format_value(k, v)}" for k, v in q.items())
        records.append("  {\n" + fields + "\n  }")
    return "[\n" + ",\n".join(records) + "\n]\n"


def write_questions(path: Path, questions: list[dict]) -> None:
    # newline="" : без подмены LF на CRLF в Windows, иначе файл зависит от ОС
    with path.open("w", encoding="utf-8", newline="") as f:
        f.write(dumps_questions(questions))


def split_questions(questions: list[dict]) -> dict[str, list[dict]]:
    """Вопросы по значению split, порядок исходный. Неизвестный split — ошибка, а не молчаливый пропуск."""
    parts: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for q in questions:
        split = q.get("split")
        if split not in parts:
            raise ValueError(f"{q.get('id')}: split {split!r}, ожидалось одно из {SPLITS}")
        parts[split].append(q)
    return parts


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) > 1:
        print("Использование: split_questions.py [questions.json]", file=sys.stderr)
        return 2
    source = Path(args[0]) if args else DEFAULT_QUESTIONS
    try:
        questions = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(questions, list):
            raise ValueError("ожидался список записей")
        parts = split_questions(questions)
    except (OSError, ValueError) as e:
        print(f"Не удалось разделить {source}: {e}", file=sys.stderr)
        return 1

    for split, items in parts.items():
        target = source.with_name(f"questions_{split}.json")
        write_questions(target, items)
        print(f"{target.name}: {len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
