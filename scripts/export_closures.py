"""Выгрузка эталона закрытий задач data/gold/closures.json из раздела closures файла data/gold/world.yaml.

    python scripts/export_closures.py
    python scripts/export_closures.py path/to/world.yaml path/to/closures.json

Запись: task_event_id (что закрыто), closing_event_id (чем закрыто), task_statement, closing_statement,
task_meeting, closing_meeting. Порядок как в world.yaml; повторный запуск даёт тот же файл побайтово (LF).
Неизвестный id события или закрытие раньше задачи — ошибка, файл не пишется.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WORLD = ROOT / "data" / "gold" / "world.yaml"
DEFAULT_OUT = ROOT / "data" / "gold" / "closures.json"


def build_closures(world: dict) -> list[dict]:
    events = {e["id"]: e for e in world["events"]}
    dates = {m["id"]: m["date"] for m in world["meetings"]}
    result = []
    for c in world["closures"]:
        task, closing = events.get(c["closes"]), events.get(c["from"])
        if task is None:
            raise ValueError(f"закрытие {c}: события {c['closes']} нет в world.yaml")
        if closing is None:
            raise ValueError(f"закрытие {c}: события {c['from']} нет в world.yaml")
        if dates[closing["meeting"]] < dates[task["meeting"]]:
            raise ValueError(f"закрытие {c['from']} → {c['closes']}: закрывающая встреча раньше задачи")
        result.append({
            "task_event_id": task["id"],
            "closing_event_id": closing["id"],
            "task_statement": task["statement"],
            "closing_statement": closing["statement"],
            "task_meeting": task["meeting"],
            "closing_meeting": closing["meeting"],
        })
    return result


def dumps_closures(items: list[dict]) -> str:
    return json.dumps(items, ensure_ascii=False, indent=2) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) > 2:
        print("Использование: export_closures.py [world.yaml [closures.json]]", file=sys.stderr)
        return 2
    world_path = Path(args[0]) if args else DEFAULT_WORLD
    out_path = Path(args[1]) if len(args) > 1 else DEFAULT_OUT
    try:
        world = yaml.safe_load(world_path.read_text(encoding="utf-8"))
        items = build_closures(world)
    except (OSError, ValueError, KeyError, yaml.YAMLError) as e:
        print(f"Не удалось выгрузить закрытия: {e}", file=sys.stderr)
        return 1
    # newline="" : LF в любой ОС, чтобы файл не зависел от платформы
    with out_path.open("w", encoding="utf-8", newline="") as f:
        f.write(dumps_closures(items))
    print(f"{out_path.name}: {len(items)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
