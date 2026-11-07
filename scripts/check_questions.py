"""Проверка вопросов data/gold/questions.json против сюжета data/gold/world.yaml.

    python scripts/check_questions.py
    python scripts/check_questions.py path/to/questions.json path/to/world.yaml

Проверяет:
  1. Схема каждой записи (лишние и недостающие поля, значения type/split/lang/as_of).
  2. id вопросов уникальны.
  3. Все source_meetings и as_of есть в world.yaml; source_meetings не позже as_of.
  4. Все event_ids есть в world.yaml; встреча каждого события входит в source_meetings
     и не позже as_of.
  5. У вопросов answerable=false оба списка пусты, тип none; у остальных списки не пусты.
  6. Вопросов на казахском (lang=kk) не меньше 3.

Код возврата 1 и сообщение на каждую ошибку, если проверки не прошли.
Модель вопроса живёт здесь, а не в src/schemas.py: эталон читают только оценка и роль A.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS = ROOT / "data" / "gold" / "questions.json"
DEFAULT_WORLD = ROOT / "data" / "gold" / "world.yaml"

MIN_KK_QUESTIONS = 3
ID_RE = re.compile(r"^q_\d{2,}$")
MEETING_RE = re.compile(r"^m_\d{3}$")
EVENT_RE = re.compile(r"^ev_m\d{3}_\d{2}$")


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    type: Literal["T1", "T2", "T3", "T4", "T5", "none"]
    split: Literal["dev", "test"]
    lang: Literal["ru", "kk"]
    question: str = Field(min_length=1)
    as_of: str
    expected_answer: str = Field(min_length=1)
    source_meetings: list[str]
    event_ids: list[str]
    answerable: bool

    @field_validator("id")
    @classmethod
    def _id_format(cls, v: str) -> str:
        if not ID_RE.match(v):
            raise ValueError("id должен быть вида q_01")
        return v

    @field_validator("question", "expected_answer")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("пустая строка")
        return v

    @field_validator("as_of")
    @classmethod
    def _as_of_format(cls, v: str) -> str:
        if not MEETING_RE.match(v):
            raise ValueError("as_of должен быть вида m_008")
        return v

    @field_validator("source_meetings")
    @classmethod
    def _meetings_format(cls, v: list[str]) -> list[str]:
        for m in v:
            if not MEETING_RE.match(m):
                raise ValueError(f"встреча {m!r} не вида m_008")
        return v

    @field_validator("event_ids")
    @classmethod
    def _events_format(cls, v: list[str]) -> list[str]:
        for e in v:
            if not EVENT_RE.match(e):
                raise ValueError(f"событие {e!r} не вида ev_m001_01")
        return v


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_world(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _schema_errors(index: int, raw: Any) -> tuple[Question | None, list[str]]:
    label = f"запись {index}" + (f" ({raw['id']})" if isinstance(raw, dict) and "id" in raw else "")
    if not isinstance(raw, dict):
        return None, [f"{label}: ожидался объект"]
    try:
        return Question.model_validate(raw), []
    except ValidationError as e:
        errors = []
        for err in e.errors():
            where = ".".join(str(p) for p in err["loc"]) or "запись"
            errors.append(f"{label}: поле {where}: {err['msg']}")
        return None, errors


def check(raw_questions: Any, world: dict) -> list[str]:
    """Возвращает список ошибок; пустой список — всё в порядке."""
    if not isinstance(raw_questions, list):
        return ["questions.json: ожидался список записей"]

    meeting_order = {m["id"]: i for i, m in enumerate(world["meetings"])}
    event_meeting = {e["id"]: e["meeting"] for e in world["events"]}

    errors: list[str] = []
    questions: list[Question] = []
    for i, raw in enumerate(raw_questions, start=1):
        q, errs = _schema_errors(i, raw)
        errors += errs
        if q is not None:
            questions.append(q)

    ids = Counter(q.id for q in questions)
    errors += [f"дубль id вопроса: {qid}" for qid, n in ids.items() if n > 1]

    for q in questions:
        errors += _reference_errors(q, meeting_order, event_meeting)
        errors += _answerable_errors(q)

    kk = sum(1 for q in questions if q.lang == "kk")
    if kk < MIN_KK_QUESTIONS:
        errors.append(f"вопросов на казахском {kk}, нужно не меньше {MIN_KK_QUESTIONS}")
    return errors


def _reference_errors(q: Question, meeting_order: dict[str, int], event_meeting: dict[str, str]) -> list[str]:
    errors: list[str] = []
    as_of_known = q.as_of in meeting_order
    if not as_of_known:
        errors.append(f"{q.id}: as_of {q.as_of} нет в world.yaml")

    for m in q.source_meetings:
        if m not in meeting_order:
            errors.append(f"{q.id}: встречи {m} нет в world.yaml")
        elif as_of_known and meeting_order[m] > meeting_order[q.as_of]:
            errors.append(f"{q.id}: source_meetings содержит {m}, это позже as_of {q.as_of}")

    for ev in q.event_ids:
        meeting = event_meeting.get(ev)
        if meeting is None:
            errors.append(f"{q.id}: события {ev} нет в world.yaml")
            continue
        if meeting not in q.source_meetings:
            errors.append(f"{q.id}: встреча {meeting} события {ev} не входит в source_meetings")
        if as_of_known and meeting_order[meeting] > meeting_order[q.as_of]:
            errors.append(f"{q.id}: событие {ev} из {meeting}, это позже as_of {q.as_of}")
    return errors


def _answerable_errors(q: Question) -> list[str]:
    errors: list[str] = []
    if not q.answerable:
        if q.source_meetings or q.event_ids:
            errors.append(f"{q.id}: answerable=false, но source_meetings или event_ids не пусты")
        if q.type != "none":
            errors.append(f"{q.id}: answerable=false требует type none, а не {q.type}")
    else:
        if not q.source_meetings or not q.event_ids:
            errors.append(f"{q.id}: answerable=true, но source_meetings или event_ids пусты")
        if q.type == "none":
            errors.append(f"{q.id}: type none допустим только при answerable=false")
    return errors


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) > 2:
        print("Использование: check_questions.py [questions.json [world.yaml]]", file=sys.stderr)
        return 2
    questions_path = Path(args[0]) if args else DEFAULT_QUESTIONS
    world_path = Path(args[1]) if len(args) > 1 else DEFAULT_WORLD

    try:
        raw = load_json(questions_path)
        world = load_world(world_path)
    except (OSError, ValueError, yaml.YAMLError) as e:
        print(f"Не удалось прочитать файл: {e}", file=sys.stderr)
        return 1

    errors = check(raw, world)
    if errors:
        for e in errors:
            print(f"ОШИБКА: {e}", file=sys.stderr)
        print(f"Ошибок: {len(errors)}", file=sys.stderr)
        return 1

    by_type = Counter(q["type"] for q in raw)
    by_split = Counter(q["split"] for q in raw)
    by_lang = Counter(q["lang"] for q in raw)
    print(f"Вопросов: {len(raw)}")
    print("По типам: " + ", ".join(f"{k} {v}" for k, v in sorted(by_type.items())))
    print("По выборкам: " + ", ".join(f"{k} {v}" for k, v in sorted(by_split.items())))
    print("По языкам: " + ", ".join(f"{k} {v}" for k, v in sorted(by_lang.items())))
    print("Все проверки пройдены.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
