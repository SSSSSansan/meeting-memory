"""Проверка вопросов data/gold/questions.json против сюжета data/gold/world.yaml.

    python scripts/check_questions.py
    python scripts/check_questions.py path/to/questions.json path/to/world.yaml

Поле decision_ids: id событий из world.yaml (решения, задачи, открытые вопросы, статусы), из которых
складывается ответ; название сохранено из TEAM.md.

Проверяет:
  1. Схема каждой записи (лишние и недостающие поля, значения type/split/lang/as_of).
  2. id вопросов уникальны.
  3. Все source_meetings и as_of есть в world.yaml (номер встречи может отсутствовать,
     например m_014 исключена); source_meetings не позже as_of. «Позже» считается
     по дате встречи в world.yaml, а не по номеру.
  4. Все decision_ids есть в world.yaml, без повторов; встреча каждого события входит
     в source_meetings и не позже as_of.
  5. У вопросов answerable=false оба списка пусты, тип none; у остальных списки не пусты.
  6. as_of_date совпадает с датой встречи as_of в world.yaml (ISO); поля as_of_date,
     required_facts, forbidden_facts есть у каждой записи.
  7. У вопросов split=test required_facts не пусты; у answerable=false в test они тоже есть
     (смысл эталона — «система сообщает, что данных нет»). В dev пустые required_facts допустимы,
     скрипт печатает предупреждение с их числом.
  8. Набор целиком: 58 вопросов, test 23, dev 35; kk не меньше 3 и в dev, и в test;
     в test не меньше 3 вопросов каждого типа T1..T5 и none.

Пункты 6 (наличие полей) и 7–8 — «строгий» режим, он включён по умолчанию. Исторические копии
(questions_m008.json, записаны до новых полей) проверяются с strict=False: схема, ссылки на
world.yaml и answerable, без требований к полному набору.

Код возврата 1 и сообщение на каждую ошибку, если проверки не прошли.
Модель вопроса живёт здесь, а не в src/schemas.py: эталон читают только оценка и роль A.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QUESTIONS = ROOT / "data" / "gold" / "questions.json"
DEFAULT_WORLD = ROOT / "data" / "gold" / "world.yaml"

EXPECTED_TOTAL = 58
EXPECTED_TEST = 23
EXPECTED_DEV = 35
MIN_KK_PER_SPLIT = 3
MIN_TEST_PER_TYPE = 3
QUESTION_TYPES = ("T1", "T2", "T3", "T4", "T5", "none")
NEW_FIELDS = ("as_of_date", "required_facts", "forbidden_facts")
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
    decision_ids: list[str]
    answerable: bool
    # Поля добавлены позже (только добавление): у исторической копии questions_m008.json их нет,
    # поэтому в модели они необязательны, а наличие в questions.json проверяет check(strict=True).
    as_of_date: str | None = None
    required_facts: list[str] = Field(default_factory=list)
    forbidden_facts: list[str] = Field(default_factory=list)

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

    @field_validator("as_of_date")
    @classmethod
    def _as_of_date_format(cls, v: str | None) -> str | None:
        if v is None:
            return v
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError("as_of_date должна быть ISO-датой вида 2026-10-26") from None
        return v

    @field_validator("required_facts", "forbidden_facts")
    @classmethod
    def _facts_not_blank(cls, v: list[str]) -> list[str]:
        if any(not f.strip() for f in v):
            raise ValueError("пустая строка в списке фактов")
        return v

    @field_validator("source_meetings")
    @classmethod
    def _meetings_format(cls, v: list[str]) -> list[str]:
        for m in v:
            if not MEETING_RE.match(m):
                raise ValueError(f"встреча {m!r} не вида m_008")
        return v

    @field_validator("decision_ids")
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


def check(raw_questions: Any, world: dict, *, strict: bool = True) -> list[str]:
    """Возвращает список ошибок; пустой список — всё в порядке.

    strict=False — для исторических копий без новых полей: без пунктов 6–8 из описания модуля.
    """
    if not isinstance(raw_questions, list):
        return ["questions.json: ожидался список записей"]

    # порядок встреч — по дате: нумерация может иметь пропуски (m_014 исключена)
    meeting_order = {m["id"]: m["date"] for m in world["meetings"]}
    event_meeting = {e["id"]: e["meeting"] for e in world["events"]}

    errors: list[str] = []
    questions: list[Question] = []
    for i, raw in enumerate(raw_questions, start=1):
        q, errs = _schema_errors(i, raw)
        errors += errs
        if q is not None:
            questions.append(q)
            if strict:
                errors += _new_field_errors(q, raw, meeting_order)

    ids = Counter(q.id for q in questions)
    errors += [f"дубль id вопроса: {qid}" for qid, n in ids.items() if n > 1]

    for q in questions:
        errors += _reference_errors(q, meeting_order, event_meeting)
        errors += _answerable_errors(q)
        if strict:
            errors += _test_facts_errors(q)

    if strict:
        errors += _set_errors(questions)
    return errors


def warnings(raw_questions: Any) -> list[str]:
    """Предупреждения (не ошибки): dev-вопросы без required_facts. Пока их дописывают отдельной порцией."""
    if not isinstance(raw_questions, list):
        return []
    empty = [
        q["id"] for q in raw_questions
        if isinstance(q, dict) and q.get("split") == "dev" and not q.get("required_facts")
    ]
    if not empty:
        return []
    return [f"у {len(empty)} вопросов dev пустые required_facts (допустимо для dev, нужно дописать)"]


def _new_field_errors(q: Question, raw: dict, meeting_order: dict[str, date]) -> list[str]:
    errors = [f"{q.id}: нет поля {f}" for f in NEW_FIELDS if f not in raw]
    if "as_of_date" in raw and q.as_of in meeting_order:
        expected = meeting_order[q.as_of].isoformat()
        if q.as_of_date != expected:
            errors.append(
                f"{q.id}: as_of_date {q.as_of_date} не совпадает с датой встречи {q.as_of} ({expected}) в world.yaml"
            )
    return errors


def _test_facts_errors(q: Question) -> list[str]:
    # в dev пустые required_facts допустимы (см. warnings), в test — нет: по ним считается таблица отчёта
    if q.split == "test" and not q.required_facts:
        return [f"{q.id}: split=test, но required_facts пусты"]
    return []


def _set_errors(questions: list[Question]) -> list[str]:
    errors: list[str] = []
    total = len(questions)
    by_split = Counter(q.split for q in questions)
    if total != EXPECTED_TOTAL:
        errors.append(f"вопросов всего {total}, ожидалось {EXPECTED_TOTAL}")
    if by_split["test"] != EXPECTED_TEST:
        errors.append(f"вопросов в выборке test {by_split['test']}, ожидалось {EXPECTED_TEST}")
    if by_split["dev"] != EXPECTED_DEV:
        errors.append(f"вопросов в выборке dev {by_split['dev']}, ожидалось {EXPECTED_DEV}")

    for split in ("dev", "test"):
        kk = sum(1 for q in questions if q.split == split and q.lang == "kk")
        if kk < MIN_KK_PER_SPLIT:
            errors.append(f"вопросов на казахском в {split} {kk}, нужно не меньше {MIN_KK_PER_SPLIT}")

    by_type = Counter(q.type for q in questions if q.split == "test")
    for t in QUESTION_TYPES:
        if by_type[t] < MIN_TEST_PER_TYPE:
            errors.append(f"вопросов типа {t} в выборке test {by_type[t]}, нужно не меньше {MIN_TEST_PER_TYPE}")
    return errors


def _reference_errors(q: Question, meeting_order: dict[str, date], event_meeting: dict[str, str]) -> list[str]:
    errors: list[str] = []
    as_of_known = q.as_of in meeting_order
    if not as_of_known:
        errors.append(f"{q.id}: as_of {q.as_of} нет в world.yaml")

    for m in q.source_meetings:
        if m not in meeting_order:
            errors.append(f"{q.id}: встречи {m} нет в world.yaml")
        elif as_of_known and meeting_order[m] > meeting_order[q.as_of]:
            errors.append(f"{q.id}: source_meetings содержит {m}, это позже as_of {q.as_of}")

    for ev in sorted(e for e, n in Counter(q.decision_ids).items() if n > 1):
        errors.append(f"{q.id}: decision_ids содержит {ev} больше одного раза")

    for ev in q.decision_ids:
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
        if q.source_meetings or q.decision_ids:
            errors.append(f"{q.id}: answerable=false, но source_meetings или decision_ids не пусты")
        if q.type != "none":
            errors.append(f"{q.id}: answerable=false требует type none, а не {q.type}")
    else:
        if not q.source_meetings or not q.decision_ids:
            errors.append(f"{q.id}: answerable=true, но source_meetings или decision_ids пусты")
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
    notes = warnings(raw)
    if errors:
        for e in errors:
            print(f"ОШИБКА: {e}", file=sys.stderr)
        for w in notes:
            print(f"ПРЕДУПРЕЖДЕНИЕ: {w}", file=sys.stderr)
        print(f"Ошибок: {len(errors)}", file=sys.stderr)
        return 1

    by_type = Counter(q["type"] for q in raw)
    by_split = Counter(q["split"] for q in raw)
    by_lang = Counter(q["lang"] for q in raw)
    print(f"Вопросов: {len(raw)}")
    print("По типам: " + ", ".join(f"{k} {v}" for k, v in sorted(by_type.items())))
    print("По выборкам: " + ", ".join(f"{k} {v}" for k, v in sorted(by_split.items())))
    print("По языкам: " + ", ".join(f"{k} {v}" for k, v in sorted(by_lang.items())))
    for w in notes:
        print(f"ПРЕДУПРЕЖДЕНИЕ: {w}")
    print("Все проверки пройдены.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
