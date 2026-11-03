"""Проверка сюжета синтетического набора data/gold/world.yaml.

    python scripts/check_world.py                 # data/gold/world.yaml
    python scripts/check_world.py path/to/world.yaml

Проверяет:
  1. id встреч, людей, проектов и событий уникальны.
  2. Все ссылки существуют: участники, проекты, встречи, id событий
     в links, closures, recaps, reissues, repeats.
  3. Даты встреч идут по возрастанию вместе с номерами; даты задач корректны (или null).
  4. project события входит в projects_discussed его встречи (если не null).
  5. Каждая связь идёт от события более поздней встречи к событию более ранней.
  6. У связей supersedes/contradicts/depends_on/refines оба конца — decision.
  7. owner, made_by, said_by, reported_by есть в people.
  8. В событиях нет неожиданных полей (типичная причина — запятая в незакавыченном note).

Код возврата 1 и сообщение на каждую ошибку, если проверки не прошли. Если прошли,
печатает итог. Неожиданные поля в остальных разделах выводятся предупреждением
и код возврата не меняют.
"""
from __future__ import annotations

import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PATH = ROOT / "data" / "gold" / "world.yaml"

EVENT_TYPES = {"decision", "task", "open_question", "status_update", "recap", "non_decision"}
LINK_TYPES = {"supersedes", "contradicts", "depends_on", "refines", "same_topic"}
DECISION_LINK_TYPES = {"supersedes", "contradicts", "depends_on", "refines"}
ACTOR_FIELDS = ("owner", "made_by", "said_by", "reported_by")
EVENT_REF_FIELDS = ("recaps", "reissues", "repeats")
KNOWN_SCENARIOS = [
    "evolution_deadline", "cross_project", "contradiction_load", "forgotten_promise",
    "cancelled_decision", "language_mix", "language_kk", "noise", "replaced_decision",
    "dependency", "thinking_aloud", "recap", "self_correction", "absentia_assignment",
]
LANGUAGE_PROFILES = ["ru", "mixed", "kk"]

ALLOWED_KEYS = {
    "people": {"id", "name", "role", "role_aliases", "written_variants", "note"},
    "projects": {"id", "name", "description", "written_variants"},
    "meetings": {
        "id", "date", "title", "main_project", "projects_discussed", "participants",
        "language_profile", "target_minutes", "scenarios",
    },
    "events": {
        "id", "meeting", "type", "project", "statement", "made_by", "owner", "due",
        "reported_by", "said_by", "note", "kind", "self_correction", *EVENT_REF_FIELDS,
    },
    "links": {"from", "to", "type", "scenario", "rationale", "cancelled_without_replacement"},
    "closures": {"from", "closes"},
}


def load_world(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        world = yaml.safe_load(f)
    if not isinstance(world, dict):
        raise TypeError(f"{path}: верхний уровень должен быть словарём")
    return world


def _items(world: dict, section: str) -> list[dict]:
    return [x for x in (world.get(section) or []) if isinstance(x, dict)]


def _dups(ids: list) -> list:
    return sorted(str(i) for i, n in Counter(ids).items() if n > 1)


def check(world: dict) -> list[str]:
    errors: list[str] = []
    people = _items(world, "people")
    projects = _items(world, "projects")
    meetings = _items(world, "meetings")
    events = _items(world, "events")
    links = _items(world, "links")
    closures = _items(world, "closures")

    # 1. уникальность id
    for name, items in (("людей", people), ("проектов", projects),
                        ("встреч", meetings), ("событий", events)):
        for d in _dups([x.get("id") for x in items]):
            errors.append(f"[1] дубль id {name}: {d}")

    people_ids = {x.get("id") for x in people}
    project_ids = {x.get("id") for x in projects}
    meeting_by_id = {x.get("id"): x for x in meetings}
    event_by_id = {x.get("id"): x for x in events}
    meeting_pos = {x.get("id"): i for i, x in enumerate(meetings)}

    # 2. ссылки (встречи)
    for m in meetings:
        mid = m.get("id")
        for p in m.get("participants") or []:
            if p not in people_ids:
                errors.append(f"[2] {mid}: участник {p!r} нет в people")
        for p in m.get("projects_discussed") or []:
            if p not in project_ids:
                errors.append(f"[2] {mid}: проект {p!r} из projects_discussed нет в projects")
        mp = m.get("main_project")
        if mp is not None and mp not in project_ids:
            errors.append(f"[2] {mid}: main_project {mp!r} нет в projects")

    # 2. ссылки (события)
    for e in events:
        eid = e.get("id")
        if e.get("meeting") not in meeting_by_id:
            errors.append(f"[2] {eid}: встречи {e.get('meeting')!r} нет в meetings")
        if e.get("type") not in EVENT_TYPES:
            errors.append(f"[2] {eid}: неизвестный тип события {e.get('type')!r}")
        pr = e.get("project")
        if pr is not None and pr not in project_ids:
            errors.append(f"[2] {eid}: проект {pr!r} нет в projects")
        for field in EVENT_REF_FIELDS:
            ref = e.get(field)
            if ref is not None and ref not in event_by_id:
                errors.append(f"[2] {eid}: {field} ссылается на несуществующее событие {ref!r}")

    # 2. ссылки (links, closures)
    for i, ln in enumerate(links, 1):
        label = f"links[{i}] {ln.get('from')} -> {ln.get('to')}"
        for end in ("from", "to"):
            if ln.get(end) not in event_by_id:
                errors.append(f"[2] {label}: {end} — несуществующее событие {ln.get(end)!r}")
        if ln.get("type") not in LINK_TYPES:
            errors.append(f"[2] {label}: неизвестный тип связи {ln.get('type')!r}")
    for i, cl in enumerate(closures, 1):
        for end in ("from", "closes"):
            if cl.get(end) not in event_by_id:
                errors.append(
                    f"[2] closures[{i}]: {end} — несуществующее событие {cl.get(end)!r}"
                )

    # 3. даты встреч и задач
    prev_num, prev_date, prev_id = -1, None, None
    for m in meetings:
        mid = str(m.get("id"))
        match = re.fullmatch(r"m_(\d{3})", mid)
        d = m.get("date")
        if not match:
            errors.append(f"[3] {mid}: id встречи должен быть вида m_NNN")
        if not isinstance(d, date):
            errors.append(f"[3] {mid}: дата {d!r} не является датой YYYY-MM-DD")
            continue
        if match:
            num = int(match.group(1))
            if num <= prev_num:
                errors.append(f"[3] {mid}: номер не больше предыдущего ({prev_id})")
            prev_num = num
        if prev_date is not None and d < prev_date:
            errors.append(f"[3] {mid}: дата {d} раньше даты предыдущей встречи {prev_id} ({prev_date})")
        prev_date, prev_id = d, mid
    for e in events:
        if e.get("type") == "task" and "due" in e:
            due = e["due"]
            if due is None:
                continue
            m_date = (meeting_by_id.get(e.get("meeting")) or {}).get("date")
            if not isinstance(due, date):
                errors.append(f"[3] {e.get('id')}: due {due!r} не является датой YYYY-MM-DD")
            elif isinstance(m_date, date) and due < m_date:
                errors.append(f"[3] {e.get('id')}: due {due} раньше даты встречи {m_date}")

    # 4. проект события ∈ projects_discussed встречи
    for e in events:
        pr, m = e.get("project"), meeting_by_id.get(e.get("meeting"))
        if pr is not None and m is not None and pr not in (m.get("projects_discussed") or []):
            errors.append(
                f"[4] {e.get('id')}: проект {pr!r} нет в projects_discussed встречи {m.get('id')}"
            )

    # 5. направление связей: от более поздней встречи к более ранней
    for i, ln in enumerate(links, 1):
        src, dst = event_by_id.get(ln.get("from")), event_by_id.get(ln.get("to"))
        if src is None or dst is None:
            continue
        p_src, p_dst = meeting_pos.get(src.get("meeting")), meeting_pos.get(dst.get("meeting"))
        if p_src is None or p_dst is None:
            continue
        if p_src <= p_dst:
            errors.append(
                f"[5] links[{i}] {ln.get('from')} ({src.get('meeting')}) -> {ln.get('to')} "
                f"({dst.get('meeting')}): связь должна идти от более поздней встречи к более ранней"
            )

    # 6. концы связей решений — decision
    for i, ln in enumerate(links, 1):
        if ln.get("type") not in DECISION_LINK_TYPES:
            continue
        for end in ("from", "to"):
            ev = event_by_id.get(ln.get(end))
            if ev is not None and ev.get("type") != "decision":
                errors.append(
                    f"[6] links[{i}] ({ln.get('type')}): {end} {ln.get(end)} "
                    f"имеет тип {ev.get('type')!r}, нужен 'decision'"
                )

    # 7. авторы и исполнители ∈ people
    for e in events:
        for field in ACTOR_FIELDS:
            val = e.get(field)
            if val is not None and val not in people_ids:
                errors.append(f"[7] {e.get('id')}: {field}={val!r} нет в people")

    # 8. неожиданные поля в событиях: запятая в незакавыченном значении внутри {...}
    # не ломает YAML, а превращает хвост значения в отдельный ключ
    for e in events:
        extra = sorted(str(k) for k in e if k not in ALLOWED_KEYS["events"])
        if extra:
            errors.append(
                f"[8] {e.get('id')}: неожиданные поля {extra}; возможно, значение с запятой "
                "внутри {...} нужно взять в кавычки"
            )

    return errors


def warnings(world: dict) -> list[str]:
    """Неожиданные поля вне событий (для событий это ошибка, см. check)."""
    out: list[str] = []
    for section, allowed in ALLOWED_KEYS.items():
        if section == "events":
            continue
        for x in _items(world, section):
            extra = sorted(str(k) for k in x if k not in allowed)
            if extra:
                out.append(
                    f"{section}: {x.get('id', x.get('from'))} — неожиданные поля {extra}; "
                    "возможно, незакавыченное значение с запятой внутри {...}"
                )
    return out


def task_status(world: dict) -> list[tuple[str, str, str]]:
    closed_by = {c.get("closes"): c.get("from") for c in _items(world, "closures")}
    rows = []
    for e in _items(world, "events"):
        if e.get("type") not in ("task", "open_question"):
            continue
        eid = e["id"]
        status = f"closed (закрыто {closed_by[eid]})" if eid in closed_by else "open"
        if e.get("reissues"):
            status += f"; повторно выдана (reissues {e['reissues']})"
        if e.get("repeats"):
            status += f"; повтор вопроса (repeats {e['repeats']})"
        rows.append((eid, e["type"], status))
    return rows


def summary(world: dict) -> str:
    meetings = _items(world, "meetings")
    events = _items(world, "events")
    lines = ["== Итог =="]

    by_lang = Counter(m.get("language_profile") for m in meetings)
    total = len(meetings)
    lines.append(f"Встреч: {total}")
    for lp in LANGUAGE_PROFILES:
        lines.append(f"  {lp}: {by_lang.get(lp, 0)}")
    for lp in sorted(k for k in by_lang if k not in LANGUAGE_PROFILES):
        lines.append(f"  {lp} (не из language_profiles): {by_lang[lp]}")
    non_ru = by_lang.get("mixed", 0) + by_lang.get("kk", 0)
    pct = 100 * non_ru / total if total else 0.0
    lines.append(f"Не-русских (mixed + kk): {non_ru} из {total} = {pct:.1f}%")

    lines.append(f"Событий: {len(events)}")
    for t, n in sorted(Counter(e.get("type") for e in events).items()):
        lines.append(f"  {t}: {n}")

    tags: dict[str, list[str]] = {}
    for m in meetings:
        for tag in m.get("scenarios") or []:
            tags.setdefault(tag, []).append(str(m.get("id")))
    for ln in _items(world, "links"):
        if ln.get("scenario"):
            tags.setdefault(ln["scenario"], [])
    lines.append("Теги сценариев (встречи, где встречаются):")
    for tag in KNOWN_SCENARIOS:
        where = ", ".join(tags.get(tag, [])) or "— не встречается"
        lines.append(f"  {tag}: {where}")
    unknown = sorted(set(tags) - set(KNOWN_SCENARIOS))
    if unknown:
        lines.append(f"  теги вне списка: {', '.join(unknown)}")

    lines.append("Статус задач и открытых вопросов:")
    for eid, etype, status in task_status(world):
        lines.append(f"  {eid} [{etype}]: {status}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    path = Path(args[0]) if args else DEFAULT_PATH
    try:
        world = load_world(path)
    except (OSError, TypeError, yaml.YAMLError) as e:
        print(f"ОШИБКА: не удалось прочитать {path}: {e}", file=sys.stderr)
        return 1

    errors = check(world)
    for w in warnings(world):
        print(f"ПРЕДУПРЕЖДЕНИЕ: {w}", file=sys.stderr)
    if errors:
        for err in errors:
            print(f"ОШИБКА: {err}", file=sys.stderr)
        print(f"Проверка не пройдена: ошибок {len(errors)}", file=sys.stderr)
        return 1

    print(f"{path}: проверки 1–8 пройдены\n")
    print(summary(world))
    return 0


if __name__ == "__main__":
    sys.exit(main())
