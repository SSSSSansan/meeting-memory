"""РОЛЬ B, шаг 1. Извлечение решений/задач/вопросов из транскрипта через LLM.

Что делает функция extract():
  1. Просит модель вернуть JSON по правилам из prompts.py.
  2. Проверяет, что каждая цитата ДОСЛОВНО есть в транскрипте.
  3. Если есть выдуманные цитаты или сломанный JSON — один раз просит исправить.
  4. Оставшиеся выдуманные пункты выбрасывает (но считает: это hallucination rate).
  5. Исправляет таймкоды по цитате, присваивает id, проверяет сроки.
  6. Пишет статистику в runs/extract_report.jsonl.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

from src.extract.prompts import SYSTEM_PROMPT, WEEKDAYS, retry_prompt, user_prompt
from src.llm_client import complete
from src.schemas import (Decision, Extraction, OpenQuestion, Source, Task,
                         TaskStatus, Transcript)

ROOT = Path(__file__).resolve().parent.parent.parent
REPORT = ROOT / "runs" / "extract_report.jsonl"


class ExtractionError(RuntimeError):
    pass


@dataclass
class ExtractStats:
    meeting_id: str
    items_raw: int = 0
    bad_quotes_first_attempt: int = 0
    dropped_final: int = 0
    merged_repeats: int = 0
    attempts: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: int = 0


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower()).strip()


def format_transcript(t: Transcript) -> str:
    return "\n".join(f"[{s.ts}] {s.speaker}: {s.text}" for s in t.segments)


def _parse_json(text: str) -> tuple[Optional[dict], Optional[str]]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, str(e)
    if not isinstance(data, dict):
        return None, "корень ответа должен быть объектом"
    return data, None


def _all_items(data: dict) -> list[tuple[str, dict]]:
    out = []
    for kind in ("decisions", "tasks", "open_questions"):
        for it in data.get(kind) or []:
            if isinstance(it, dict):
                out.append((kind, it))
    return out


def _bad_items(data: dict, transcript: Transcript) -> list[dict]:
    full = _norm(transcript.full_text())
    return [it for _, it in _all_items(data) if _norm(str(it.get("quote", ""))) not in full]


def _fix_ts(quote: str, claimed_ts: str, transcript: Transcript) -> str:
    q = _norm(quote)
    for seg in transcript.segments:
        if q in _norm(seg.text):
            return seg.ts
    return claimed_ts if re.fullmatch(r"\d{2}:\d{2}:\d{2}", str(claimed_ts)) else "00:00:00"


def _check_due(due, meeting_date: str) -> Optional[str]:
    """Срок должен быть корректной датой не раньше встречи и не позже чем через год."""
    if not due:
        return None
    try:
        d = datetime.strptime(str(due), "%Y-%m-%d").date()
        m = datetime.strptime(meeting_date, "%Y-%m-%d").date()
    except ValueError:
        return None
    return d.isoformat() if m <= d <= m + timedelta(days=365) else None


def _person(name, participants: list[str]) -> Optional[str]:
    if not name or str(name).lower() in ("null", "none"):
        return None
    name = str(name).strip()
    for p in participants:
        if _norm(p) == _norm(name):
            return p
    return name


def extract(transcript: Transcript, *, stats: Optional[ExtractStats] = None) -> Extraction:
    mid = transcript.meeting_id
    sid = mid.replace("_", "")  # m_001 -> m001, как в id: d_m001_01
    stats = stats or ExtractStats(meeting_id=mid)
    weekday = WEEKDAYS[datetime.strptime(transcript.date, "%Y-%m-%d").weekday()]

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt(transcript.title, transcript.date, weekday,
                                                transcript.participants, format_transcript(transcript))},
    ]

    data: Optional[dict] = None
    for attempt in (1, 2):
        stats.attempts = attempt
        r = complete(messages, json_schema={"type": "object"}, temperature=0.0,
                     max_tokens=3000, purpose="extract")
        stats.prompt_tokens += r.prompt_tokens
        stats.completion_tokens += r.completion_tokens
        stats.cost_usd += r.cost_usd
        stats.latency_ms += r.latency_ms

        parsed, err = _parse_json(r.text)
        bad = _bad_items(parsed, transcript) if parsed else []
        if attempt == 1 and parsed:
            stats.items_raw = len(_all_items(parsed))
            stats.bad_quotes_first_attempt = len(bad)
        if parsed is not None:
            data = parsed  # лучший из доступных ответов
        if parsed is not None and not bad:
            break
        if attempt == 1:
            messages += [{"role": "assistant", "content": r.text},
                         {"role": "user", "content": retry_prompt(
                             [str(i.get("quote", "")) for i in bad], err)}]

    if data is None:
        raise ExtractionError(f"{mid}: модель не вернула корректный JSON за 2 попытки")

    full = _norm(transcript.full_text())
    parts = transcript.participants
    projects: list[str] = [str(p).strip() for p in data.get("projects_discussed") or [] if str(p).strip()]

    def keep(it: dict) -> bool:
        ok = _norm(str(it.get("quote", ""))) in full and str(it.get("statement", "")).strip() \
            and str(it.get("project", "")).strip()
        if not ok:
            stats.dropped_final += 1
        return bool(ok)

    def src(it: dict) -> Source:
        q = str(it["quote"]).strip()
        return Source(ts=_fix_ts(q, str(it.get("ts", "")), transcript), quote=q)

    def proj(it: dict) -> str:
        p = str(it["project"]).strip()
        if p not in projects:
            projects.append(p)
        return p

    decisions, tasks, questions = [], [], []
    for it in (i for i in data.get("decisions") or [] if isinstance(i, dict)):
        if it.get("repeats_earlier") is True and decisions:
            stats.merged_repeats += 1  # подтверждение уже записанного решения — не новое решение
            continue
        if keep(it):
            try:
                conf = max(0.0, min(1.0, float(it.get("confidence", 0.5))))
            except (TypeError, ValueError):
                conf = 0.5
            decisions.append(Decision(id=f"d_{sid}_{len(decisions)+1:02d}", project=proj(it),
                                      statement=str(it["statement"]).strip(),
                                      made_by=_person(it.get("made_by"), parts),
                                      confidence=conf, source=src(it)))
    for it in (i for i in data.get("tasks") or [] if isinstance(i, dict)):
        if keep(it):
            status = str(it.get("status", "open")).lower()
            status = status if status in {s.value for s in TaskStatus} else "open"
            tasks.append(Task(id=f"t_{sid}_{len(tasks)+1:02d}", project=proj(it),
                              statement=str(it["statement"]).strip(),
                              owner=_person(it.get("owner"), parts),
                              due=_check_due(it.get("due"), transcript.date),
                              status=status, source=src(it)))
    for it in (i for i in data.get("open_questions") or [] if isinstance(i, dict)):
        if keep(it):
            questions.append(OpenQuestion(id=f"q_{sid}_{len(questions)+1:02d}", project=proj(it),
                                          statement=str(it["statement"]).strip(), source=src(it)))

    _write_report(stats)
    return Extraction(meeting_id=mid, projects_discussed=projects, decisions=decisions,
                      tasks=tasks, open_questions=questions)


def _write_report(stats: ExtractStats) -> None:
    REPORT.parent.mkdir(exist_ok=True)
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **stats.__dict__}
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
