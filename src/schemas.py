"""Контракт данных между ролями A / B / C.

Менять только по согласованию всех троих. Описание — docs/K02_MEETING_MEMORY_CONTEXT.md, раздел 4.
"""
from __future__ import annotations

from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------- 4.1 Входной транскрипт ----------

class Segment(BaseModel):
    ts: str = Field(pattern=r"^\d{2}:\d{2}:\d{2}$", description="HH:MM:SS от начала встречи")
    speaker: str
    text: str
    # Необязательные поля из Scribal (адаптер src/ingest/scribal_adapter.py)
    start_s: Optional[float] = None
    end_s: Optional[float] = None
    speaker_raw: Optional[str] = None  # исходная метка SPEAKER_00
    lang: Optional[str] = None


class Transcript(BaseModel):
    meeting_id: str
    title: str
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    participants: list[str]
    segments: list[Segment]
    language: Optional[str] = None
    source: Optional[Literal["synthetic", "own_asr", "scribal"]] = None
    started_at: Optional[str] = None

    def full_text(self) -> str:
        return "\n".join(s.text for s in self.segments)


# ---------- Сырой формат Scribal (вход адаптера) ----------

class ScribalWord(BaseModel):
    word: str
    start: float
    end: float
    score: Optional[float] = None


class ScribalSegment(BaseModel):
    start: float
    end: float
    speaker: Optional[str] = None
    text: str
    words: list[ScribalWord] = []


class ScribalTranscript(BaseModel):
    """Файл transcript.json по presigned-ссылке из GET /meetings/{id}/transcript."""
    text: str = ""
    language: Optional[str] = None
    detected_language: Optional[str] = None
    segments: list[ScribalSegment]


class ScribalUser(BaseModel):
    full_name: str = ""


class ScribalParticipant(BaseModel):
    user: ScribalUser


class ScribalMeeting(BaseModel):
    """Ответ GET /api/v1/meetings/{public_id} (нужные нам поля)."""
    public_id: str = ""
    title: str = ""
    language: Optional[str] = None
    started_at: str
    speaker_names: Optional[dict[str, str]] = None  # в примере куратора null
    participants: list[ScribalParticipant] = []


# ---------- 4.2 Результат извлечения ----------

class Source(BaseModel):
    ts: str = Field(pattern=r"^\d{2}:\d{2}:\d{2}$")
    quote: str = Field(min_length=3, description="ДОСЛОВНАЯ подстрока транскрипта")


class Decision(BaseModel):
    id: str
    project: str
    statement: str
    made_by: Optional[str] = None
    confidence: float = Field(ge=0.0, le=1.0)
    source: Source


class TaskStatus(str, Enum):
    open = "open"
    done = "done"
    cancelled = "cancelled"


class Task(BaseModel):
    id: str
    project: str
    statement: str
    owner: Optional[str] = None
    due: Optional[str] = Field(default=None, pattern=r"^\d{4}-\d{2}-\d{2}$")
    status: TaskStatus = TaskStatus.open
    source: Source


class OpenQuestion(BaseModel):
    id: str
    project: str
    statement: str
    source: Source


class Extraction(BaseModel):
    meeting_id: str
    projects_discussed: list[str]
    decisions: list[Decision] = []
    tasks: list[Task] = []
    open_questions: list[OpenQuestion] = []


# ---------- 4.3 Связи ----------

LinkType = Literal["supersedes", "contradicts", "depends_on", "refines", "same_topic"]

LINK_CONFIDENCE_THRESHOLD = 0.6  # ниже — в links_rejected.json


class Link(BaseModel):
    from_id: str = Field(alias="from")
    to_id: str = Field(alias="to")
    type: LinkType
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str

    model_config = {"populate_by_name": True}


# ---------- 4.6 Единый интерфейс ответа (baseline и graph) ----------

class Citation(BaseModel):
    meeting_id: str
    ts: str
    quote: str


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


class Answer(BaseModel):
    answer: str
    citations: list[Citation] = []
    mode: Literal["baseline", "graph"]
    usage: Usage = Usage()
    latency_ms: int = 0
    # Встречи, попавшие в контекст ДО генерации (порядок ранжирования). Нужны для полноты поиска (A4):
    # она отделяет «поиск не нашёл» от «модель нашла, но не использовала». Аддитивное поле, 2026-11-13.
    retrieved_meetings: list[str] = []
