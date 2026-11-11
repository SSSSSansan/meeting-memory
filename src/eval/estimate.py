"""РОЛЬ A. Оценка числа вызовов и стоимости прогона до его начала. Модель не вызывается.

Грубо: 1 токен ≈ 3 символа русского/казахского текста, ответ ≈ 300 токенов. Цены — из .env
(PRICE_IN_PER_M, PRICE_OUT_PER_M, PRICE_EMBED_PER_M); нули значат «цены не заданы».
Контекст baseline = k средних чанков по встречам до as_of; контекст graph считается по-настоящему
(retrieve_subgraph не ходит в модель).
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Sequence

from scripts import cost_guard
from src.answer.prompts import SYSTEM_PROMPT
from src.eval import judge as judge_mod
from src.eval import systems
from src.retrieval.baseline import EMBED_BATCH, chunk_transcript

CHARS_PER_TOKEN = 3
ANSWER_TOKENS = 300
JUDGE_COMPLETION_TOKENS = 100   # JSON с reason до ~200 символов
JUDGE_MIN_MEETINGS = 3          # ответ может сослаться на встречи вне source_meetings: добираем до трёх (≈7000 токенов)


@dataclass
class Estimate:
    chat_calls_min: int = 0
    chat_calls_max: int = 0       # с одним ретраем на каждый вопрос
    embed_calls: int = 0
    prompt_tokens: int = 0        # на один проход без ретраев
    completion_tokens: int = 0
    embed_tokens: int = 0
    notes: list[str] = field(default_factory=list)

    @staticmethod
    def priced() -> bool:
        return any(float(os.environ.get(k, "0") or 0) for k in ("PRICE_IN_PER_M", "PRICE_OUT_PER_M"))

    def cost_usd(self) -> tuple[float, float]:
        """(без ретраев, с ретраями)."""
        p_in, p_out, p_emb = (float(os.environ.get(k, "0") or 0)
                              for k in ("PRICE_IN_PER_M", "PRICE_OUT_PER_M", "PRICE_EMBED_PER_M"))
        base = (self.prompt_tokens * p_in + self.completion_tokens * p_out + self.embed_tokens * p_emb) / 1e6
        extra = (self.prompt_tokens * 1.4 * p_in + self.completion_tokens * p_out) / 1e6  # ретрай ≈ 2.4× вход
        return base, base + extra

    def lines(self) -> list[str]:
        lo, hi = self.cost_usd()
        out = [f"вызовов модели (chat): {self.chat_calls_min}–{self.chat_calls_max}",
               f"вызовов эмбеддингов: {self.embed_calls} (≈{self.embed_tokens} токенов)",
               f"токенов chat: ≈{self.prompt_tokens} вход + ≈{self.completion_tokens} выход (без ретраев)"]
        if not any(float(os.environ.get(k, "0") or 0) for k in ("PRICE_IN_PER_M", "PRICE_OUT_PER_M")):
            out.append("стоимость: цены не заданы (PRICE_IN_PER_M / PRICE_OUT_PER_M в .env = 0), оценить нельзя")
        else:
            out.append(f"стоимость: ≈${lo:.4f} без ретраев, до ≈${hi:.4f} с ретраями")
        return out + self.notes


def estimate_run(plan: dict[tuple[str, str], list[str]], *, raw_dir: Path, k: int, depth: int) -> Estimate:
    """plan: {(mode, as_of): [тексты вопросов]}. Заглушки (graph без графа) вызовов не делают."""
    est = Estimate()
    for (mode, as_of), questions in sorted(plan.items()):
        n = len(questions)
        if mode == "graph":
            reason = systems.graph_problem(raw_dir, as_of)
            if reason:
                est.notes.append(f"graph до {as_of}: заглушка, вызовов 0 ({reason})")
                continue
            from src.retrieval.graph_rag import retrieve_subgraph
            ctx_chars = [len(retrieve_subgraph(q, depth=depth)["context"]) for q in questions]
        else:
            chunks = [c for t in systems.load_transcripts(raw_dir, as_of) for c in chunk_transcript(t)]
            if not chunks:
                est.notes.append(f"baseline до {as_of}: нет встреч, вызовов 0")
                continue
            chars = sum(len(c["text"]) for c in chunks)
            est.embed_calls += -(-len(chunks) // EMBED_BATCH) + n  # индекс + по запросу на вопрос
            est.embed_tokens += chars // CHARS_PER_TOKEN + sum(len(q) for q in questions) // CHARS_PER_TOKEN
            ctx_chars = [k * chars // len(chunks) + 150] * n
        est.chat_calls_min += n
        est.chat_calls_max += 2 * n
        chars_in = sum(len(SYSTEM_PROMPT) + len(q) + c for q, c in zip(questions, ctx_chars))
        est.prompt_tokens += chars_in // CHARS_PER_TOKEN
        est.completion_tokens += n * ANSWER_TOKENS
    return est


# ---------- судья (--judge) ----------

@dataclass
class JudgeEstimate:
    calls: int = 0                  # по числу ответов; одинаковые входы harness пропустит, так что это верхняя граница
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_lo: Optional[float] = None   # None: цены судьи не заданы
    cost_hi: Optional[float] = None   # если каждый вердикт придётся повторить из-за формата

    def lines(self) -> list[str]:
        out = [f"вызовов судьи: до {self.calls} (по одному на ответ; повторяющиеся входы не оцениваются дважды)",
               f"токенов судьи: ≈{self.prompt_tokens} вход + ≈{self.completion_tokens} выход "
               f"(транскрипты: source_meetings, добранные до {JUDGE_MIN_MEETINGS} встреч — в ответе могут быть другие)"]
        if self.cost_lo is None:
            out.append("стоимость судьи: цены не заданы (JUDGE_PRICE_* / PRICE_* в .env = 0), оценить нельзя")
        else:
            out.append(f"стоимость судьи: ≈${self.cost_lo:.4f} без ретраев, до ≈${self.cost_hi:.4f} с ретраями")
        return out


def _avg_meeting_chars(raw_dir: Path, as_of: str) -> int:
    sizes = []
    for mid in systems.raw_meeting_ids(raw_dir, as_of):
        doc = judge_mod.load_meeting_doc(raw_dir, mid)
        if doc is not None:
            sizes.append(len(doc.text))
    return sum(sizes) // len(sizes) if sizes else 0


def estimate_judge(questions: Sequence, modes: Sequence[str], *, raw_dir: Path) -> JudgeEstimate:
    """Судья вызывается на каждый настоящий (не stub) ответ каждого режима. Ответы ещё не получены, поэтому вход
    считаем по настоящему сообщению судьи с ответом средней длины. Модель не вызывается."""
    est = JudgeEstimate()
    chars = 0
    avg_cache: dict[str, int] = {}
    graph_stub: dict[str, bool] = {}
    for mode in modes:
        for q in questions:
            if mode == "graph":
                if q.as_of not in graph_stub:
                    graph_stub[q.as_of] = bool(systems.graph_problem(raw_dir, q.as_of))
                if graph_stub[q.as_of]:
                    continue
            docs = list(judge_mod.select_meetings(q.source_meetings, [], q.as_of, raw_dir).docs)
            if len(docs) < JUDGE_MIN_MEETINGS:
                if q.as_of not in avg_cache:
                    avg_cache[q.as_of] = _avg_meeting_chars(raw_dir, q.as_of)
                filler = judge_mod.MeetingDoc("", "", "", "x" * avg_cache[q.as_of])
                docs += [filler] * (JUDGE_MIN_MEETINGS - len(docs))
            msgs = judge_mod.build_messages(q.question, q.expected_answer, "x" * ANSWER_TOKENS * CHARS_PER_TOKEN,
                                            q.answerable, meetings=docs)
            chars += sum(len(m["content"]) for m in msgs)
            est.calls += 1
    est.prompt_tokens = chars // CHARS_PER_TOKEN
    est.completion_tokens = est.calls * JUDGE_COMPLETION_TOKENS
    p_in, p_out = judge_mod.judge_prices()
    if p_in or p_out:
        est.cost_lo = (est.prompt_tokens * p_in + est.completion_tokens * p_out) / 1e6
        est.cost_hi = est.cost_lo + (est.prompt_tokens * 1.4 * p_in + est.completion_tokens * p_out) / 1e6
    return est


def total_cost(answers: Estimate, judge: Optional[JudgeEstimate]) -> Optional[tuple[float, float]]:
    """(без ретраев, с ретраями) ответы + судья; None, если какой-то из цен нет."""
    if not answers.priced() or (judge is not None and judge.cost_lo is None):
        return None
    lo, hi = answers.cost_usd()
    if judge is not None:
        lo, hi = lo + judge.cost_lo, hi + judge.cost_hi
    return lo, hi


def remaining_budget() -> float:
    """Остаток BUDGET_USD по runs/budget.json (cost_guard)."""
    path = cost_guard.BUDGET_FILE
    spent = json.loads(path.read_text(encoding="utf-8"))["spent_usd"] if path.exists() else 0.0
    return float(os.environ.get("BUDGET_USD", "3.0")) - spent
