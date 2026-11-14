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
from src.eval import judge_facts as facts_mod
from src.eval import systems
from src.retrieval.baseline import chunk_transcript

CHARS_PER_TOKEN = 3
ANSWER_TOKENS = 300
JUDGE_COMPLETION_TOKENS = 100   # JSON с reason до ~200 символов
FACTS_CLAIMS_GUESS = 6          # атомарных утверждений в среднем ответе (ответ ≈ 300 токенов)
FACTS_TOKENS_PER_FLAG = 15      # {"id": "R1", "present": true} ≈ 15 токенов
FACTS_TOKENS_PER_CLAIM = 30     # {"text": "...", "label": "..."} ≈ 30 токенов
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


def _chunks_until(raw_dir: Path, as_of: str) -> list[dict]:
    return [c for t in systems.load_transcripts(raw_dir, as_of) for c in chunk_transcript(t)]


def reindex_note(raw_dir: Path) -> str:
    """Сколько стоит пересобрать индекс baseline по всем встречам raw_dir (делает не harness)."""
    chunks = [c for t in systems.load_transcripts(raw_dir, "9999-12-31") for c in chunk_transcript(t)]
    tokens = sum(len(c["text"]) + 150 for c in chunks) // CHARS_PER_TOKEN   # + заголовок чанка
    price = float(os.environ.get("PRICE_EMBED_PER_M", "0") or 0)
    cost = f", ≈${tokens * price / 1e6:.4f}" if price else ""
    return f"Переиндексация: ≈{len(chunks)} чанков, ≈{tokens} токенов эмбеддингов{cost} (в оценку ниже не входит)."


def estimate_run(plan: dict[tuple[str, str], list[str]], *, raw_dir: Path, k: int, depth: int,
                 repeats: int = 1) -> Estimate:
    """plan: {(mode, as_of_date): [тексты вопросов]}; as_of — ISO-дата (или meeting_id). Заглушки (graph без графа)
    вызовов не делают. Индекс baseline готовый (harness его не строит): эмбеддинги — только вопросы, по одному на
    ответ (hybrid тоже один: вектор вопроса переиспользуется). Оценка — при готовом индексе.
    repeats: каждый повтор — полный проход по вопросам мимо кэша, поэтому вызовы, токены и эмбеддинги запросов ×repeats.
    hybrid: контекст graph + до HYBRID_CHUNKS чанков."""
    from src.retrieval.graph_rag import retrieve_subgraph
    from src.retrieval.hybrid import HYBRID_CHUNKS

    est = Estimate()
    for (mode, as_of), questions in sorted(plan.items()):
        n = len(questions)
        as_of_date = systems.as_of_date_of(as_of, raw_dir) or as_of
        ctx_chars = [0] * n
        if mode in systems.GRAPH_MODES:
            reason = systems.graph_problem(raw_dir, as_of_date)
            if reason:
                est.notes.append(f"{mode} до {as_of_date}: заглушка, вызовов 0 ({reason})")
                continue
            ctx_chars = [len(retrieve_subgraph(q, depth=depth, as_of=as_of_date)["context"]) for q in questions]
        if mode in systems.INDEX_MODES:
            chunks = _chunks_until(raw_dir, as_of_date)
            if not chunks:
                est.notes.append(f"{mode} до {as_of_date}: нет встреч, вызовов 0")
                continue
            avg = sum(len(c["text"]) for c in chunks) // len(chunks)
            n_chunks = HYBRID_CHUNKS if mode == "hybrid" else k
            est.embed_calls += n * repeats  # индекс готов: только вектор вопроса на ответ и повтор
            est.embed_tokens += repeats * sum(len(q) for q in questions) // CHARS_PER_TOKEN
            ctx_chars = [c + n_chunks * avg + 150 for c in ctx_chars]
        est.chat_calls_min += n * repeats
        est.chat_calls_max += 2 * n * repeats
        chars_in = sum(len(SYSTEM_PROMPT) + len(q) + c for q, c in zip(questions, ctx_chars))
        est.prompt_tokens += repeats * chars_in // CHARS_PER_TOKEN
        est.completion_tokens += repeats * n * ANSWER_TOKENS
    if repeats > 1:
        est.notes.append(f"повторов: {repeats}; ответы повторов идут мимо кэша и оплачиваются все "
                         f"(оценка уже ×{repeats})")
    return est


# ---------- судья (--judge) ----------

@dataclass
class JudgeEstimate:
    calls: int = 0                  # по числу ответов; одинаковые входы harness пропустит, так что это верхняя граница
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_lo: Optional[float] = None   # None: цены судьи не заданы
    cost_hi: Optional[float] = None   # если каждый вердикт придётся повторить из-за формата
    repeats: int = 1

    def lines(self) -> list[str]:
        again = (f" и повтор: {self.repeats} повторов оцениваются отдельно, но повтор с тем же ответом судья берёт "
                 "из кэша по хэшу входа за $0, так что это верхняя граница" if self.repeats > 1 else "")
        out = [f"вызовов судьи: до {self.calls} (по одному на ответ{again}; повторяющиеся входы не оцениваются дважды)",
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


def _graph_stub(mode: str, q, raw_dir: Path, memo: dict[str, bool]) -> bool:
    """graph и hybrid станут заглушкой, если графа нет или в нём не хватает встреч до as_of_date вопроса."""
    if mode not in systems.GRAPH_MODES:
        return False
    d = getattr(q, "as_of_date", None) or systems.as_of_date_of(q.as_of, raw_dir) or q.as_of
    if d not in memo:
        memo[d] = bool(systems.graph_problem(raw_dir, d))
    return memo[d]


def estimate_judge(questions: Sequence, modes: Sequence[str], *, raw_dir: Path, repeats: int = 1) -> JudgeEstimate:
    """Судья вызывается на каждый настоящий (не stub) ответ каждого режима и каждого повтора. Ответы ещё не получены,
    поэтому вход считаем по настоящему сообщению судьи с ответом средней длины. Модель не вызывается."""
    est = JudgeEstimate(repeats=repeats)
    chars = 0
    avg_cache: dict[str, int] = {}
    graph_stub: dict[str, bool] = {}
    for mode in modes:
        for q in questions:
            if _graph_stub(mode, q, raw_dir, graph_stub):
                continue
            docs = list(judge_mod.select_meetings(q.source_meetings, [], q.as_of, raw_dir).docs)
            if len(docs) < JUDGE_MIN_MEETINGS:
                if q.as_of not in avg_cache:
                    avg_cache[q.as_of] = _avg_meeting_chars(raw_dir, q.as_of)
                filler = judge_mod.MeetingDoc("", "", "", "x" * avg_cache[q.as_of])
                docs += [filler] * (JUDGE_MIN_MEETINGS - len(docs))
            msgs = judge_mod.build_messages(q.question, q.expected_answer, "x" * ANSWER_TOKENS * CHARS_PER_TOKEN,
                                            q.answerable, meetings=docs)
            chars += repeats * sum(len(m["content"]) for m in msgs)
            est.calls += repeats
    est.prompt_tokens = chars // CHARS_PER_TOKEN
    est.completion_tokens = est.calls * JUDGE_COMPLETION_TOKENS
    p_in, p_out = judge_mod.judge_prices()
    if p_in or p_out:
        est.cost_lo = (est.prompt_tokens * p_in + est.completion_tokens * p_out) / 1e6
        est.cost_hi = est.cost_lo + (est.prompt_tokens * 1.4 * p_in + est.completion_tokens * p_out) / 1e6
    return est


def total_cost(answers: Estimate, judge: Optional[JudgeEstimate],
               facts: Optional["FactsJudgeEstimate"] = None) -> Optional[tuple[float, float]]:
    """(без ретраев, с ретраями) ответы + судья (v2 и/или по фактам); None, если какой-то из цен нет."""
    judges = [j for j in (judge, facts) if j is not None]
    if not answers.priced() or any(j.cost_lo is None for j in judges):
        return None
    lo, hi = answers.cost_usd()
    for j in judges:
        lo, hi = lo + j.cost_lo, hi + j.cost_hi
    return lo, hi


# ---------- судья по фактам (--judge-facts) ----------

@dataclass
class FactsJudgeEstimate(JudgeEstimate):
    """Судья по фактам: без транскриптов, вход — вопрос, ответ и списки фактов; выход — флаги и claims."""

    def lines(self) -> list[str]:
        again = (f"; повторов ответов и судьи вместе {self.repeats}, каждый повтор судьи идёт мимо кэша и оплачивается"
                 if self.repeats > 1 else "")
        out = [f"вызовов судьи по фактам: до {self.calls} (ответы × повторы ответов × повторы судьи{again}; "
               "одинаковые ответы судья берёт из памяти прогона, так что это верхняя граница)",
               f"токенов судьи по фактам: ≈{self.prompt_tokens} вход + ≈{self.completion_tokens} выход "
               f"(без транскриптов; ответ ≈{ANSWER_TOKENS} токенов, ≈{FACTS_CLAIMS_GUESS} утверждений на ответ)"]
        if self.cost_lo is None:
            out.append("стоимость судьи по фактам: цены не заданы (JUDGE_PRICE_* / PRICE_* в .env = 0), оценить нельзя")
        else:
            out.append(f"стоимость судьи по фактам: ≈${self.cost_lo:.4f} без ретраев, "
                       f"до ≈${self.cost_hi:.4f} с ретраями")
        return out


def estimate_judge_facts(questions: Sequence, modes: Sequence[str], *, raw_dir: Path, repeats: int = 1,
                         judge_repeats: int = 1) -> FactsJudgeEstimate:
    """Вызовов = (настоящие ответы всех режимов и повторов) × judge_repeats. Модель не вызывается. Вход считаем по
    настоящему сообщению судьи с ответом средней длины, выход — по числу фактов вопроса."""
    est = FactsJudgeEstimate(repeats=repeats * judge_repeats)
    chars = completion = 0
    graph_stub: dict[str, bool] = {}
    for mode in modes:
        for q in questions:
            if _graph_stub(mode, q, raw_dir, graph_stub):
                continue
            if not q.required_facts and not q.forbidden_facts:
                continue    # харнес такому вопросу судью не зовёт (questions_m008.json без фактов)
            msgs = facts_mod.build_messages(q.question, "x" * ANSWER_TOKENS * CHARS_PER_TOKEN, q.required_facts,
                                            q.forbidden_facts, q.answerable)
            n_calls = repeats * judge_repeats
            chars += n_calls * sum(len(m["content"]) for m in msgs)
            completion += n_calls * (FACTS_TOKENS_PER_FLAG * (len(q.required_facts) + len(q.forbidden_facts))
                                     + FACTS_TOKENS_PER_CLAIM * FACTS_CLAIMS_GUESS)
            est.calls += n_calls
    est.prompt_tokens = chars // CHARS_PER_TOKEN
    est.completion_tokens = completion
    p_in, p_out = judge_mod.judge_prices()
    if p_in or p_out:
        est.cost_lo = (est.prompt_tokens * p_in + est.completion_tokens * p_out) / 1e6
        est.cost_hi = est.cost_lo + (est.prompt_tokens * 1.4 * p_in + est.completion_tokens * p_out) / 1e6
    return est


def remaining_budget() -> float:
    """Остаток BUDGET_USD по runs/budget.json (cost_guard)."""
    path = cost_guard.BUDGET_FILE
    spent = json.loads(path.read_text(encoding="utf-8"))["spent_usd"] if path.exists() else 0.0
    return float(os.environ.get("BUDGET_USD", "3.0")) - spent
