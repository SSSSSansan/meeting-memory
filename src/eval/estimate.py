"""РОЛЬ A. Оценка числа вызовов и стоимости прогона до его начала. Модель не вызывается.

Грубо: 1 токен ≈ 3 символа русского/казахского текста, ответ ≈ 300 токенов. Цены — из .env
(PRICE_IN_PER_M, PRICE_OUT_PER_M, PRICE_EMBED_PER_M); нули значат «цены не заданы».
Контекст baseline = k средних чанков по встречам до as_of; контекст graph считается по-настоящему
(retrieve_subgraph не ходит в модель).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.answer.prompts import SYSTEM_PROMPT
from src.eval import systems
from src.retrieval.baseline import EMBED_BATCH, chunk_transcript

CHARS_PER_TOKEN = 3
ANSWER_TOKENS = 300


@dataclass
class Estimate:
    chat_calls_min: int = 0
    chat_calls_max: int = 0       # с одним ретраем на каждый вопрос
    embed_calls: int = 0
    prompt_tokens: int = 0        # на один проход без ретраев
    completion_tokens: int = 0
    embed_tokens: int = 0
    notes: list[str] = field(default_factory=list)

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
