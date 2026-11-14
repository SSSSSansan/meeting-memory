"""Контрольный режим longctx (A5): без поиска, в контекст идут ВСЕ встречи с датой не позже as_of.

    from src.retrieval.longctx import build_longctx
    ctx = build_longctx(index, as_of="2026-09-01")   # {"context", "meetings", "context_tokens", "limit_tokens"}

Зачем: верхняя граница «модель видит всё». Если longctx не лучше baseline/graph, поиск не теряет нужное.
CONTEXT_K к режиму не применяется: это контроль, бюджет контекста не ограничен.
Тексты берутся из чанков того же индекса, что у baseline (header-v2: имена говорящих уже подставлены): встреча
собирается из своих окон без повторов перекрытия, реплики по времени, встречи по дате, у каждой заголовок. Строки
несут метку [meeting_id ts] — по ней answerer проверяет цитаты, как в остальных режимах.
Не влезает в лимит модели — ошибка LongContextTooLarge, а не молчаливая обрезка (обрезка тихо меняет сам контроль).
"""
from __future__ import annotations

import os
from typing import Optional

from src.retrieval.baseline import BaselineIndex

CHARS_PER_TOKEN = 3          # как в src/eval/estimate.py: грубо для ru/kk, с запасом (токенизатора в зависимостях нет)
# max_tokens ответа (1500) + ретрай answerer._generate: в его запрос уходят первый ответ и текст ошибок
ANSWER_RESERVE_TOKENS = 3500
DEFAULT_LIMIT_TOKENS = 128_000
MODEL_LIMITS = {"gpt-4o-mini": 128_000, "gpt-4o": 128_000, "gpt-4.1": 1_047_576, "gpt-4.1-mini": 1_047_576,
                "gpt-4.1-nano": 1_047_576}


class LongContextTooLarge(RuntimeError):
    pass


def model_limit_tokens(model: Optional[str] = None) -> int:
    """Окно модели: LLM_CONTEXT_TOKENS из окружения, иначе по имени LLM_MODEL, иначе 128k."""
    if os.environ.get("LLM_CONTEXT_TOKENS"):
        return int(os.environ["LLM_CONTEXT_TOKENS"])
    model = model or os.environ.get("LLM_MODEL", "")
    return MODEL_LIMITS.get(model, DEFAULT_LIMIT_TOKENS)


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def meeting_blocks(index: BaselineIndex, as_of: Optional[str] = None) -> list[tuple[str, str]]:
    """[(meeting_id, текст встречи с заголовком)] по дате; только встречи не позже as_of (включительно)."""
    meetings: dict[str, dict] = {}
    for c in index.chunks:
        if as_of is not None and c["date"] > as_of:
            continue
        m = meetings.setdefault(c["meeting_id"], {"date": c["date"], "title": c["title"], "lines": {}})
        for s in c["segments"]:
            m["lines"].setdefault((s["ts"], s["text"]), s["speaker"])
    out = []
    for mid, m in sorted(meetings.items(), key=lambda kv: (kv[1]["date"], kv[0])):
        lines = [f"[{mid} {ts}] {speaker}: {text}"
                 for (ts, text), speaker in sorted(m["lines"].items(), key=lambda kv: kv[0][0])]
        out.append((mid, f"=== Встреча {mid} «{m['title']}», {m['date']} ===\n" + "\n".join(lines)))
    return out


def build_longctx(index: BaselineIndex, as_of: Optional[str] = None, *, prompt_overhead: str = "",
                  limit_tokens: Optional[int] = None) -> dict:
    """prompt_overhead — остальной текст промпта (системный + вопрос), чтобы проверять лимит по всему запросу."""
    blocks = meeting_blocks(index, as_of)
    context = "\n\n".join(text for _, text in blocks)
    tokens = estimate_tokens(context)
    limit = limit_tokens or model_limit_tokens()
    total = tokens + estimate_tokens(prompt_overhead) + ANSWER_RESERVE_TOKENS
    if total > limit:
        raise LongContextTooLarge(
            f"longctx: контекст ≈{tokens} токенов ({len(blocks)} встреч до {as_of or 'конца'}), с промптом и ответом "
            f"≈{total} > лимит модели {limit} (LLM_CONTEXT_TOKENS / LLM_MODEL). Контекст не обрезается: "
            "уменьшите набор встреч или возьмите модель с большим окном.")
    return {"context": context, "meetings": [mid for mid, _ in blocks], "context_tokens": tokens,
            "limit_tokens": limit}
