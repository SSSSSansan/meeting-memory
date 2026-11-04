"""РОЛЬ C. Генерация ответа с цитатами. ОДИН промпт для baseline и graph (раздел 11.3).

    python -m src.answer.answerer "Когда релиз?"                  # baseline по data/transcripts
    python -m src.answer.answerer "Как менялся срок?" --mode graph
    python -m src.answer.answerer "Как менялся срок?" --mode auto   # ветку выбирает src.retrieval.router

Возвращает src.schemas.Answer — единый формат 4.6 для обеих веток:
    answer(question, mode="baseline")   — поиск по чанкам → контекст → LLM
    answer(question, mode="graph")      — подграф (src.retrieval.graph_rag) → контекст → тот же LLM-шаг
    generate(question, context, mode)   — только LLM-часть, общая для обеих веток

Что делает generate():
  1. Просит модель ответить JSON с цитатами по промпту 11.3.
  2. Проверяет каждую цитату: она должна дословно (после normalize) быть в строке контекста с той же
     меткой [meeting_id ts]. Верная цитата с чужой меткой переносится на единственную подходящую строку.
  3. Если цитаты не подтвердились или JSON сломан — один раз просит исправить, показав ошибки.
  4. Оставшиеся неподтверждённые цитаты выбрасывает и помечает это в тексте ответа.
  5. Пишет статистику в runs/answer_report.jsonl — из неё считается доля невалидных цитат.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional

from src.answer.prompts import SYSTEM_PROMPT, retry_prompt, user_prompt
from src.llm_client import complete
from src.retrieval.baseline import ROOT, BaselineIndex, load_transcripts
from src.retrieval.graph_rag import retrieve_subgraph
from src.retrieval.router import route
from src.schemas import Answer, Citation, Usage
from src.utils.text import normalize

REPORT = ROOT / "runs" / "answer_report.jsonl"
NO_CONTEXT_ANSWER = "В контексте нет информации для ответа на этот вопрос."

_LINE = re.compile(r"^\[(\S+) (\d{2}:\d{2}:\d{2})\] .*?: (.*)$", re.M)
_QUOTE_MARKS = " \"'«»„“”…"

_index: Optional[BaselineIndex] = None


@dataclass
class AnswerStats:
    mode: str                    # какой веткой реально ответили
    question: str = ""
    requested_mode: str = ""     # что попросили: baseline | graph | auto
    route_qtype: str = ""        # для auto: как роутер понял вопрос (T1..T5) и почему
    route_reason: str = ""
    attempts: int = 0
    json_failed_first: bool = False
    citations_first: int = 0
    invalid_first: int = 0       # не подтвердились с первой попытки — это и есть доля выдуманных цитат
    relocated: int = 0           # цитата дословная, но метка не та: исправили по контексту
    citations_final: int = 0
    dropped_final: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


def default_index() -> BaselineIndex:
    """Индекс по data/transcripts, собирается один раз на процесс."""
    global _index
    if _index is None:
        _index = BaselineIndex()
        _index.add(load_transcripts(ROOT / "data" / "transcripts"))
    return _index


def format_context(hits: list[dict]) -> str:
    """Чанки → текст для модели: встречи по дате, реплики по времени, без повторов.

    Соседние чанки перекрываются, без склейки одна реплика попала бы в контекст дважды.
    Каждая строка несёт метку [meeting_id ts] — из неё модель берёт ссылку для цитаты.
    """
    meetings: dict[str, dict] = {}
    for h in hits:
        m = meetings.setdefault(h["meeting_id"], {"date": h["date"], "title": h["title"], "lines": {}})
        for s in h["segments"]:
            m["lines"].setdefault((s["ts"], s["text"]), s["speaker"])
    blocks = []
    for mid, m in sorted(meetings.items(), key=lambda kv: (kv[1]["date"], kv[0])):
        lines = [f"[{mid} {ts}] {speaker}: {text}"
                 for (ts, text), speaker in sorted(m["lines"].items(), key=lambda kv: kv[0][0])]
        blocks.append(f"=== Встреча {mid} «{m['title']}», {m['date']} ===\n" + "\n".join(lines))
    return "\n\n".join(blocks)


def _parse(text: str) -> tuple[Optional[dict], Optional[str]]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        return None, str(e)
    if not isinstance(data, dict) or not isinstance(data.get("answer"), str):
        return None, 'нужен объект с полями "answer" (строка) и "citations" (список)'
    return data, None


def _citations(raw) -> list[Citation]:
    out = []
    for c in raw if isinstance(raw, list) else []:
        if not isinstance(c, dict):
            continue
        fields = {k: str(c.get(k) or "").strip() for k in ("meeting_id", "ts", "quote")}
        fields["quote"] = fields["quote"].strip(_QUOTE_MARKS)
        if all(fields.values()):
            out.append(Citation(**fields))
    return out


def _verify(citations: list[Citation], context: str) -> tuple[list[Citation], list[Citation], int]:
    """Делит цитаты на подтверждённые и нет. Возвращает (good, bad, relocated)."""
    lines = [(mid, ts, normalize(text)) for mid, ts, text in _LINE.findall(context)]
    good, bad, relocated, seen = [], [], 0, set()
    for c in citations:
        q = normalize(c.quote)
        if any(mid == c.meeting_id and ts == c.ts and q in text for mid, ts, text in lines):
            fixed = c
        else:
            places = {(mid, ts) for mid, ts, text in lines if q in text}
            if len(places) != 1:  # нигде нет — выдумана; в нескольких местах — неясно, на что ссылаться
                bad.append(c)
                continue
            (mid, ts), = places
            fixed = Citation(meeting_id=mid, ts=ts, quote=c.quote)
            relocated += 1
        key = (fixed.meeting_id, fixed.ts, normalize(fixed.quote))
        if key not in seen:
            seen.add(key)
            good.append(fixed)
    return good, bad, relocated


def generate(question: str, context: str, mode: str,
             stats: Optional[AnswerStats] = None) -> Answer:
    """context — строки вида «[meeting_id ts] Говорящий: текст»: по ним проверяются цитаты."""
    stats = stats or AnswerStats(mode=mode, question=question)
    if not context.strip():
        _write_report(stats)
        return Answer(answer=NO_CONTEXT_ANSWER, mode=mode)

    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(question, context)}]
    best: Optional[tuple[dict, list[Citation], list[Citation], int]] = None
    raw_text = ""
    for attempt in (1, 2):
        stats.attempts = attempt
        r = complete(messages, json_schema={"type": "object"}, temperature=0.0, max_tokens=1500,
                     purpose=f"answer_{mode}" if attempt == 1 else f"answer_{mode}_retry")
        stats.prompt_tokens += r.prompt_tokens
        stats.completion_tokens += r.completion_tokens
        stats.cost_usd += r.cost_usd
        raw_text = raw_text or r.text

        data, err = _parse(r.text)
        good, bad, relocated = _verify(_citations(data.get("citations")), context) if data else ([], [], 0)
        if attempt == 1:
            stats.json_failed_first = data is None
            stats.citations_first = len(good) + len(bad)
            stats.invalid_first = len(bad)
        if data is not None:
            best = (data, good, bad, relocated)
        if data is not None and not bad:
            break
        if attempt == 1:
            # Тот же запрос вернулся бы из кэша llm_client — поэтому повтор только с текстом ошибки.
            messages += [{"role": "assistant", "content": r.text},
                         {"role": "user", "content": retry_prompt([c.model_dump() for c in bad], err)}]

    usage = Usage(prompt_tokens=stats.prompt_tokens, completion_tokens=stats.completion_tokens,
                  cost_usd=stats.cost_usd)
    if best is None:
        # Ни разу не JSON — отдаём текст как есть, но без цитат: непроверяемая ссылка хуже отсутствующей.
        _write_report(stats)
        return Answer(answer=raw_text.strip() or NO_CONTEXT_ANSWER, mode=mode, usage=usage)

    data, good, bad, relocated = best
    stats.relocated, stats.citations_final, stats.dropped_final = relocated, len(good), len(bad)
    text = data["answer"].strip()
    if bad:
        text += f"\n\n(Удалено неподтверждённых цитат: {len(bad)} — их нет дословно в источниках.)"
    _write_report(stats)
    return Answer(answer=text, citations=good, mode=mode, usage=usage)


def answer(question: str, mode: str = "baseline", k: int = 5,
           index: Optional[BaselineIndex] = None, stats: Optional[AnswerStats] = None,
           depth: int = 2, graph=None) -> Answer:
    """mode: baseline | graph | auto (выбирает src.retrieval.router).

    k — сколько чанков берёт baseline; depth и graph — параметры graph-ветки.
    """
    t0 = time.perf_counter()
    requested, context, chosen = mode, None, None
    if mode == "auto":
        chosen = route(question, graph)
        mode = chosen.mode
        if mode == "graph":
            sub = retrieve_subgraph(question, depth=depth, G=graph)
            if sub["items"]:
                context = sub["context"]
            else:
                mode = "baseline"
                chosen.reason += "; в графе по вопросу ничего нет — отвечает baseline"
    if mode == "baseline":
        context = format_context((index or default_index()).search(question, k=k))
    elif mode == "graph":
        if context is None:
            context = retrieve_subgraph(question, depth=depth, G=graph)["context"]
    else:
        raise ValueError(f"mode={mode!r}: допустимо baseline | graph | auto")

    stats = stats or AnswerStats(mode=mode, question=question)
    stats.mode, stats.requested_mode = mode, requested
    if chosen:
        stats.route_qtype, stats.route_reason = chosen.qtype, chosen.reason
    result = generate(question, context, mode, stats)
    result.latency_ms = int((time.perf_counter() - t0) * 1000)
    return result


def _write_report(stats: AnswerStats) -> None:
    REPORT.parent.mkdir(exist_ok=True)
    row = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **stats.__dict__}
    with open(REPORT, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--mode", choices=["baseline", "graph", "auto"], default="baseline")
    ap.add_argument("--depth", type=int, default=2)
    a = ap.parse_args()
    print(answer(a.question, mode=a.mode, k=a.k, depth=a.depth).model_dump_json(indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
