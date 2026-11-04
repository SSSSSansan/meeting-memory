"""Единственная точка входа в LLM и эмбеддинги.

Весь остальной код ходит в модель ТОЛЬКО через complete() и embed().
Backend переключается переменной LLM_BACKEND в .env:

    stub   — фиксированные ответы, ничего не стоит, для разработки без модели
    openai — OpenAI API
    local  — любой OpenAI-совместимый сервер (vLLM / Ollama) по LOCAL_BASE_URL

Каждый вызов пишется в runs/calls.jsonl (backend, purpose, токены, задержка, стоимость),
проходит через cost_guard, ответы чата кэшируются по хэшу в runs/cache/.
Сетевые сбои, 429 и 5xx повторяет сам клиент openai (LLM_MAX_RETRIES, по умолчанию 3);
повтор при невалидном JSON — забота вызывающего кода (extract, answer), он знает, что исправлять.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "runs"
CACHE = RUNS / "cache"


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    latency_ms: int = 0
    cost_usd: float = 0.0
    model: str = ""
    from_cache: bool = False


def _env(name: str, default: str = "") -> str:
    return os.environ.get(name, default)


def _load_dotenv() -> None:
    path = ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


_load_dotenv()


def _price(model: str, kind: str) -> float:
    """USD за 1M токенов из .env (PRICE_IN_PER_M / PRICE_OUT_PER_M).

    Цены НЕ зашиты в код: они меняются. Возьмите актуальные со страницы провайдера.
    """
    key = "PRICE_IN_PER_M" if kind == "in" else "PRICE_OUT_PER_M"
    return float(_env(key, "0"))


def _cache_key(backend: str, model: str, messages: list[dict], temperature: float,
               json_schema: Optional[dict]) -> str:
    blob = json.dumps([backend, model, messages, temperature, json_schema],
                      ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _log_call(resp: LLMResponse, purpose: str, kind: str = "chat") -> None:
    RUNS.mkdir(exist_ok=True)
    with open(RUNS / "calls.jsonl", "a", encoding="utf-8") as f:
        row = asdict(resp)
        row.update(purpose=purpose, kind=kind, backend=_env("LLM_BACKEND", "stub"),
                   ts=time.strftime("%Y-%m-%dT%H:%M:%S"),
                   # cost_usd — цена ответа (для сравнения режимов), spent_usd — реально потрачено:
                   # у ответа из кэша они различаются, для отчёта о расходах суммировать spent_usd
                   spent_usd=0.0 if resp.from_cache else resp.cost_usd)
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def complete(messages: list[dict], *, json_schema: Optional[dict] = None,
             temperature: float = 0.0, max_tokens: int = 1024,
             purpose: str = "generic") -> LLMResponse:
    """Один вызов чат-модели. Возвращает LLMResponse с usage и задержкой."""
    from scripts.cost_guard import check_budget, record_spend  # локальный импорт: нет циклов

    backend = _env("LLM_BACKEND", "stub")
    model = _env("LLM_MODEL", "stub-model")

    key = _cache_key(backend, model, messages, temperature, json_schema)
    CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = CACHE / f"{key}.json"
    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding="utf-8"))
        data["from_cache"] = True
        resp = LLMResponse(**data)
        _log_call(resp, purpose)
        return resp

    check_budget()
    t0 = time.perf_counter()

    if backend == "stub":
        resp = _complete_stub(messages, json_schema)
    elif backend in ("openai", "local"):
        resp = _complete_openai_compatible(backend, model, messages, json_schema,
                                           temperature, max_tokens)
    else:
        raise ValueError(f"Неизвестный LLM_BACKEND={backend!r}. Допустимо: stub | openai | local")

    resp.latency_ms = int((time.perf_counter() - t0) * 1000)
    resp.model = model
    resp.cost_usd = (resp.prompt_tokens * _price(model, "in")
                     + resp.completion_tokens * _price(model, "out")) / 1_000_000

    cache_file.write_text(json.dumps(asdict(resp), ensure_ascii=False), encoding="utf-8")
    record_spend(resp.cost_usd)
    _log_call(resp, purpose)
    return resp


def _complete_stub(messages: list[dict], json_schema: Optional[dict]) -> LLMResponse:
    text = '{"stub": true}' if json_schema else "[stub] LLM_BACKEND=stub — подключите модель в .env"
    approx = sum(len(m.get("content", "")) for m in messages) // 4
    return LLMResponse(text=text, prompt_tokens=approx, completion_tokens=8)


def _client(backend: str):
    from openai import OpenAI  # импорт здесь: для stub пакет не нужен

    retry = dict(max_retries=int(_env("LLM_MAX_RETRIES", "3")),
                 timeout=float(_env("LLM_TIMEOUT_S", "120")))
    if backend == "openai":
        return OpenAI(api_key=_env("OPENAI_API_KEY"), **retry)
    return OpenAI(base_url=_env("LOCAL_BASE_URL", "http://localhost:8000/v1"),
                  api_key=_env("LOCAL_API_KEY", "not-needed"), **retry)


def _complete_openai_compatible(backend: str, model: str, messages: list[dict],
                                json_schema: Optional[dict], temperature: float,
                                max_tokens: int) -> LLMResponse:
    kwargs = dict(model=model, messages=messages, temperature=temperature,
                  max_tokens=max_tokens)
    if json_schema:
        kwargs["response_format"] = {"type": "json_object"}
    r = _client(backend).chat.completions.create(**kwargs)
    usage = r.usage
    cached = 0
    details = getattr(usage, "prompt_tokens_details", None)
    if details is not None:
        cached = getattr(details, "cached_tokens", 0) or 0
    return LLMResponse(text=r.choices[0].message.content or "",
                       prompt_tokens=usage.prompt_tokens,
                       completion_tokens=usage.completion_tokens,
                       cached_tokens=cached)


def embed(texts: list[str], *, purpose: str = "embed") -> list[list[float]]:
    """Эмбеддинги. stub возвращает детерминированные векторы из хэша (бесплатно, не логируется)."""
    backend = _env("LLM_BACKEND", "stub")
    if backend == "stub":
        out = []
        for t in texts:
            h = hashlib.sha256(t.encode("utf-8")).digest()
            out.append([b / 255.0 for b in h[:16]])
        return out
    if backend not in ("openai", "local"):
        raise ValueError(f"Неизвестный LLM_BACKEND={backend!r}. Допустимо: stub | openai | local")
    from scripts.cost_guard import check_budget, record_spend  # локальный импорт: нет циклов

    check_budget()
    model = _env("EMBED_MODEL", "text-embedding-3-small")
    t0 = time.perf_counter()
    r = _client(backend).embeddings.create(model=model, input=texts)
    tokens = getattr(getattr(r, "usage", None), "prompt_tokens", 0) or 0
    resp = LLMResponse(text="", prompt_tokens=tokens, model=model,
                       latency_ms=int((time.perf_counter() - t0) * 1000),
                       cost_usd=tokens * float(_env("PRICE_EMBED_PER_M", "0")) / 1_000_000)
    record_spend(resp.cost_usd)
    _log_call(resp, purpose, kind="embed")
    return [d.embedding for d in r.data]
