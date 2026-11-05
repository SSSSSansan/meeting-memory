import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import scripts.cost_guard as cg
import src.llm_client as llm

ROOT = Path(__file__).resolve().parent.parent
MSGS = [{"role": "user", "content": "Сколько будет 2+2?"}]


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "RUNS", tmp_path)
    monkeypatch.setattr(llm, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(cg, "BUDGET_FILE", tmp_path / "budget.json")
    for k in ("LLM_BACKEND", "LLM_MODEL", "EMBED_MODEL", "PRICE_IN_PER_M", "PRICE_OUT_PER_M",
              "PRICE_EMBED_PER_M", "BUDGET_USD"):
        monkeypatch.delenv(k, raising=False)


def _rows(tmp_path) -> list[dict]:
    path = tmp_path / "calls.jsonl"
    return [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines()] if path.exists() else []


class FakeOpenAI:
    """Повторяет форму ответа openai: chat.completions.create и embeddings.create."""

    def __init__(self, text='{"answer": "4"}'):
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))
        self.embeddings = SimpleNamespace(create=self._embed)
        self.text = text

    def _chat(self, **kw):
        self.calls.append(("chat", kw))
        usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=200,
                                prompt_tokens_details=SimpleNamespace(cached_tokens=300))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.text))],
                               usage=usage)

    def _embed(self, **kw):
        self.calls.append(("embed", kw))
        return SimpleNamespace(data=[SimpleNamespace(embedding=[0.1, 0.2]) for _ in kw["input"]],
                               usage=SimpleNamespace(prompt_tokens=50))


def _use_fake(monkeypatch, backend, fake):
    monkeypatch.setenv("LLM_BACKEND", backend)
    monkeypatch.setenv("LLM_MODEL", "m-test")
    monkeypatch.setattr(llm, "_client", lambda b: fake)


@pytest.mark.parametrize("backend", ["openai", "local"])
def test_backends_return_same_shape_with_usage_and_cost(monkeypatch, tmp_path, backend):
    fake = FakeOpenAI()
    _use_fake(monkeypatch, backend, fake)
    monkeypatch.setenv("PRICE_IN_PER_M", "0.5")
    monkeypatch.setenv("PRICE_OUT_PER_M", "2")
    r = llm.complete(MSGS, json_schema={"type": "object"}, purpose="answer_baseline")
    assert (r.text, r.prompt_tokens, r.completion_tokens, r.cached_tokens) == ('{"answer": "4"}', 1000, 200, 300)
    assert r.cost_usd == pytest.approx((1000 * 0.5 + 200 * 2) / 1e6) and r.model == "m-test"
    assert fake.calls[0][1]["response_format"] == {"type": "json_object"}
    row = _rows(tmp_path)[0]
    assert (row["backend"], row["purpose"], row["kind"]) == (backend, "answer_baseline", "chat")
    assert row["from_cache"] is False
    assert row["spent_usd"] == pytest.approx(r.cost_usd)
    assert cg._read()["spent_usd"] == pytest.approx(r.cost_usd)


def test_no_response_format_without_schema(monkeypatch):
    fake = FakeOpenAI(text="просто текст")
    _use_fake(monkeypatch, "openai", fake)
    assert llm.complete(MSGS).text == "просто текст"
    assert "response_format" not in fake.calls[0][1]


def test_repeat_call_comes_from_cache_and_costs_nothing(monkeypatch, tmp_path):
    fake = FakeOpenAI()
    _use_fake(monkeypatch, "openai", fake)
    monkeypatch.setenv("PRICE_IN_PER_M", "1")
    first = llm.complete(MSGS, purpose="x")
    spent = cg._read()["spent_usd"]
    second = llm.complete(MSGS, purpose="x")
    assert len(fake.calls) == 1 and second.from_cache and second.text == first.text
    assert cg._read()["spent_usd"] == spent
    assert [r["spent_usd"] for r in _rows(tmp_path)] == [pytest.approx(first.cost_usd), 0.0]

    llm.complete(MSGS, temperature=0.7, purpose="x")   # другой параметр — другой ключ кэша
    assert len(fake.calls) == 2


def test_budget_exceeded_stops_before_calling_model(monkeypatch):
    fake = FakeOpenAI()
    _use_fake(monkeypatch, "openai", fake)
    monkeypatch.setenv("BUDGET_USD", "0.001")
    cg.record_spend(0.001)
    with pytest.raises(cg.BudgetExceeded):
        llm.complete(MSGS)
    with pytest.raises(cg.BudgetExceeded):
        llm.embed(["текст"])
    assert fake.calls == []


def test_tiny_spends_are_not_rounded_away():
    for _ in range(10):
        cg.record_spend(1e-7)
    assert cg._read()["spent_usd"] == pytest.approx(1e-6)


def test_embed_logs_tokens_and_cost(monkeypatch, tmp_path):
    fake = FakeOpenAI()
    _use_fake(monkeypatch, "local", fake)
    monkeypatch.setenv("EMBED_MODEL", "BAAI/bge-m3")
    monkeypatch.setenv("PRICE_EMBED_PER_M", "0.02")
    assert llm.embed(["а", "б"], purpose="embed_index") == [[0.1, 0.2], [0.1, 0.2]]
    assert fake.calls[0][1] == {"model": "BAAI/bge-m3", "input": ["а", "б"]}
    row = _rows(tmp_path)[0]
    assert (row["kind"], row["purpose"], row["prompt_tokens"]) == ("embed", "embed_index", 50)
    assert row["cost_usd"] == pytest.approx(50 * 0.02 / 1e6)


def test_stub_is_deterministic_and_free(tmp_path):
    a = llm.complete(MSGS, json_schema={"type": "object"})
    assert a.text == '{"stub": true}' and a.cost_usd == 0
    assert llm.embed(["x"]) == llm.embed(["x"]) and llm.embed(["x"]) != llm.embed(["y"])
    assert cg._read()["spent_usd"] == 0


def test_unknown_backend_rejected(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "gpt5-direct")
    with pytest.raises(ValueError):
        llm.complete(MSGS)
    with pytest.raises(ValueError):
        llm.embed(["x"])


def test_client_gets_retry_and_timeout_settings(monkeypatch):
    seen = {}

    class Spy:
        def __init__(self, **kw):
            seen.update(kw)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=Spy))
    monkeypatch.setenv("LLM_MAX_RETRIES", "5")
    llm._client("local")
    assert seen["max_retries"] == 5 and seen["timeout"] == 120.0 and seen["base_url"]


def test_only_llm_client_talks_to_model_providers():
    pattern = re.compile(r"^\s*(import openai|from openai\b|import requests|from requests\b)", re.M)
    offenders = [str(p.relative_to(ROOT)) for folder in ("src", "scripts", "app")
                 for p in (ROOT / folder).rglob("*.py")
                 if p.name != "llm_client.py" and pattern.search(p.read_text(encoding="utf-8"))]
    assert offenders == []


@pytest.mark.parametrize("args", [["-m", "scripts.cost_guard", "--status"],
                                  ["scripts/cost_guard.py", "--status"],
                                  ["-m", "scripts.cost_guard"]])
def test_cost_guard_status_cli(args):
    out = subprocess.run([sys.executable, *args], cwd=ROOT, capture_output=True, text=True,
                         encoding="utf-8", env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
    assert out.returncode == 0, out.stderr
    assert "Потрачено $" in out.stdout and "осталось" in out.stdout
