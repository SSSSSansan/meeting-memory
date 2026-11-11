import json

import pytest

import src.answer.answerer as ans
from src.llm_client import LLMResponse
from src.retrieval.baseline import BaselineIndex
from src.schemas import Answer, Citation, Segment, Transcript


@pytest.fixture(autouse=True)
def _stub_backend(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_BACKEND", "stub")
    monkeypatch.setenv("EMBED_MODEL", "")
    monkeypatch.setattr(ans, "REPORT", tmp_path / "answer_report.jsonl")


def _fake_llm(monkeypatch, *texts: str) -> list:
    """Подменяет LLM: отвечает текстами по очереди, последний повторяется."""
    calls = []

    def fake(messages, **kw):
        calls.append((messages, kw))
        text = texts[min(len(calls), len(texts)) - 1]
        return LLMResponse(text=text, prompt_tokens=120, completion_tokens=30, cost_usd=0.0002)

    monkeypatch.setattr(ans, "complete", fake)
    return calls


def _json(answer: str, *citations: tuple[str, str, str]) -> str:
    return json.dumps({"answer": answer, "citations": [
        {"meeting_id": m, "ts": ts, "quote": q} for m, ts, q in citations]}, ensure_ascii=False)


def _report_rows() -> list[dict]:
    return [json.loads(line) for line in ans.REPORT.read_text(encoding="utf-8").splitlines()]


def _index() -> BaselineIndex:
    def t(mid, date, texts):
        return Transcript(meeting_id=mid, title=f"Zhol {mid}", date=date, participants=[],
                          segments=[Segment(ts=f"00:01:{i:02d}", speaker="Амина", text=x)
                                    for i, x in enumerate(texts)])
    index = BaselineIndex()
    index.add([t("m_002", "2026-08-10", ["Релиз Zhol переносим на пятое сентября.", "Согласны."]),
               t("m_001", "2026-08-03", ["Релиз Zhol будет двадцатого августа.", "Хорошо.", "Начинаем."])])
    return index


GOOD = {"answer": "Срок менялся: 20 августа, затем 5 сентября.",
        "citations": [{"meeting_id": "m_001", "ts": "00:01:00", "quote": "Релиз Zhol будет двадцатого августа"},
                      {"meeting_id": "m_002", "ts": "00:01:00", "quote": "переносим на пятое сентября"}]}


def test_answer_matches_contract_4_6(monkeypatch):
    _fake_llm(monkeypatch, json.dumps(GOOD, ensure_ascii=False))
    a = ans.answer("Когда релиз Zhol?", index=_index())
    assert Answer.model_validate(a.model_dump()) == a
    assert a.mode == "baseline" and a.answer == GOOD["answer"]
    assert [(c.meeting_id, c.ts) for c in a.citations] == [("m_001", "00:01:00"), ("m_002", "00:01:00")]
    assert (a.usage.prompt_tokens, a.usage.completion_tokens) == (120, 30)
    assert a.usage.cost_usd == pytest.approx(0.0002) and a.latency_ms >= 0


def test_prompt_gets_question_and_labeled_context(monkeypatch):
    calls = _fake_llm(monkeypatch, json.dumps(GOOD, ensure_ascii=False))
    ans.answer("Когда релиз Zhol?", index=_index(), k=5)
    messages, kw = calls[0]
    user = messages[1]["content"]
    assert "ВОПРОС: Когда релиз Zhol?" in user
    assert "[m_001 00:01:00] Амина: Релиз Zhol будет двадцатого августа." in user
    assert kw["temperature"] == 0.0 and kw["purpose"] == "answer_baseline"


def test_context_is_chronological_and_deduplicated():
    hits = _index().search("Zhol", k=10)
    ctx = ans.format_context(hits)
    assert ctx.index("=== Встреча m_001") < ctx.index("=== Встреча m_002")
    assert ctx.count("[m_001 00:01:02]") == 1          # реплика из двух перекрывающихся чанков
    m001 = ctx.split("=== Встреча m_002")[0]
    assert m001.index("00:01:00") < m001.index("00:01:01") < m001.index("00:01:02")
    assert "2026-08-03" in ctx and "«Zhol m_001»" in ctx


def test_broken_json_retried_then_returned_as_text_without_citations(monkeypatch):
    calls = _fake_llm(monkeypatch, "Релиз 5 сентября [m_002 00:01:00]")
    a = ans.answer("Когда релиз?", index=_index())
    assert a.answer == "Релиз 5 сентября [m_002 00:01:00]" and a.citations == []
    assert len(calls) == 2 and "JSON" in calls[1][0][-1]["content"]
    assert a.usage.prompt_tokens == 240


def test_invented_quote_fixed_by_retry_with_feedback(monkeypatch):
    calls = _fake_llm(monkeypatch,
                      _json("Релиз 20 августа", ("m_001", "00:01:00", "релиз точно в августе")),
                      _json("Релиз 20 августа", ("m_001", "00:01:00", "Релиз Zhol будет двадцатого августа")))
    stats = ans.AnswerStats(mode="baseline")
    a = ans.answer("Когда релиз?", index=_index(), stats=stats)
    retry = calls[1][0]
    assert retry[-2]["role"] == "assistant" and "релиз точно в августе" in retry[-1]["content"]
    assert [c.quote for c in a.citations] == ["Релиз Zhol будет двадцатого августа"]
    assert "Удалено" not in a.answer
    assert (stats.attempts, stats.citations_first, stats.invalid_first, stats.dropped_final) == (2, 1, 1, 0)
    assert a.usage.completion_tokens == 60


def test_invented_quote_dropped_and_marked_if_retry_fails(monkeypatch):
    bad = _json("Релиз 20 августа", ("m_001", "00:01:00", "Релиз Zhol будет двадцатого августа"),
                ("m_002", "00:01:00", "релиз отменили совсем"))
    _fake_llm(monkeypatch, bad)
    a = ans.answer("Когда релиз?", index=_index())
    assert [c.ts for c in a.citations] == ["00:01:00"] and a.citations[0].meeting_id == "m_001"
    assert "Удалено неподтверждённых цитат: 1" in a.answer
    row = _report_rows()[-1]
    assert (row["citations_first"], row["invalid_first"], row["dropped_final"]) == (2, 1, 1)


def test_quote_with_wrong_label_is_relocated(monkeypatch):
    _fake_llm(monkeypatch, _json("Перенесли", ("m_001", "00:01:02", "переносим на пятое сентября")))
    stats = ans.AnswerStats(mode="baseline")
    a = ans.answer("?", index=_index(), stats=stats)
    assert [(c.meeting_id, c.ts) for c in a.citations] == [("m_002", "00:01:00")]
    assert stats.relocated == 1 and stats.attempts == 1


def test_ambiguous_quote_without_matching_label_is_rejected():
    ctx = "[m_001 00:00:01] А: Релиз в среду.\n[m_002 00:00:05] Б: Релиз в среду, да."
    good, bad, _ = ans._verify([Citation(meeting_id="m_003", ts="00:00:01", quote="Релиз в среду")], ctx)
    assert good == [] and len(bad) == 1


def test_quote_matching_ignores_case_spaces_and_quote_marks():
    ctx = "[m_001 00:00:01] Нұралы: Давайте   перенесём релиз на май."
    cits = [Citation(meeting_id="m_001", ts="00:00:01", quote=q)
            for q in ["«давайте перенесём релиз»", "ДАВАЙТЕ  ПЕРЕНЕСЁМ"]]
    good, bad, _ = ans._verify(ans._citations([c.model_dump() for c in cits]), ctx)
    assert bad == [] and len(good) == 2


def test_quote_matching_kazakh_and_decomposed_unicode():
    ctx = ("[m_002 00:00:07] Сандуғаш: Жақсы, келістік.\n"
           "[m_002 00:00:08] Амина: Хорошо, мой вариант.")
    decomposed = "мой вариант"  # «й» как «и» + кратка: так иногда отдаёт ASR
    good, bad, _ = ans._verify([Citation(meeting_id="m_002", ts="00:00:07", quote="жақсы, КЕЛІСТІК"),
                                Citation(meeting_id="m_002", ts="00:00:08", quote=decomposed)], ctx)
    assert bad == [] and len(good) == 2


def test_duplicate_citations_collapsed(monkeypatch):
    c = ("m_001", "00:01:00", "Релиз Zhol будет двадцатого августа")
    _fake_llm(monkeypatch, _json("ok", c, c))
    assert len(ans.answer("?", index=_index()).citations) == 1


def test_model_saying_no_info_is_kept_without_retry(monkeypatch):
    calls = _fake_llm(monkeypatch, _json("В контексте нет информации о бюджете Qamqor."))
    a = ans.answer("Какой бюджет у Qamqor?", index=_index())
    assert a.answer.startswith("В контексте нет информации") and a.citations == [] and len(calls) == 1


def test_malformed_citations_are_dropped(monkeypatch):
    bad = {"answer": "ok", "citations": [{"meeting_id": "m_001", "ts": "00:01:00"},
                                         "строка", {"meeting_id": "m_002", "ts": "00:01:00", "quote": "Согласны"}]}
    _fake_llm(monkeypatch, "```json\n" + json.dumps(bad, ensure_ascii=False) + "\n```")
    a = ans.answer("?", index=_index())
    assert [c.meeting_id for c in a.citations] == ["m_002"]


def test_empty_index_answers_no_context_without_llm(monkeypatch):
    calls = _fake_llm(monkeypatch, "не должно вызываться")
    a = ans.answer("Когда релиз?", index=BaselineIndex())
    assert a.answer == ans.NO_CONTEXT_ANSWER and a.citations == [] and calls == []


def test_graph_mode_uses_subgraph_context_and_same_verification(monkeypatch):
    from src.graph.queries import load_graph
    calls = _fake_llm(monkeypatch, _json("Сначала 20 апреля, потом 5 мая.",
                                         ("m_001", "00:04:12", "перенесём релиз на двадцатое апреля"),
                                         ("m_002", "00:01:05", "Переносим релиз Атласа на пятое мая")))
    G = load_graph(ans.ROOT / "data/sample/graph.json")
    a = ans.answer("Как менялся срок релиза Атласа?", mode="graph", graph=G)
    assert a.mode == "graph" and len(a.citations) == 2 and "Удалено" not in a.answer
    assert "=== Проект Атлас ===" in calls[0][0][1]["content"]
    assert calls[0][1]["purpose"] == "answer_graph"


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        ans.answer("?", mode="hybrid", index=_index())


def test_report_has_question_latency_and_its_parts(monkeypatch):
    calls = _fake_llm(monkeypatch, json.dumps(GOOD, ensure_ascii=False))
    a = ans.answer("Когда релиз Zhol?", index=_index())
    row = _report_rows()[-1]
    assert len(calls) == 1
    assert row["latency_ms"] == a.latency_ms >= row["retrieval_ms"] >= 0
    assert row["llm_ms"] == 0 and row["from_cache"] is False   # подменённая LLM: задержка 0, не из кэша
    assert (row["mode"], row["requested_mode"]) == ("baseline", "baseline")


def test_report_marks_fully_cached_answers(monkeypatch):
    def cached(messages, **kw):
        return LLMResponse(text=json.dumps({"answer": "ok", "citations": []}), latency_ms=7, from_cache=True)

    monkeypatch.setattr(ans, "complete", cached)
    ans.generate("?", "[m_001 00:00:01] А: текст", "graph")
    row = _report_rows()[-1]
    assert row["from_cache"] is True and row["llm_ms"] == 7 and row["latency_ms"] >= 0
