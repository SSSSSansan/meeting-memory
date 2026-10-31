import json
from pathlib import Path

import src.extract.extractor as ex
from src.llm_client import LLMResponse
from src.schemas import Transcript

ROOT = Path(__file__).resolve().parent.parent


def _t():
    return Transcript.model_validate_json(
        (ROOT / "data" / "transcripts" / "m_001.json").read_text(encoding="utf-8"))


GOOD = {
    "projects_discussed": ["Атлас"],
    "decisions": [{"project": "Атлас", "statement": "Релиз на 20 апреля", "made_by": "руслан ким",
                   "confidence": 0.9, "ts": "00:00:01",
                   "quote": "перенесём релиз на двадцатое апреля"}],
    "tasks": [{"project": "Атлас", "statement": "Скрипт миграции", "owner": "Айгерим Нурланова",
               "due": "2026-02-10", "status": "open", "ts": "00:06:40",
               "quote": "подготовь скрипт миграции базы данных до десятого февраля"}],
    "open_questions": [],
}


def _fake(responses, calls):
    it = iter(responses)

    def fake(messages, **kw):
        calls.append(messages)
        return LLMResponse(text=next(it), prompt_tokens=10, completion_tokens=5)
    return fake


def test_happy_path_assigns_ids_fixes_ts_and_names(monkeypatch):
    calls = []
    monkeypatch.setattr(ex, "complete", _fake([json.dumps(GOOD)], calls))
    monkeypatch.setattr(ex, "_write_report", lambda s: None)
    e = ex.extract(_t())
    assert len(calls) == 1
    d = e.decisions[0]
    assert d.id == "d_m001_01"
    assert d.source.ts == "00:04:12"          # таймкод исправлен по цитате
    assert d.made_by == "Руслан Ким"          # имя приведено к участнику
    assert e.tasks[0].id == "t_m001_01" and e.tasks[0].due == "2026-02-10"


def test_hallucinated_quote_triggers_retry_then_dropped(monkeypatch):
    bad = json.loads(json.dumps(GOOD))
    bad["decisions"].append({"project": "Атлас", "statement": "Выдумка", "confidence": 0.9,
                             "ts": "00:00:02", "quote": "эта фраза нигде не звучала"})
    calls = []
    monkeypatch.setattr(ex, "complete", _fake([json.dumps(bad), json.dumps(bad)], calls))
    monkeypatch.setattr(ex, "_write_report", lambda s: None)
    stats = ex.ExtractStats(meeting_id="m_001")
    e = ex.extract(_t(), stats=stats)
    assert len(calls) == 2                      # была повторная просьба
    assert stats.bad_quotes_first_attempt == 1
    assert stats.dropped_final == 1
    assert len(e.decisions) == 1                # выдуманный выброшен


def test_bad_json_retries(monkeypatch):
    calls = []
    monkeypatch.setattr(ex, "complete", _fake(["не json", json.dumps(GOOD)], calls))
    monkeypatch.setattr(ex, "_write_report", lambda s: None)
    e = ex.extract(_t())
    assert len(calls) == 2 and len(e.decisions) == 1


def test_due_out_of_range_dropped():
    assert ex._check_due("2030-01-01", "2026-02-03") is None
    assert ex._check_due("2026-01-01", "2026-02-03") is None
    assert ex._check_due("2026-02-10", "2026-02-03") == "2026-02-10"


def test_repeated_decision_is_merged(monkeypatch):
    data = json.loads(json.dumps(GOOD))
    data["decisions"].append({"project": "Атлас", "statement": "Подтверждаем 20 апреля",
                              "confidence": 0.8, "repeats_earlier": True, "ts": "00:04:40",
                              "quote": "двадцатое апреля реалистично"})
    calls = []
    monkeypatch.setattr(ex, "complete", _fake([json.dumps(data)], calls))
    monkeypatch.setattr(ex, "_write_report", lambda s: None)
    stats = ex.ExtractStats(meeting_id="m_001")
    e = ex.extract(_t(), stats=stats)
    assert len(e.decisions) == 1 and stats.merged_repeats == 1
