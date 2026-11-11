import json
from pathlib import Path

import pytest

from src.eval.metrics import (MeetingTexts, citation_is_valid, events_found, is_abstention, meeting_number,
                              score_answer, ts_to_sec)
from src.schemas import Citation


@pytest.fixture
def texts(tmp_path: Path) -> MeetingTexts:
    segs = [{"start": 0.0, "text": "Привет, начинаем."},
            {"start": 65.0, "text": "Договорились сдвинуть релиз Zhol на 4 сентября."},
            {"start": 130.0, "text": "Қамқор жобасы бойынша прототипті жұмаға дейін жасаймыз."}]
    (tmp_path / "m_001.transcript.json").write_text(json.dumps({"segments": segs}, ensure_ascii=False),
                                                    encoding="utf-8")
    return MeetingTexts(tmp_path)


GOLD = {"ev_1": {"id": "ev_1", "meeting": "m_001", "start_s": 65.0,
                 "quote": "Договорились сдвинуть релиз Zhol на 4 сентября."},
        "ev_2": {"id": "ev_2", "meeting": "m_001", "start_s": 130.0,
                 "quote": "прототипті жұмаға дейін жасаймыз"}}


def cit(mid="m_001", ts="00:01:05", quote="сдвинуть релиз Zhol на 4 сентября"):
    return Citation(meeting_id=mid, ts=ts, quote=quote)


def test_helpers():
    assert ts_to_sec("01:02:03") == 3723 and ts_to_sec("bad") is None
    assert meeting_number("m_008") == 8 and meeting_number("x") is None


def test_citation_valid_uses_gold_normalization(texts):
    assert citation_is_valid(cit(), texts)
    assert citation_is_valid(cit(quote="ПРОТОТИПТІ  жұмаға ДЕЙІН", ts="00:02:10"), texts)  # регистр, пробелы
    assert citation_is_valid(cit(quote="прототипти жумага"), texts)  # kk-буквы сведены, как в build_gold
    assert not citation_is_valid(cit(quote="перенесли релиз на октябрь"), texts)
    assert not citation_is_valid(cit(mid="m_099"), texts)  # такой встречи нет


def test_events_found_counts_event_once_for_many_citations():
    two_on_one = [cit(), cit(ts="00:01:07", quote="Договорились сдвинуть релиз")]
    assert events_found(two_on_one, ["ev_1"], GOLD) == 1
    assert events_found(two_on_one, ["ev_1", "ev_2"], GOLD) == 1


def test_events_found_by_timestamp_window_or_quote_not_other_meeting():
    assert events_found([cit(ts="00:01:20", quote="что-то совсем другое")], ["ev_1"], GOLD) == 1  # 15 с от 65
    assert events_found([cit(ts="00:03:00", quote="что-то совсем другое")], ["ev_1"], GOLD) == 0  # далеко
    assert events_found([cit(mid="m_002")], ["ev_1"], GOLD) == 0  # та же цитата, чужая встреча
    assert events_found([cit()], ["ev_unknown"], GOLD) == 0


def test_abstention_is_decided_by_the_refusal_phrase():
    assert is_abstention("В контексте нет информации о бюджете.")
    assert is_abstention("Контексте жоқ мәлімет.")
    assert not is_abstention("Релиз сдвинули на 4 сентября.")


def test_abstention_with_attached_citation_is_still_abstention(texts):
    # q_34: честный отказ, но система приложила цитату из встречи — раньше это считалось «не отказом»
    refusal = "В контексте нет информации о дате выхода английской версии Qamqor бота."
    m = _score(texts, [cit()], answer=refusal, answerable=False, source_meetings=[], decision_ids=[])
    assert m.abstained is True and m.citations_total == 1
    # ответ с фактом и цитатой отказом по-прежнему не считается
    assert _score(texts, [cit()], answer="Релиз сдвинули на 4 сентября.").abstained is False


def _score(texts, citations, **kw):
    args = dict(answer="ответ", citations=citations, answerable=True, source_meetings=["m_001"],
                decision_ids=["ev_1"], as_of="m_008", gold=GOLD, texts=texts)
    return score_answer(**{**args, **kw})


def test_score_answerable(texts):
    m = _score(texts, [cit(), cit(quote="выдумка")])
    assert m.correct_meetings is True and m.meetings_hit == 1.0
    assert (m.events_found, m.events_total) == (1, 1)
    assert (m.citations_total, m.citations_valid) == (2, 1)


def test_score_wrong_meeting_and_no_citations(texts):
    assert _score(texts, [cit(mid="m_002")]).correct_meetings is False
    m = _score(texts, [])
    assert m.correct_meetings is False and m.events_found == 0 and m.citations_total == 0


def test_score_flags_citation_after_as_of(texts):
    assert _score(texts, [cit(mid="m_009")], as_of="m_008").cited_after_as_of == 1
    assert _score(texts, [cit()], as_of="m_008").cited_after_as_of == 0


def test_score_unanswerable_has_no_source_metrics(texts):
    m = _score(texts, [], answer="В контексте нет информации.", answerable=False, source_meetings=[],
               decision_ids=[])
    assert m.abstained is True
    assert m.correct_meetings is None and m.events_found is None
