"""Судья (src/eval/judge.py) на заглушках: сеть и настоящая модель не нужны."""
import inspect
import json
from types import SimpleNamespace

import pytest

import scripts.cost_guard as cg
import src.llm_client as llm
from src.eval import judge as J

Q, EXP, ANS = "ВОПРОС_УНИКАЛЬНЫЙ", "ЭТАЛОН_УНИКАЛЬНЫЙ", "ОТВЕТ_УНИКАЛЬНЫЙ"


class FakeComplete:
    """Заменяет llm_client.complete: отдаёт заранее заданные тексты по очереди, запоминает вызовы."""

    def __init__(self, *texts: str):
        self.texts = list(texts)
        self.calls: list[dict] = []

    def __call__(self, messages, **kw):
        self.calls.append({"messages": messages, **kw})
        return SimpleNamespace(text=self.texts.pop(0) if len(self.texts) > 1 else self.texts[0])


def verdict(score=2, fabrication=0, reason="совпало с эталоном") -> str:
    return json.dumps({"reason": reason, "score": score, "fabrication": fabrication}, ensure_ascii=False)


def test_returns_score_fabrication_reason():
    fake = FakeComplete(verdict(1, 1, "часть фактов"))
    r = J.judge(Q, EXP, ANS, True, complete_fn=fake)
    assert (r.score, r.fabrication, r.reason) == (1, 1, "часть фактов")
    score, fab, reason = r  # NamedTuple распаковывается
    assert (score, fab, reason) == (1, 1, "часть фактов")


def test_temperature_zero_json_mode_and_purpose():
    fake = FakeComplete(verdict())
    J.judge(Q, EXP, ANS, True, complete_fn=fake)
    call = fake.calls[0]
    assert call["temperature"] == 0.0
    assert call["json_schema"] == J.JUDGE_JSON_SCHEMA
    assert call["purpose"] == "judge"
    assert call["max_tokens"] > 0


def test_prompt_versions():
    assert J.JUDGE_PROMPT_VERSION_V1 == "judge-v1" and J.JUDGE_PROMPT_VERSION_V2 == "judge-v2"
    assert J.JUDGE_PROMPT_VERSION == "judge-v2"  # текущая версия по умолчанию


# ---------- что получает судья ----------

def test_judge_signature_has_only_allowed_inputs():
    params = list(inspect.signature(J.judge).parameters)
    assert params[:4] == ["question", "expected_answer", "system_answer", "answerable"]
    # дальше только режим v2 (транскрипты) и подмена модели в тестах; ручным оценкам и метрикам места нет
    assert params[4:] == ["with_transcripts", "meetings", "complete_fn"]


@pytest.mark.parametrize("answerable, flag", [(True, "да"), (False, "нет")])
def test_judge_gets_only_four_fields(answerable, flag):
    fake = FakeComplete(verdict(2 if answerable else 0))
    J.judge(Q, EXP, ANS, answerable, complete_fn=fake)
    msgs = fake.calls[0]["messages"]
    assert [m["role"] for m in msgs] == ["system", "user"]
    user = msgs[1]["content"]
    assert user == J.JUDGE_USER_TEMPLATE.format(question=Q, expected_answer=EXP, system_answer=ANS, answerable=flag)
    assert Q in user and EXP in user and ANS in user and f"есть в данных: {flag}" in user


@pytest.mark.parametrize("word", ["human", "split", "T1", "T2", "T3", "T4", "T5", "cited", "метрик", "question_id",
                                  "m_0", "decision_ids", "комментар"])
def test_system_prompt_has_no_forbidden_hints(word):
    assert word not in J.JUDGE_SYSTEM_PROMPT


def test_prompt_is_russian_and_describes_scale():
    p = J.JUDGE_SYSTEM_PROMPT
    for part in ("score", "fabrication", "ложный отказ", "честный отказ", "JSON"):
        assert part in p


# ---------- формат JSON ----------

def test_fenced_json_is_accepted():
    fake = FakeComplete("```json\n" + verdict(2, 0) + "\n```")
    assert J.judge(Q, EXP, ANS, True, complete_fn=fake).score == 2


def test_reason_is_one_line_and_bounded():
    long_reason = "строка один\nстрока два " + "я" * 1000
    r = J.judge(Q, EXP, ANS, True, complete_fn=FakeComplete(verdict(reason=long_reason)))
    assert "\n" not in r.reason and len(r.reason) <= J.MAX_REASON_CHARS


@pytest.mark.parametrize("bad", [
    "не json",
    "[1, 2]",
    '{"score": 2, "fabrication": 0}',                                   # нет reason
    '{"reason": "x", "score": 3, "fabrication": 0}',                    # score вне шкалы
    '{"reason": "x", "score": "2", "fabrication": 0}',                  # строка вместо числа
    '{"reason": "x", "score": 1.5, "fabrication": 0}',
    '{"reason": "x", "score": true, "fabrication": 0}',                 # bool — не 1
    '{"reason": "x", "score": 2, "fabrication": 2}',
    '{"reason": "", "score": 2, "fabrication": 0}',                     # пустая причина
])
def test_invalid_verdict_is_rejected(bad):
    with pytest.raises(ValueError):
        J.parse_verdict(bad, True)


def test_unanswerable_question_cannot_get_score_one():
    with pytest.raises(ValueError):
        J.parse_verdict(verdict(1), False)
    assert J.parse_verdict(verdict(0, 1), False).score == 0


def test_invalid_json_gets_one_retry_with_error_text():
    fake = FakeComplete("мусор", verdict(2))
    r = J.judge(Q, EXP, ANS, True, complete_fn=fake)
    assert r.score == 2 and len(fake.calls) == 2
    retry = fake.calls[1]["messages"]
    assert retry[-2] == {"role": "assistant", "content": "мусор"}  # иначе кэш вернул бы тот же ответ
    assert "Формат неверный" in retry[-1]["content"]


def test_two_bad_answers_raise_format_error():
    fake = FakeComplete("мусор", "опять мусор")
    with pytest.raises(J.JudgeFormatError) as e:
        J.judge(Q, EXP, ANS, True, complete_fn=fake)
    assert len(fake.calls) == 2 and e.value.raw == "опять мусор"


# ---------- пустой ответ ----------

@pytest.mark.parametrize("empty", ["", "   ", "\n\t"])
@pytest.mark.parametrize("answerable", [True, False])
def test_empty_answer_scores_zero_without_model_call(empty, answerable):
    fake = FakeComplete(verdict())
    r = J.judge(Q, EXP, empty, answerable, complete_fn=fake)
    assert (r.score, r.fabrication) == (0, 0) and "пуст" in r.reason
    assert fake.calls == []


@pytest.mark.parametrize("q, e", [("", EXP), ("  ", EXP), (Q, ""), (Q, " ")])
def test_empty_question_or_reference_is_an_error(q, e):
    with pytest.raises(ValueError):
        J.judge(q, e, ANS, True, complete_fn=FakeComplete(verdict()))


# ---------- cost_guard остаётся включённым ----------

def test_default_path_goes_through_llm_client_and_cost_guard(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "RUNS", tmp_path)
    monkeypatch.setattr(llm, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(cg, "BUDGET_FILE", tmp_path / "budget.json")
    monkeypatch.setenv("LLM_BACKEND", "stub")
    monkeypatch.setenv("BUDGET_USD", "1.0")
    # stub отвечает '{"stub": true}' — не вердикт, но до модели вызов дошёл через llm_client
    with pytest.raises(J.JudgeFormatError):
        J.judge(Q, EXP, ANS, True)
    assert (tmp_path / "calls.jsonl").exists()
    # бюджет исчерпан → BudgetExceeded не глотается ни ретраем, ни обработкой формата
    (tmp_path / "budget.json").write_text(json.dumps({"spent_usd": 5.0}), encoding="utf-8")
    monkeypatch.setattr(llm, "CACHE", tmp_path / "cache2")
    with pytest.raises(cg.BudgetExceeded):
        J.judge(Q, EXP, ANS, True)


# ---------- v2: транскрипты ----------

DOC = J.MeetingDoc("m_077", "2026-09-01", "Название_ВСТРЕЧИ", "SPEAKER_00: ТЕКСТ_ВСТРЕЧИ_УНИКАЛЬНЫЙ")


def test_v2_prompt_contains_transcripts_with_date_and_title_but_not_meeting_id():
    fake = FakeComplete(verdict())
    J.judge(Q, EXP, ANS, True, with_transcripts=True, meetings=[DOC], complete_fn=fake)
    msgs = fake.calls[0]["messages"]
    assert msgs[0]["content"] == J.JUDGE_SYSTEM_PROMPT_V2 and msgs[0]["content"] != J.JUDGE_SYSTEM_PROMPT
    user = msgs[1]["content"]
    assert "=== Встреча от 2026-09-01: Название_ВСТРЕЧИ ===" in user and "ТЕКСТ_ВСТРЕЧИ_УНИКАЛЬНЫЙ" in user
    assert Q in user and EXP in user and ANS in user and "m_077" not in user


def test_v2_without_meetings_says_so_instead_of_inventing_them():
    fake = FakeComplete(verdict())
    J.judge(Q, EXP, ANS, True, with_transcripts=True, meetings=[], complete_fn=fake)
    assert J.NO_TRANSCRIPTS_TEXT in fake.calls[0]["messages"][1]["content"]


def test_v1_messages_are_byte_identical_to_the_first_release():
    """v1 не должен меняться: его ответы лежат в кэше, а сравнение v1/v2 опирается на тот же текст."""
    msgs = J.build_messages(Q, EXP, ANS, True)
    assert msgs[0]["content"] == J.JUDGE_SYSTEM_PROMPT
    assert msgs[1]["content"] == J.JUDGE_USER_TEMPLATE.format(
        question=Q, expected_answer=EXP, system_answer=ANS, answerable="да")
    assert "Транскрипт" not in msgs[0]["content"] + msgs[1]["content"]


def test_transcripts_and_flag_must_agree():
    fake = FakeComplete(verdict())
    with pytest.raises(ValueError):
        J.judge(Q, EXP, ANS, True, with_transcripts=True, complete_fn=fake)
    with pytest.raises(ValueError):
        J.judge(Q, EXP, ANS, True, meetings=[DOC], complete_fn=fake)
    assert fake.calls == []


def test_v2_empty_answer_still_skips_the_model():
    fake = FakeComplete(verdict())
    assert J.judge(Q, EXP, "", True, with_transcripts=True, meetings=[DOC], complete_fn=fake).score == 0
    assert fake.calls == []


@pytest.mark.parametrize("word", ["human", "split", "T1", "T2", "T3", "T4", "T5", "cited", "метрик", "question_id",
                                  "m_0", "decision_ids", "комментар", "source_meetings"])
def test_v2_system_prompt_has_no_forbidden_hints(word):
    assert word not in J.JUDGE_SYSTEM_PROMPT_V2


def test_v2_prompt_rules_about_fabrication_and_missing_meetings():
    p = J.JUDGE_SYSTEM_PROMPT_V2
    for part in ("ни в эталоне, ни в переданных транскриптах", "ТОЛЬКО по смыслу", "не передана", "отметь в reason"):
        assert part in p


# ---------- модель и цены судьи ----------

def test_judge_model_defaults_to_llm_model(monkeypatch):
    monkeypatch.setenv("LLM_MODEL", "answer-model")
    monkeypatch.delenv("JUDGE_MODEL", raising=False)
    assert J.judge_model() == "answer-model"
    monkeypatch.setenv("JUDGE_MODEL", "")  # пустое значение из .env тоже «не задано»
    assert J.judge_model() == "answer-model"
    monkeypatch.setenv("JUDGE_MODEL", "judge-model")
    assert J.judge_model() == "judge-model"


def test_judge_prices_default_to_common_prices(monkeypatch):
    monkeypatch.setenv("PRICE_IN_PER_M", "2.5")
    monkeypatch.setenv("PRICE_OUT_PER_M", "10")
    for k in ("JUDGE_PRICE_IN_PER_M", "JUDGE_PRICE_OUT_PER_M"):
        monkeypatch.delenv(k, raising=False)
    assert J.judge_prices() == (2.5, 10.0)
    monkeypatch.setenv("JUDGE_PRICE_IN_PER_M", "1")
    monkeypatch.setenv("JUDGE_PRICE_OUT_PER_M", "")
    assert J.judge_prices() == (1.0, 10.0)


def test_judge_call_uses_judge_model_and_prices_and_restores_env(monkeypatch, tmp_path):
    """Через настоящий llm_client: модель судьи уходит в API, cost_guard пишет цену судьи, окружение возвращается."""
    monkeypatch.setattr(llm, "RUNS", tmp_path)
    monkeypatch.setattr(llm, "CACHE", tmp_path / "cache")
    monkeypatch.setattr(cg, "BUDGET_FILE", tmp_path / "budget.json")
    for k, v in dict(LLM_BACKEND="openai", LLM_MODEL="answer-model", PRICE_IN_PER_M="1", PRICE_OUT_PER_M="2",
                     JUDGE_MODEL="judge-model", JUDGE_PRICE_IN_PER_M="10", JUDGE_PRICE_OUT_PER_M="20",
                     BUDGET_USD="3").items():
        monkeypatch.setenv(k, v)
    calls = []

    class Fake:
        def __init__(self):
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._chat))

        def _chat(self, **kw):
            calls.append(kw)
            usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=200, prompt_tokens_details=None)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=verdict()))],
                                   usage=usage)

    monkeypatch.setattr(llm, "_client", lambda backend: Fake())
    assert J.judge(Q, EXP, ANS, True).score == 2
    assert calls[0]["model"] == "judge-model" and calls[0]["temperature"] == 0.0
    spent = json.loads((tmp_path / "budget.json").read_text(encoding="utf-8"))["spent_usd"]
    assert spent == pytest.approx((1000 * 10 + 200 * 20) / 1e6)       # цена судьи, не отвечающей системы
    import os
    assert os.environ["LLM_MODEL"] == "answer-model" and os.environ["PRICE_IN_PER_M"] == "1"
    assert os.environ["PRICE_OUT_PER_M"] == "2"


def test_judge_env_restores_even_if_the_call_fails(monkeypatch):
    import os
    monkeypatch.setenv("LLM_MODEL", "answer-model")
    monkeypatch.delenv("PRICE_IN_PER_M", raising=False)
    monkeypatch.setenv("JUDGE_MODEL", "judge-model")
    with pytest.raises(RuntimeError):
        with J.judge_env():
            assert os.environ["LLM_MODEL"] == "judge-model"
            raise RuntimeError("сбой")
    assert os.environ["LLM_MODEL"] == "answer-model" and "PRICE_IN_PER_M" not in os.environ
