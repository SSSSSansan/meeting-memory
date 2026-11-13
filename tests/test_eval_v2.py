"""Метрики v2 без сети и модели: метрики по цитатам, судья по фактам (заглушки), парный бутстрап."""
import json
import os
from types import SimpleNamespace

import pytest

from src.eval import bootstrap as bs
from src.eval import judge_facts as jf
from src.eval import metrics as mx
from src.eval.judge_facts import Claim, FactsJudgeFormatError, FactsVerdict


# ---------- метрики по цитатам ----------

def test_citation_recall_and_precision_on_known_example():
    source, cited = ["m_001", "m_002"], ["m_002", "m_005", "m_006"]
    assert mx.citation_recall(cited, source) == 0.5          # нашли 1 из 2
    assert mx.citation_precision(cited, source) == pytest.approx(1 / 3)   # из трёх цитируемых верна одна


def test_citation_metrics_for_answer_without_citations_and_for_unanswerable():
    assert mx.citation_recall([], ["m_001"]) == 0.0
    assert mx.citation_precision([], ["m_001"]) == 0.0       # нет цитаты — не засчитано, а не «пропуск»
    assert mx.citation_recall(["m_001"], []) is None         # answerable=false: не считается
    assert mx.citation_precision(["m_001"], []) is None


def test_retrieval_recall_is_none_when_not_reported_not_zero():
    assert mx.retrieval_recall(None, ["m_001"]) is None      # «не измерено» ≠ 0.0
    assert mx.retrieval_recall([], ["m_001"]) == 0.0
    assert mx.retrieval_recall(["m_001", "m_009"], ["m_001", "m_002"]) == 0.5
    assert mx.retrieval_recall(["m_001"], []) is None        # answerable=false


def test_score_answer_fills_v2_metrics():
    from src.schemas import Citation

    class NoTexts:
        def get(self, _):
            return None

    kw = dict(answer="ответ", citations=[Citation(meeting_id="m_001", ts="00:00:01", quote="x")], answerable=True,
              source_meetings=["m_001", "m_002"], decision_ids=[], as_of="m_015", gold={}, texts=NoTexts())
    m = mx.score_answer(**kw)
    assert (m.citation_recall, m.citation_precision, m.retrieval_recall) == (0.5, 1.0, None)
    m = mx.score_answer(**kw, retrieved_meetings=["m_001", "m_002", "m_003"])
    assert m.retrieval_recall == 1.0
    m = mx.score_answer(**{**kw, "answerable": False, "source_meetings": []}, retrieved_meetings=["m_001"])
    assert (m.citation_recall, m.citation_precision, m.retrieval_recall) == (None, None, None)


# ---------- судья по фактам: метрики ----------

def verdict(required, forbidden=(), claims=()):
    return FactsVerdict(required=list(required), forbidden=list(forbidden), claims=[Claim(*c) for c in claims])


def test_eleven_claims_eight_of_them_forbidden_give_low_fact_precision():
    claims = [("подтверждённый факт", "supported"), ("лишняя деталь", "extra"), ("ещё деталь", "extra")] + [
        (f"запрещённое утверждение {i}", "wrong") for i in range(8)]
    assert len(claims) == 11
    v = verdict([("факт 1", True), ("факт 2", True)], [(f"запрет {i}", True) for i in range(8)], claims)
    m = jf.fact_metrics(v, answerable=True, answer="длинный ответ")
    assert m.fact_precision == pytest.approx(3 / 11)         # (supported + extra) / все claims
    assert m.fact_precision < 0.3
    assert m.fact_precision_strict == pytest.approx(1 / 11)  # только supported
    assert m.fact_recall == 1.0
    assert m.forbidden_hits == 8
    assert m.fact_f1 == pytest.approx(2 * 1.0 * (3 / 11) / (1.0 + 3 / 11))
    assert m.fact_f1 < 0.5       # полный recall не спасает ответ, набитый запрещённым


def test_fact_metrics_basic_recall_precision_and_f1():
    v = verdict([("a", True), ("b", False), ("c", True), ("d", False)], [("x", False)],
                [("утв. 1", "supported"), ("утв. 2", "extra"), ("утв. 3", "wrong"), ("утв. 4", "supported")])
    m = jf.fact_metrics(v, answerable=True, answer="ответ")
    assert m.fact_recall == 0.5
    assert m.fact_precision == 0.75
    assert m.fact_precision_strict == 0.5
    assert m.forbidden_hits == 0
    assert m.fact_f1 == pytest.approx(2 * 0.5 * 0.75 / 1.25)


def test_f1_is_zero_when_recall_and_precision_are_both_zero():
    v = verdict([("a", False)], [], [("неверное", "wrong")])
    m = jf.fact_metrics(v, answerable=True, answer="ответ")
    assert (m.fact_recall, m.fact_precision, m.fact_f1) == (0.0, 0.0, 0.0)


def test_honest_refusal_on_unanswerable_question_scores_perfect():
    v = verdict([("В данных нет ответа о бюджете Boost", True)], [("бюджет 5 млн", False)], [])
    m = jf.fact_metrics(v, answerable=False, answer="В контексте нет информации о бюджете Boost.")
    assert (m.fact_recall, m.fact_precision, m.fact_precision_strict, m.fact_f1) == (1.0, 1.0, 1.0, 1.0)
    assert m.forbidden_hits == 0


def test_empty_claims_without_refusal_give_precision_zero():
    # ответимый вопрос: пустой список утверждений — это не «всё верно»
    m = jf.fact_metrics(verdict([("a", True)], [], []), answerable=True, answer="В контексте нет информации.")
    assert m.fact_precision == 0.0 and m.fact_f1 == 0.0
    # вопрос без ответа, но системы не отказалась (ничего содержательного, отказа нет)
    m = jf.fact_metrics(verdict([("нет данных", False)], [], []), answerable=False, answer="Ответ был дан раньше.")
    assert m.fact_precision == 0.0


def test_made_up_answer_to_unanswerable_question_is_wrong():
    v = verdict([("В данных нет ответа", False)], [("бюджет 5 млн", True)], [("бюджет составляет 5 млн", "wrong")])
    m = jf.fact_metrics(v, answerable=False, answer="Бюджет составляет 5 млн.")
    assert (m.fact_recall, m.fact_precision, m.forbidden_hits, m.fact_f1) == (0.0, 0.0, 1, 0.0)


def test_extra_claims_are_not_penalised_known_limitation():
    # судья без транскриптов не может проверить факт вне списка: extra не штрафуется (см. memory.md)
    v = verdict([("a", True)], [], [("a", "supported"), ("выдуманная деталь", "extra")])
    assert jf.fact_metrics(v, answerable=True, answer="ответ").fact_precision == 1.0


def test_no_required_facts_leaves_recall_and_f1_undefined():
    m = jf.fact_metrics(verdict([], [("x", False)], [("утв.", "extra")]), answerable=True, answer="ответ")
    assert m.fact_recall is None and m.fact_f1 is None


# ---------- судья по фактам: вход, разбор, повторы ----------

REQ, FORB = ["срок релиза 4 сентября", "ответственный Амина"], ["срок 15 сентября"]
GOOD = {"required": [{"id": "R1", "present": True}, {"id": "R2", "present": False}],
        "forbidden": [{"id": "F1", "asserted": False}],
        "claims": [{"text": "релиз 4 сентября", "label": "supported"}]}


def fake_complete(replies, calls=None):
    it = iter(replies)

    def complete(messages, **kw):
        if calls is not None:
            calls.append({"messages": messages, **kw})
        text = next(it)
        return SimpleNamespace(text=text if isinstance(text, str) else json.dumps(text, ensure_ascii=False),
                               prompt_tokens=100, completion_tokens=20, cost_usd=0.001)
    return complete


def test_judge_gets_only_the_five_allowed_inputs_and_temperature_zero():
    calls = []
    v = jf.judge_facts("Когда релиз Zhol?", "Релиз 4 сентября.", REQ, FORB, True,
                       complete_fn=fake_complete([GOOD], calls))
    c = calls[0]
    assert c["temperature"] == 0.0 and c["json_schema"] is jf.JUDGE_FACTS_JSON_SCHEMA
    text = "\n".join(m["content"] for m in c["messages"]).lower()
    user = c["messages"][1]["content"]
    for part in ("Когда релиз Zhol?", "Релиз 4 сентября.", REQ[0], REQ[1], FORB[0], "есть в данных: да"):
        assert part in user
    # режим, тип, split, эталонный ответ, цитаты и транскрипты судье не передаются
    for banned in ("baseline", "graph", "t1", "t2", "split", "expected_answer", "эталонный ответ", "транскрипт",
                   "source_meetings", "decision_ids", "citation"):
        assert banned not in user.lower(), banned
    assert "режим" not in user.lower() and "тип вопроса" not in user.lower()
    assert "эталонный ответ" not in text
    assert v.required == [(REQ[0], True), (REQ[1], False)]
    assert v.forbidden == [(FORB[0], False)]
    assert v.claims == [Claim("релиз 4 сентября", "supported")]
    assert (v.prompt_tokens, v.completion_tokens) == (100, 20)


def test_build_messages_signature_takes_exactly_five_fields():
    import inspect
    assert list(inspect.signature(jf.build_messages).parameters) == [
        "question", "system_answer", "required_facts", "forbidden_facts", "answerable"]
    assert jf.JUDGE_FACTS_PROMPT_VERSION == "judge-facts-v1"


def test_answerable_false_is_stated_in_the_prompt_and_input():
    msgs = jf.build_messages("вопрос", "ответ", ["в данных этого нет"], [], False)
    assert "есть в данных: нет" in msgs[1]["content"]
    assert "честно сообщает" in msgs[0]["content"]
    assert jf.NO_FACTS_TEXT in msgs[1]["content"]   # пустой forbidden не ломает вход


@pytest.mark.parametrize("bad, expect", [
    ("не json", "не JSON"),
    ({"required": [], "forbidden": []}, "нет полей"),
    ({**GOOD, "required": [{"id": "R1", "present": True}]}, "нужны id"),
    ({**GOOD, "required": [{"id": "R1", "present": 1}, {"id": "R2", "present": False}]}, "true или false"),
    ({**GOOD, "claims": [{"text": "x", "label": "good"}]}, "label"),
])
def test_bad_format_is_retried_once_with_error_text_then_raises(bad, expect):
    calls = []
    with pytest.raises(FactsJudgeFormatError) as e:
        jf.judge_facts("в", "о", REQ, FORB, True, complete_fn=fake_complete([bad, bad], calls))
    assert expect in str(e.value)
    assert len(calls) == 2 and "Формат неверный" in calls[1]["messages"][-1]["content"]


def test_format_error_then_good_answer_counts_both_calls_in_usage():
    v = jf.judge_facts("в", "о", REQ, FORB, True, complete_fn=fake_complete(["мусор", GOOD]))
    assert v.prompt_tokens == 200 and v.completion_tokens == 40


def test_empty_answer_is_judged_without_model_call_as_nothing_found():
    runner = jf.JudgeFactsRunner(fn=lambda *a: pytest.fail("модель не должна вызываться"))
    out = runner.evaluate(question="в", answer="", required_facts=REQ, forbidden_facts=FORB, answerable=True)
    assert out.metrics["fact_recall"] == 0.0 and out.metrics["fact_f1"] == 0.0 and out.metrics["forbidden_hits"] == 0
    out = runner.evaluate(question="в", answer="текст", required_facts=REQ, forbidden_facts=FORB, answerable=True,
                          answered=False)       # статус не ok
    assert out.metrics["fact_f1"] == 0.0 and runner.calls == 0


def test_judge_repeats_average_metrics_report_std_and_use_separate_caches():
    import contextlib

    import src.llm_client as llm_client
    seen_caches = []
    verdicts = iter([verdict([("a", True), ("b", True)], [], [("x", "supported")]),
                     verdict([("a", True), ("b", False)], [], [("x", "supported"), ("y", "wrong")]),
                     verdict([("a", False), ("b", False)], [], [("y", "wrong")])])

    @contextlib.contextmanager
    def cache_for(rep):
        seen_caches.append((rep, llm_client.CACHE))
        yield

    def fn(*a):
        return next(verdicts)

    runner = jf.JudgeFactsRunner(judge_repeats=3, fn=fn, repeat_cache=cache_for)
    out = runner.evaluate(question="в", answer="о", required_facts=["a", "b"], forbidden_facts=[], answerable=True)
    assert runner.calls == 3 and out.repeats_ok == 3
    assert [r for r, _ in seen_caches] == [1, 2, 3]               # свой контекст кэша на каждый повтор
    assert out.metrics["fact_recall"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert out.std["fact_recall"] == pytest.approx(0.5)            # выборочное std (n−1)
    assert out.metrics["fact_precision"] == pytest.approx((1.0 + 0.5 + 0.0) / 3)


def test_same_input_is_judged_once_per_judge_repeat():
    runner = jf.JudgeFactsRunner(judge_repeats=2, fn=lambda *a: verdict([("a", True)], [], [("a", "supported")]))
    for _ in range(3):      # один и тот же ответ (два режима или повторы ответов)
        runner.evaluate(question="в", answer="о", required_facts=["a"], forbidden_facts=[], answerable=True)
    assert runner.calls == 2 and runner.reused == 4


def test_format_failure_of_all_repeats_gives_no_metrics_and_is_counted():
    def fn(*a):
        raise FactsJudgeFormatError("плохой формат")
    runner = jf.JudgeFactsRunner(judge_repeats=2, fn=fn)
    out = runner.evaluate(question="в", answer="о", required_facts=["a"], forbidden_facts=[], answerable=True)
    assert out.metrics is None and runner.failed == 2 and out.errors


def test_judge_model_prices_come_from_judge_env(monkeypatch):
    monkeypatch.setenv("JUDGE_MODEL", "gpt-judge")
    seen = {}

    def fake_llm_complete(messages, **kw):
        seen["model"] = os.environ["LLM_MODEL"]
        return SimpleNamespace(text=json.dumps(GOOD), prompt_tokens=1, completion_tokens=1, cost_usd=0.0)

    monkeypatch.setattr("src.llm_client.complete", fake_llm_complete)
    monkeypatch.setenv("LLM_MODEL", "gpt-answer")
    jf.judge_facts("в", "о", REQ, FORB, True)
    assert seen["model"] == "gpt-judge" and os.environ["LLM_MODEL"] == "gpt-answer"


# ---------- парный бутстрап ----------

def values(xs):
    return {f"q_{i:02d}": x for i, x in enumerate(xs)}


def test_paired_uses_common_questions_and_is_reproducible():
    a, b = values([0.9, 0.8, 0.7, 0.9, 0.8, 0.6, 0.9]), values([0.5, 0.6, 0.4, 0.7, 0.5, 0.5, 0.6])
    a["q_only_graph"], b["q_only_baseline"] = 1.0, 0.0       # не общие вопросы не участвуют
    r1, r2 = bs.paired_bootstrap(a, b), bs.paired_bootstrap(a, b)
    assert r1 == r2                                           # seed = 42: результат воспроизводим
    assert r1.n == 7 and r1.n_common == 7 and r1.n_excluded == 0
    assert r1.diff.mean == pytest.approx((0.4 + 0.2 + 0.3 + 0.2 + 0.3 + 0.1 + 0.3) / 7)
    assert r1.diff.ci_low <= r1.diff.mean <= r1.diff.ci_high
    assert bs.N_BOOT == 10_000 and bs.SEED == 42 and bs.PERCENTILES == (2.5, 97.5)


def test_seed_changes_the_resamples_and_default_seed_is_fixed():
    xs = [((i * 37) % 101) / 100 for i in range(40)]        # много разных значений: интервал не «слипается»
    a, b = values(xs), values([x * 0.8 for x in xs])
    base = bs.paired_bootstrap(a, b)
    assert bs.paired_bootstrap(a, b, seed=bs.SEED) == base
    other = bs.paired_bootstrap(a, b, seed=7)
    assert (other.diff.ci_low, other.diff.ci_high) != (base.diff.ci_low, base.diff.ci_high)
    assert other.diff.mean == base.diff.mean               # среднее от seed не зависит


def test_pairing_is_kept_resampled_difference_not_difference_of_independent_resamples():
    # разность постоянна (+0.2 на каждом вопросе): при парном бутстрапе интервал разности вырождается в точку,
    # при независимых ресэмплах режимов он был бы широким
    a = values([0.1, 0.9, 0.3, 0.7, 0.5, 0.2, 0.8])
    b = {q: v - 0.2 for q, v in a.items()}
    r = bs.paired_bootstrap(a, b)
    assert r.diff.mean == pytest.approx(0.2)
    assert r.diff.ci_low == pytest.approx(0.2) and r.diff.ci_high == pytest.approx(0.2)
    assert r.verdict == bs.VERDICT_ZERO_EXCLUDED
    assert r.a.ci_high - r.a.ci_low > 0.1                     # а у самих режимов интервалы широкие


def test_rows_without_estimate_are_dropped_from_both_modes_and_counted():
    a = values([0.9, None, 0.7, 0.9, 0.8, 0.6])
    b = values([0.5, 0.6, None, 0.7, 0.5, 0.5])
    r = bs.paired_bootstrap(a, b)
    assert r.n == 4 and r.n_common == 6 and r.n_excluded == 2
    # средние обоих режимов — по одним и тем же 4 вопросам
    assert r.a.mean == pytest.approx((0.9 + 0.9 + 0.8 + 0.6) / 4)
    assert r.b.mean == pytest.approx((0.5 + 0.7 + 0.5 + 0.5) / 4)
    assert r.a.n == r.b.n == r.diff.n == 4


def test_interval_containing_zero_is_not_proven():
    a = values([0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6])
    b = values([0.3, 0.8, 0.2, 0.9, 0.4, 0.7, 0.5])
    r = bs.paired_bootstrap(a, b)
    assert r.diff.ci_low <= 0 <= r.diff.ci_high
    assert r.verdict == "различие не доказано"


def test_less_than_five_questions_marks_interval_uninformative():
    r = bs.paired_bootstrap(values([0.9, 0.8, 0.9, 0.7]), values([0.1, 0.2, 0.1, 0.3]))
    assert r.n == 4 and r.verdict == "n<5, интервал не информативен"      # даже если интервал не содержит 0
    assert bs.mean_ci(values([0.5, 0.6])).note == "n<5, интервал не информативен"
    assert bs.mean_ci(values([0.5] * 5)).note is None
    empty = bs.paired_bootstrap(values([None, None]), values([0.5, 0.5]))
    assert empty.n == 0 and empty.n_excluded == 2 and empty.diff.mean is None
    assert bs.mean_ci(values([None])).n == 0


def test_works_for_any_pair_of_modes_and_swapping_flips_the_sign():
    a, b = values([0.9, 0.8, 0.7, 0.9, 0.8]), values([0.5, 0.6, 0.4, 0.7, 0.5])
    ab, ba = bs.paired_bootstrap(a, b), bs.paired_bootstrap(b, a)
    assert ab.diff.mean == pytest.approx(-ba.diff.mean)
    assert ab.diff.ci_low == pytest.approx(-ba.diff.ci_high)


def test_interval_note_says_it_is_about_question_set_not_repeats():
    assert "зависимость от набора вопросов" in bs.INTERVAL_NOTE and "температуре 0" in bs.INTERVAL_NOTE
