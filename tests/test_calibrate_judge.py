"""Калибровка судьи (scripts/calibrate_judge.py): kappa, чтение листа, отчёт, защита от платного запуска. Без сети."""
import csv
import json

import pytest

import scripts.calibrate_judge as C
from src.eval.judge import JudgeFormatError, JudgeResult

CITED_TOKEN, COMMENT_TOKEN = "m_099_ТОКЕН", "КОММЕНТ_ТОКЕН"


# ---------- kappa на известных примерах ----------

def test_kappa_classic_two_by_two():
    # 20 да/да, 5 да/нет, 10 нет/да, 15 нет/нет: po=0.7, pe=0.5, kappa=0.4
    a = [1] * 20 + [1] * 5 + [0] * 10 + [0] * 15
    b = [1] * 20 + [0] * 5 + [1] * 10 + [0] * 15
    assert C.cohen_kappa(a, b, (0, 1)) == pytest.approx(0.4)
    assert C.exact_agreement(a, b) == pytest.approx(0.7)


def test_kappa_three_classes_plain_and_quadratic():
    # Ручной расчёт: po=4/6, pe=1/3 → kappa=0.5; взвешенное несовпадение 0.5 против ожидаемого 2 → 1-0.25=0.75
    a = [0, 0, 1, 1, 2, 2]
    b = [0, 1, 1, 2, 2, 2]
    assert C.cohen_kappa(a, b, C.SCORES) == pytest.approx(0.5)
    assert C.cohen_kappa(a, b, C.SCORES, weights="quadratic") == pytest.approx(0.75)


def test_kappa_perfect_and_opposite():
    a = [0, 1, 2, 0, 1, 2]
    assert C.cohen_kappa(a, a, C.SCORES) == pytest.approx(1.0)
    assert C.cohen_kappa(a, a, C.SCORES, weights="quadratic") == pytest.approx(1.0)
    assert C.cohen_kappa([0, 0, 2, 2], [2, 2, 0, 0], C.SCORES, weights="quadratic") < 0


def test_quadratic_kappa_penalizes_far_errors_more():
    human = [0, 1, 2, 0, 1, 2, 0, 1, 2]
    near = [0, 1, 2, 1, 1, 2, 0, 2, 2]      # две ошибки на 1
    far = [0, 1, 2, 2, 1, 2, 0, 0, 2]       # две ошибки: 0→2 и 1→0, но 0→2 на 2 балла
    assert C.exact_agreement(human, near) == C.exact_agreement(human, far)
    assert C.cohen_kappa(human, near, C.SCORES, weights="quadratic") > \
        C.cohen_kappa(human, far, C.SCORES, weights="quadratic")


def test_kappa_undefined_when_everyone_says_the_same():
    assert C.cohen_kappa([0, 0, 0], [0, 0, 0], C.FABS) is None
    assert C.cohen_kappa([], [], C.FABS) is None
    # один оценщик константен, другой нет → kappa = 0, а не «не определён»
    assert C.cohen_kappa([0, 0, 0, 0], [0, 0, 0, 1], C.FABS) == pytest.approx(0.0)


def test_confusion_matrix_rows_are_human_columns_are_judge():
    m = C.confusion_matrix([0, 0, 1, 2], [0, 2, 1, 1], C.SCORES)
    assert m == [[1, 0, 1], [0, 1, 0], [0, 1, 0]]


# ---------- чтение листа ----------

def _write_sheet(path, rows, *, excel=False):
    cols = ["label_id", "question", "expected_answer", "system_answer", "cited_meetings",
            "human_score", "human_fabrication", "comment"]
    delim = ";" if excel else ","
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        if excel:  # так сохраняет Excel-таблица: лишняя строка Column1..Column8 над шапкой
            f.write(delim.join(f"Column{i}" for i in range(1, 9)) + "\r\n")
        w = csv.writer(f, delimiter=delim)
        w.writerow(cols)
        for r in rows:
            w.writerow([r[c] for c in cols])


def _row(i, *, score="2", fab="нет", answerable=True, answer=None):
    return dict(label_id=f"L{i:02d}", question=f"Вопрос {i}?",
                expected_answer="Эталон" if answerable else "В данных нет ответа",
                system_answer=answer if answer is not None else f"Ответ {i}", cited_meetings=CITED_TOKEN,
                human_score=score, human_fabrication=fab, comment=COMMENT_TOKEN)


@pytest.mark.parametrize("excel", [False, True])
def test_load_sheet_both_formats(tmp_path, excel):
    p = tmp_path / "sheet.csv"
    _write_sheet(p, [_row(1, score="2", fab="нет"), _row(2, score="0", fab="да")], excel=excel)
    rows, notes = C.load_sheet(p, {"Вопрос 1?": True, "Вопрос 2?": True})
    assert [(r.label_id, r.human_score, r.human_fabrication) for r in rows] == [("L01", 2, 0), ("L02", 0, 1)]
    assert notes == []


def test_fabrication_accepts_numbers_and_words():
    assert [C.parse_fabrication(v) for v in ("да", "Нет", "1", "0", " ДА ")] == [1, 0, 1, 0, 1]
    assert C.parse_fabrication("2") is None and C.parse_fabrication("") is None
    assert C.parse_score("2") == 2 and C.parse_score("3") is None and C.parse_score("") is None


def test_invalid_human_labels_are_reported(tmp_path):
    p = tmp_path / "sheet.csv"
    _write_sheet(p, [_row(1, score="2", fab="2"), _row(2, score="", fab="нет")])
    rows, notes = C.load_sheet(p, {"Вопрос 1?": True, "Вопрос 2?": True})
    problems = C.label_problems(notes)
    assert any("L01" in n and "human_fabrication" in n for n in problems)
    assert any("L02" in n and "human_score" in n for n in problems)
    assert rows[0].human_fabrication is None and rows[1].human_score is None


def test_answerable_comes_from_questions_with_fallback_to_reference(tmp_path):
    p = tmp_path / "sheet.csv"
    _write_sheet(p, [_row(1), _row(2, answerable=False), _row(3)])
    rows, notes = C.load_sheet(p, {"Вопрос 1?": True, "Вопрос 2?": False})
    assert [r.answerable for r in rows] == [True, False, True]
    assert len(notes) == 1 and notes[0].startswith("L03")  # нет в questions.json → вывели по эталону


def test_sheet_without_header_is_an_error(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(C.SheetError):
        C.load_sheet(p, {})


def test_real_labeling_sheet_is_readable():
    """Настоящий лист в репозитории читается (формат Excel с точкой с запятой); его оценки не проверяем."""
    if not C.DEFAULT_SHEET.exists():
        pytest.skip("labeling_sheet.csv нет")
    rows, _ = C.load_sheet(C.DEFAULT_SHEET, C.load_answerable(C.DEFAULT_QUESTIONS))
    assert len(rows) == 20 and rows[0].label_id == "L01"
    assert all(r.question and r.expected_answer and r.system_answer for r in rows)


# ---------- прогон и отчёт ----------

def _rows(n=4):
    return [C.SheetRow(f"L{i:02d}", f"Вопрос {i}?", "Эталон", f"Ответ {i}", True, hs, hf)
            for i, (hs, hf) in enumerate([(2, 0), (1, 0), (0, 0), (2, 1)][:n], 1)]


def test_run_judge_passes_only_four_fields_and_survives_format_error():
    seen = []

    def fake(q, e, a, answerable):
        seen.append((q, e, a, answerable))
        if q == "Вопрос 2?":
            raise JudgeFormatError("плохой формат", "x")
        return JudgeResult(2, 0, "ок")

    out = C.run_judge(_rows(), fake)
    assert all(len(s) == 4 for s in seen)
    assert [o.score for o in out] == [2, None, 2, 2]
    assert "ОШИБКА ФОРМАТА" in out[1].reason


def test_run_judge_does_not_swallow_other_errors():
    def boom(*a):
        raise RuntimeError("лимит")
    with pytest.raises(RuntimeError):
        C.run_judge(_rows(), boom)


def test_report_has_all_required_numbers_and_disagreements():
    human = _rows()
    # судья: L01 верно; L02 1→2 (на 1); L03 0→2 (на 2 → в списке); L04 fabrication 1→0 (в списке)
    judged = [C.JudgeRow("L01", 2, 0, "а"), C.JudgeRow("L02", 2, 0, "б"),
              C.JudgeRow("L03", 2, 0, "в"), C.JudgeRow("L04", 2, 0, "г")]
    rep = C.render_report(human, judged, date="2026-11-11", backend="openai", model="gpt-4o")
    for part in ("точное совпадение", "Cohen kappa", "квадратично", "fabrication", "Матрица ошибок",
                 "| человек \\ судья |", "JUDGE_PROMPT_VERSION", "одной модели (gpt-4o)", "n = 4"):
        assert part in rep, part
    assert "50%" in rep                                    # совпали L01 и L04 по score: 2 из 4
    bad = C.disagreements(human, {j.label_id: j for j in judged})
    assert [h.label_id for h, _ in bad] == ["L03", "L04"]  # L02 расходится на 1 балл и не входит
    assert "**L03**" in rep and "**L04**" in rep and "**L02**" not in rep


def test_report_marks_failed_rows_and_undefined_kappa():
    human = _rows(2)
    judged = [C.JudgeRow("L01", 2, 0, "а"), C.JudgeRow("L02", None, None, "ОШИБКА")]
    rep = C.render_report(human, judged, date="d", backend="b", model="m")
    assert "в расчёте: 1" in rep and "L02" in rep and "не определён" in rep


def test_judge_csv_columns(tmp_path):
    p = tmp_path / "out.csv"
    C.write_judge_csv(p, [C.JudgeRow("L01", 2, 0, "ок, да"), C.JudgeRow("L02", None, None, "ОШИБКА")])
    rows = list(csv.reader(p.open(encoding="utf-8-sig", newline="")))
    assert rows[0] == ["label_id", "judge_score", "judge_fabrication", "reason"]
    assert rows[1] == ["L01", "2", "0", "ок, да"] and rows[2][1] == ""


# ---------- main: запрещённые поля, защита от платного запуска ----------

@pytest.fixture
def env(tmp_path, monkeypatch):
    sheet = tmp_path / "sheet.csv"
    _write_sheet(sheet, [_row(i, score=str(i % 3), fab="нет") for i in range(1, 6)], excel=True)
    questions = tmp_path / "q.json"
    questions.write_text(json.dumps([{"question": f"Вопрос {i}?", "answerable": True} for i in range(1, 6)]),
                         encoding="utf-8")
    monkeypatch.setattr(C.cost_guard, "BUDGET_FILE", tmp_path / "budget.json")
    monkeypatch.setenv("BUDGET_USD", "3.0")
    monkeypatch.setenv("PRICE_IN_PER_M", "2.5")
    monkeypatch.setenv("PRICE_OUT_PER_M", "10")
    args = ["--sheet", str(sheet), "--questions", str(questions), "--out-dir", str(tmp_path / "out")]
    return args, tmp_path / "out"


def test_judge_never_sees_human_columns_end_to_end(env, monkeypatch):
    """Настоящий judge() на заглушке complete: в сообщения не попадают human_*, комментарий, cited_meetings."""
    args, out = env
    monkeypatch.setenv("LLM_BACKEND", "openai")
    sent: list[str] = []

    def fake_complete(messages, **kw):
        sent.append(json.dumps(messages, ensure_ascii=False))
        return type("R", (), {"text": json.dumps({"reason": "ок", "score": 2, "fabrication": 0})})()

    from src.eval import judge as J
    code = C.main(args + ["--confirm-paid"],
                  judge_fn=lambda q, e, a, ok: J.judge(q, e, a, ok, complete_fn=fake_complete))
    assert code == 0 and len(sent) == 5
    blob = "\n".join(sent)
    for forbidden in (CITED_TOKEN, COMMENT_TOKEN, "human_score", "human_fabrication", "cited_meetings"):
        assert forbidden not in blob
    assert (out / "judge_output.csv").exists() and (out / "agreement.md").exists()


def test_paid_backend_requires_confirmation(env, monkeypatch, capsys):
    args, out = env
    monkeypatch.setenv("LLM_BACKEND", "openai")
    called = []
    code = C.main(args, judge_fn=lambda *a: called.append(a))
    assert code == 3 and called == [] and not out.exists()
    printed = capsys.readouterr().out
    assert "вызовов модели 5" in printed and "Стоимость" in printed and "cost_guard включён" in printed


def test_dry_run_never_calls_judge(env, monkeypatch):
    args, out = env
    monkeypatch.setenv("LLM_BACKEND", "openai")
    called = []
    assert C.main(args + ["--dry-run"], judge_fn=lambda *a: called.append(a)) == 0
    assert called == [] and not out.exists()


def test_stub_backend_is_refused(env, monkeypatch):
    args, _ = env
    monkeypatch.setenv("LLM_BACKEND", "stub")
    assert C.main(args + ["--confirm-paid"], judge_fn=lambda *a: pytest.fail("вызван")) == 2


def test_invalid_human_labels_stop_before_any_call(tmp_path, env, monkeypatch):
    args, _ = env
    monkeypatch.setenv("LLM_BACKEND", "openai")
    _write_sheet(tmp_path / "sheet.csv", [_row(1, score="2", fab="2")] + [_row(i) for i in range(2, 6)], excel=True)
    assert C.main(args + ["--confirm-paid"], judge_fn=lambda *a: pytest.fail("вызван")) == 2


def test_estimate_exceeding_budget_is_refused(env, monkeypatch, tmp_path):
    args, _ = env
    monkeypatch.setenv("LLM_BACKEND", "openai")
    (tmp_path / "budget.json").write_text(json.dumps({"spent_usd": 3.0}), encoding="utf-8")
    assert C.main(args + ["--confirm-paid"], judge_fn=lambda *a: pytest.fail("вызван")) == 4


def test_cost_estimate_scales_with_calls(env):
    args, _ = env
    rows = _rows()
    est = C.estimate_cost(rows)
    assert est.calls == 4 and est.cost_hi > est.cost_lo > 0
