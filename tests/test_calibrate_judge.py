"""Калибровка судьи (scripts/calibrate_judge.py): kappa, чтение листа, отчёт, защита от платного запуска. Без сети."""
import csv
import json

import pytest

import scripts.calibrate_judge as C
from src.eval import judge as J
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


# ---------- main и v2: фикстура с транскриптами ----------

class Env:
    """Лист на 5 строк, вопросы, key_private, чистые транскрипты m_001..m_005 (даты по порядку) и пути вывода."""

    def __init__(self, tmp_path):
        self.tmp = tmp_path
        self.out = tmp_path / "out"
        self.raw = tmp_path / "raw"
        self.raw.mkdir()
        for i in range(1, 6):
            (self.raw / f"m_00{i}.meeting.json").write_text(json.dumps(
                {"title": f"Название_{i}", "started_at": f"2026-08-0{i}T10:00:00+05:00"}), encoding="utf-8")
            (self.raw / f"m_00{i}.transcript.json").write_text(json.dumps(
                {"segments": [{"start": 0, "speaker": "SPEAKER_00", "text": f"ТЕКСТ_ВСТРЕЧИ_{i}"}]}),
                encoding="utf-8")
        # вопрос i: источник m_00i, as_of m_003 → встречи 4 и 5 для судьи недоступны
        self.cited = {1: "m_002", 2: "m_002", 3: "", 4: "m_005", 5: "m_001"}
        _write_sheet(tmp_path / "sheet.csv",
                     [{**_row(i, score=str(i % 3), fab="нет"), "cited_meetings": self.cited[i]}
                      for i in range(1, 6)], excel=True)
        (tmp_path / "q.json").write_text(json.dumps(
            [{"id": f"q_{i}", "question": f"Вопрос {i}?", "answerable": True, "as_of": "m_003",
              "source_meetings": [f"m_00{i}"]} for i in range(1, 6)]), encoding="utf-8")
        (tmp_path / "key.csv").write_text("label_id,question_id,type,split,lang\n" + "\n".join(
            f"L{i:02d},q_{i},T9,SPLITTOKEN,kk" for i in range(1, 6)), encoding="utf-8")
        self.sheet, self.questions, self.key = tmp_path / "sheet.csv", tmp_path / "q.json", tmp_path / "key.csv"

    def args(self, *extra):
        return ["--sheet", str(self.sheet), "--questions", str(self.questions), "--key", str(self.key),
                "--raw-dir", str(self.raw), "--out-dir", str(self.out), *extra]


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr(C.cost_guard, "BUDGET_FILE", tmp_path / "budget.json")
    monkeypatch.setenv("BUDGET_USD", "3.0")
    monkeypatch.setenv("PRICE_IN_PER_M", "2.5")
    monkeypatch.setenv("PRICE_OUT_PER_M", "10")
    for k in ("JUDGE_MODEL", "JUDGE_PRICE_IN_PER_M", "JUDGE_PRICE_OUT_PER_M"):
        monkeypatch.delenv(k, raising=False)
    return Env(tmp_path)


def _loaded(env):
    return C.load_sheet(env.sheet, questions=C.load_questions(env.questions), key=C.load_key(env.key))


# ---------- выбор встреч ----------

def test_sheet_rows_are_linked_to_questions_through_key(env):
    rows, notes = _loaded(env)
    assert [r.question_id for r in rows] == [f"q_{i}" for i in range(1, 6)]
    assert rows[0].source_meetings == ("m_001",) and rows[0].as_of == "m_003"
    assert rows[0].cited_meetings == ("m_002",) and rows[2].cited_meetings == ()
    assert C.meta_problems(notes) == []


def test_load_key_reads_only_two_columns(env):
    assert C.load_key(env.key) == {f"L{i:02d}": f"q_{i}" for i in range(1, 6)}


def test_key_pointing_to_other_question_is_reported(env):
    env.key.write_text("label_id,question_id\nL01,q_2\n", encoding="utf-8")
    rows, notes = _loaded(env)
    assert any("L01" in n and "другой" in n for n in C.meta_problems(notes))
    assert rows[0].question_id == "q_1"  # запасная связь по тексту вопроса


def test_select_union_of_source_and_cited_without_duplicates(env):
    rows, _ = _loaded(env)
    sel = C.select_for_rows(rows, env.raw)
    assert [d.meeting_id for d in sel["L01"].docs] == ["m_001", "m_002"]   # source ∪ cited
    assert [d.meeting_id for d in sel["L02"].docs] == ["m_002"]            # source = cited → одна встреча
    assert [d.meeting_id for d in sel["L03"].docs] == ["m_003"]            # cited пуст


def test_select_drops_meetings_after_as_of(env):
    rows, _ = _loaded(env)
    sel = C.select_for_rows(rows, env.raw)
    assert sel["L04"].docs == [] and len(sel["L04"].notes) == 2            # source m_004 и cited m_005 позже m_003
    assert all("позже as_of" in n for n in sel["L04"].notes)
    assert [d.meeting_id for d in sel["L05"].docs] == ["m_001"]


def test_as_of_is_compared_by_meeting_date_not_number(tmp_path):
    for mid, day in (("m_009", "2026-09-10"), ("m_010", "2026-09-05")):   # у m_010 номер больше, дата раньше
        (tmp_path / f"{mid}.meeting.json").write_text(
            json.dumps({"title": mid, "started_at": f"{day}T10:00:00+05:00"}), encoding="utf-8")
        (tmp_path / f"{mid}.transcript.json").write_text(json.dumps({"segments": []}), encoding="utf-8")
    sel = J.select_meetings(["m_010"], ["m_009"], "m_009", tmp_path)
    assert [d.meeting_id for d in sel.docs] == ["m_010", "m_009"] and sel.notes == []   # по дате: m_010 раньше


def test_select_falls_back_to_number_without_metadata_and_reports_missing(tmp_path):
    sel = J.select_meetings(["m_002", "m_007"], [], "m_003", tmp_path)
    assert sel.docs == [] and any("m_002" in n and "не найден" in n for n in sel.notes)
    assert any("m_007" in n and "позже" in n for n in sel.notes)


def test_meeting_doc_has_date_title_and_clean_text(env):
    doc = J.load_meeting_doc(env.raw, "m_002")
    assert (doc.date, doc.title, doc.text) == ("2026-08-02", "Название_2", "SPEAKER_00: ТЕКСТ_ВСТРЕЧИ_2")
    assert J.load_meeting_doc(env.raw, "m_077") is None


# ---------- v2: что уходит судье ----------

def _capture_complete(sent):
    def fake(messages, **kw):
        sent.append(json.dumps(messages, ensure_ascii=False))
        return type("R", (), {"text": json.dumps({"reason": "ок", "score": 2, "fabrication": 0})})()
    return fake


def test_v2_prompt_has_transcripts_but_no_forbidden_fields(env, monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "openai")
    sent: list[str] = []
    fake = _capture_complete(sent)
    code = C.main(env.args("--version", "v2", "--confirm-paid"),
                  judge_fn=lambda q, e, a, ok, **kw: J.judge(q, e, a, ok, complete_fn=fake, **kw))
    assert code == 0 and len(sent) == 5
    blob = "\n".join(sent)
    assert "ТЕКСТ_ВСТРЕЧИ_1" in sent[0] and "ТЕКСТ_ВСТРЕЧИ_2" in sent[0] and "Название_1" in sent[0]
    assert "2026-08-01" in sent[0]
    assert "ТЕКСТ_ВСТРЕЧИ_4" not in blob and "ТЕКСТ_ВСТРЕЧИ_5" not in sent[3]   # позже as_of
    assert "ТЕКСТ_ВСТРЕЧИ_5" not in blob
    for forbidden in (COMMENT_TOKEN, "SPLITTOKEN", "human_score", "human_fabrication", "cited_meetings",
                      "source_meetings", "question_id", "m_001", "m_002", "m_003", "q_1"):
        assert forbidden not in blob, forbidden
    assert J.JUDGE_PROMPT_VERSION_V2 in (env.out / "agreement_v2.md").read_text(encoding="utf-8")


def test_v1_prompt_still_has_no_transcripts(env, monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "openai")
    sent: list[str] = []
    code = C.main(env.args("--version", "v1", "--confirm-paid"),
                  judge_fn=lambda q, e, a, ok, **kw: J.judge(q, e, a, ok, complete_fn=_capture_complete(sent), **kw))
    assert code == 0 and len(sent) == 5
    blob = "\n".join(sent)
    assert "ТЕКСТ_ВСТРЕЧИ" not in blob and "Транскрипты" not in blob and CITED_TOKEN not in blob
    assert (env.out / "judge_output_v1.csv").exists() and (env.out / "agreement_v1.md").exists()


def test_v2_run_passes_meetings_kwargs_to_judge(env, monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "openai")
    seen = []

    def fake(q, e, a, ok, **kw):
        seen.append(kw)
        return J.JudgeResult(2, 0, "ок")

    assert C.main(env.args("--version", "v2", "--confirm-paid"), judge_fn=fake) == 0
    assert all(kw["with_transcripts"] is True for kw in seen)
    assert [len(kw["meetings"]) for kw in seen] == [2, 1, 1, 0, 1]


def test_old_v1_files_are_not_overwritten_and_comparison_is_appended(env, monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "openai")
    env.out.mkdir()
    (env.out / "agreement.md").write_text("СТАРЫЙ ОТЧЁТ", encoding="utf-8")
    C.write_judge_csv(env.out / "judge_output.csv",
                      [C.JudgeRow(f"L{i:02d}", i % 3, 0, f"v1 причина {i}") for i in range(1, 6)])
    old_csv = (env.out / "judge_output.csv").read_bytes()
    assert C.main(env.args("--confirm-paid"), judge_fn=lambda *a, **kw: J.JudgeResult(2, 1, "v2 причина")) == 0
    assert (env.out / "agreement.md").read_text(encoding="utf-8") == "СТАРЫЙ ОТЧЁТ"
    assert (env.out / "judge_output.csv").read_bytes() == old_csv
    rep = (env.out / "agreement_v2.md").read_text(encoding="utf-8")
    assert "## Сравнение v1 и v2" in rep and "| score: kappa |" in rep and "v2 причина" in rep
    assert (env.out / "judge_output_v2.csv").exists()


def test_comparison_table_and_changed_rows():
    human = [C.SheetRow(f"L0{i}", f"В{i}", "Э", "О", True, hs, hf) for i, (hs, hf) in
             enumerate([(2, 0), (1, 0), (0, 0), (2, 1)], 1)]
    v1 = [C.JudgeRow("L01", 2, 0, "а"), C.JudgeRow("L02", 2, 0, "б"), C.JudgeRow("L03", 2, 1, "в"),
          C.JudgeRow("L04", 2, 0, "г")]
    v2 = [C.JudgeRow("L01", 2, 0, "а"), C.JudgeRow("L02", 1, 0, "б2"), C.JudgeRow("L03", 0, 0, "в2"),
          C.JudgeRow("L04", 2, 1, "г2")]
    rep = C.render_comparison(human, v1, v2)
    assert "| score: точное совпадение | 50% | 100% |" in rep      # v1: L01, L04; v2: все четыре
    assert "| fabrication: совпадение | 50% | 100% |" in rep
    assert "изменился (score/fab): 3" in rep
    assert "| L02 | 1/0 | 2/0 | 1/0 | б2 |" in rep and "L01" not in rep.split("изменился")[1]


def test_judge_csv_roundtrip(tmp_path):
    rows = [C.JudgeRow("L01", 2, 0, "ок"), C.JudgeRow("L02", None, None, "ОШИБКА")]
    C.write_judge_csv(tmp_path / "x.csv", rows)
    assert C.load_judge_csv(tmp_path / "x.csv") == rows


# ---------- main: защита от платного запуска ----------

def test_paid_backend_requires_confirmation(env, monkeypatch, capsys):
    monkeypatch.setenv("LLM_BACKEND", "openai")
    called = []
    assert C.main(env.args(), judge_fn=lambda *a, **k: called.append(a)) == 3
    assert called == [] and not env.out.exists()
    printed = capsys.readouterr().out
    assert "вызовов модели 5" in printed and "Стоимость" in printed and "cost_guard включён" in printed
    assert "токенов на входе" in printed and "на вызов" in printed


def test_dry_run_prints_meetings_and_never_calls_judge(env, monkeypatch, capsys):
    monkeypatch.setenv("LLM_BACKEND", "openai")
    called = []
    assert C.main(env.args("--dry-run"), judge_fn=lambda *a, **k: called.append(a)) == 0
    assert called == [] and not env.out.exists()
    printed = capsys.readouterr().out
    assert "L01 (q_1, as_of m_003): m_001, m_002" in printed and "L04 (q_4, as_of m_003): нет" in printed


def test_stub_backend_is_refused(env, monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "stub")
    assert C.main(env.args("--confirm-paid"), judge_fn=lambda *a, **k: pytest.fail("вызван")) == 2


def test_invalid_human_labels_stop_before_any_call(env, monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "openai")
    _write_sheet(env.sheet, [_row(1, score="2", fab="2")] + [_row(i) for i in range(2, 6)], excel=True)
    assert C.main(env.args("--confirm-paid"), judge_fn=lambda *a, **k: pytest.fail("вызван")) == 2


def test_v2_without_link_to_questions_stops_but_v1_does_not(env, monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "openai")
    env.questions.write_text("[]", encoding="utf-8")
    assert C.main(env.args("--version", "v2", "--confirm-paid"),
                  judge_fn=lambda *a, **k: pytest.fail("вызван")) == 2
    assert C.main(env.args("--version", "v1", "--confirm-paid"),
                  judge_fn=lambda *a, **k: J.JudgeResult(2, 0, "ок")) == 0


def test_estimate_exceeding_budget_is_refused(env, monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "openai")
    (env.tmp / "budget.json").write_text(json.dumps({"spent_usd": 3.0}), encoding="utf-8")
    assert C.main(env.args("--confirm-paid"), judge_fn=lambda *a, **k: pytest.fail("вызван")) == 4


def test_cost_estimate_v2_is_larger_than_v1_and_uses_judge_prices(env, monkeypatch):
    rows, _ = _loaded(env)
    v1 = C.estimate_cost(rows)
    v2 = C.estimate_cost(rows, C.select_for_rows(rows, env.raw))
    assert v1.calls == v2.calls == 5 and v2.prompt_tokens > v1.prompt_tokens
    assert v2.avg_prompt_tokens == v2.prompt_tokens // 5 and v2.cost_hi > v2.cost_lo > 0
    monkeypatch.setenv("JUDGE_PRICE_IN_PER_M", "25")
    monkeypatch.setenv("JUDGE_PRICE_OUT_PER_M", "100")
    assert C.estimate_cost(rows).cost_lo == pytest.approx(v1.cost_lo * 10)


def test_report_names_different_models_when_judge_model_differs(env):
    rows, _ = _loaded(env)
    judged = [C.JudgeRow(r.label_id, 2, 0, "ок") for r in rows]
    rep = C.render_report(rows, judged, date="d", backend="openai", model="judge-m", version="v2",
                          answer_model="answer-m", selected=C.select_for_rows(rows, env.raw))
    assert "разные модели" in rep and "одной модели" not in rep and "judge-v2" in rep
    assert "| встречи |" in rep
