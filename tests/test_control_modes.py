"""A5: контрольные режимы bm25 и longctx. Всё на заглушке LLM, без сети."""
from pathlib import Path

import pytest
from pydantic import ValidationError

import src.answer.answerer as answerer
from scripts.build_gold import KK_FOLD as GOLD_KK_FOLD
from scripts.build_gold import normalize_quote
from src import api
from src.eval import results_v2, systems
from src.eval import run as ev
from src.eval.metrics import MeetingTexts
from src.retrieval import bm25 as bm25_mod
from src.retrieval import longctx as longctx_mod
from src.retrieval.baseline import CONTEXT_K, BaselineIndex
from src.schemas import Answer

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_BACKEND", "stub")
    monkeypatch.setenv("EMBED_MODEL", "")
    monkeypatch.delenv("LLM_CONTEXT_TOKENS", raising=False)
    monkeypatch.setattr(answerer, "REPORT", tmp_path / "answer_report.jsonl")
    monkeypatch.setattr(ev, "RESULTS", tmp_path / "results")
    yield
    api.use_index_dir(api.DEFAULT_INDEX_DIR)


def chunk(mid: str, date: str, n: int, text: str, title: str = "Синк", ts: str = "00:00:10") -> dict:
    return {"chunk_id": f"c_{mid.replace('_', '')}_{n:02d}", "meeting_id": mid, "date": date, "title": title,
            "projects": [], "speakers": ["Амина"], "header": f"[{mid} | {date}]",
            "text": f"[{ts}] Амина: {text}",
            "segments": [{"ts": ts, "speaker": "Амина", "text": text, "start_s": 10.0}]}


def stub_index(chunks: list[dict]) -> BaselineIndex:
    index = BaselineIndex(projects=[])
    index.chunks = chunks
    return index


FILLER = ["обсудили планы на неделю", "договорились созвониться завтра", "посмотрели дизайн экрана",
          "проверили метрики продаж", "решили обновить документацию", "команда отчиталась по задачам",
          "согласовали формат демо", "обсудили найм тестировщика", "разобрали баги прошлого релиза",
          "запланировали ретро", "уточнили бюджет", "обсудили отпуск"]


# ---------- bm25 ----------

def test_bm25_finds_chunk_by_keyword():
    chunks = [chunk("m_001", "2026-08-01", i + 1, t) for i, t in enumerate(FILLER)]
    chunks.append(chunk("m_002", "2026-08-03", 1, "Релиз Zhol переносим на двадцатое сентября"))
    hits = bm25_mod.bm25_search(stub_index(chunks), "Когда релиз Zhol?", as_of="2026-12-31")
    assert hits[0]["chunk_id"] == "c_m002_01" and hits[0]["score"] > 0
    assert all(h["chunk_id"] != "c_m001_01" for h in hits)    # нет общих слов — score 0, в выдачу не идёт


def test_bm25_kazakh_letters_are_folded_like_build_gold():
    assert bm25_mod.KK_FOLD == GOLD_KK_FOLD
    assert bm25_mod.tokenize("Қамқор") == bm25_mod.tokenize("Камкор") == ["камкор"]
    assert " ".join(bm25_mod.tokenize("Қамқорға [шум] Әсем, Ғалым")) == normalize_quote("Камкорга Асем Галым")
    chunks = [chunk("m_001", "2026-08-01", 1, "по Камкор всё по плану"),
              chunk("m_001", "2026-08-01", 2, "обсудили дизайн")]
    hits = bm25_mod.bm25_search(stub_index(chunks), "Что по Қамқор?")
    assert [h["chunk_id"] for h in hits] == ["c_m001_01"]


def test_bm25_tokens_are_letters_and_digits():
    assert bm25_mod.tokenize("Релиз 20-го, v2_beta!") == ["релиз", "20", "го", "v2", "beta"]


def test_bm25_as_of_is_inclusive_and_cuts_future():
    chunks = [chunk("m_001", "2026-08-01", 1, "релиз Zhol в августе"),
              chunk("m_002", "2026-08-05", 1, "релиз Zhol перенесли"),
              chunk("m_003", "2026-08-10", 1, "релиз Zhol отменили")]
    hits = bm25_mod.bm25_search(stub_index(chunks), "релиз Zhol", as_of="2026-08-05")
    assert {h["meeting_id"] for h in hits} == {"m_001", "m_002"}


def test_bm25_respects_context_k():
    chunks = [chunk("m_001", "2026-08-01", i + 1, f"релиз Zhol пункт {i}", ts=f"00:{i:02d}:00")
              for i in range(CONTEXT_K + 5)]
    assert len(bm25_mod.bm25_search(stub_index(chunks), "релиз Zhol")) == CONTEXT_K
    assert len(bm25_mod.bm25_search(stub_index(chunks), "релиз Zhol", k=3)) == 3


def test_bm25_answer_mode_and_retrieved_meetings():
    chunks = [chunk("m_001", "2026-08-01", 1, "Релиз Zhol переносим на двадцатое"),
              chunk("m_002", "2026-08-09", 1, "Релиз Zhol снова переносим")]
    ans = answerer.answer("Когда релиз Zhol?", mode="bm25", index=stub_index(chunks), as_of="2026-08-05")
    assert ans.mode == "bm25" and ans.retrieved_meetings == ["m_001"]


# ---------- longctx ----------

def test_longctx_takes_only_meetings_until_as_of():
    chunks = [chunk("m_002", "2026-08-05", 1, "второе"), chunk("m_001", "2026-08-01", 1, "первое"),
              chunk("m_001", "2026-08-01", 2, "первое ещё", ts="00:00:20"),
              chunk("m_003", "2026-08-10", 1, "будущее")]
    lc = longctx_mod.build_longctx(stub_index(chunks), as_of="2026-08-05")
    assert lc["meetings"] == ["m_001", "m_002"]                     # по дате, as_of включительно
    assert "будущее" not in lc["context"] and "m_003" not in lc["context"]
    assert lc["context"].index("=== Встреча m_001") < lc["context"].index("=== Встреча m_002")
    assert "[m_001 00:00:20] Амина: первое ещё" in lc["context"] and lc["context_tokens"] > 0


def test_longctx_overlap_lines_are_not_duplicated():
    a, b = chunk("m_001", "2026-08-01", 1, "одна реплика"), chunk("m_001", "2026-08-01", 2, "одна реплика")
    assert longctx_mod.build_longctx(stub_index([a, b]))["context"].count("одна реплика") == 1


def test_longctx_too_large_is_an_error_not_truncation():
    chunks = [chunk("m_001", "2026-08-01", 1, "слово " * 3000)]
    with pytest.raises(longctx_mod.LongContextTooLarge, match="не обрезается"):
        longctx_mod.build_longctx(stub_index(chunks), limit_tokens=4000)


def test_longctx_limit_from_env(monkeypatch):
    monkeypatch.setenv("LLM_CONTEXT_TOKENS", "5000")
    assert longctx_mod.model_limit_tokens() == 5000
    monkeypatch.delenv("LLM_CONTEXT_TOKENS")
    assert longctx_mod.model_limit_tokens("gpt-4o-mini") == 128_000


def test_longctx_answer_ignores_k_and_reports_all_meetings(capsys):
    chunks = [chunk(f"m_{i:03d}", f"2026-08-{i:02d}", 1, f"встреча {i}") for i in range(1, CONTEXT_K + 4)]
    ans = answerer.answer("Что было?", mode="longctx", index=stub_index(chunks), as_of="2026-08-31", k=2)
    assert ans.mode == "longctx" and len(ans.retrieved_meetings) == CONTEXT_K + 3
    assert "longctx: контекст ≈" in capsys.readouterr().err


# ---------- схема и старые режимы ----------

@pytest.mark.parametrize("mode", ["baseline", "graph", "hybrid", "bm25", "longctx"])
def test_answer_mode_accepts_all_modes(mode):
    assert Answer(answer="x", mode=mode).mode == mode


def test_answer_mode_rejects_unknown():
    with pytest.raises(ValidationError):
        Answer(answer="x", mode="vector")


def test_old_modes_unchanged_and_all_excludes_controls():
    assert ev.MODES["all"] == ["baseline", "graph", "hybrid"] and ev.MODES["both"] == ["baseline", "graph"]
    assert ev.MODES["bm25"] == ["bm25"] and ev.MODES["longctx"] == ["longctx"]
    assert systems.MODES[:3] == ("baseline", "graph", "hybrid")
    assert ev.modes_of("all") == ["baseline", "graph", "hybrid"]
    assert ev.modes_of(ev.mode_arg("baseline, graph,bm25")) == ["baseline", "graph", "bm25"]
    for bad in ("bm25,bm25", "vector", "all,bm25"):
        with pytest.raises(Exception):
            ev.mode_arg(bad)


def test_baseline_context_unchanged_by_controls():
    """baseline на тех же чанках идёт прежней веткой (вектор), а не bm25."""
    chunks = [chunk("m_001", "2026-08-01", 1, "релиз")]
    index = stub_index(chunks)
    seen = {}
    index.search = lambda q, k, as_of: seen.setdefault("args", (q, k, as_of)) and chunks
    ans = answerer.answer("релиз?", mode="baseline", index=index, as_of="2026-08-01")
    assert ans.mode == "baseline" and seen["args"] == ("релиз?", CONTEXT_K, "2026-08-01")


# ---------- harness ----------

def row(mode="baseline", **kw):
    base = dict(question_id="q_01", type="T1", split="test", lang="ru", as_of="m_015", mode=mode, system="real",
                backend="stub", status="ok", answer="Релиз 4 сентября.")
    return ev.Row(**{**base, **kw})


def test_final_table_has_bm25_not_longctx_and_symmetry_marks_longctx():
    rows = [row("baseline", context_elements=CONTEXT_K), row("graph", context_elements=3),
            row("bm25", context_elements=CONTEXT_K),
            row("longctx", context_meetings=14, context_tokens=60000)]
    modes = ["baseline", "graph", "bm25", "longctx"]
    final = ev.render_final_table(rows, modes, date="d", k=CONTEXT_K, depth=2, notes={})
    assert "accuracy: bm25" in final and ": longctx" not in final
    assert "longctx**: в таблицу не входит (контроль, бюджет контекста не ограничен)" in final
    sym = "\n".join(ev.symmetry_lines(rows, modes))
    assert "longctx (контроль, бюджет контекста не ограничен)" in sym and "Предупреждение" not in sym
    assert "встреч max 14" in sym and "≈60000" in sym
    cfg = results_v2.build_config(modes=modes, split="test", n_questions=1, as_of="m_015", judge_repeats=1,
                                  repeats=1, backend="stub", commit="x", judge_facts=False)
    result = ev.annotate_result(results_v2.build_result(run_id="r", timestamp="t", config=cfg, rows=rows), rows, modes)
    assert result["config"]["final_table_modes"] == ["baseline", "graph", "bm25"]


def test_final_table_without_bm25_unchanged_columns():
    rows = [row("baseline"), row("graph")]
    final = ev.render_final_table(rows, ["baseline", "graph"], date="d", k=CONTEXT_K, depth=2, notes={})
    assert "accuracy: baseline | accuracy: graph | fabrication_rate" in final and "bm25" not in final


@pytest.fixture
def index_dir(tmp_path):
    path = tmp_path / "idx"
    api.use_index_dir(path)
    for _ in api.index_batches(systems.load_transcripts(RAW, "m_005")):
        pass
    api.use_index_dir(api.DEFAULT_INDEX_DIR)
    return path


@pytest.mark.parametrize("mode", ["bm25", "longctx"])
def test_harness_control_modes_cut_future(index_dir, mode):
    system = systems.build_system(mode, raw_dir=RAW, index_dir=index_dir, as_of_dates=["2026-08-05"])
    assert not system.is_stub and system.uses_as_of
    q = ev.Question.model_validate(dict(
        id="q_01", type="T1", split="dev", lang="ru", question="Что решили по срокам релиза Zhol?", as_of="m_002",
        as_of_date="2026-08-05", expected_answer="x", source_meetings=["m_001"], decision_ids=[], answerable=True))
    (r,) = ev.run_eval([q], [mode], lambda m, a: system, gold={}, texts=MeetingTexts(RAW), backend="stub")
    assert r.status == "ok" and set(r.retrieved_meetings) <= {"m_001", "m_002"}
    if mode == "bm25":
        assert 0 < r.context_chunks <= CONTEXT_K
    else:
        assert r.context_elements is None and r.context_meetings == 2
    assert answerer.build_longctx.__module__ == "src.retrieval.longctx"     # обёртка снята
