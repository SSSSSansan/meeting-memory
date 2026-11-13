import numpy as np
import pytest

from src.retrieval.baseline import BaselineIndex, _pieces, chunk_transcript, embed_text, mentioned_projects
from src.schemas import Segment, Transcript
from src.utils.text import normalize


@pytest.fixture(autouse=True)
def _stub_backend(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "stub")
    monkeypatch.setenv("EMBED_MODEL", "")


def _transcript(n: int, meeting_id: str = "m_009", text=None) -> Transcript:
    segs = [Segment(ts=f"00:00:{i:02d}", speaker=f"SPEAKER_0{i % 3}",
                    text=text or f"Реплика номер {i} про проект Zhol.", start_s=float(i))
            for i in range(n)]
    return Transcript(meeting_id=meeting_id, title="Zhol: тест", date="2026-08-03",
                      participants=[], segments=segs)


def test_chunk_keeps_meeting_ts_speaker():
    c = chunk_transcript(_transcript(5))[1]
    assert c["chunk_id"] == "c_m009_02" and c["meeting_id"] == "m_009"
    assert [s["ts"] for s in c["segments"]] == ["00:00:02", "00:00:03", "00:00:04"]
    assert c["segments"][0]["speaker"] == "SPEAKER_02" and c["segments"][0]["start_s"] == 2.0
    assert c["text"].splitlines()[0] == "[00:00:02] SPEAKER_02: Реплика номер 2 про проект Zhol."


@pytest.mark.parametrize("n, windows", [
    (0, []),
    (1, [[0]]),
    (3, [[0, 1, 2]]),
    (5, [[0, 1, 2], [2, 3, 4]]),
    (6, [[0, 1, 2], [2, 3, 4], [4, 5]]),
    (7, [[0, 1, 2], [2, 3, 4], [4, 5, 6]]),
])
def test_windows_overlap_without_tail_duplicate(n, windows):
    got = [[int(s["start_s"]) for s in c["segments"]] for c in chunk_transcript(_transcript(n))]
    assert got == windows


def test_long_segment_is_split_into_substrings():
    sentence = " ".join(["слово"] * 30) + "."
    long_text = " ".join([sentence] * 7) + " " + " ".join(["хвост"] * 120)
    pieces = _pieces(long_text, 80)
    assert len(pieces) > 1 and all(len(p.split()) <= 80 for p in pieces)
    assert all(normalize(p) in normalize(long_text) for p in pieces)
    assert normalize(" ".join(pieces)) == normalize(long_text)

    chunks = chunk_transcript(_transcript(1, text=long_text))
    assert {s["ts"] for c in chunks for s in c["segments"]} == {"00:00:00"}


def test_bad_window_params_rejected():
    with pytest.raises(ValueError):
        chunk_transcript(_transcript(3), window=2, step=3)


def test_search_ranks_identical_text_first():
    index = BaselineIndex(projects=[])
    index.add([_transcript(7)])
    target = index.chunks[1]
    hits = index.search(embed_text(target), k=2)  # эмбеддится заголовок + текст
    assert hits[0]["chunk_id"] == target["chunk_id"]
    assert hits[0]["sim"] == pytest.approx(1.0, abs=1e-5)
    assert hits[0]["score"] >= hits[1]["score"]


# ---------- C1: метаданные в чанках ----------

PROJECTS = ["Boost", "Qamqor", "Squad", "Zhol"]


def _meeting(meeting_id: str, date: str, title: str, texts: list[str]) -> Transcript:
    segs = [Segment(ts=f"00:00:{i:02d}", speaker=["Амина", "Нұралы"][i % 2], text=t, start_s=float(i))
            for i, t in enumerate(texts)]
    return Transcript(meeting_id=meeting_id, title=title, date=date, participants=[], segments=segs)


@pytest.mark.parametrize("text, expected", [
    ("Что по Жолу и по Камкору?", ["Zhol", "Qamqor"]),
    ("Zhol, Джол, Қамқорға, Qamqor", ["Zhol", "Qamqor"]),
    ("Сквад и Буст", ["Squad", "Boost"]),
    ("Жалпы алғанда бәрі жақсы, желтый экран", []),  # казахское «жалпы» — не Zhol
])
def test_mentioned_projects_spelling_variants(text, expected):
    assert sorted(mentioned_projects(text, PROJECTS)) == sorted(expected)


def test_header_has_projects_of_whole_meeting_date_and_speakers():
    t = _meeting("m_007", "2026-08-31", "Squad: планирование",
                 ["Давайте перенесём на двадцатое.", "Ок.", "Да.", "По Qamqor нужен бэкендер.",
                  "Qamqor подождёт.", "Один раз Boost."])
    c = chunk_transcript(t, projects=PROJECTS)[0]
    assert "Squad" not in c["text"] and "Qamqor" not in c["text"]
    assert c["projects"] == ["Squad", "Qamqor"]  # Boost назван один раз — случайное упоминание
    assert c["speakers"] == ["Амина", "Нұралы"]
    assert c["header"] == ("[m_007 | 2026-08-31 | «Squad: планирование» | проекты: Squad, Qamqor | "
                           "Амина, Нұралы | 00:00:00]")
    assert embed_text(c).startswith(c["header"] + "\n")


def test_project_question_finds_chunk_whose_text_never_names_project():
    """Критерий C1: реплика без названия проекта находится по вопросу про проект — через заголовок."""
    index = BaselineIndex(projects=PROJECTS)
    index.add([_meeting("m_001", "2026-08-03", "Zhol: старт", ["Давайте перенесём на двадцатое."])])
    hit = index.search("Zhol", k=1)[0]
    assert "Zhol" not in hit["text"] and hit["projects"] == ["Zhol"]


def test_load_rejects_index_without_headers(tmp_path, monkeypatch):
    index = BaselineIndex(projects=PROJECTS)
    index.add([_transcript(3)])
    index.save(tmp_path)
    assert BaselineIndex.load(tmp_path).projects == PROJECTS
    monkeypatch.setattr("src.retrieval.baseline.CHUNK_FORMAT", "plain")
    with pytest.raises(ValueError):
        BaselineIndex.load(tmp_path)


# ---------- C2: фильтр по дате → широкий поиск → пересортировка → обрезка ----------

def _two_meetings_index() -> BaselineIndex:
    index = BaselineIndex(projects=PROJECTS)
    index.add([_meeting("m_001", "2026-08-03", "Zhol: старт", [f"Реплика {i}." for i in range(9)]),
               _meeting("m_009", "2026-09-20", "Qamqor: статус", [f"Реплика {i}." for i in range(9)])])
    return index


def test_as_of_cuts_future_meetings():
    index = _two_meetings_index()
    assert {h["meeting_id"] for h in index.search("Реплика", k=20)} == {"m_001", "m_009"}
    assert {h["meeting_id"] for h in index.search("Реплика", k=20, as_of="2026-09-01")} == {"m_001"}
    assert index.search("Реплика", as_of="2026-01-01") == []


def test_project_boosts_but_never_filters():
    index = _two_meetings_index()
    hits = index.search("Что решили по Qamqor?", k=20)
    assert {h["meeting_id"] for h in hits} == {"m_001", "m_009"}  # чужой проект не выброшен
    boosted = [h for h in hits if h["meeting_id"] == "m_009"]
    assert all(h["score"] - h["sim"] >= 0.05 - 1e-9 for h in boosted)
    assert all(h["score"] - h["sim"] < 0.05 for h in hits if h["meeting_id"] == "m_001")


def test_k_cuts_after_rerank_and_pool_is_wider():
    index = _two_meetings_index()
    assert len(index.search("Реплика", k=3)) == 3
    assert len(index.search("Реплика", k=3, pool=1)) == 3  # пул не уже k


def test_near_duplicate_chunks_dropped():
    index = BaselineIndex(projects=PROJECTS)
    index.add([_meeting("m_001", "2026-08-03", "Zhol: старт", ["A.", "B.", "C."])])
    dup = {**index.chunks[0], "chunk_id": "c_m001_99"}
    index.chunks.append(dup)
    index._vectors = np.vstack([index._vectors, index._vectors[:1]])
    assert [h["chunk_id"] for h in index.search("A", k=5)] == ["c_m001_01"]


def test_add_is_idempotent_and_incremental():
    index = BaselineIndex()
    assert index.add([_transcript(5, "m_001")]) == 2
    assert index.add([_transcript(5, "m_001"), _transcript(3, "m_002")]) == 1
    assert index.add([_transcript(3, "m_002")]) == 0
    assert [c["chunk_id"] for c in index.chunks] == ["c_m001_01", "c_m001_02", "c_m002_01"]
    assert len(index.search("Zhol", k=10)) == 3


def test_empty_index_search():
    assert BaselineIndex().search("что угодно") == []


def test_save_load_roundtrip(tmp_path):
    index = BaselineIndex()
    index.add([_transcript(5)])
    index.save(tmp_path)
    loaded = BaselineIndex.load(tmp_path)
    assert loaded.chunks == index.chunks
    assert [h["chunk_id"] for h in loaded.search("Реплика номер 3", k=2)] == \
           [h["chunk_id"] for h in index.search("Реплика номер 3", k=2)]


def test_load_rejects_index_from_other_embed_model(tmp_path, monkeypatch):
    index = BaselineIndex()
    index.add([_transcript(3)])
    index.save(tmp_path)
    monkeypatch.setenv("EMBED_MODEL", "BAAI/bge-m3")
    with pytest.raises(ValueError):
        BaselineIndex.load(tmp_path)
