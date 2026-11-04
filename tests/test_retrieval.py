import pytest

from src.retrieval.baseline import BaselineIndex, _pieces, chunk_transcript
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
    index = BaselineIndex()
    index.add([_transcript(7)])
    target = index.chunks[1]
    hits = index.search(target["text"], k=2)
    assert hits[0]["chunk_id"] == target["chunk_id"]
    assert hits[0]["score"] == pytest.approx(1.0, abs=1e-5)
    assert hits[0]["score"] >= hits[1]["score"]


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
