import json
from pathlib import Path

from scripts.validate_extraction import validate_one, EXTRACTED
from src.schemas import Extraction, Transcript, Link
from src.llm_client import complete, embed

ROOT = Path(__file__).resolve().parent.parent


def test_sample_transcripts_match_schema():
    for f in (ROOT / "data" / "transcripts").glob("*.json"):
        Transcript.model_validate_json(f.read_text(encoding="utf-8"))


def test_sample_extractions_valid_and_quotes_verbatim():
    files = list(EXTRACTED.glob("*.json"))
    assert files
    for f in files:
        errors, total, bad = validate_one(f)
        assert errors == []
        assert bad == 0, f"{f.name}: выдуманные цитаты"


def test_validator_catches_fabricated_quote(tmp_path, monkeypatch):
    import scripts.validate_extraction as v
    ext = json.loads((EXTRACTED / "m_001.json").read_text(encoding="utf-8"))
    ext["decisions"][0]["source"]["quote"] = "эта фраза нигде не звучала"
    bad = tmp_path / "m_001.json"
    bad.write_text(json.dumps(ext, ensure_ascii=False), encoding="utf-8")
    errors, total, n_bad = v.validate_one(bad)
    assert n_bad == 1


def test_link_alias_from_to():
    link = Link.model_validate({"from": "d2", "to": "d1", "type": "supersedes",
                                "confidence": 0.8, "rationale": "x"})
    assert link.from_id == "d2"


def test_llm_client_stub(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_BACKEND", "stub")
    r = complete([{"role": "user", "content": "привет"}], purpose="test")
    assert r.text
    assert len(embed(["a", "b"])) == 2
