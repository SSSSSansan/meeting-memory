import json
from pathlib import Path

import yaml

from scripts import language_share as ls


def _write_raw(raw: Path, mid: str, texts: list[str]) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    segs = [{"start": i, "end": i + 1, "speaker": "SPEAKER_00", "text": t} for i, t in enumerate(texts)]
    (raw / f"{mid}.transcript.json").write_text(json.dumps({"segments": segs}, ensure_ascii=False), encoding="utf-8")


def _world(tmp_path: Path, profiles: dict[str, str]) -> Path:
    meetings = [{"id": m, "language_profile": p} for m, p in profiles.items()]
    path = tmp_path / "world.yaml"
    path.write_text(yaml.safe_dump({"meetings": meetings}, allow_unicode=True), encoding="utf-8")
    return path


def test_letters_and_words_counted():
    c = ls.analyze_texts(["Привет, как дела", "Бұл керек деп ойлаймын", "ну да"])
    assert c["utterances"] == 3 and c["utt_kk"] == 1
    # «бұл» (буква ұ), «керек», «деп» — казахские; «ойлаймын» и русские — нет
    assert c["words"] == 3 + 4 + 2 and c["words_kk"] == 3


def test_word_list_catches_kk_word_without_special_letters():
    c = ls.analyze_texts(["ол керек"])
    assert c["utt_kk"] == 0 and c["words_kk"] == 2


def test_names_with_kk_letters_do_not_count():
    c = ls.analyze_texts(["Нұралы, привет", "Сандуғашқа скажи", "Қамқор готов"])
    assert c["utt_kk"] == 0 and c["words_kk"] == 0


def test_nfc_and_case_insensitive():
    decomposed = "Й" + " Ә"  # составной символ + заглавная казахская буква
    assert ls.has_kk_letters(decomposed)
    assert ls.has_kk_letters("ҚАЗІР")


def test_empty_meeting_has_zero_shares():
    c = ls.with_shares(ls.analyze_texts([]))
    assert c["utt_share"] == c["word_share"] == 0.0
    assert not ls.has_kazakh(c, 0.1)


def test_threshold_is_inclusive_and_by_utterances_or_words():
    ten = ls.analyze_texts(["қазір"] + ["привет всем"] * 9)  # 1 из 10 реплик = ровно 10%
    assert ls.has_kazakh(ten, 0.10)
    assert not ls.has_kazakh(ls.analyze_texts(["қазір"] + ["привет всем"] * 10), 0.10)
    # мало реплик с буквами, но много казахских слов в одной длинной реплике
    long_kk = ls.analyze_texts(["ол керек деп сосын мен бар"] + ["привет"] * 20)
    assert long_kk["utt_kk"] == 0 and ls.has_kazakh(long_kk, 0.10)


def test_main_writes_report_with_profiles_and_target(tmp_path):
    raw = tmp_path / "raw"
    _write_raw(raw, "m_001", ["Привет", "Делаем релиз"] * 5)
    _write_raw(raw, "m_002", ["Сәлем, қалай", "Жақсы", "ну да", "ол керек"])
    _write_raw(raw, "m_003", ["Бұл жоқ", "Иә, қазір"])
    _write_raw(raw, "m_004", ["Привет", "Пока"])
    world = _world(tmp_path, {"m_001": "ru", "m_002": "mixed", "m_003": "kk", "m_004": "ru"})
    out = tmp_path / "out" / "language_share.md"
    assert ls.main(["--raw-dir", str(raw), "--world", str(world), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "| m_001 | ru |" in text and "| m_003 | kk |" in text
    assert "**Набор**" in text and "2 из 4" in text  # m_002, m_003 содержат казахский
    assert "50,0%" in text and "вне цели" in text
    for p in ("ru", "mixed", "kk"):
        assert f"| {p} |" in text
    assert "Метод" in text


def test_missing_transcript_returns_error(tmp_path, capsys):
    world = _world(tmp_path, {"m_001": "ru"})
    assert ls.main(["--raw-dir", str(tmp_path / "raw"), "--world", str(world), "--out", str(tmp_path / "o.md")]) == 1
    assert "нет транскрипта" in capsys.readouterr().err


def test_real_set_meets_basic_expectations(tmp_path):
    """На реальных данных (data/raw в git): казахский есть, kk-встречи самые казахские."""
    meetings = ls.load_world(ls.DEFAULT_WORLD)
    rows = ls.collect(ls.DEFAULT_RAW_DIR, meetings)
    assert len(rows) == len(meetings) == 14
    by_id = {r["meeting"]: ls.with_shares(r) for r in rows}
    assert by_id["m_001"]["utt_share"] == 0.0
    assert by_id["m_005"]["utt_share"] > 0.8 and by_id["m_012"]["utt_share"] > 0.5
