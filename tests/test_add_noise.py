import json
import re
from itertools import pairwise
from pathlib import Path

import pytest
import yaml

from scripts.add_noise import (
    CONFIG_PATH,
    INPUT_DIR,
    SPEAKER_MAPS_PATH,
    WORLD_PATH,
    build_entity_index,
    dump_json,
    load_config,
    main,
    process,
    split_token,
)
from src.ingest.scribal_adapter import convert_files
from src.schemas import ScribalMeeting, ScribalTranscript, Transcript

COMBOS = [f"{lv}_{var}" for lv in ("clean", "medium", "strong") for var in ("labeled", "unlabeled")]
KK_MAP = {"ә": "а", "ғ": "г", "қ": "к", "ң": "н", "ө": "о", "ұ": "у", "ү": "у", "һ": "х", "і": "и"}
TRANSLIT = {"рэг": "rag", "файнтюнинг": "fine-tuning", "опенсорс": "open-source",
            "эксель": "excel", "пуш": "push", "апи": "api"}
TEMPLATES = [
    "Привет всем. Эээ, давайте начнем... Нуралы, ты готов?",
    "Да.",
    "Коля сделал миграцию для Zhol-дың релиза, эээ, на 15 сентября.",
    "Қамқор-ға готовую модель берем, RAG-пен и fine-tuning тоже. Ұзақ жұмыс болды ...",
    "Excel выгрузку и push нужно проверить, API ответил за 3,5 секунды!",
    "Колю попросим, а Абду и Сандуғаш посмотрят интеграцию Boost.",
    "Ладно, договорились. Тогда ммм до пятницы, да?",
    "Жақсы, келістік. Қазақша да жұмыс істеуі керек, мына жерде қате бар...",
]
LABELS = ["SPEAKER_00", "SPEAKER_01", "SPEAKER_02", "SPEAKER_03"]
MAPS = json.loads(SPEAKER_MAPS_PATH.read_text(encoding="utf-8"))
WORLD = yaml.safe_load(WORLD_PATH.read_text(encoding="utf-8"))
INDEX = build_entity_index(WORLD)


def make_raw(texts: list[str], labels: list[str] | None = None) -> dict:
    labels = labels or [LABELS[i % 4] for i in range(len(texts))]
    segments = [{"start": float(i * 10), "end": float(i * 10 + 8), "speaker": labels[i],
                 "text": t} for i, t in enumerate(texts)]
    return {"text": " ".join(texts), "language": "ru", "detected_language": "rus",
            "segments": segments}


def write_input(directory: Path, texts: list[str], meeting: str = "m_001") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{meeting}.transcript.json").write_bytes(dump_json(make_raw(texts)))
    (directory / f"{meeting}.meeting.json").write_bytes(dump_json(
        {"public_id": "x", "title": "t", "language": "ru",
         "started_at": "2026-08-03T10:00:00+05:00", "speaker_names": None, "participants": []}))
    return directory


def run_main(src: Path, tmp: Path, name: str, *extra: str) -> tuple[int, Path, Path]:
    out, logs = tmp / f"{name}_out", tmp / f"{name}_logs"
    code = main([str(src), "--out", str(out), "--logs", str(logs), *extra])
    return code, out, logs


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("noise")
    texts = [TEMPLATES[i % len(TEMPLATES)] for i in range(160)]
    src = write_input(tmp / "src", texts)
    code, out, logs = run_main(src, tmp, "a", "--text")
    assert code == 0
    return {"src": src, "out": out, "logs": logs, "tmp": tmp, "texts": texts}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tokens_of(text: str) -> list[str]:
    return [t for t in text.split() if not re.fullmatch(r"\[[^\[\]]+\]", t)]


# ---------- детерминизм и сохранение входа ----------

def test_two_runs_are_byte_identical(run):
    code, out2, logs2 = run_main(run["src"], run["tmp"], "b", "--text")
    assert code == 0
    for first, second in ((run["out"], out2), (run["logs"], logs2)):
        files = sorted(p.relative_to(first) for p in first.rglob("*") if p.is_file())
        assert files
        assert files == sorted(p.relative_to(second) for p in second.rglob("*") if p.is_file())
        for rel in files:
            assert (first / rel).read_bytes() == (second / rel).read_bytes(), rel


def test_all_six_combinations_are_written(run):
    for combo in COMBOS:
        for name in ("m_001.transcript.json", "m_001.meeting.json"):
            assert (run["out"] / combo / name).exists()
        assert (run["logs"] / combo / "m_001.json").exists()
    assert (run["out"] / "medium_unlabeled" / "m_001.txt").exists()
    assert not (run["out"] / "medium_labeled" / "m_001.txt").exists()


def test_meeting_file_is_copied_unchanged(run):
    want = (run["src"] / "m_001.meeting.json").read_bytes()
    for combo in COMBOS:
        assert (run["out"] / combo / "m_001.meeting.json").read_bytes() == want


def test_clean_labeled_is_byte_identical_to_input(run):
    got = (run["out"] / "clean_labeled" / "m_001.transcript.json").read_bytes()
    assert got == (run["src"] / "m_001.transcript.json").read_bytes()


def test_clean_unlabeled_only_lacks_speaker(run):
    src = load(run["src"] / "m_001.transcript.json")
    got = load(run["out"] / "clean_unlabeled" / "m_001.transcript.json")
    for seg in src["segments"]:
        del seg["speaker"]
    assert got == src


def test_unlabeled_has_no_speaker_anywhere(run):
    for combo in COMBOS:
        if combo.endswith("_unlabeled"):
            text = (run["out"] / combo / "m_001.transcript.json").read_text(encoding="utf-8")
            assert '"speaker"' not in text
        else:
            assert '"speaker"' in (run["out"] / combo / "m_001.transcript.json").read_text("utf-8")


# ---------- время ----------

@pytest.mark.parametrize("combo", COMBOS)
def test_segments_ordered_and_not_overlapping(run, combo):
    segs = load(run["out"] / combo / "m_001.transcript.json")["segments"]
    prev_end = 0.0
    for s in segs:
        assert s["end"] >= s["start"] >= prev_end
        prev_end = s["end"]


@pytest.mark.parametrize("combo", COMBOS)
def test_text_field_is_segments_joined_by_space(run, combo):
    raw = load(run["out"] / combo / "m_001.transcript.json")
    assert raw["text"] == " ".join(s["text"] for s in raw["segments"])


# ---------- содержимое ----------

def norm(token: str) -> str:
    core = split_token(token)[1].lower()
    return core.translate(str.maketrans(KK_MAP))


def restore(token: str, mention: dict | None) -> str:
    lead, core, trail = split_token(token)
    if mention is not None:
        assert core.startswith(mention["final_form"])
        core = mention["source_form"] + core[len(mention["final_form"]):]
    low = core.lower()
    for ru, lat in TRANSLIT.items():
        if low == ru or low.startswith(ru + "-"):
            low = lat + low[len(ru):]
            break
    return norm(lead + low + trail)


@pytest.mark.parametrize("combo", COMBOS)
def test_word_sequence_equals_source_minus_fillers(run, combo):
    log = load(run["logs"] / combo / "m_001.json")
    out_segs = load(run["out"] / combo / "m_001.transcript.json")["segments"]
    mentions = {(m["segment"], m["token"]): m for m in log["entity_mentions"]}
    removed = {(r["source"]["segment"], r["source"]["token"])
               for r in log["replacements"] if r.get("removed")}

    got = []
    for si, s in enumerate(out_segs):
        for ti, tok in enumerate(tokens_of(s["text"])):
            got.append(restore(tok, mentions.get((si, ti))))
    want = []
    for si, text in enumerate(run["texts"]):
        for ti, tok in enumerate(text.split()):
            if (si, ti) not in removed:
                want.append(norm(tok))
    assert [w for w in got if w] == [w for w in want if w]


@pytest.mark.parametrize("combo", COMBOS)
def test_numbers_are_untouched(run, combo):
    pattern = re.compile(r"\d+(?:[.,:]\d+)*")
    out = " ".join(s["text"] for s in load(run["out"] / combo / "m_001.transcript.json")["segments"])
    assert pattern.findall(out) == pattern.findall(" ".join(run["texts"]))


@pytest.mark.parametrize("combo", COMBOS)
def test_log_describes_every_text_change(run, combo):
    log = load(run["logs"] / combo / "m_001.json")
    out_segs = load(run["out"] / combo / "m_001.transcript.json")["segments"]
    removed = {(r["source"]["segment"], r["source"]["token"])
               for r in log["replacements"] if r.get("removed")}
    source = [(si, ti, tok) for si, t in enumerate(run["texts"])
              for ti, tok in enumerate(t.split()) if (si, ti) not in removed]
    noisy = [tok for s in out_segs for tok in tokens_of(s["text"])]
    assert len(source) == len(noisy)
    before = {(si, ti): a for si, ti, a in source}
    after = {(si, ti): b for (si, ti, _), b in zip(source, noisy, strict=True)}
    chains: dict[tuple[int, int], list[dict]] = {}
    for r in log["replacements"]:
        if not r.get("removed") and r["operation"] not in ("speaker_mislabel", "noise_tag"):
            chains.setdefault((r["source"]["segment"], r["source"]["token"]), []).append(r)
    # каждое отличие от чистого текста описано в журнале
    assert {pos for pos in before if before[pos] != after[pos]} <= set(chains)
    # а каждая запись — непрерывная цепочка «было → стало» от исходного слова до итогового
    for pos, chain in chains.items():
        assert chain[0]["was"] == before[pos]
        assert all(a["now"] == b["was"] for a, b in pairwise(chain))
        assert chain[-1]["now"] == after[pos]
    ops = {}
    for r in log["replacements"]:
        ops[r["operation"]] = ops.get(r["operation"], 0) + 1
    for op, n in ops.items():
        assert log["counters"][op] == n


def test_clean_logs_have_mentions_but_no_changes(run):
    for combo in ("clean_labeled", "clean_unlabeled"):
        log = load(run["logs"] / combo / "m_001.json")
        assert log["replacements"] == []
        assert log["entity_mentions"] and not any(m["changed"] for m in log["entity_mentions"])


def test_log_has_params_seeds_and_sources(run):
    log = load(run["logs"] / "strong_unlabeled" / "m_001.json")
    assert log["global_seed"] == 42 and log["params"]["noise_tag_prob"] == 0.04
    assert set(log["seeds"]) >= {"fillers", "entities", "blocks", "noise_tag"}
    assert sum(src["words"] for seg in log["segments"] for src in seg["sources"]) == sum(
        seg["words"] for seg in log["segments"])
    assert all(src["person"] for seg in log["segments"] for src in seg["sources"])


# ---------- блоки ----------

@pytest.mark.parametrize("level", ["medium", "strong"])
def test_block_sizes_within_limits(run, level):
    segs = load(run["out"] / f"{level}_unlabeled" / "m_001.transcript.json")["segments"]
    sizes = [len(tokens_of(s["text"])) for s in segs]
    assert len(sizes) > 1
    assert min(sizes) >= 1 and max(sizes) <= 250 + 4  # остаток до 4 слов присоединяется
    assert sizes[-1] >= 5


def test_clean_unlabeled_blocks_equal_replies(run):
    segs = load(run["out"] / "clean_unlabeled" / "m_001.transcript.json")["segments"]
    assert [s["text"] for s in segs] == run["texts"]


def test_txt_format_matches_real_transcript_layout(run):
    segs = load(run["out"] / "medium_unlabeled" / "m_001.transcript.json")["segments"]
    txt = (run["out"] / "medium_unlabeled" / "m_001.txt").read_text(encoding="utf-8")
    blocks = txt.split("\n\n")
    assert blocks.pop() == ""
    assert len(blocks) == len(segs)
    for block, seg in zip(blocks, segs, strict=True):
        label, text = block.split("\n")
        sec = int(seg["start"])
        assert label == f"{sec // 60:02d}:{sec % 60:02d}"
        assert text == seg["text"]


def test_txt_uses_hours_for_meetings_longer_than_an_hour(tmp_path):
    raw = make_raw(["Первый блок.", "Второй блок."])
    raw["segments"][1].update(start=3700.0, end=3708.0)
    src = tmp_path / "src"
    src.mkdir()
    (src / "m_001.transcript.json").write_bytes(dump_json(raw))
    (src / "m_001.meeting.json").write_bytes(b"{}\n")
    code, out, _ = run_main(src, tmp_path, "h", "--text")
    assert code == 0
    assert (out / "clean_unlabeled" / "m_001.txt").read_text("utf-8").startswith("0:00:00\n")


# ---------- адаптер ----------

@pytest.mark.parametrize("combo", COMBOS)
def test_adapter_accepts_every_combination(run, combo):
    t_path = run["out"] / combo / "m_001.transcript.json"
    m_path = run["out"] / combo / "m_001.meeting.json"
    ScribalTranscript.model_validate(load(t_path))
    ScribalMeeting.model_validate(load(m_path))
    result = convert_files(t_path, m_path, "m_001", source="synthetic")
    assert isinstance(result, Transcript)
    assert len(result.segments) == len(load(t_path)["segments"])


# ---------- отдельные операции на маленьких примерах ----------

def noisy(texts: list[str], variant: str = "labeled", labels: list[str] | None = None,
          seed: int = 42, **params) -> list[dict]:
    cfg = load_config(CONFIG_PATH)
    cfg["levels"]["medium"] = {**cfg["levels"]["clean"], **params}
    raw = make_raw(texts, labels)
    return process("m_001", "medium", variant, raw, cfg, INDEX, MAPS["m_001"], seed).segments


def test_filler_removal_keeps_punctuation_on_previous_word():
    segs = noisy(["Слово эээ, дальше", "Эээ, давайте", "Ладно ммм."], filler_remove_prob=1)
    assert [s["text"] for s in segs] == ["Слово, дальше", "давайте", "Ладно."]


def test_reply_made_only_of_fillers_is_dropped_with_log_entry():
    cfg = load_config(CONFIG_PATH)
    cfg["levels"]["medium"] = {**cfg["levels"]["clean"], "filler_remove_prob": 1}
    raw = make_raw(["Эээ.", "Нормально."])
    combo = process("m_001", "medium", "labeled", raw, cfg, INDEX, MAPS["m_001"], 42)
    assert [s["text"] for s in combo.segments] == ["Нормально."]
    assert combo.log["dropped_source_segments"] == [0]


@pytest.mark.parametrize("source, allowed", [
    ("слово... дальше", {"слово, дальше", "слово дальше"}),
    ("слово ... дальше", {"слово, дальше", "слово дальше"}),
    ("слово...", {"слово.", "слово"}),
    ("слово… дальше", {"слово, дальше", "слово дальше"}),
])
def test_ellipsis_is_dropped_or_replaced(source, allowed):
    seen = {noisy([source], ellipsis_mode="drop", seed=seed)[0]["text"] for seed in range(40)}
    assert seen == allowed  # оба исхода встречаются, многоточий не остаётся


def test_entity_variant_keeps_kazakh_suffix_and_ignores_other_cases():
    seg = noisy(["Qamqor-ға и Zhol-дың о Колю и Qamqor"], entity_variant_prob=1)[0]["text"]
    tokens = seg.split()
    assert tokens[0].endswith("-ға") and not tokens[0].startswith("Qamqor-")
    assert tokens[2].endswith("-дың") and not tokens[2].startswith("Zhol-")
    assert tokens[4] == "Колю"
    assert tokens[6] != "Qamqor"


def test_entity_variant_preserves_first_letter_case():
    low = noisy(["вчера николай ушел"], entity_variant_prob=1)[0]["text"].split()[1]
    up = noisy(["Вчера Николай ушел"], entity_variant_prob=1)[0]["text"].split()[1]
    assert low[0].islower() and up[0].isupper()


def test_kk_letter_loss_maps_letters_and_spares_entities():
    segs = noisy(["қазақша ұзақ Нұралы ӘН"], kk_letter_loss_prob=1)
    assert segs[0]["text"] == "казакша узак Нұралы АН"


def test_transliteration_table_with_suffix_and_case():
    segs = noisy(["RAG-пен и Excel, push API fine-tuning open-source Zhol"], transliteration_prob=1)
    assert segs[0]["text"] == "рэг-пен и эксель, пуш апи файнтюнинг опенсорс Zhol"


def test_sentence_end_drop_and_capitalization_use_original_boundaries():
    segs = noisy(["привет. как дела? хорошо"], sentence_end_drop_prob=1,
                 sentence_start_capitalize_prob=1)
    assert segs[0]["text"] == "Привет Как дела Хорошо"


def test_sentence_ops_leave_ellipsis_and_numbers_alone():
    segs = noisy(["сумма 3.5 тыс. ну... да"], sentence_end_drop_prob=1)
    assert segs[0]["text"] == "сумма 3.5 тыс ну... да"


def test_mislabel_only_for_short_replies():
    segs = noisy(["Да", "это длинная реплика из пяти слов"], speaker_mislabel_prob=1,
                 labels=["SPEAKER_00", "SPEAKER_01"])
    assert segs[0]["speaker"] != "SPEAKER_00"
    assert segs[1]["speaker"] == "SPEAKER_01"


def test_noise_tag_is_appended_to_text_end():
    segs = noisy(["один", "два три"], noise_tag_prob=1)
    assert [s["text"] for s in segs] == ["один [шум]", "два три [шум]"]


def test_blocks_are_cut_between_words_and_keep_time_order():
    texts = [" ".join(f"слово{i}_{j}" for j in range(12)) for i in range(40)]
    segs = noisy(texts, variant="unlabeled", block_words={"median": 21, "sigma": 1.24, "min": 1, "max": 250})
    words = [w for s in segs for w in s["text"].split()]
    assert words == [w for t in texts for w in t.split()]
    assert all(a["end"] <= b["start"] for a, b in pairwise(segs))
    assert all("speaker" not in s for s in segs)


# ---------- ошибки входа ----------

def test_person_without_written_variants_is_an_error(tmp_path, capsys):
    world = yaml.safe_load(WORLD_PATH.read_text(encoding="utf-8"))
    del world["people"][0]["written_variants"]
    path = tmp_path / "world.yaml"
    path.write_text(yaml.safe_dump(world, allow_unicode=True), encoding="utf-8")
    src = write_input(tmp_path / "src", ["Привет"])
    code, out, _ = run_main(src, tmp_path, "e", "--world", str(path))
    assert code == 1 and "amina" in capsys.readouterr().err
    assert not out.exists()


def test_empty_reply_is_an_error(tmp_path, capsys):
    src = write_input(tmp_path / "src", ["Привет", "   "])
    code, out, _ = run_main(src, tmp_path, "e")
    assert code == 1 and "пустая реплика в сегменте 1" in capsys.readouterr().err
    assert not out.exists()


def test_meeting_without_speaker_maps_is_an_error(tmp_path, capsys):
    maps = tmp_path / "maps.json"
    maps.write_text(json.dumps({"m_002": MAPS["m_002"]}), encoding="utf-8")
    src = write_input(tmp_path / "src", ["Привет"])
    code, out, _ = run_main(src, tmp_path, "e", "--speaker-maps", str(maps))
    assert code == 1 and "speaker_maps" in capsys.readouterr().err
    assert not out.exists()


def test_unknown_speaker_label_is_an_error(tmp_path, capsys):
    src = tmp_path / "src"
    write_input(src, ["Привет"])
    raw = load(src / "m_001.transcript.json")
    raw["segments"][0]["speaker"] = "SPEAKER_09"
    (src / "m_001.transcript.json").write_bytes(dump_json(raw))
    assert run_main(src, tmp_path, "e")[0] == 1
    assert "SPEAKER_09" in capsys.readouterr().err


def test_probability_out_of_range_is_an_error(tmp_path, capsys):
    cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    cfg["levels"]["medium"]["noise_tag_prob"] = 1.5
    path = tmp_path / "levels.yaml"
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    src = write_input(tmp_path / "src", ["Привет"])
    assert run_main(src, tmp_path, "e", "--config", str(path))[0] == 1
    assert "noise_tag_prob" in capsys.readouterr().err


# ---------- настоящие чистые файлы, если они есть ----------

def test_real_raw_files_run_and_clean_labeled_matches(tmp_path):
    files = sorted(INPUT_DIR.glob("m_*.transcript.json")) if INPUT_DIR.exists() else []
    if not files:
        pytest.skip("data/raw/ не заполнен")
    code, out, _ = run_main(INPUT_DIR, tmp_path, "real", "--text")
    assert code == 0
    for f in files:
        meeting = f.name.split(".")[0]
        assert (out / "clean_labeled" / f.name).read_bytes() == f.read_bytes()
        for combo in COMBOS:
            t_path = out / combo / f.name
            convert_files(t_path, out / combo / f"{meeting}.meeting.json", meeting, source="synthetic")
