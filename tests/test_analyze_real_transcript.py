import json
from pathlib import Path

from scripts.analyze_real_transcript import (
    MIN_MEDIAN_P90_MAX,
    analyze,
    dist,
    main,
    parse,
)


def stats_of(text: str, **kw) -> dict:
    return analyze(parse(text), **kw)


def test_parser_empty_block_multiline_blank_start_and_hours_format():
    text = (
        "0:05\n\n"
        "0:10\nпервая строка\nвторая строка\n\n"
        "0:15\n\nтекст после пустой строки\n\n"
        "1:00:05\nконец\n"
    )
    p = parse(text)
    assert [b.start for b in p.blocks] == [5, 10, 15, 3605]
    assert [len(b.lines) for b in p.blocks] == [0, 2, 1, 1]
    assert [bool(b.starts_with_blank and b.lines) for b in p.blocks] == [False, False, True, False]
    s = analyze(p)
    assert s["blocks"]["total"] == 4 and s["blocks"]["empty"] == 1
    assert s["format_check"]["blocks_starting_with_blank_line"] == 1
    assert s["normalization"]["blocks_with_line_breaks_inside"] == 1
    assert s["blocks"]["total_seconds"] == 3600


def test_tag_positions_counts_and_block_share():
    text = (
        "0:01\n[шум] привет\n\n0:02\nпривет [шум] мир\n\n"
        "0:03\nпривет [шум]\n\n0:04\n[шум]\n\n0:05\nпросто слова\n"
    )
    tags = stats_of(text, min_tag_count=1)["tags"]
    t = tags["listed"]["[шум]"]
    assert t["count"] == 4
    assert t["position"] == {"start": 1, "middle": 1, "end": 1, "whole_block": 1}
    assert t["blocks_pct"] == 80.0  # 4 из 5 непустых блоков
    assert tags["total_occurrences"] == 4


def test_rare_tags_are_not_listed():
    s = stats_of("0:01\n[Иванов] привет\n\n0:02\n[шум] а [шум] б [шум] в\n", min_tag_count=3)
    assert list(s["tags"]["listed"]) == ["[шум]"]
    assert s["tags"]["rare_unlisted"] == {"distinct": 1, "occurrences": 1}


def test_repeats_are_case_insensitive_and_comma_separate():
    s = stats_of("0:01\nчтобы чтобы мы шли, шли домой. Дом дом\n")
    assert s["blocks"]["total_words"] == 8
    assert s["repeats"]["adjacent_same_word"] == 2
    assert s["repeats"]["adjacent_same_word_per_1000_words"] == 250.0
    assert s["repeats"]["same_word_through_comma"] == 1
    assert s["repeats"]["same_word_through_comma_per_1000_words"] == 125.0


def test_breaks():
    text = (
        "0:01\nпервый блок без точки\n\n"
        "0:05\nи продолжение заканчивается.\n\n"
        "0:09\nСлово-\n\n"
        "0:12\nВопрос?\n"
    )
    b = stats_of(text)["breaks"]
    assert b["blocks_not_ending_with_terminal_pct"] == 50.0
    assert b["blocks_starting_lowercase_pct"] == 50.0
    assert b["blocks_ending_with_dash_or_hyphen"] == 1
    assert b["blocks_continuing_in_next_block"] == 1


def test_fillers_are_case_insensitive():
    f = stats_of("0:01\nНУ вот ВОТ Как Бы так\n")["fillers"]
    assert f["ну"]["count"] == 1
    assert f["вот"]["count"] == 2 and f["вот"]["per_1000_words"] == 333.33
    assert f["так"]["count"] == 1
    assert f["как бы"]["count_substring"] == 1 and f["как бы"]["count_whole_words"] == 1


def test_multiword_filler_counts_substring_as_asked():
    f = stats_of("0:01\nкак быстро это было\n")["fillers"]["как бы"]
    assert f["count_substring"] == 1 and f["count_whole_words"] == 0


def test_capitalized_words_outside_block_start():
    p = stats_of("0:01\nМосква и Алматы, это Города. Вот\n")["punctuation"]
    assert p["words_outside_block_start"] == 5
    assert p["capitalized_outside_block_start_pct"] == 60.0
    assert p["capitalized_not_after_sentence_end_pct"] == 40.0


def test_scripts_kazakh_digits_and_numerals():
    s = stats_of("0:01\nМен қазір жұмыс істеймін hello teст\n\n"
                 "0:02\nнас 12 человек, два часа и Два дня, 3,5 раза\n")
    assert s["kazakh"]["words_with_kazakh_letters"] == 3
    assert s["latin_and_numbers"]["cyrillic_words_with_latin_letters"] == 1
    assert s["latin_and_numbers"]["digit_numbers"] == 2
    assert s["latin_and_numbers"]["numerals_as_words"] == 2
    total = s["blocks"]["total_words"]
    assert s["latin_and_numbers"]["latin_words_pct"] == round(100 * 1 / total, 2)
    assert s["kazakh"]["words_with_kazakh_letters_pct"] == round(100 * 3 / total, 2)


def test_long_pauses():
    text = "".join(f"{ts}\nслово\n\n" for ts in ("0:00", "0:05", "0:20", "1:00", "1:00:05"))
    pauses = stats_of(text)["pauses"]
    assert pauses == {"gaps_over_10_s": 3, "gaps_over_30_s": 2}


def test_normalization_flags():
    n = stats_of("0:01\nдва  пробела\n\n0:02\nстрока\nвторая\n\n0:03\nслово , запятая\n")[
        "normalization"]
    assert n == {"blocks_with_double_spaces": 1, "blocks_with_line_breaks_inside": 1,
                 "blocks_with_space_before_punctuation": 1}


def test_speech_rate_and_distributions():
    ten = " ".join("абвгдежзик")
    text = f"0:00\n{ten}\n\n0:10\nа б в г д\n\n0:30\nконец\n"
    s = stats_of(text)
    assert s["speech_rate"]["words_per_second_overall"] == 0.5
    short = s["speech_rate"]["short_blocks_words_per_second"]
    assert (short["p10"], short["median"], short["p90"]) == (0.325, 0.625, 0.925)
    d = s["blocks"]["duration_seconds"]
    assert (d["min"], d["median"], d["p90"], d["max"]) == (10.0, 15.0, 19.0, 20.0)
    w = s["blocks"]["words_per_block"]
    assert (w["min"], w["median"], w["p90"], w["max"]) == (1.0, 5.0, 9.0, 10.0)
    assert s["blocks"]["share_of_1_3_word_blocks_pct"] == 33.33


def test_dist_helper():
    d = dist([1, 2, 3, 4], MIN_MEDIAN_P90_MAX)
    assert d == {"n": 4, "min": 1.0, "median": 2.5, "p90": 3.7, "max": 4.0}
    assert dist([], MIN_MEDIAN_P90_MAX)["median"] is None


def test_odd_lines_land_in_anomalies_not_dropped():
    text = (
        "шапка документа\n"      # 1: до первой метки
        "0:01\nтекст\n\n"        # 2-4
        "12\n\n"                 # 5: номер страницы
        "0:05\nтекст два\n"      # 7-8
        "\f\n"                   # 9: разрыв страницы
        "0:07\nещё\n\n"          # 10-11
        "1:5\n"                  # 13: кривая метка
        "10:30 дальше\n"         # 14: метка с текстом
    )
    s = stats_of(text)
    assert s["anomalies"] == [
        {"line": 1, "length": 15, "kind": "before_first_timestamp"},
        {"line": 5, "length": 2, "kind": "digits_only"},
        {"line": 9, "length": 1, "kind": "page_break_char"},
        {"line": 13, "length": 3, "kind": "malformed_timestamp"},
        {"line": 14, "length": 12, "kind": "timestamp_with_text"},
    ]
    assert s["format_check"]["text_lines_in_blocks"] == 3
    assert s["format_check"]["timestamp_lines"] == 3


def test_timestamp_glued_to_previous_text_is_counted():
    p = parse("0:01\nтекст\n0:05\nещё\n")
    assert p.ts_not_after_blank == 1


SAMPLE = (
    "0:00\nНу вот, секретнаяфраза [шум] про Иванова.\n\n"
    "0:07\n[шум]\n\n"
    "0:09\n[Иванов] да, мы будем [шум] делать.\n\n"
    "0:30\nсобрались [шум]\n"
)


def run_main(tmp_path: Path, name: str, capsys) -> tuple[bytes, str]:
    src = tmp_path / "in.txt"
    src.write_text(SAMPLE, encoding="utf-8")
    out = tmp_path / name
    assert main([str(src), str(out)]) == 0
    return out.read_bytes(), capsys.readouterr().out


def test_output_is_deterministic_and_has_no_text_content(tmp_path, capsys):
    first, printed1 = run_main(tmp_path, "a.json", capsys)
    second, printed2 = run_main(tmp_path, "b.json", capsys)
    assert first == second and printed1 == printed2.replace("b.json", "a.json")
    assert json.loads(first)["tags"]["listed"]["[шум]"]["count"] == 4
    for leaked in ("секретнаяфраза", "Иванов", "будем"):
        assert leaked not in first.decode("utf-8")
        assert leaked not in printed1
    assert "[шум]" in printed1


def test_no_timestamps_fails_with_numbers_only(tmp_path, capsys):
    src = tmp_path / "in.txt"
    src.write_text("просто текст без меток\nещё строка\n", encoding="utf-8")
    assert main([str(src), str(tmp_path / "out.json")]) == 1
    err = capsys.readouterr().err
    assert "не найдено ни одной метки" in err and "просто текст" not in err
    assert not (tmp_path / "out.json").exists()


def test_non_utf8_input_fails(tmp_path, capsys):
    src = tmp_path / "in.txt"
    src.write_bytes("0:01\nтекст\n".encode("cp1251"))
    assert main([str(src), str(tmp_path / "out.json")]) == 1
    assert "не в UTF-8" in capsys.readouterr().err
