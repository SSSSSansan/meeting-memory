"""Агрегаты по одной настоящей расшифровке: только числа, без содержимого текста.

    python scripts/analyze_real_transcript.py ../drafts/real_sample.txt ../drafts/real_sample_stats.json

Вход — файл из блоков «метка времени, под ней текст, пустая строка», без меток
говорящих. Метка: М:СС или Ч:ММ:СС. В JSON и в консоль не попадают ни реплики,
ни имена; из текста выводятся только служебные теги в квадратных скобках
(редкие теги, встретившиеся реже --min-tag-count раз, сворачиваются в счётчик:
в скобках может оказаться имя), а также номера строк и их длины.

Строки, не подходящие ни под метку, ни под текст блока (текст до первой метки,
номера страниц, разрывы страниц, метка с текстом на той же строке и т. п.),
в статистику не входят и перечисляются в anomalies с номером, длиной и видом.
Служебный текст колонтитулов отличить от речи по форме нельзя: он остаётся
в блоках, а число повторяющихся длинных строк показано в format_check.

Порядок ключей фиксирован, числа округлены, повторный запуск даёт тот же файл.
Код возврата 1, если в файле не нашлось ни одной метки времени или он не UTF-8.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

TS_RE = re.compile(r"^\s*(?:(\d{1,2}):(\d{2}):(\d{2})|(\d{1,3}):(\d{2}))\s*$")
TAG_RE = re.compile(r"\[[^\[\]\n]{1,40}\]")
DIGIT_RE = re.compile(r"\d+(?:[.,:]\d+)*")
EDGE_RE = re.compile(r"^\W+|\W+$")
WORD_CHAR_RE = re.compile(r"\w")
SPACE_BEFORE_PUNCT_RE = re.compile(r"[ \u00a0][,.;:!?…]")
SINGLE_PERIOD_RE = re.compile(r"(?<!\.)\.(?!\.)")
DASH_RE = re.compile(r"[–—]|(?<!\w)-(?!\w)")

# виды аномальных строк (кроме page_break_char и before_first_timestamp)
ANOMALY_PATTERNS = [
    ("malformed_timestamp", re.compile(r"^\s*\d+:\d+(?::\d+)*\s*$")),
    ("timestamp_with_text", re.compile(r"^\s*\d{1,3}:\d{2}(?::\d{2})?[^\d:\s]|^\s*\d{1,3}:\d{2}(?::\d{2})?\s+\S")),
    ("digits_only", re.compile(r"^\s*\d+\s*$")),
    ("page_label", re.compile(
        r"(?i)^\s*(?:(?:стр|страница|page)\.?\s*\d+(?:\s*(?:из|of|/)\s*\d+)?"
        r"|\d+\s*(?:из|of|/)\s*\d+|[-–—]\s*\d+\s*[-–—])\s*$")),
    ("separator_line", re.compile(r"^\s*[-–—_=*]{3,}\s*$")),
]

TERMINAL = ".?!…"
CLOSERS = "\"'»”’)"
DASHES = "-–—‐‑"
KK_LETTERS = set("әғқңөұүһі")
SHORT_BLOCK_SECONDS = 20
FILLER_WORDS = ["эээ", "эм", "ммм", "ну", "так", "вот", "короче", "типа", "значит",
                "условно", "скажем", "да"]
FILLER_PHRASES = ["как бы", "как говорится", "в общем", "то есть"]
NUMERALS = frozenset([
    "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять", "десять",
    "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать",
    "семнадцать", "восемнадцать", "девятнадцать", "двадцать", "тридцать", "сорок",
    "пятьдесят", "сто", "тысяча",
])


@dataclass
class Block:
    line_no: int
    start: int
    lines: list[str] = field(default_factory=list)
    starts_with_blank: bool = False
    internal_blank: bool = False
    _pending_blank: bool = False


@dataclass
class Anomaly:
    line_no: int
    length: int
    kind: str


@dataclass
class Parsed:
    blocks: list[Block]
    anomalies: list[Anomaly]
    total_lines: int
    blank_lines: int
    timestamp_lines: int
    text_lines: int
    ts_not_after_blank: int
    repeated_long_lines: Counter


def parse(text: str) -> Parsed:
    blocks: list[Block] = []
    anomalies: list[Anomaly] = []
    blank = ts_lines = text_lines = ts_not_after_blank = 0
    seen: Counter = Counter()
    current: Block | None = None
    prev_blank = True  # начало файла считается «после пустой строки»
    prev_was_ts = False
    # split по \n, а не splitlines(): в тексте могут быть другие разделители строк
    raw_lines = text.split("\n")
    for no, raw in enumerate(raw_lines, 1):
        line = raw.removesuffix("\r")
        if "\f" in line:
            anomalies.append(Anomaly(no, len(line), "page_break_char"))
            line = line.replace("\f", "")
        if not line.strip():
            blank += 1
            if current is not None:
                if prev_was_ts:
                    current.starts_with_blank = True
                if current.lines:
                    current._pending_blank = True
            prev_blank, prev_was_ts = True, False
            continue
        m = TS_RE.match(line)
        if m:
            if m.group(1) is not None:
                start = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
            else:
                start = int(m.group(4)) * 60 + int(m.group(5))
            if not prev_blank:
                ts_not_after_blank += 1
            current = Block(no, start)
            blocks.append(current)
            ts_lines += 1
            prev_blank, prev_was_ts = False, True
            continue
        kind = next((k for k, rx in ANOMALY_PATTERNS if rx.match(line)), None)
        if kind is None and current is None:
            kind = "before_first_timestamp"
        prev_blank, prev_was_ts = False, False
        if kind is not None:
            anomalies.append(Anomaly(no, len(line), kind))
            continue
        assert current is not None
        stripped = line.strip()
        if current._pending_blank:
            current.internal_blank = True
            current._pending_blank = False
        current.lines.append(stripped)
        text_lines += 1
        if len(stripped) >= 20:
            seen[stripped] += 1
    repeated = Counter({k: v for k, v in seen.items() if v >= 3})
    return Parsed(blocks, anomalies, len(raw_lines), blank, ts_lines, text_lines,
                  ts_not_after_blank, repeated)


# ---------- числа ----------

def r3(x: float | None) -> float | None:
    return None if x is None else round(x, 3)


def pct(part: int, whole: int) -> float | None:
    return round(100 * part / whole, 2) if whole else None


def per(part: int, whole: int, k: int) -> float | None:
    return round(part * k / whole, 2) if whole else None


def percentile(sorted_vals: list[float], q: float) -> float | None:
    if not sorted_vals:
        return None
    pos = (len(sorted_vals) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(sorted_vals) - 1)
    return sorted_vals[lo] + (sorted_vals[hi] - sorted_vals[lo]) * (pos - lo)


def dist(values: list[float], keys: tuple[tuple[str, float], ...]) -> dict:
    vals = sorted(values)
    out: dict = {"n": len(vals)}
    for name, q in keys:
        out[name] = r3(percentile(vals, q))
    return out


MIN_MEDIAN_P90_MAX = (("min", 0.0), ("median", 0.5), ("p90", 0.9), ("max", 1.0))
P10_MEDIAN_P90 = (("p10", 0.1), ("median", 0.5), ("p90", 0.9))


@dataclass
class View:
    block: Block
    text: str          # строки блока через \n
    clean: str         # текст без тегов
    raw_tokens: list[str]
    words: list[str]   # те же слова в нижнем регистре, без пунктуации по краям

    @property
    def n_words(self) -> int:
        return len(self.words)


def make_view(block: Block) -> View:
    text = "\n".join(block.lines)
    clean = TAG_RE.sub(" ", text)
    raw_tokens = [t for t in clean.split() if WORD_CHAR_RE.search(t)]
    words = [EDGE_RE.sub("", t).lower() for t in raw_tokens]
    return View(block, text, clean, raw_tokens, words)


def first_alnum(clean: str) -> str:
    return next((c for c in clean if c.isalnum()), "")


def script_class(word: str) -> str:
    names = [unicodedata.name(c, "") for c in word if c.isalpha()]
    latin = any(n.startswith("LATIN") for n in names)
    cyr = any(n.startswith("CYRILLIC") for n in names)
    if latin and cyr:
        return "mixed"
    return "latin" if latin else ("cyrillic" if cyr else "other")


def tag_position(view: View, m: re.Match, lead: int, body: str) -> str:
    if not view.clean.strip():
        return "whole_block"
    a, b = m.start() - lead, m.end() - lead
    if a == 0:
        return "start"
    if b == len(body):
        return "end"
    return "middle"


def analyze(parsed: Parsed, min_tag_count: int = 3) -> dict:
    blocks = parsed.blocks
    views = [make_view(b) for b in blocks]
    n = len(blocks)
    with_text = [v for v in views if v.block.lines]
    with_words = [v for v in views if v.n_words]
    total_words = sum(v.n_words for v in views)

    starts = [b.start for b in blocks]
    gaps = [starts[i + 1] - starts[i] for i in range(n - 1)]
    positive = [(i, g) for i, g in enumerate(gaps) if g > 0]

    out: dict = {}
    out["format_check"] = {
        "total_lines": parsed.total_lines,
        "blank_lines": parsed.blank_lines,
        "timestamp_lines": parsed.timestamp_lines,
        "text_lines_in_blocks": parsed.text_lines,
        "anomalous_lines": len(parsed.anomalies),
        "anomalous_lines_by_kind": dict(sorted(Counter(a.kind for a in parsed.anomalies).items())),
        "timestamps_not_preceded_by_blank_line": parsed.ts_not_after_blank,
        "blocks_starting_with_blank_line": sum(1 for b in blocks if b.starts_with_blank and b.lines),
        "blocks_with_blank_line_between_text": sum(b.internal_blank for b in blocks),
        "distinct_repeated_long_lines": len(parsed.repeated_long_lines),
        "occurrences_of_repeated_long_lines": sum(parsed.repeated_long_lines.values()),
        "non_increasing_timestamps": sum(1 for g in gaps if g <= 0),
    }
    out["anomalies"] = [
        {"line": a.line_no, "length": a.length, "kind": a.kind} for a in parsed.anomalies
    ]

    total_seconds = starts[-1] - starts[0] if n else 0
    out["blocks"] = {
        "total": n,
        "empty": sum(1 for b in blocks if not b.lines),
        "tag_only": len(with_text) - len(with_words),
        "with_words": len(with_words),
        "total_seconds": total_seconds,
        "duration_seconds": dist([g for _, g in positive], MIN_MEDIAN_P90_MAX),
        "words_per_block": dist([v.n_words for v in with_words], MIN_MEDIAN_P90_MAX),
        "share_of_1_3_word_blocks_pct": pct(sum(1 for v in with_words if v.n_words <= 3),
                                            len(with_words)),
        "total_words": total_words,
    }

    known = [(views[i], g) for i, g in positive]
    rate_blocks = [(v, g) for v, g in known if v.n_words and g <= SHORT_BLOCK_SECONDS]
    out["speech_rate"] = {
        "words_per_second_overall": r3(
            sum(v.n_words for v, _ in known) / sum(g for _, g in known) if known else None),
        "short_blocks_max_seconds": SHORT_BLOCK_SECONDS,
        "short_blocks_words_per_second": dist([v.n_words / g for v, g in rate_blocks],
                                              P10_MEDIAN_P90),
    }

    not_terminated = starts_lower = ends_dash = continues = 0
    for i, v in enumerate(views):
        if not v.n_words:
            continue
        tail = v.clean.rstrip()
        tail_core = tail.rstrip(CLOSERS)
        if not tail_core or tail_core[-1] not in TERMINAL:
            not_terminated += 1
        if tail and tail[-1] in DASHES:
            ends_dash += 1
        if first_alnum(v.clean).islower():
            starts_lower += 1
            if i > 0 and views[i - 1].n_words:
                continues += 1
    out["breaks"] = {
        "blocks_not_ending_with_terminal_pct": pct(not_terminated, len(with_words)),
        "blocks_starting_lowercase_pct": pct(starts_lower, len(with_words)),
        "blocks_ending_with_dash_or_hyphen": ends_dash,
        "blocks_continuing_in_next_block": continues,
    }

    tag_counts: Counter = Counter()
    tag_blocks: Counter = Counter()
    tag_pos: dict[str, Counter] = {}
    for v in with_text:
        body = v.text.strip()
        lead = len(v.text) - len(v.text.lstrip())
        in_block = set()
        for m in TAG_RE.finditer(v.text):
            key = " ".join(m.group(0).lower().split())
            tag_counts[key] += 1
            in_block.add(key)
            tag_pos.setdefault(key, Counter())[tag_position(v, m, lead, body)] += 1
        for key in in_block:
            tag_blocks[key] += 1
    tags: dict = {}
    rare = [k for k, c in tag_counts.items() if c < min_tag_count]
    for key, count in sorted(tag_counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if count < min_tag_count:
            continue
        pos = tag_pos[key]
        tags[key] = {
            "count": count,
            "blocks_pct": pct(tag_blocks[key], len(with_text)),
            "position": {p: pos.get(p, 0) for p in ("start", "middle", "end", "whole_block")},
        }
    out["tags"] = {
        "total_occurrences": sum(tag_counts.values()),
        "distinct": len(tag_counts),
        "min_count_to_list": min_tag_count,
        "listed": tags,
        "rare_unlisted": {"distinct": len(rare), "occurrences": sum(tag_counts[k] for k in rare)},
    }

    word_counter: Counter = Counter(w for v in views for w in v.words)
    fillers: dict = {}
    total_fillers = 0
    for w in FILLER_WORDS:
        c = word_counter[w]
        total_fillers += c
        fillers[w] = {"count": c, "per_1000_words": per(c, total_words, 1000)}
    collapsed = [" ".join(v.clean.lower().split()) for v in views]
    joined_words = [" " + " ".join(v.words) + " " for v in views]
    for ph in FILLER_PHRASES:
        # по заданию считаем подстроку («как бы» внутри «как быстро» тоже попадёт);
        # рядом даём счёт по целым словам, чтобы оценить долю таких случаев
        sub = sum(t.count(ph) for t in collapsed)
        whole = sum(t.count(f" {ph} ") for t in joined_words)
        total_fillers += sub
        fillers[ph] = {"count_substring": sub, "count_whole_words": whole,
                       "per_1000_words": per(sub, total_words, 1000)}
    fillers["_total_per_1000_words"] = per(total_fillers, total_words, 1000)
    out["fillers"] = fillers

    plain = comma = 0
    for v in views:
        for i in range(1, len(v.words)):
            if v.words[i] and v.words[i] == v.words[i - 1]:
                prev = v.raw_tokens[i - 1]
                if prev.endswith(","):
                    comma += 1
                elif prev[-1].isalnum():
                    plain += 1
    out["repeats"] = {
        "adjacent_same_word": plain,
        "adjacent_same_word_per_1000_words": per(plain, total_words, 1000),
        "same_word_through_comma": comma,
        "same_word_through_comma_per_1000_words": per(comma, total_words, 1000),
    }

    clean_all = "\n".join(v.clean for v in views)
    outside = capital = capital_mid = 0
    for v in views:
        for i in range(1, len(v.words)):
            outside += 1
            if EDGE_RE.sub("", v.raw_tokens[i])[:1].isupper():
                capital += 1
                if v.raw_tokens[i - 1][-1] not in TERMINAL:
                    capital_mid += 1
    out["punctuation"] = {
        "commas_per_100_words": per(clean_all.count(","), total_words, 100),
        "periods_per_100_words": per(len(SINGLE_PERIOD_RE.findall(clean_all)), total_words, 100),
        "question_marks_per_100_words": per(clean_all.count("?"), total_words, 100),
        "dashes_per_100_words": per(len(DASH_RE.findall(clean_all)), total_words, 100),
        "words_outside_block_start": outside,
        "capitalized_outside_block_start_pct": pct(capital, outside),
        "capitalized_not_after_sentence_end_pct": pct(capital_mid, outside),
    }

    scripts: Counter = Counter(script_class(w) for v in views for w in v.words)
    out["latin_and_numbers"] = {
        "latin_words_pct": pct(scripts["latin"], total_words),
        "cyrillic_words_with_latin_letters": scripts["mixed"],
        "digit_numbers": sum(len(DIGIT_RE.findall(v.clean)) for v in views),
        "numerals_as_words": sum(c for w, c in word_counter.items() if w in NUMERALS),
    }

    kk = sum(1 for v in views for w in v.words if any(c in KK_LETTERS for c in w))
    out["kazakh"] = {"words_with_kazakh_letters": kk,
                     "words_with_kazakh_letters_pct": pct(kk, total_words)}

    out["pauses"] = {"gaps_over_10_s": sum(1 for g in gaps if g > 10),
                     "gaps_over_30_s": sum(1 for g in gaps if g > 30)}

    out["normalization"] = {
        "blocks_with_double_spaces": sum(
            1 for b in blocks if any("  " in ln or "\t" in ln for ln in b.lines)),
        "blocks_with_line_breaks_inside": sum(1 for b in blocks if len(b.lines) > 1),
        "blocks_with_space_before_punctuation": sum(
            1 for b in blocks if any(SPACE_BEFORE_PUNCT_RE.search(ln) for ln in b.lines)),
    }
    return out


# ---------- вывод ----------

def summary(stats: dict) -> str:
    b, s, k = stats["blocks"], stats["speech_rate"], stats["breaks"]
    t, f, rp = stats["tags"], stats["fillers"], stats["repeats"]
    p, ln, kz = stats["punctuation"], stats["latin_and_numbers"], stats["kazakh"]
    fc = stats["format_check"]
    lines: list[str] = []
    add = lines.append
    add(f"Строк: {fc['total_lines']}, пустых: {fc['blank_lines']}, меток: {fc['timestamp_lines']}, "
        f"текстовых строк в блоках: {fc['text_lines_in_blocks']}, аномальных: {fc['anomalous_lines']}")
    add(f"Блоков: {b['total']} (пустых {b['empty']}, только теги {b['tag_only']}), "
        f"слов: {b['total_words']}, длительность {b['total_seconds']} с")
    d, w = b["duration_seconds"], b["words_per_block"]
    add(f"Длительность блока, с: min {d['min']}, медиана {d['median']}, p90 {d['p90']}, max {d['max']}")
    add(f"Слов в блоке: min {w['min']}, медиана {w['median']}, p90 {w['p90']}, max {w['max']}; "
        f"блоков из 1-3 слов {b['share_of_1_3_word_blocks_pct']}%")
    sr = s["short_blocks_words_per_second"]
    add(f"Скорость, слов/с: всего {s['words_per_second_overall']}; блоки ≤{SHORT_BLOCK_SECONDS} с: "
        f"p10 {sr['p10']}, медиана {sr['median']}, p90 {sr['p90']}")
    add(f"Обрывы: без конечной пунктуации {k['blocks_not_ending_with_terminal_pct']}%, "
        f"со строчной буквы {k['blocks_starting_lowercase_pct']}%, на дефис/тире "
        f"{k['blocks_ending_with_dash_or_hyphen']}, продолжаются в следующем "
        f"{k['blocks_continuing_in_next_block']}")
    add(f"Теги: всего {t['total_occurrences']}, видов {t['distinct']}, в списке {len(t['listed'])}")
    for tag, info in t["listed"].items():
        add(f"  {tag}: {info['count']}, блоков {info['blocks_pct']}%, позиции {info['position']}")
    nm = stats["normalization"]
    add(f"Паразиты на 1000 слов (суммарно): {f['_total_per_1000_words']}")
    add(f"Повторы на 1000 слов: подряд {rp['adjacent_same_word_per_1000_words']}, "
        f"через запятую {rp['same_word_through_comma_per_1000_words']}")
    add(f"На 100 слов: запятых {p['commas_per_100_words']}, точек {p['periods_per_100_words']}, "
        f"вопросов {p['question_marks_per_100_words']}, тире {p['dashes_per_100_words']}; "
        f"с заглавной вне начала блока {p['capitalized_outside_block_start_pct']}%")
    add(f"Латиница {ln['latin_words_pct']}% слов, смесь алфавитов "
        f"{ln['cyrillic_words_with_latin_letters']}, цифровых записей {ln['digit_numbers']}, "
        f"числительных словами {ln['numerals_as_words']}")
    add(f"Казахские буквы: {kz['words_with_kazakh_letters']} слов "
        f"({kz['words_with_kazakh_letters_pct']}%)")
    add(f"Паузы: >10 с {stats['pauses']['gaps_over_10_s']}, >30 с {stats['pauses']['gaps_over_30_s']}")
    add(f"Нормализация: двойные пробелы {nm['blocks_with_double_spaces']}, "
        f"переносы внутри {nm['blocks_with_line_breaks_inside']}, "
        f"пробел перед знаком {nm['blocks_with_space_before_punctuation']}")
    if stats["anomalies"]:
        lines.append("Аномальные строки (номер, длина, вид):")
        lines += [f"  {a['line']}, {a['length']}, {a['kind']}" for a in stats["anomalies"]]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--min-tag-count", type=int, default=3,
                    help="теги реже этого числа не перечисляются (в скобках может быть имя)")
    a = ap.parse_args(argv)
    try:
        text = a.input.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError as e:
        print(f"ОШИБКА: файл не в UTF-8 (байт {e.start})", file=sys.stderr)
        return 1
    except OSError as e:
        print(f"ОШИБКА: {e.strerror}: {a.input}", file=sys.stderr)
        return 1
    parsed = parse(text)
    if not parsed.blocks:
        print(f"ОШИБКА: не найдено ни одной метки времени; строк {parsed.total_lines}, "
              f"пустых {parsed.blank_lines}, аномальных {len(parsed.anomalies)}: "
              f"{dict(sorted(Counter(x.kind for x in parsed.anomalies).items()))}",
              file=sys.stderr)
        return 1
    stats = analyze(parsed, a.min_tag_count)
    # LF и UTF-8 независимо от ОС: повторный запуск даёт тот же файл
    a.output.write_bytes((json.dumps(stats, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print(summary(stats))
    print(f"\nJSON: {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
