"""Слой шума: чистые транскрипты Scribal → «испорченные» версии трёх уровней.

    python scripts/add_noise.py                       # data/raw → data/noisy, журналы в data/gold/noise_logs
    python scripts/add_noise.py DIR --text            # + m_NNN.txt в формате настоящей расшифровки

Для каждой встречи (m_NNN.transcript.json + m_NNN.meeting.json) и каждой из шести комбинаций
уровень (clean, medium, strong) × вариант (labeled, unlabeled) пишется
data/noisy/<level>_<variant>/m_NNN.transcript.json и m_NNN.meeting.json (meeting копируется
как есть). Журнал с ответами — data/gold/noise_logs/<level>_<variant>/m_NNN.json: он лежит
в gold и не должен попадать на вход системы.

Параметры — в scripts/noise_levels.yaml (скрипт пишет их в журнал). Любой случайный выбор
берёт отдельный генератор с зерном sha256("<зерно>:<встреча>:<уровень>:<вариант>:<операция>"),
повторный запуск даёт побайтово те же файлы. Порядок операций:

  1 междометия (удаляется только слово-междометие; пунктуация уходит на предыдущее слово)
  2 многоточия (запятая или ничего; в конце реплики точка или ничего)
  3 названия и имена из world.yaml (упоминания пишутся в журнал, замена на другую форму)
  4 потеря казахских букв (гипотеза)   5 латиница по звучанию
  6 конечные знаки предложений и заглавные буквы   7 ошибки меток (только labeled)
  8 блоки по словам (только unlabeled, medium и strong)   9 тег шума в конце блока

Блоки (операция 8): размер n берётся из логнормального распределения. С вероятностью
boundary_snap_prob разрез сдвигается к ближайшему токену, кончающему предложение или реплику
(окно ±max(3, round(0.3·n)), при равенстве берётся более дальний). Границы определяются по
чистому тексту и едут вместе с токеном; при удалении слова граница переходит на предыдущий
оставшийся токен. Привязка использует отдельный генератор, поэтому при вероятности 0
результат тот же, что у нарезки без привязки. В журнале у каждого блока поля snapped и
boundary (sentence_end, reply_end, mid_sentence) и счётчики block_boundaries.

Решения там, где задание допускает толкования:
  * граница «конец предложения» ищется после снятия закрывающих кавычек и скобок;
  * слова, распознанные как упоминания сущностей (заменённые или нет), не затрагиваются
    операциями 4 и 5, чтобы форма сущности оставалась в списке написаний;
  * многоточие в начале токена переносится на предыдущее слово (в начале реплики пропадает),
    токен из одного многоточия удаляется, многоточие внутри слова не трогается и считается
    в счётчике ellipsis_skipped;
  * заглавная буква ставится только на строчную первую букву: слова, уже написанные
    с заглавной, не меняются;
  * блок короче 5 слов в конце потока присоединяется к предыдущему, поэтому максимум блока
    может быть до 254 слов.
Код возврата 1, если вход или параметры некорректны (файлы при этом не пишутся).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
INPUT_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "noisy"
LOG_DIR = ROOT / "data" / "gold" / "noise_logs"
CONFIG_PATH = ROOT / "scripts" / "noise_levels.yaml"
WORLD_PATH = ROOT / "data" / "gold" / "world.yaml"
SPEAKER_MAPS_PATH = ROOT / "data" / "gold" / "speaker_maps.json"

LEVELS = ["clean", "medium", "strong"]
VARIANTS = ["labeled", "unlabeled"]
LEVEL_KEYS = [
    "filler_remove_prob", "ellipsis_mode", "entity_variant_prob", "kk_letter_loss_prob",
    "transliteration_prob", "sentence_end_drop_prob", "sentence_start_capitalize_prob",
    "noise_tag_prob", "speaker_mislabel_prob", "block_words", "boundary_snap_prob",
]
OPS = ["fillers", "ellipsis", "entities", "kk_loss", "translit", "sentence_end",
       "sentence_start", "mislabel", "blocks", "block_snap", "noise_tag"]

SPLIT_RE = re.compile(r"^(\W*)(.*?)(\W*)$", re.DOTALL)
ELL = r"(?:\.{3,}|…)"
ELL_RE = re.compile(ELL)
ONLY_ELL_RE = re.compile(rf"{ELL}+")
LEAD_ELL_RE = re.compile(rf"^{ELL}")
TRAIL_ELL_RE = re.compile(rf"{ELL}$")
TERM_TRAIL_RE = re.compile(r"^([.?!]+)([\"»”’')\]]*)$")
TAG_RE = re.compile(r"\[[^\[\]\n]{1,40}\]")
KK_RE = re.compile("[әғқңөұүһіӘҒҚҢӨҰҮҺІ]")
DIGITS_RE = re.compile(r"\d+(?:[.,:]\d+)*")
MEETING_FILE_RE = re.compile(r"^(m_\d{3})\.transcript\.json$")
CLOSERS = "\"'»”’)"
HYPHEN_SUFFIX_RE = re.compile(r"^-[^\W\d_]+$")


class NoiseError(ValueError):
    """Некорректный вход или параметры: сообщение показывается пользователю."""


# ---------- токены ----------

def split_token(text: str) -> tuple[str, str, str]:
    m = SPLIT_RE.match(text)
    assert m is not None
    return m.group(1), m.group(2), m.group(3)


def ends_alnum(text: str) -> bool:
    return bool(text) and text[-1].isalnum()


def has_terminal(text: str) -> bool:
    """Токен заканчивается . ? ! (многоточие не считается)."""
    _, core, trail = split_token(text)
    m = TERM_TRAIL_RE.match(trail)
    return bool(core) and m is not None and "..." not in m.group(1)


@dataclass
class Tok:
    text: str
    start: float
    end: float
    seg: int
    idx: int
    entity: dict | None = None
    events: list[dict] = field(default_factory=list)
    # границы считаются по чистому тексту до всех операций и едут вместе с токеном
    sent_end: bool = False   # токен кончает предложение (. ? ! …)
    reply_end: bool = False  # последний токен реплики

    @property
    def boundary(self) -> bool:
        return self.sent_end or self.reply_end

    def inherit_boundary(self, removed: Tok) -> None:
        """Удалённый токен передаёт границу предыдущему оставшемуся."""
        self.sent_end |= removed.sent_end
        self.reply_end |= removed.reply_end

    def change(self, op: str, new: str, **extra: object) -> None:
        if new != self.text:
            self.events.append({"operation": op, "was": self.text, "now": new, **extra})
            self.text = new

    @property
    def source(self) -> dict:
        return {"segment": self.seg, "token": self.idx}


@dataclass
class Seg:
    idx: int
    speaker: str
    start: float
    end: float
    toks: list[Tok]
    mislabel: dict | None = None


@dataclass
class Out:
    """Выходной сегмент: реплика (labeled) или блок (unlabeled)."""
    toks: list[Tok]
    start: float
    end: float
    speaker: str | None
    tag: str | None = None
    mislabel: dict | None = None
    snapped: bool | None = None  # только для блоков: разрез привязан к границе

    @property
    def text(self) -> str:
        body = " ".join(t.text for t in self.toks)
        return f"{body} {self.tag}" if self.tag else body


def make_rng(seed: int, meeting: str, level: str, variant: str, op: str) -> random.Random:
    key = f"{seed}:{meeting}:{level}:{variant}:{op}"
    return random.Random(int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "big"))


# ---------- параметры и справочники ----------

def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    for key in ("global_seed", "fillers", "noise_tag", "kk_letter_map",
                "transliteration_table", "levels", "block_min_tail_words"):
        if key not in cfg:
            raise NoiseError(f"{path.name}: нет ключа {key}")
    for level in LEVELS:
        params = (cfg["levels"] or {}).get(level)
        if not isinstance(params, dict):
            raise NoiseError(f"{path.name}: нет уровня {level}")
        missing = [k for k in LEVEL_KEYS if k not in params]
        if missing:
            raise NoiseError(f"{path.name}: у уровня {level} нет параметров {missing}")
        for k in LEVEL_KEYS:
            if k.endswith("_prob") and not 0 <= params[k] <= 1:
                raise NoiseError(f"{path.name}: {level}.{k} должно быть от 0 до 1")
        if params["ellipsis_mode"] not in ("none", "drop"):
            raise NoiseError(f"{path.name}: {level}.ellipsis_mode должен быть none или drop")
    return cfg


def build_entity_index(world: dict) -> dict[str, dict]:
    """Форма (в нижнем регистре) → сущность. Ошибка, если у сущности нет written_variants."""
    index: dict[str, dict] = {}
    for kind in ("people", "projects"):
        for e in world.get(kind) or []:
            forms = e.get("written_variants")
            if not isinstance(forms, list):
                raise NoiseError(f"world.yaml: у {e.get('id')} нет written_variants")
            entry = {"id": e["id"], "name": e.get("name"), "forms": list(forms)}
            for form in forms:
                prev = index.get(form.lower())
                if prev is not None and prev["id"] != e["id"]:
                    raise NoiseError(
                        f"world.yaml: форма {form!r} у {prev['id']} и {e['id']} одновременно")
                index[form.lower()] = entry
    return index


# ---------- операции ----------

def op_fillers(segs: list[Seg], rng: random.Random, prob: float, fillers: set[str],
               removed: list[dict]) -> None:
    for seg in segs:
        kept: list[Tok] = []
        for tok in seg.toks:
            _, core, trail = split_token(tok.text)
            if core.lower() in fillers and rng.random() < prob:
                removed.append({"operation": "fillers", "was": tok.text, "now": "",
                                "source": tok.source})
                if kept:
                    kept[-1].inherit_boundary(tok)
                    if trail and ends_alnum(kept[-1].text):
                        kept[-1].change("fillers", kept[-1].text + trail)
                continue
            kept.append(tok)
        seg.toks = kept


def op_ellipsis(segs: list[Seg], rng: random.Random, removed: list[dict],
                skipped: Counter) -> None:
    def pick(at_end: bool) -> str:
        return ("." if at_end else ",") if rng.random() < 0.5 else ""

    for seg in segs:
        kept: list[Tok] = []
        last = len(seg.toks) - 1
        for i, tok in enumerate(seg.toks):
            text = tok.text
            if not ELL_RE.search(text):
                kept.append(tok)
                continue
            if ONLY_ELL_RE.fullmatch(text):
                rep = pick(i == last)
                removed.append({"operation": "ellipsis", "was": text, "now": "",
                                "source": tok.source})
                if kept:
                    kept[-1].inherit_boundary(tok)
                    if rep and ends_alnum(kept[-1].text):
                        kept[-1].change("ellipsis", kept[-1].text + rep)
                continue
            new = text
            lead = LEAD_ELL_RE.match(new)
            if lead:
                rep = pick(False)
                new = new[lead.end():]
                if rep and kept and ends_alnum(kept[-1].text):
                    kept[-1].change("ellipsis", kept[-1].text + rep)
            trail = TRAIL_ELL_RE.search(new)
            if trail:
                new = new[:trail.start()] + pick(i == last)
            if ELL_RE.search(new):
                skipped["ellipsis_skipped"] += 1
            tok.change("ellipsis", new)
            kept.append(tok)
        seg.toks = kept


def match_entity(text: str, index: dict[str, dict]) -> tuple[dict, str, str, str, str] | None:
    lead, core, trail = split_token(text)
    base, suffix = core, ""
    if "-" in core:
        base, rest = core.split("-", 1)
        suffix = "-" + rest
        if not HYPHEN_SUFFIX_RE.match(suffix):
            return None
    entity = index.get(base.lower())
    if entity is None:
        return None
    return entity, lead, base, suffix, trail


def match_case(form: str, like: str) -> str:
    if not form or not like:
        return form
    return form[0].upper() + form[1:] if like[0].isupper() else form[0].lower() + form[1:]


def op_entities(segs: list[Seg], rng: random.Random, prob: float,
                index: dict[str, dict]) -> None:
    for seg in segs:
        for tok in seg.toks:
            hit = match_entity(tok.text, index)
            if hit is None:
                continue
            entity, lead, base, suffix, trail = hit
            tok.entity = {"id": entity["id"], "name": entity["name"], "form": base,
                          "suffix": suffix}
            others = [f for f in entity["forms"] if f.lower() != base.lower()]
            if prob > 0 and others and rng.random() < prob:
                new_form = match_case(rng.choice(others), base)
                tok.change("entity_variant", lead + new_form + suffix + trail,
                           entity=entity["id"])


def op_kk_loss(segs: list[Seg], rng: random.Random, prob: float, table: dict[str, str]) -> None:
    full = {**table, **{k.upper(): v.upper() for k, v in table.items()}}
    translation = str.maketrans(full)
    for seg in segs:
        for tok in seg.toks:
            if tok.entity is None and KK_RE.search(tok.text) and rng.random() < prob:
                tok.change("kk_letter_loss", tok.text.translate(translation), hypothesis=True)


def translit_match(core: str, keys: list[str]) -> tuple[str, str] | None:
    low = core.lower()
    for key in keys:
        if low == key:
            return key, ""
        if low.startswith(key + "-"):
            return key, core[len(key):]
    return None


def op_translit(segs: list[Seg], rng: random.Random, prob: float, table: dict[str, str]) -> None:
    lowered = {k.lower(): v for k, v in table.items()}
    keys = sorted(lowered, key=len, reverse=True)
    for seg in segs:
        for tok in seg.toks:
            if tok.entity is not None:
                continue
            lead, core, trail = split_token(tok.text)
            hit = translit_match(core, keys)
            if hit and rng.random() < prob:
                key, rest = hit
                tok.change("transliteration", lead + lowered[key] + rest + trail)


def first_alpha(text: str) -> int:
    return next((i for i, c in enumerate(text) if c.isalpha()), -1)


def op_sentences(segs: list[Seg], rng_end: random.Random, rng_start: random.Random,
                 p_drop: float, p_cap: float) -> None:
    for seg in segs:
        starts = [i == 0 or has_terminal(seg.toks[i - 1].text) for i in range(len(seg.toks))]
        if p_drop > 0:
            for tok in seg.toks:
                if has_terminal(tok.text) and rng_end.random() < p_drop:
                    lead, core, trail = split_token(tok.text)
                    m = TERM_TRAIL_RE.match(trail)
                    assert m is not None
                    tok.change("sentence_end_drop", lead + core + m.group(2))
        if p_cap > 0:
            for tok, is_start in zip(seg.toks, starts, strict=True):
                pos = first_alpha(tok.text)
                if is_start and pos >= 0 and tok.text[pos].islower() and rng_start.random() < p_cap:
                    tok.change("sentence_start_capitalize",
                               tok.text[:pos] + tok.text[pos].upper() + tok.text[pos + 1:])


def op_mislabel(segs: list[Seg], rng: random.Random, prob: float, labels: list[str]) -> None:
    for seg in segs:
        if len(seg.toks) <= 3 and rng.random() < prob:
            others = [lb for lb in labels if lb != seg.speaker]
            if others:
                new = rng.choice(others)
                seg.mislabel = {"was": seg.speaker, "now": new}
                seg.speaker = new


def snap_window(size: int) -> int:
    return max(3, round(0.3 * size))


def snap_position(boundary: list[bool], size: int) -> int | None:
    """Размер блока, при котором разрез ложится на границу, ближайшую к size.

    boundary[k] — токен k оставшегося потока кончает предложение или реплику. Окно
    [size - d, size + d]; при равном расстоянии берётся более дальний от начала."""
    d = snap_window(size)
    best: tuple[tuple[int, int], int] | None = None
    for j in range(max(1, size - d), min(len(boundary), size + d) + 1):
        if boundary[j - 1]:
            key = (abs(j - size), -j)
            if best is None or key < best[0]:
                best = (key, j)
    return None if best is None else best[1]


def make_blocks(tokens: list[Tok], rng: random.Random, spec: dict, min_tail: int,
                snap_rng: random.Random | None = None,
                snap_prob: float = 0.0) -> list[tuple[list[Tok], bool]]:
    """Режет поток токенов на блоки; второй элемент пары — разрез привязан к границе.

    При snap_prob == 0 генератор привязки не трогается: результат тот же, что без привязки."""
    mu, sigma = math.log(spec["median"]), spec["sigma"]
    blocks: list[tuple[list[Tok], bool]] = []
    i, n = 0, len(tokens)
    while i < n:
        size = min(max(round(math.exp(rng.gauss(mu, sigma))), spec["min"]), spec["max"])
        snapped = False
        if snap_prob > 0 and snap_rng is not None and snap_rng.random() < snap_prob:
            reach = size + snap_window(size)
            j = snap_position([t.boundary for t in tokens[i:i + reach]], size)
            if j is not None:
                snapped = j <= spec["max"]  # за максимумом режем на максимуме, это уже не привязка
                size = min(j, spec["max"])
        rest = n - i - size
        if rest < min_tail:  # остаток (в том числе нулевой) уходит в этот же блок
            size = n - i
        blocks.append((tokens[i:i + size], snapped))
        i += size
    return blocks


def boundary_kind(tok: Tok) -> str:
    if tok.reply_end:
        return "reply_end"
    return "sentence_end" if tok.sent_end else "mid_sentence"


# ---------- один прогон: встреча × уровень × вариант ----------

def segs_from_raw(raw_segments: list[dict]) -> list[Seg]:
    segs = []
    for si, s in enumerate(raw_segments):
        words = s["text"].split()
        n = len(words)
        span = s["end"] - s["start"]
        toks = [Tok(w, s["start"] + span * i / n, s["start"] + span * (i + 1) / n, si, i,
                    sent_end=w.rstrip(CLOSERS)[-1:] in tuple(".?!…"), reply_end=i == n - 1)
                for i, w in enumerate(words)]
        segs.append(Seg(si, s["speaker"], s["start"], s["end"], toks))
    return segs


@dataclass
class Combo:
    level: str
    variant: str
    transcript: dict
    segments: list[dict]
    texts: list[str]
    log: dict
    counters: Counter
    txt: str | None = None


def dump_json(obj: object) -> bytes:
    # LF и UTF-8 независимо от ОС: файлы должны совпадать побайтово
    return (json.dumps(obj, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def process(meeting: str, level: str, variant: str, raw: dict, cfg: dict, index: dict,
            mapping: dict[str, str], seed: int) -> Combo:
    params = cfg["levels"][level]
    rngs = {op: make_rng(seed, meeting, level, variant, op) for op in OPS}
    segs = segs_from_raw(raw["segments"])
    removed: list[dict] = []
    skipped: Counter = Counter()

    if params["filler_remove_prob"] > 0:
        op_fillers(segs, rngs["fillers"], params["filler_remove_prob"], set(cfg["fillers"]), removed)
    if params["ellipsis_mode"] == "drop":
        op_ellipsis(segs, rngs["ellipsis"], removed, skipped)
    op_entities(segs, rngs["entities"], params["entity_variant_prob"], index)
    if params["kk_letter_loss_prob"] > 0:
        op_kk_loss(segs, rngs["kk_loss"], params["kk_letter_loss_prob"], cfg["kk_letter_map"])
    if params["transliteration_prob"] > 0:
        op_translit(segs, rngs["translit"], params["transliteration_prob"],
                    cfg["transliteration_table"])
    if params["sentence_end_drop_prob"] > 0 or params["sentence_start_capitalize_prob"] > 0:
        op_sentences(segs, rngs["sentence_end"], rngs["sentence_start"],
                     params["sentence_end_drop_prob"], params["sentence_start_capitalize_prob"])
    if variant == "labeled" and params["speaker_mislabel_prob"] > 0:
        op_mislabel(segs, rngs["mislabel"], params["speaker_mislabel_prob"], sorted(mapping))

    dropped = [s.idx for s in segs if not s.toks]
    live = [s for s in segs if s.toks]
    if variant == "labeled":
        outs = [Out(s.toks, round(s.toks[0].start, 2), round(s.toks[-1].end, 2), s.speaker,
                    mislabel=s.mislabel) for s in live]
    elif params["block_words"] is None:
        outs = [Out(s.toks, round(s.toks[0].start, 2), round(s.toks[-1].end, 2), None)
                for s in live]
    else:
        stream = [t for s in live for t in s.toks]
        blocks = make_blocks(stream, rngs["blocks"], params["block_words"],
                             cfg["block_min_tail_words"], rngs["block_snap"],
                             params["boundary_snap_prob"])
        outs = [Out(b, round(b[0].start, 2), round(b[-1].end, 2), None, snapped=snapped)
                for b, snapped in blocks]
    if params["noise_tag_prob"] > 0:
        for o in outs:
            o.tag = cfg["noise_tag"] if rngs["noise_tag"].random() < params["noise_tag_prob"] else None

    texts = [o.text for o in outs]
    if level == "clean":
        # чистый уровень обязан совпадать со входом: проверяем, что токенизация обратима
        original = [s["text"] for s in raw["segments"]]
        if variant == "labeled" and texts != original:
            raise NoiseError(f"{meeting}: текст сегментов не восстанавливается из токенов "
                             "(лишние пробелы во входе?)")
        segments = [{k: v for k, v in s.items() if variant == "labeled" or k != "speaker"}
                    for s in raw["segments"]]
        if variant == "unlabeled":
            texts = original
    else:
        segments = []
        for o in outs:
            seg = {"start": o.start, "end": o.end}
            if o.speaker is not None:
                seg["speaker"] = o.speaker
            seg["text"] = o.text
            segments.append(seg)

    transcript = {}
    for key, value in raw.items():
        if key == "text":
            transcript["text"] = " ".join(s["text"] for s in segments)
        elif key == "segments":
            transcript["segments"] = segments
        else:
            transcript[key] = value

    log, counters = build_log(meeting, level, variant, params, seed, rngs_seeds(
        seed, meeting, level, variant), raw, mapping, outs, removed, dropped, skipped)
    txt = make_txt(segments) if variant == "unlabeled" else None
    return Combo(level, variant, transcript, segments, texts, log, counters, txt)


def rngs_seeds(seed: int, meeting: str, level: str, variant: str) -> dict:
    return {op: int.from_bytes(hashlib.sha256(
        f"{seed}:{meeting}:{level}:{variant}:{op}".encode()).digest()[:8], "big") for op in OPS}


def make_txt(segments: list[dict]) -> str:
    long_meeting = bool(segments) and segments[-1]["end"] > 3600
    parts = []
    for s in segments:
        sec = int(s["start"])
        label = (f"{sec // 3600}:{sec % 3600 // 60:02d}:{sec % 60:02d}" if long_meeting
                 else f"{sec // 60:02d}:{sec % 60:02d}")
        parts.append(f"{label}\n{s['text']}\n\n")
    return "".join(parts)


def final_form(tok: Tok) -> str:
    _, core, _ = split_token(tok.text)
    suffix = tok.entity["suffix"] if tok.entity else ""
    return core[:len(core) - len(suffix)] if suffix and core.endswith(suffix) else core


def build_log(meeting: str, level: str, variant: str, params: dict, seed: int,
              seeds: dict, raw: dict, mapping: dict[str, str], outs: list[Out],
              removed: list[dict], dropped: list[int],
              skipped: Counter) -> tuple[dict, Counter]:
    segments_log, mentions, replacements = [], [], []
    boundaries: Counter = Counter()
    for oi, o in enumerate(outs):
        sources: dict[int, int] = {}
        for t in o.toks:
            sources[t.seg] = sources.get(t.seg, 0) + 1
        entry = {
            "index": oi, "start": o.start, "end": o.end, "words": len(o.toks),
            "sources": [{"segment": si, "speaker": raw["segments"][si]["speaker"],
                         "person": mapping.get(raw["segments"][si]["speaker"]), "words": n}
                        for si, n in sources.items()],
        }
        if o.snapped is not None:  # блок: чем кончается и привязан ли разрез к границе
            kind = boundary_kind(o.toks[-1])
            entry["snapped"], entry["boundary"] = o.snapped, kind
            boundaries[kind] += 1
            boundaries["snapped"] += o.snapped
        segments_log.append(entry)
        for ti, t in enumerate(o.toks):
            if t.entity is not None:
                final = final_form(t)
                mentions.append({
                    "segment": oi, "token": ti, "entity": t.entity["id"],
                    "canonical": t.entity["name"], "source_form": t.entity["form"],
                    "final_form": final, "changed": final != t.entity["form"],
                    "source": t.source})
            for ev in t.events:
                replacements.append({"operation": ev["operation"], "segment": oi, "token": ti,
                                     "was": ev["was"], "now": ev["now"],
                                     **({"hypothesis": True} if ev.get("hypothesis") else {}),
                                     "source": t.source})
        if o.mislabel:
            replacements.append({"operation": "speaker_mislabel", "segment": oi, "token": None,
                                 "was": o.mislabel["was"], "now": o.mislabel["now"],
                                 "source": {"segment": o.toks[0].seg, "token": None}})
        if o.tag:
            replacements.append({"operation": "noise_tag", "segment": oi, "token": None,
                                 "was": "", "now": o.tag,
                                 "source": {"segment": o.toks[-1].seg, "token": None}})
    for r in removed:
        replacements.append({"operation": r["operation"], "segment": None, "token": None,
                             "was": r["was"], "now": "", "removed": True,
                             "source": r["source"]})
    counters: Counter = Counter(r["operation"] for r in replacements)
    counters.update(skipped)
    counters["entity_mentions"] = len(mentions)
    counters["entity_mentions_changed"] = sum(m["changed"] for m in mentions)
    log = {
        "meeting_id": meeting, "level": level, "variant": variant, "global_seed": seed,
        "params": params, "seeds": seeds,
        "dropped_source_segments": dropped,
        "segments": segments_log, "entity_mentions": mentions, "replacements": replacements,
        "counters": dict(sorted(counters.items())),
    }
    if boundaries:
        log["block_boundaries"] = {k: boundaries[k] for k in
                                   ("sentence_end", "reply_end", "mid_sentence", "snapped")}
    return log, counters


# ---------- статистика ----------

def percentile(vals: list[float], q: float) -> float:
    s = sorted(vals)
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def stats(texts: list[str], fillers: set[str]) -> dict:
    bodies = [TAG_RE.sub("", t).strip() for t in texts]
    lengths = [len(b.split()) for b in bodies]
    words = sum(lengths)
    tokens = [w for b in bodies for w in b.split()]
    fill = sum(split_token(w)[1].lower() in fillers for w in tokens)
    ellipses = sum(len(ELL_RE.findall(b)) for b in bodies)
    commas = sum(b.count(",") for b in bodies)
    no_end = sum(1 for b in bodies if b.rstrip().rstrip(CLOSERS)[-1:] not in tuple(".?!…"))
    lower = sum(1 for b in bodies if next((c for c in b if c.isalnum()), "").islower())
    n = len(texts)
    return {
        "segments": n, "words": words,
        "len": (min(lengths), percentile(lengths, 0.5), percentile(lengths, 0.9), max(lengths)),
        "no_end_pct": 100 * no_end / n, "lower_pct": 100 * lower / n,
        "fillers_per_1000": 1000 * fill / words, "ellipsis_per_100": 100 * ellipses / words,
        "commas_per_100": 100 * commas / words,
        "tag_pct": 100 * sum(bool(TAG_RE.search(t)) for t in texts) / n,
    }


def stats_line(name: str, st: dict) -> str:
    lo, med, p90, hi = st["len"]
    return (f"  {name:<6} сегм {st['segments']:>4}  слов {st['words']:>5}  блок {lo:>2}/{med:>5.1f}/"
            f"{p90:>6.1f}/{hi:>3}  без.конца {st['no_end_pct']:>5.1f}%  строчн {st['lower_pct']:>5.1f}%  "
            f"межд/1000 {st['fillers_per_1000']:>5.2f}  мнг/100 {st['ellipsis_per_100']:>4.2f}  "
            f"зпт/100 {st['commas_per_100']:>5.2f}  [шум] {st['tag_pct']:>4.1f}%")


REAL_SAMPLE = {"len": (None, 21, 103, 247), "no_end_pct": 38.5, "lower_pct": 34.6,
               "fillers_per_1000": 0.0, "ellipsis_per_100": 0.0, "commas_per_100": 16.85,
               "tag_pct": 1.0}


# ---------- запуск ----------

def discover(input_dir: Path) -> list[str]:
    ids = sorted(m.group(1) for p in input_dir.iterdir() if (m := MEETING_FILE_RE.match(p.name)))
    if not ids:
        raise NoiseError(f"в {input_dir} нет файлов m_NNN.transcript.json")
    return ids


def validate_input(meeting: str, input_dir: Path, maps: dict, world: dict) -> dict:
    meta = input_dir / f"{meeting}.meeting.json"
    if not meta.exists():
        raise NoiseError(f"{meeting}: нет {meta.name} рядом с транскриптом")
    mapping = maps.get(meeting)
    if not mapping:
        raise NoiseError(f"{meeting}: нет соответствия меток в speaker_maps.json")
    people = {p["id"] for p in world.get("people") or []}
    for label, person in mapping.items():
        if person not in people:
            raise NoiseError(f"{meeting}: {label} → {person!r} нет в world.yaml")
    raw = json.loads((input_dir / f"{meeting}.transcript.json").read_text(encoding="utf-8"))
    segments = raw.get("segments")
    if not isinstance(segments, list) or not segments:
        raise NoiseError(f"{meeting}: нет сегментов")
    for i, s in enumerate(segments):
        if not str(s.get("text", "")).strip():
            raise NoiseError(f"{meeting}: пустая реплика в сегменте {i}")
        if s.get("speaker") not in mapping:
            raise NoiseError(f"{meeting}: сегмент {i}: метка {s.get('speaker')!r} нет "
                             "в speaker_maps.json")
    return raw


def run(a: argparse.Namespace) -> int:
    cfg = load_config(a.config)
    seed = cfg["global_seed"] if a.seed is None else a.seed
    world = yaml.safe_load(a.world.read_text(encoding="utf-8"))
    index = build_entity_index(world)
    maps = json.loads(a.speaker_maps.read_text(encoding="utf-8"))
    meetings = discover(a.input_dir)
    raws = {m: validate_input(m, a.input_dir, maps, world) for m in meetings}

    results: dict[tuple[str, str], dict[str, Combo]] = {}
    for meeting in meetings:
        for level in LEVELS:
            for variant in VARIANTS:
                results.setdefault((level, variant), {})[meeting] = process(
                    meeting, level, variant, raws[meeting], cfg, index, maps[meeting], seed)

    for (level, variant), per_meeting in results.items():
        name = f"{level}_{variant}"
        for meeting, combo in per_meeting.items():
            out_dir, log_dir = a.out / name, a.logs / name
            out_dir.mkdir(parents=True, exist_ok=True)
            log_dir.mkdir(parents=True, exist_ok=True)
            if level == "clean" and variant == "labeled":
                data = (a.input_dir / f"{meeting}.transcript.json").read_bytes()
            else:
                data = dump_json(combo.transcript)
            (out_dir / f"{meeting}.transcript.json").write_bytes(data)
            (out_dir / f"{meeting}.meeting.json").write_bytes(
                (a.input_dir / f"{meeting}.meeting.json").read_bytes())
            (log_dir / f"{meeting}.json").write_bytes(dump_json(combo.log))
            if a.text and combo.txt is not None:
                (out_dir / f"{meeting}.txt").write_bytes(combo.txt.encode("utf-8"))

    fillers = set(cfg["fillers"])
    print(f"Файлов записано: {sum(len(v) for v in results.values())} × (transcript, meeting, журнал"
          f"{', txt' if a.text else ''}); зерно {seed}\n")
    totals: dict[str, dict] = {}
    for (level, variant), per_meeting in results.items():
        name = f"{level}_{variant}"
        print(f"== {name} ==")
        all_texts: list[str] = []
        ops: Counter = Counter()
        for meeting, combo in per_meeting.items():
            print(stats_line(meeting, stats(combo.texts, fillers)))
            all_texts += combo.texts
            ops.update(combo.counters)
        total = stats(all_texts, fillers)
        totals[name] = total
        print(stats_line("ИТОГО", total))
        print("  замены: " + (", ".join(f"{k}={v}" for k, v in sorted(ops.items())) or "нет"))
        bounds: Counter = Counter()
        for combo in per_meeting.values():
            bounds.update(combo.log.get("block_boundaries", {}))
        if bounds:
            print(f"  границы блоков: по концу предложения {bounds['sentence_end']}, "
                  f"по концу реплики {bounds['reply_end']}, посреди предложения "
                  f"{bounds['mid_sentence']}; привязано {bounds['snapped']} из {total['segments']}")
        print()
    print("== medium_unlabeled против настоящего образца "
          "(boundary_snap_prob выбран по этому образцу, остальное не подгонялось) ==")
    t = totals["medium_unlabeled"]
    rows = [
        ("блок: медиана, слов", REAL_SAMPLE["len"][1], t["len"][1]),
        ("блок: p90, слов", REAL_SAMPLE["len"][2], t["len"][2]),
        ("блок: max, слов", REAL_SAMPLE["len"][3], t["len"][3]),
        ("без . ? ! … в конце, %", REAL_SAMPLE["no_end_pct"], t["no_end_pct"]),
        ("со строчной буквы, %", REAL_SAMPLE["lower_pct"], t["lower_pct"]),
        ("междометий на 1000 слов", REAL_SAMPLE["fillers_per_1000"], t["fillers_per_1000"]),
        ("многоточий на 100 слов", REAL_SAMPLE["ellipsis_per_100"], t["ellipsis_per_100"]),
        ("запятых на 100 слов", REAL_SAMPLE["commas_per_100"], t["commas_per_100"]),
        ("блоков с [шум], %", REAL_SAMPLE["tag_pct"], t["tag_pct"]),
    ]
    print(f"  {'метрика':<28}{'образец':>10}{'у нас':>10}")
    for name, real, ours in rows:
        print(f"  {name:<28}{real:>10.2f}{ours:>10.2f}")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("input_dir", nargs="?", type=Path, default=INPUT_DIR)
    ap.add_argument("--out", type=Path, default=OUT_DIR)
    ap.add_argument("--logs", type=Path, default=LOG_DIR)
    ap.add_argument("--config", type=Path, default=CONFIG_PATH)
    ap.add_argument("--world", type=Path, default=WORLD_PATH)
    ap.add_argument("--speaker-maps", type=Path, default=SPEAKER_MAPS_PATH)
    ap.add_argument("--seed", type=int, default=None, help="по умолчанию global_seed из конфига")
    ap.add_argument("--text", action="store_true",
                    help="для unlabeled писать ещё m_NNN.txt в формате настоящей расшифровки")
    a = ap.parse_args(argv)
    try:
        return run(a)
    except NoiseError as e:
        print(f"ОШИБКА: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
