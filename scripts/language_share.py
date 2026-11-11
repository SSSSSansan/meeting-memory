"""РОЛЬ A. Доля казахской речи в наборе по чистым транскриптам data/raw. Модель не вызывается.

    python scripts/language_share.py
    python scripts/language_share.py --threshold 0.10 --out results/language_share.md

Метод (оценка, не разметка): реплика = сегмент транскрипта.
  1. Реплика «с казахскими буквами», если в ней есть хотя бы одна из ә ғ қ ң ө ұ ү һ і.
  2. Слово считается казахским, если в нём есть такая буква или оно входит в список частых
     казахских слов без особых букв (KK_WORDS: «ол», «керек», «деп», «бар», ...). Общие для двух
     языков слова («да», «не», «на») в список не входят. Слова — последовательности букв, регистр
     и NFC сняты; цифры не считаются. Знаменатель — все слова, включая латиницу (англ. термины).
  Имена с казахскими буквами (Нұралы, Сандуғаш, Қамқор) языком не считаются и из оценки исключены.
  Оценка занижена для казахских слов, совпадающих с русскими или не попавших в список, и
  завышена, если «і» встретится в чужом тексте; для порога 10% этого достаточно.
Встреча «содержит казахскую речь», если доля реплик с казахскими буквами или доля казахских слов
не меньше порога (по умолчанию 10%). Набор встреч и профили ru/mixed/kk берутся из data/gold/world.yaml.
Результат: results/language_share.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RAW_DIR = ROOT / "data" / "raw"
DEFAULT_WORLD = ROOT / "data" / "gold" / "world.yaml"
DEFAULT_OUT = ROOT / "results" / "language_share.md"

KK_LETTERS = frozenset("әғқңөұүһі")
THRESHOLD = 0.10
TARGET = (0.25, 0.30)
SENSITIVITY = (0.10, 0.20, 0.30)
PROFILES = ("ru", "mixed", "kk")

# Частые казахские слова без особых букв (выбраны по встречам m_002, m_005, m_007, m_012).
KK_WORDS = frozenset("""
ол олар оны мен сен сол солай осы содан сосын ал бар деп де екен емес керек болады болмайды болды
болса болып жатыр жатырмыз алады алып жасап жасау жазып жалпы тек жауап кейде кеше мысалы адам тез
жаман ауыр жел барып отырып отырмын келгенде дайын ойланып дегендей жарайды сау рахмет ма па пе ба ше
""".split())

# Имена людей и проектов с казахскими буквами: встречаются и в русской речи, языка не показывают.
NAME_STEMS = ("нұралы", "сандуғаш", "қамқор")

WORD_RE = re.compile(r"[^\W\d_]+")


def split_words(text: str) -> list[str]:
    return WORD_RE.findall(unicodedata.normalize("NFC", text).lower())


def is_name(word: str) -> bool:
    return word.startswith(NAME_STEMS)


def has_kk_letters(text: str) -> bool:
    return any(KK_LETTERS & set(w) for w in split_words(text) if not is_name(w))


def is_kk_word(word: str) -> bool:
    if is_name(word):
        return False
    return bool(KK_LETTERS & set(word)) or word in KK_WORDS


def share(part: int, total: int) -> float:
    return part / total if total else 0.0


def analyze_texts(texts: list[str]) -> dict:
    """Счётчики по списку реплик: реплик, с казахскими буквами, слов, казахских слов."""
    utt = len(texts)
    utt_kk = sum(has_kk_letters(t) for t in texts)
    words = [w for t in texts for w in split_words(t)]
    words_kk = sum(is_kk_word(w) for w in words)
    return {"utterances": utt, "utt_kk": utt_kk, "words": len(words), "words_kk": words_kk}


def add_counts(a: dict, b: dict) -> dict:
    return {k: a[k] + b[k] for k in a}


def with_shares(c: dict) -> dict:
    return {**c, "utt_share": share(c["utt_kk"], c["utterances"]), "word_share": share(c["words_kk"], c["words"])}


def has_kazakh(c: dict, threshold: float) -> bool:
    # Сравнение целыми числами: 8 реплик из 80 ровно 10%, без ошибок float.
    return (c["utt_kk"] >= threshold * c["utterances"] - 1e-9 and c["utt_kk"] > 0) or \
           (c["words_kk"] >= threshold * c["words"] - 1e-9 and c["words_kk"] > 0)


def load_world(path: Path) -> list[dict]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))["meetings"]


def load_meeting_texts(raw_dir: Path, meeting_id: str) -> list[str]:
    path = raw_dir / f"{meeting_id}.transcript.json"
    if not path.exists():
        raise FileNotFoundError(f"нет транскрипта {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    return [s.get("text", "") for s in data["segments"] if s.get("text", "").strip()]


def collect(raw_dir: Path, meetings: list[dict]) -> list[dict]:
    rows = []
    for m in meetings:
        c = analyze_texts(load_meeting_texts(raw_dir, m["id"]))
        rows.append({"meeting": m["id"], "profile": m["language_profile"], **c})
    return rows


def pct(x: float) -> str:
    return f"{x * 100:.1f}%".replace(".", ",")


def render(rows: list[dict], threshold: float = THRESHOLD) -> str:
    total = {k: 0 for k in ("utterances", "utt_kk", "words", "words_kk")}
    for r in rows:
        total = add_counts(total, r)
    n = len(rows)
    flags = {r["meeting"]: has_kazakh(r, threshold) for r in rows}
    n_kk = sum(flags.values())
    out = ["# Доля казахской речи в наборе", "",
           f"Источник: чистые транскрипты `data/raw` (по сегментам), {n} встреч из `world.yaml`. "
           "Скрипт: `scripts/language_share.py`.",
           "", "## Метод", "",
           "Оценка по буквам и частым словам, не разметка. Реплика = сегмент транскрипта.",
           "- **Реплика с казахскими буквами** — есть хотя бы одна из `ә ғ қ ң ө ұ ү һ і` "
           "(кроме имён Нұралы, Сандуғаш, Қамқор).",
           "- **Казахское слово** — в слове есть такая буква или оно в списке частых казахских слов без особых букв "
           f"({len(KK_WORDS)} слов: «ол», «керек», «деп», «бар», «сосын» и т. п.); "
           "общие для ru и kk слова в список не входят. "
           "Знаменатель — все слова (латиница и русские тоже).",
           f"- **Встреча содержит казахскую речь**, если доля реплик с казахскими буквами **или** доля казахских слов "
           f"не меньше {pct(threshold)}.",
           "- Оценка занижена для казахских слов вне списка и русско-казахских омонимов.", "",
           "## По встречам", "",
           "| Встреча | Профиль | Реплик | С казах. буквами | Доля реплик | Слов | Казах. слов | Доля слов "
           "| Казахская речь |",
           "|---|---|---:|---:|---:|---:|---:|---:|:---:|"]
    for r in rows:
        s = with_shares(r)
        yes = "да" if flags[r["meeting"]] else "нет"
        out.append(f"| {r['meeting']} | {r['profile']} | {r['utterances']} | {r['utt_kk']} | {pct(s['utt_share'])} | "
                   f"{r['words']} | {r['words_kk']} | {pct(s['word_share'])} | {yes} |")
    t = with_shares(total)
    out.append(f"| **Набор** | — | {total['utterances']} | {total['utt_kk']} | {pct(t['utt_share'])} | "
               f"{total['words']} | {total['words_kk']} | {pct(t['word_share'])} | {n_kk} из {n} |")

    out += ["", "## По профилям из world.yaml", "",
            "| Профиль | Встреч | Реплик | Доля реплик с казах. буквами | Слов | Доля казах. слов "
            "| Встреч с казахской речью |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for p in PROFILES:
        sub = [r for r in rows if r["profile"] == p]
        c = {k: sum(r[k] for r in sub) for k in total}
        s = with_shares(c)
        out.append(f"| {p} | {len(sub)} | {c['utterances']} | {pct(s['utt_share'])} | {c['words']} | "
                   f"{pct(s['word_share'])} | {sum(flags[r['meeting']] for r in sub)} из {len(sub)} |")

    lo, hi = TARGET
    verdict = "в пределах цели" if lo <= share(n_kk, n) <= hi else "вне цели"
    out += ["", "## Цель задания: 25–30% встреч с казахским", "",
            f"- При пороге {pct(threshold)}: казахская речь есть в **{n_kk} из {n}** встреч = "
            f"**{pct(share(n_kk, n))}** (цель {int(lo * 100)}–{int(hi * 100)}%; {verdict}).",
            f"- Встречи: {', '.join(m for m, f in flags.items() if f) or '—'}.",
            "", "Чувствительность к порогу (доля реплик с казах. буквами или казах. слов не меньше порога):", "",
            "| Порог | Встреч | Доля |", "|---:|---:|---:|"]
    for th in SENSITIVITY:
        k = sum(has_kazakh(r, th) for r in rows)
        out.append(f"| {pct(th)} | {k} из {n} | {pct(share(k, n))} |")
    by_profile = sum(r["profile"] in ("mixed", "kk") for r in rows)
    out += ["", f"По заявленным профилям (`mixed` или `kk`): {by_profile} из {n} = {pct(share(by_profile, n))}.", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    ap.add_argument("--world", type=Path, default=DEFAULT_WORLD)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--threshold", type=float, default=THRESHOLD)
    args = ap.parse_args(argv)
    try:
        rows = collect(args.raw_dir, load_world(args.world))
    except FileNotFoundError as e:
        print(f"ошибка: {e}", file=sys.stderr)
        return 1
    text = render(rows, args.threshold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    n_kk = sum(has_kazakh(r, args.threshold) for r in rows)
    print(f"встреч с казахской речью (порог {pct(args.threshold)}): "
          f"{n_kk} из {len(rows)} = {pct(share(n_kk, len(rows)))}")
    print(f"записано: {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
