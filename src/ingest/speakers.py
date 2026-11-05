"""B9. SPEAKER_00 → имя, когда Scribal не дал speaker_names.

Эвристика обращений: если в реплике говорящего A звучит имя участника N, то следующая
реплика другого говорящего B, скорее всего, принадлежит N (а A — не N). Имя назначается
только при достаточном числе голосов и перевесе; иначе остаётся SPEAKER_XX, имя не выдумываем.

    python -m src.ingest.speakers      # точность против data/gold/speaker_maps.json
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from src.schemas import Transcript

ROOT = Path(__file__).resolve().parent.parent.parent
_FOLD = str.maketrans("ұүқғңәөіһё", "уукгнаоихе")
MIN_VOTES, MIN_SHARE = 2, 0.6


def fold(s: str) -> str:
    return s.lower().translate(_FOLD)


def _stem(name: str) -> str:
    return fold(name.split()[0])[:4]


def _mentions(text: str, stems: dict[str, str]) -> set[str]:
    words = re.findall(r"\w+", fold(text))
    return {n for n, st in stems.items() if any(w.startswith(st) for w in words)}


def attribute_speakers(t: Transcript) -> dict[str, str]:
    """{SPEAKER_XX: имя} только для уверенных случаев."""
    stems = {p: _stem(p) for p in t.participants}
    votes: dict[str, Counter] = defaultdict(Counter)
    segs = t.segments
    for a, b in zip(segs, segs[1:]):
        la, lb = a.speaker_raw or a.speaker, b.speaker_raw or b.speaker
        if la == lb:
            continue
        for n in _mentions(a.text, stems):
            votes[lb][n] += 1
            votes[la][n] -= 1  # говорящий не обращается к себе
    result: dict[str, str] = {}
    cand = []
    for label, c in votes.items():
        pos = {n: v for n, v in c.items() if v > 0}
        total = sum(pos.values())
        if not total:
            continue
        name, v = max(pos.items(), key=lambda kv: kv[1])
        if v >= MIN_VOTES and v / total >= MIN_SHARE:
            cand.append((v, label, name))
    for v, label, name in sorted(cand, reverse=True):
        if name not in result.values():
            result[label] = name
    return result


def apply_speakers(t: Transcript) -> Transcript:
    """Подставляет имена вместо SPEAKER_XX там, где их не дал Scribal; speaker_raw сохраняется."""
    mapping = attribute_speakers(t)
    for s in t.segments:
        raw = s.speaker_raw or s.speaker
        if s.speaker == raw and raw in mapping:
            s.speaker = mapping[raw]
    return t


def evaluate() -> None:
    from src.ingest.scribal_adapter import convert_files
    gold = json.loads((ROOT / "data/gold/speaker_maps.json").read_text(encoding="utf-8"))
    world = (ROOT / "data/gold/world.yaml").read_text(encoding="utf-8")
    ids = dict(re.findall(r"id: (\w+), name: ([^,]+),", world))
    ok = wrong = unknown = 0
    for mid, gmap in gold.items():
        f = ROOT / "data/raw" / f"{mid}.transcript.json"
        if not f.exists():
            continue
        t = convert_files(f, ROOT / "data/raw" / f"{mid}.meeting.json", mid)
        got = attribute_speakers(t)
        for label, gid in gmap.items():
            if label not in got:
                unknown += 1
            elif fold(got[label]) == fold(ids.get(gid, gid)):
                ok += 1
            else:
                wrong += 1
        print(mid, {k: got.get(k, "?") for k in gmap})
    print(f"верно={ok} неверно={wrong} не определено={unknown}")


if __name__ == "__main__":
    evaluate()
