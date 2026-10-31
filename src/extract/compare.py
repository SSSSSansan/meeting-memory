"""Сверка извлечения модели с ручным эталоном.

    python -m src.extract.compare                    # data/extracted_llm vs data/extracted
    python -m src.extract.compare data/extracted_llm data/extracted

Пункты считаются совпавшими, если их цитаты пересекаются по словам (Jaccard >= 0.5)
или одна цитата содержит другую. Считается отдельно для решений, задач и вопросов.
Выводит, что модель пропустила и что добавила лишнего, — по этому правят промпт.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

from src.schemas import Extraction

ROOT = Path(__file__).resolve().parent.parent.parent


def _words(s: str) -> set[str]:
    return set(re.findall(r"\w+", s.lower()))


def _match(a: str, b: str) -> bool:
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return False
    if a.lower() in b.lower() or b.lower() in a.lower():
        return True
    return len(wa & wb) / len(wa | wb) >= 0.5


def compare(pred: Extraction, gold: Extraction) -> dict:
    result = {}
    for kind in ("decisions", "tasks", "open_questions"):
        p_items, g_items = list(getattr(pred, kind)), list(getattr(gold, kind))
        used, matched = set(), 0
        missed = []
        for g in g_items:
            hit = next((i for i, p in enumerate(p_items)
                        if i not in used and p.project == g.project and _match(p.source.quote, g.source.quote)), None)
            if hit is None:
                missed.append(g)
            else:
                used.add(hit)
                matched += 1
        extra = [p for i, p in enumerate(p_items) if i not in used]
        result[kind] = dict(matched=matched, gold=len(g_items), pred=len(p_items),
                            missed=missed, extra=extra)
    return result


def main() -> int:
    pred_dir = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "data/extracted_llm")
    gold_dir = ROOT / (sys.argv[2] if len(sys.argv) > 2 else "data/extracted")
    tot = {k: [0, 0, 0] for k in ("decisions", "tasks", "open_questions")}
    for gf in sorted(gold_dir.glob("*.json")):
        pf = pred_dir / gf.name
        if not pf.exists():
            print(f"{gf.name}: нет результата модели")
            continue
        gold = Extraction.model_validate_json(gf.read_text(encoding="utf-8"))
        pred = Extraction.model_validate_json(pf.read_text(encoding="utf-8"))
        print(f"\n=== {gf.stem} ===")
        for kind, r in compare(pred, gold).items():
            tot[kind][0] += r["matched"]; tot[kind][1] += r["gold"]; tot[kind][2] += r["pred"]
            print(f"{kind}: совпало {r['matched']} из {r['gold']} эталонных, модель выдала {r['pred']}")
            for m in r["missed"]:
                print(f"   ПРОПУЩЕНО: [{m.project}] {m.statement}")
            for e in r["extra"]:
                print(f"   ЛИШНЕЕ:    [{e.project}] {e.statement}")
    print("\n=== ИТОГО (precision / recall) ===")
    for kind, (m, g, p) in tot.items():
        prec = m / p if p else 0.0
        rec = m / g if g else 0.0
        print(f"{kind}: precision={prec:.2f} recall={rec:.2f}  (совпало {m}, эталон {g}, модель {p})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
