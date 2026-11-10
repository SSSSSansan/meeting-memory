"""Сверка извлечения с gold Даны (data/gold/decisions.json) по близости цитат.

    python scripts/compare_gold.py [data/extracted_llm_raw]
Совпадение: цитаты близки по словам ИЛИ таймкод в пределах WINDOW_S.
Gold-события decision/task/open_question считаем эталоном; recall — доля найденных,
precision — доля выдачи модели, попавшая в gold (лишнее может быть настоящим решением, которого нет в gold).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.extract.compare import _match  # noqa: E402

WINDOW_S = 20  # цитата модели в пределах 20 с от эталонной реплики считается тем же событием


def _sec(ts):
    h, m, sec = map(int, ts.split(":"))
    return h * 3600 + m * 60 + sec


def load_gold() -> list[dict]:
    return [g for g in json.loads((ROOT / "data/gold/decisions.json").read_text(encoding="utf-8"))
            if g["type"] in ("decision", "task", "open_question")]


def load_preds(meeting_id: str, pred_dir: Path) -> list[tuple[str, str]]:
    f = pred_dir / f"{meeting_id}.json"
    if not f.exists():
        return []
    d = json.loads(f.read_text(encoding="utf-8"))
    return [(i["source"]["quote"], i["source"]["ts"])
            for k in ("decisions", "tasks", "open_questions") for i in d.get(k, [])]


def is_match(pred: tuple[str, str], gold_item: dict) -> bool:
    p, ts = pred
    return _match(p, gold_item["quote"]) or abs(_sec(ts) - gold_item["start_s"]) <= WINDOW_S


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    pred_dir = ROOT / (argv[0] if argv else "data/extracted_llm_raw")
    gold = load_gold()
    hit = n_pred = n_gold = 0
    for mid in sorted({g["meeting"] for g in gold}):
        preds = load_preds(mid, pred_dir)
        gs = [g for g in gold if g["meeting"] == mid]
        found = [g for g in gs if any(is_match(p, g) for p in preds)]
        print(f"{mid}: gold={len(gs)} найдено={len(found)} выдано={len(preds)}")
        for g in gs:
            if g not in found:
                print("   пропущено:", g["type"], "|", g["quote"][:70])
        hit += len(found)
        n_gold += len(gs)
        n_pred += len(preds)
    print(f"recall={hit}/{n_gold}  выдано моделью={n_pred}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
