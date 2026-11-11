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


KIND_OF_TYPE = {"decision": "decisions", "task": "tasks", "open_question": "open_questions"}


def load_preds(meeting_id: str, pred_dir: Path, kind: str | None = None) -> list[tuple[str, str]]:
    f = pred_dir / f"{meeting_id}.json"
    if not f.exists():
        return []
    d = json.loads(f.read_text(encoding="utf-8"))
    kinds = (kind,) if kind else ("decisions", "tasks", "open_questions")
    return [(i["source"]["quote"], i["source"]["ts"]) for k in kinds for i in d.get(k, [])]


def is_match(pred: tuple[str, str], gold_item: dict) -> bool:
    p, ts = pred
    return _match(p, gold_item["quote"]) or abs(_sec(ts) - gold_item["start_s"]) <= WINDOW_S


def match_meeting(preds: list[tuple[str, str]], gs: list[dict]) -> tuple[set[int], set[int]]:
    """Жадное сопоставление пунктов одной встречи. Две фазы, а не один проход с первым попавшимся:
    в фазе 1 совпадение по ТЕКСТУ (надёжное) забирает пару раньше, чем до неё доберётся совпадение
    только по таймкоду (WINDOW_S=20с иногда ловит СОСЕДНЮЮ, а не ту же реплику — так d_m015_01
    («срок остаётся») мог бы по таймкоду перехватить пару у d_m015_02 («фичи не добавляем»), которая
    на самом деле совпадает текстом; порядок фаз это не даёт сделать.
    Возвращает (индексы matched gold, индексы matched preds)."""
    used_pred: set[int] = set()
    matched_gold: set[int] = set()
    for g_i, g in enumerate(gs):
        hit = next((p_i for p_i, p in enumerate(preds) if p_i not in used_pred and _match(p[0], g["quote"])), None)
        if hit is not None:
            used_pred.add(hit)
            matched_gold.add(g_i)
    for g_i, g in enumerate(gs):
        if g_i in matched_gold:
            continue
        hit = next((p_i for p_i, p in enumerate(preds)
                    if p_i not in used_pred and abs(_sec(p[1]) - g["start_s"]) <= WINDOW_S), None)
        if hit is not None:
            used_pred.add(hit)
            matched_gold.add(g_i)
    return matched_gold, used_pred


def _prf(matched_gold: int, n_gold: int, matched_pred: int, n_pred: int) -> tuple[float, float, float]:
    r = matched_gold / n_gold if n_gold else 0.0
    p = matched_pred / n_pred if n_pred else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    pred_dir = ROOT / (argv[0] if argv else "data/extracted_llm_raw")
    gold = load_gold()
    meetings = sorted({g["meeting"] for g in gold})

    hit = n_pred = n_gold = 0
    for mid in meetings:
        preds = load_preds(mid, pred_dir)
        gs = [g for g in gold if g["meeting"] == mid]
        matched_gold, _ = match_meeting(preds, gs)
        print(f"{mid}: gold={len(gs)} найдено={len(matched_gold)} выдано={len(preds)}")
        for g_i, g in enumerate(gs):
            if g_i not in matched_gold:
                print("   пропущено:", g["type"], "|", g["quote"][:70])
        hit += len(matched_gold)
        n_gold += len(gs)
        n_pred += len(preds)
    print(f"recall={hit}/{n_gold}  выдано моделью={n_pred}")

    # P/R/F1 по типу: совпадение только в пределах своего типа (decision с decision и т.д.),
    # иначе recall мог бы засчитываться по цитате из чужой категории.
    print("\n=== P/R/F1 по типу (gold decisions.json) ===")
    totals = []
    for type_, kind in KIND_OF_TYPE.items():
        gt = [g for g in gold if g["type"] == type_]
        if not gt:
            continue
        matched_gold_n = matched_pred_n = total_pred = 0
        for mid in sorted({g["meeting"] for g in gt}):
            preds = load_preds(mid, pred_dir, kind)
            gs = [g for g in gt if g["meeting"] == mid]
            mg, mp = match_meeting(preds, gs)
            matched_gold_n += len(mg)
            matched_pred_n += len(mp)
            total_pred += len(preds)
        p, r, f1 = _prf(matched_gold_n, len(gt), matched_pred_n, total_pred)
        totals.append((type_, len(gt), total_pred, p, r, f1))
        print(f"{type_}: gold={len(gt)} выдано={total_pred} precision={p:.2f} recall={r:.2f} f1={f1:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
