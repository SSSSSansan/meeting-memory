"""B5. Минимальные связи между решениями: кандидаты (тот же проект, другая встреча) → классификация (промпт 11.2).

    python -m src.graph.link                      # все проекты, кандидаты по умолчанию
    python -m src.graph.link --project Zhol        # только один проект
    python -m src.graph.link --extra d_m007_01:d_m008_02   # добавить межпроектную пару вручную

Кандидаты — не эмбеддинг-поиск (это полная версия B5, не делали из-за бюджета/времени): на 10 встречах
проще и дешевле перебрать все пары решений одного проекта из разных встреч. --extra позволяет добавить
пары из разных проектов, когда кандидат известен вручную (например, конфликт загрузки одного человека).
Результат — data/graph/links.json (confidence >= 0.6) и data/graph/links_rejected.json (остальное).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.ingest.scribal_adapter import convert_files  # noqa: E402
from src.llm_client import complete  # noqa: E402
from src.schemas import Extraction  # noqa: E402

LINK_TYPES = ("supersedes", "contradicts", "depends_on", "refines", "same_topic", "none")
THRESHOLD = 0.6

SYSTEM_PROMPT = """Два решения из разных встреч одного или смежных проектов.
Определи отношение между ними.

Типы:
- supersedes: A принимает тот же вопрос заново и меняет содержание B (включая отмену B без замены)
- contradicts: A и B оба действуют, но несовместимы друг с другом
- depends_on: A выполнимо только при условии B
- refines: A уточняет детали B, не отменяя его
- same_topic: про одно и то же, но логической связи нет
- none: связи нет

Верни JSON: {"type": "...", "confidence": 0.0-1.0, "rationale": "одно предложение"}

Будь строг. none и same_topic — нормальные ответы.
Ставь supersedes только если это буквально тот же вопрос, а не похожая тема."""


def user_prompt(a: dict, b: dict) -> str:
    return (f'РЕШЕНИЕ A (встреча {a["date"]}, проект {a["project"]}):\n{a["statement"]}\n\n'
            f'РЕШЕНИЕ B (встреча {b["date"]}, проект {b["project"]}, более раннее):\n{b["statement"]}')


def _load_decisions(extracted_dir: Path, raw_dir: Path) -> list[dict]:
    """Решения всех встреч с датой и meeting_id, отсортированные по дате (для кандидатов и для A/B в промпте)."""
    out = []
    for f in sorted(extracted_dir.glob("*.json")):
        ext = Extraction.model_validate_json(f.read_text(encoding="utf-8"))
        tr_path = raw_dir / f"{ext.meeting_id}.transcript.json"
        meeting_path = raw_dir / f"{ext.meeting_id}.meeting.json"
        date = convert_files(tr_path, meeting_path, ext.meeting_id).date if tr_path.exists() else ""
        for d in ext.decisions:
            out.append({"id": d.id, "meeting_id": ext.meeting_id, "date": date,
                        "project": d.project, "statement": d.statement})
    return sorted(out, key=lambda x: (x["date"], x["meeting_id"]))


def candidates(decisions: list[dict], project: str | None = None) -> list[tuple[dict, dict]]:
    """Пары (A позже, B раньше) одного проекта из разных встреч, отсортированные по дате."""
    pairs = []
    for i, a in enumerate(decisions):
        for b in decisions[:i]:
            if a["meeting_id"] == b["meeting_id"]:
                continue
            if project and a["project"] != project:
                continue
            if a["project"] != b["project"]:
                continue
            pairs.append((a, b))
    return pairs


def classify(a: dict, b: dict) -> dict:
    messages = [{"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt(a, b)}]
    r = complete(messages, json_schema={"type": "object"}, temperature=0.0, max_tokens=200, purpose="link")
    text = r.text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    data = json.loads(text)
    if data.get("type") not in LINK_TYPES:
        data["type"] = "none"
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--extracted", default="data/extracted_llm_raw")
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--project", default=None)
    ap.add_argument("--extra", action="append", default=[], help="ID_A:ID_B — пара решений вручную (любые проекты)")
    ap.add_argument("--out-dir", default="data/graph")
    a = ap.parse_args()

    decisions = _load_decisions(ROOT / a.extracted, ROOT / a.raw)
    by_id = {d["id"]: d for d in decisions}
    pairs = candidates(decisions, project=a.project)
    for spec in a.extra:
        id_a, id_b = spec.split(":")
        pairs.append((by_id[id_a], by_id[id_b]))

    out_dir = ROOT / a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    links_path, rejected_path = out_dir / "links.json", out_dir / "links_rejected.json"
    # дополняем существующие файлы, а не перезатираем — повторный запуск с другим --project
    # (или --extra) не должен стирать связи, найденные в предыдущих запусках
    links = {(r["from"], r["to"]): r for r in json.loads(links_path.read_text(encoding="utf-8"))} \
        if links_path.exists() else {}
    rejected = {(r["from"], r["to"]): r for r in json.loads(rejected_path.read_text(encoding="utf-8"))} \
        if rejected_path.exists() else {}

    for da, db in pairs:
        res = classify(da, db)
        rec = {"from": da["id"], "to": db["id"], "type": res["type"],
               "confidence": res["confidence"], "rationale": res["rationale"]}
        key = (da["id"], db["id"])
        if res["type"] != "none" and res["confidence"] >= THRESHOLD:
            links[key] = rec
            rejected.pop(key, None)
        else:
            rejected[key] = rec
            links.pop(key, None)
        print(f'{da["id"]} -> {db["id"]}: {res["type"]} ({res["confidence"]:.2f}) {res["rationale"]}')

    links_path.write_text(json.dumps(list(links.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    rejected_path.write_text(json.dumps(list(rejected.values()), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nПар проверено: {len(pairs)}. links.json: {len(links)}, links_rejected.json: {len(rejected)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
