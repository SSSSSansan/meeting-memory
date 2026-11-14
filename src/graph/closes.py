"""РОЛЬ B. Связь CLOSES: более позднее высказывание закрывает ранее выданную задачу или отвечает на открытый вопрос.

    python -m src.graph.closes --dry-run              # кандидаты, размер запросов и оценка стоимости, без модели
    python -m src.graph.closes --limit 5              # пять платных примеров (нужен LLM_BACKEND=openai и согласие)
    python -m src.graph.closes                        # все открытые задачи и вопросы
    python -m src.graph.closes --apply                # дописать рёбра CLOSES в data/graph/graph.json

Извлечение не перезапускается: берутся уже извлечённые открытые задачи и вопросы графа и транскрипты data/raw
(вход системы). Для каждой: (1) лексически отбираем до CANDIDATES реплик из ПОЗДНЕЙШИХ встреч того же проекта,
(2) одним запросом спрашиваем модель, сообщается ли в одной из них, что задача выполнена или на вопрос дан ответ,
(3) цитата обязана быть дословной подстрокой реплики кандидата, иначе запись отбрасывается.

Результат: data/graph/closes.json (confidence >= 0.6) и data/graph/closes_rejected.json (с причиной).
Запись: {"from": id пункта-закрытия, "to": id задачи, "type": "closes", "confidence", "rationale", "quote",
"meeting_id", "ts", "date"}. Если в той же встрече рядом по времени уже есть извлечённый пункт (например, «Миграция
БД выполнена»), он и становится источником ребра; иначе создаётся Task со статусом done: `t_<встреча>_c<NN>`.

Эталон data/gold/closures.json сюда не подаётся: ни в промпт, ни в отбор. Его читает только scripts/eval_closes.py,
чтобы посчитать точность и полноту.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

import networkx as nx

from src.graph.status import _keywords, _project_meetings
from src.graph.traverse import RAW_DIR, _load_meeting, _same_stem, _ts_to_s, _words
from src.utils.text import normalize

ROOT = Path(__file__).resolve().parent.parent.parent
CLOSES_CONFIDENCE = 0.6        # как у связей между решениями (LINK_CONFIDENCE_THRESHOLD)
CANDIDATES = 5                 # реплик-кандидатов на одну задачу
MAX_PER_MEETING = 2            # и не больше двух с одной встречи: иначе одна длинная встреча вытесняет остальные
NEAR_ITEM_S = 20               # пункт той же встречи ближе этого к реплике закрытия считается её источником
PROMPT_VERSION = "closes-v1"
_DONE_MARKERS = ("готов", "сделан", "сделал", "выполн", "отправл", "отправил", "закончил", "заверш", "закрыт",
                 "сдал", "подключил", "собрал", "принят", "отдал", "есть")

SYSTEM_PROMPT = """Ты проверяешь, закрыта ли задача (или получен ли ответ на открытый вопрос) на более поздних \
встречах. Тебе дают задачу или вопрос и пронумерованные реплики из более поздних встреч того же проекта.

Закрытие — это реплика, где сообщается, что работа УЖЕ сделана, отправлена, готова или принята, либо что на вопрос \
УЖЕ дан ответ (названо, кто делает; принято решение).
Закрытием НЕ считается: планы и обещания («сделаю», «будет готово к пятнице»), напоминание о задаче, перенос срока, \
новая постановка той же задачи, обсуждение без итога, похожая, но другая задача.

Правила:
1. Выбери не больше одной реплики — ту, где закрытие сообщено явно. Если такой нет, closed=false.
2. quote — дословная подстрока именно строки «РЕПЛИКА» выбранного кандидата (не соседних строк «до» и «после»), \
от 3 до 25 слов, без исправлений и пересказа.
3. confidence от 0 до 1: насколько явно в реплике сказано, что дело сделано. Сомневаешься — ставь closed=false.
4. rationale — одно предложение: что именно сообщено.
Ответ — только JSON: {"closed": true|false, "candidate": номер или null, "quote": "..." или null, \
"confidence": число, "rationale": "..."}."""


def open_items(G: nx.MultiDiGraph) -> list[str]:
    """Цели прохода: задачи, которые извлечение считает открытыми, и открытые вопросы, ещё без входящего CLOSES."""
    out = []
    for n, d in sorted(G.nodes(data=True)):
        if d.get("type") == "Task" and d.get("status") == "open" or d.get("type") == "OpenQuestion":
            if not any(e["type"] == "CLOSES" for _, _, e in G.in_edges(n, data=True)):
                out.append(n)
    return out


def _segments(mid: str, raw_dir: Path):
    paths = (raw_dir / f"{mid}.transcript.json", raw_dir / f"{mid}.meeting.json")
    t = _load_meeting(mid, str(raw_dir), tuple(p.stat().st_mtime if p.exists() else 0 for p in paths))
    return t.segments if t is not None else []


def candidates(G: nx.MultiDiGraph, item: str, raw_dir: Optional[Path] = None, limit: int = CANDIDATES) -> list[dict]:
    """Реплики поздних встреч проекта, где есть основы ключевых слов задачи. Ранг: число совпавших ключевых слов,
    затем наличие слов «готово/сделано/отправлено», затем раньше. Каждому кандидату даны соседние реплики
    (до/после) как контекст: «готово» без них непонятно, о чём."""
    d = G.nodes[item]
    keys = _keywords(G, item)
    if not keys:
        return []
    base = Path(raw_dir or RAW_DIR)
    found = []
    for mdate, mid in _project_meetings(G, d.get("project", "")):
        if mdate <= d.get("date", ""):
            continue
        segs = _segments(mid, base)
        for i, seg in enumerate(segs):
            words = set(_words(seg.text))
            hits = sum(any(_same_stem(k, w) for w in words) for k in keys)
            if not hits:
                continue
            marker = any(w.startswith(_DONE_MARKERS) for w in re.findall(r"\w+", normalize(seg.text)))
            found.append({"meeting_id": mid, "date": mdate, "ts": seg.ts, "speaker": seg.speaker, "text": seg.text,
                          "before": segs[i - 1].text if i > 0 else "",
                          "after": segs[i + 1].text if i + 1 < len(segs) else "",
                          "score": hits + (0.5 if marker else 0.0)})
    found.sort(key=lambda c: (-c["score"], c["date"], c["ts"]))
    out, per_meeting = [], {}
    for c in found:
        if per_meeting.get(c["meeting_id"], 0) >= MAX_PER_MEETING:
            continue
        per_meeting[c["meeting_id"]] = per_meeting.get(c["meeting_id"], 0) + 1
        out.append(c)
        if len(out) == limit:
            break
    return out


def build_messages(G: nx.MultiDiGraph, item: str, cands: list[dict]) -> list[dict]:
    d = G.nodes[item]
    kind = "Открытый вопрос" if d["type"] == "OpenQuestion" else "Задача"
    head = [f"{kind}: «{d.get('statement', '')}»", f"Проект: {d.get('project', '—')}"]
    if d.get("owner"):
        head.append(f"Исполнитель: {d['owner']}")
    if d.get("due"):
        head.append(f"Срок: {d['due']}")
    head.append(f"Выдана на встрече {d['meeting_id']} ({d.get('date', '')}).")
    blocks = []
    for i, c in enumerate(cands, 1):
        blocks.append(f"[{i}] встреча {c['meeting_id']}, {c['date']}, {c['ts']}\n"
                      f"  до: {c['before']}\n  РЕПЛИКА: {c['speaker']}: {c['text']}\n  после: {c['after']}")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",
             "content": "\n".join(head) + "\n\nРеплики более поздних встреч:\n\n" + "\n\n".join(blocks)}]


def parse_answer(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.S)
    data = json.loads(text)
    if not isinstance(data, dict) or not isinstance(data.get("closed"), bool):
        raise ValueError('нужен объект с булевым полем "closed"')
    return data


def validate(answer: dict, cands: list[dict]) -> tuple[Optional[dict], str]:
    """(выбранный кандидат, причина отказа). Цитата — дословная подстрока реплики кандидата, иначе запись отброшена."""
    if not answer["closed"]:
        return None, "модель: не закрыто"
    idx = answer.get("candidate")
    if not isinstance(idx, int) or not 1 <= idx <= len(cands):
        return None, "нет номера кандидата"
    cand = cands[idx - 1]
    quote = (answer.get("quote") or "").strip()
    if not quote:
        return None, "нет цитаты"
    if normalize(quote) not in normalize(cand["text"]):
        return None, "цитата не подстрока реплики кандидата"
    try:
        conf = float(answer.get("confidence", 0))
    except (TypeError, ValueError):
        return None, "confidence не число"
    if conf < CLOSES_CONFIDENCE:
        return None, f"confidence {conf:.2f} < {CLOSES_CONFIDENCE}"
    return cand, ""


def closing_source(G: nx.MultiDiGraph, item: str, cand: dict) -> Optional[str]:
    """Уже извлечённый пункт той же встречи рядом по времени с репликой закрытия (не сама цель)."""
    centre = _ts_to_s(cand["ts"])
    near = [(abs(_ts_to_s(d["ts"]) - centre), n) for n, d in G.nodes(data=True)
            if d.get("type") in ("Decision", "Task", "OpenQuestion") and d.get("meeting_id") == cand["meeting_id"]
            and n != item and d.get("ts") and abs(_ts_to_s(d["ts"]) - centre) <= NEAR_ITEM_S]
    return min(near)[1] if near else None


def new_closing_id(G: nx.MultiDiGraph, meeting_id: str, taken: set[str]) -> str:
    short = meeting_id.replace("_", "")      # m_004 -> m004, как в остальных id
    k = 1
    while f"t_{short}_c{k:02d}" in G or f"t_{short}_c{k:02d}" in taken:
        k += 1
    return f"t_{short}_c{k:02d}"


def classify(G: nx.MultiDiGraph, item: str, raw_dir: Optional[Path] = None) -> dict:
    """Один запрос по одной задаче. Возвращает {"item", "cands", "answer", "record" | None, "reason", "usage"}."""
    from src.llm_client import complete

    cands = candidates(G, item, raw_dir)
    out = {"item": item, "cands": cands, "answer": None, "record": None, "reason": "нет кандидатов", "cost_usd": 0.0}
    if not cands:
        return out
    r = complete(build_messages(G, item, cands), json_schema={"type": "object"}, temperature=0.0, max_tokens=300,
                 purpose="closes")
    out["cost_usd"] = r.cost_usd
    try:
        out["answer"] = parse_answer(r.text)
    except (ValueError, json.JSONDecodeError) as e:
        out["reason"] = f"ответ не разобран: {e}"
        return out
    cand, reason = validate(out["answer"], cands)
    out["reason"] = reason
    if cand is not None:
        src = closing_source(G, item, cand)
        out["record"] = {"from": src, "to": item, "type": "closes",
                         "confidence": float(out["answer"]["confidence"]),
                         "rationale": out["answer"].get("rationale", ""), "quote": out["answer"]["quote"].strip(),
                         "meeting_id": cand["meeting_id"], "ts": cand["ts"], "date": cand["date"],
                         "speaker": cand["speaker"], "statement": cand["text"], "prompt_version": PROMPT_VERSION}
    return out


def estimate(G: nx.MultiDiGraph, items: list[str], raw_dir: Optional[Path] = None) -> dict:
    """Оценка стоимости без модели: размер запросов в символах → токены (≈ 2.5 символа на токен для русского) и
    цены из .env (PRICE_IN_PER_M / PRICE_OUT_PER_M). Выход — не больше 120 токенов JSON."""
    p_in, p_out = float(os.environ.get("PRICE_IN_PER_M", "0.15")), float(os.environ.get("PRICE_OUT_PER_M", "0.60"))
    rows, tok_in = [], 0
    for n in items:
        cands = candidates(G, n, raw_dir)
        if not cands:
            rows.append({"item": n, "cands": 0, "chars": 0, "tokens": 0})
            continue
        chars = sum(len(m["content"]) for m in build_messages(G, n, cands))
        rows.append({"item": n, "cands": len(cands), "chars": chars, "tokens": round(chars / 2.5)})
        tok_in += round(chars / 2.5)
    calls = sum(1 for r in rows if r["cands"])
    cost = (tok_in * p_in + calls * 120 * p_out) / 1_000_000
    return {"rows": rows, "calls": calls, "tokens_in": tok_in, "cost_usd": cost}


# ---------- файлы и граф ----------

def _load_json(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else []


def save_results(results: list[dict], G: nx.MultiDiGraph, out_dir: Path) -> tuple[int, int]:
    """Дописывает closes.json / closes_rejected.json: повторный запуск для той же задачи перезаписывает её запись,
    а не дублирует. Для закрытий без извлечённого пункта выдаёт id `t_<встреча>_cNN`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cpath, rpath = out_dir / "closes.json", out_dir / "closes_rejected.json"
    closes = {r["to"]: r for r in _load_json(cpath)}
    rejected = {r["to"]: r for r in _load_json(rpath)}
    for r in results:
        item = r["item"]
        closes.pop(item, None)
        rejected.pop(item, None)
        taken = {x["from"] for x in closes.values() if x["from"]}
        if r["record"] is not None:
            rec = dict(r["record"])
            if rec["from"] is None:
                rec["from"] = new_closing_id(G, rec["meeting_id"], taken)
                rec["new_node"] = True
                rec["origin"] = "closes"
            closes[item] = rec
        else:
            rejected[item] = {"to": item, "reason": r["reason"], "answer": r["answer"],
                              "candidates": [{k: c[k] for k in ("meeting_id", "ts", "text")} for c in r["cands"]]}
    cpath.write_text(json.dumps(sorted(closes.values(), key=lambda x: x["to"]), ensure_ascii=False, indent=2),
                     encoding="utf-8")
    rpath.write_text(json.dumps(sorted(rejected.values(), key=lambda x: x["to"]), ensure_ascii=False, indent=2),
                     encoding="utf-8")
    return len(closes), len(rejected)


def apply_closes(G: nx.MultiDiGraph, closes: list[dict]) -> int:
    """Добавляет в граф рёбра CLOSES (пункт закрытия → задача). Цитата, таймкод и встреча закрытия лежат на самом
    ребре: по ним traverse отдаёт `closed_by` у закрытой задачи. Статус цели ставится done с датой закрытия
    (`closed_at`): status.task_state на дату раньше закрытия считает задачу ещё открытой. Идемпотентно.

    Если среди извлечённых пунктов источника нет, создаётся узел типа Closure с origin="closes". Тип не Task
    намеренно: чужой код считает и ищет пункты по типам Decision/Task/OpenQuestion, и служебный узел не должен
    попадать в такие счётчики и выдачу."""
    n_new = 0
    for c in closes:
        if c["to"] not in G or c.get("confidence", 1) < CLOSES_CONFIDENCE:
            continue
        src, target = c["from"], G.nodes[c["to"]]
        if src not in G:
            text = c.get("rationale") or c["quote"]
            G.add_node(src, type="Closure", origin="closes", label=text, statement=text,
                       project=target.get("project"), meeting_id=c["meeting_id"], date=c["date"], ts=c["ts"],
                       quote=c["quote"])
            G.add_edge(src, f"project:{target.get('project')}", key="BELONGS_TO", type="BELONGS_TO")
            if f"meeting:{c['meeting_id']}" in G:
                G.add_edge(src, f"meeting:{c['meeting_id']}", key="STATED_IN", type="STATED_IN")
        if not G.has_edge(src, c["to"], key="CLOSES"):
            G.add_edge(src, c["to"], key="CLOSES", type="CLOSES", confidence=c["confidence"],
                       rationale=c.get("rationale", ""), quote=c["quote"], ts=c["ts"], meeting_id=c["meeting_id"],
                       date=c["date"], origin="closes")
            n_new += 1
        if target["type"] == "Task":
            target["status"] = "done"
            target["closed_at"] = min(c["date"], target.get("closed_at") or c["date"])
    return n_new


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--graph", default="data/graph/graph.json")
    ap.add_argument("--out", default="data/graph", help="папка для closes.json и closes_rejected.json")
    ap.add_argument("--limit", type=int, default=None, help="только первые N задач (для проверки на 5 примерах)")
    ap.add_argument("--dry-run", action="store_true", help="без модели: кандидаты, размер запросов, оценка стоимости")
    ap.add_argument("--apply", action="store_true", help="дописать рёбра CLOSES из closes.json в граф --graph")
    a = ap.parse_args(argv)

    from src.graph.queries import load_graph
    from src.graph.builder import save_graph

    G = load_graph(ROOT / a.graph)
    if a.apply:
        n = apply_closes(G, _load_json(ROOT / a.out / "closes.json"))
        save_graph(G, ROOT / a.graph)
        print(f"Добавлено рёбер CLOSES: {n}. Граф сохранён: {a.graph}")
        return 0
    items = open_items(G)[: a.limit]
    if a.dry_run:
        est = estimate(G, items)
        for r in est["rows"]:
            print(f"{r['item']:10} кандидатов {r['cands']}  ≈{r['tokens']} токенов на входе")
        print(f"\nЗапросов к модели: {est['calls']} (из {len(items)} задач). Вход ≈ {est['tokens_in']} токенов; "
              f"оценка стоимости при ценах из .env: ≈ ${est['cost_usd']:.4f}")
        return 0

    results, spent = [], 0.0
    for n in items:
        r = classify(G, n)
        spent += r["cost_usd"]
        results.append(r)
        mark = "ЗАКРЫТА" if r["record"] else "—"
        print(f"{n:10} {mark:8} {r['reason'] or (r['record'] or {}).get('quote', '')[:70]}   (накоплено ${spent:.4f})")
    k, rej = save_results(results, G, ROOT / a.out)
    print(f"\nclose.json: {k}, rejected: {rej}. Потрачено в этом прогоне: ${spent:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
