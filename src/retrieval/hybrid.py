"""РОЛЬ C. Гибридный режим (C4): граф даёт навигацию, векторный поиск — полноту.

    from src.retrieval.hybrid import retrieve_hybrid
    r = retrieve_hybrid("Как менялся срок релиза Zhol?", index, as_of="2026-10-01")
    r["graph_context"], r["chunks"], r["retrieved_meetings"]

Бюджет тот же, что у baseline и graph: всего CONTEXT_K элементов (пункт графа или чанк транскрипта = один
элемент), иначе гибрид выигрывал бы объёмом. Контекст из трёх частей, в таком порядке:
  1. Пункты графа — первые HYBRID_GRAPH_K из того же ранжирования, что у режима graph (тот же обход, тот же порядок
     важности, цепочки замен целиком), со статусами, заменами и конфликтами.
  2. Backfill: к этим пунктам — чанк транскрипта вокруг таймкода пункта (не дальше BACKFILL_WINDOW_S), до BACKFILL_MAX.
     Граф знает только то, что поймал экстракт; соседние реплики добирают пропущенное.
     Если кандидатов больше, проходят ближайшие к вопросу по косинусу: граф решает, ГДЕ смотреть, вектор — что важнее.
  3. Обычный поиск baseline (as_of, буст проекта, без почти-дублей) добивает до CONTEXT_K элементов.
Чанк, пришедший и из графа, и из поиска, занимает одно место: сравнение по chunk_id и по общим репликам
(как почти-дубли в baseline). Фиксированное окно «встреча + N секунд» не нужно — чанк уже окно реплик.

Когда роль B отдаст обход графа (seed → expand → backfill, план B6), заменить graph_part().
"""
from __future__ import annotations

from collections import defaultdict
from typing import Optional

import networkx as nx

from src.retrieval.baseline import CONTEXT_K, BaselineIndex, chunk_lines, is_near_duplicate
from src.retrieval.graph_rag import get_graph, retrieve_subgraph

BACKFILL_WINDOW_S = 45  # план B6: «секунд по сорок пять в обе стороны» от таймкода пункта
HYBRID_GRAPH_K = 4      # 4 пункта графа + 3 чанка вокруг них + не меньше 3 из поиска = CONTEXT_K
BACKFILL_MAX = 3        # навигация графа и полнота поиска получают примерно поровну места
HYBRID_CHUNKS = CONTEXT_K  # верхняя граница чанков (если граф ничего не нашёл); её читает src/eval/estimate.py


def ts_seconds(ts: str) -> int:
    h, m, s = ts.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


def _segment_seconds(seg: dict) -> float:
    return seg["start_s"] if seg.get("start_s") is not None else ts_seconds(seg["ts"])


def graph_part(question: str, depth: int, G: Optional[nx.MultiDiGraph], as_of: Optional[str],
               max_items: int = HYBRID_GRAPH_K) -> dict:
    """Пункты графа и их якоря (встреча, таймкод) для backfill. Сюда встанет обход роли B."""
    G = G if G is not None else get_graph()
    sub = retrieve_subgraph(question, depth=depth, G=G, as_of=as_of, max_items=max_items)
    anchors = [(G.nodes[n]["meeting_id"], G.nodes[n]["ts"]) for n in sub["items"]
               if G.nodes[n].get("meeting_id") and G.nodes[n].get("ts")]
    meetings = list(dict.fromkeys(m for m, _ in anchors))
    return {"items": sub["items"], "context": sub["context"] if sub["items"] else "", "anchors": anchors,
            "meetings": meetings, "fallback": sub["fallback"]}


def backfill_candidates(index: BaselineIndex, anchors: list[tuple[str, str]], as_of: Optional[str],
                        window_s: int = BACKFILL_WINDOW_S) -> list[dict]:
    """На каждый якорь — ближайший по времени чанк его встречи, если он не дальше window_s. Без повторов."""
    by_meeting: dict[str, list[dict]] = defaultdict(list)
    for c in index.chunks:
        if as_of is None or c["date"] <= as_of:
            by_meeting[c["meeting_id"]].append(c)
    out, seen = [], set()
    for mid, ts in anchors:
        center = ts_seconds(ts)
        best, best_d = None, None
        for c in by_meeting.get(mid, []):
            d = min(abs(_segment_seconds(s) - center) for s in c["segments"])
            if d <= window_s and (best_d is None or d < best_d):
                best, best_d = c, d
        if best is not None and best["chunk_id"] not in seen:
            seen.add(best["chunk_id"])
            out.append(best)
    return out


def retrieve_hybrid(question: str, index: BaselineIndex, *, depth: int = 2, G: Optional[nx.MultiDiGraph] = None,
                    as_of: Optional[str] = None, context_k: int = CONTEXT_K, graph_k: int = HYBRID_GRAPH_K,
                    backfill_max: int = BACKFILL_MAX) -> dict:
    """Возвращает {graph_context, graph_items, chunks, n_backfill, n_vector, retrieved_meetings}.

    Пунктов графа + чанков всего не больше context_k (тот же бюджет, что у baseline и graph).
    chunks — в формате baseline (для answerer.format_context), у каждого поле via: "graph" | "vector".
    retrieved_meetings — встречи пунктов графа, затем встречи чанков, без повторов.
    """
    graph = graph_part(question, depth, G, as_of, max_items=min(graph_k, context_k))
    max_chunks = context_k - len(graph["items"])
    q = index.query_vector(question) if index.chunks and max_chunks > 0 else None

    backfill = backfill_candidates(index, graph["anchors"], as_of)
    if q is not None:
        backfill.sort(key=lambda c: -index.similarity(c["chunk_id"], q))
    chunks: list[dict] = []
    taken: list[set] = []
    for c in backfill[:min(backfill_max, max_chunks)]:
        chunks.append({**c, "via": "graph"})
        taken.append(chunk_lines(c))
    n_backfill = len(chunks)

    if q is not None and len(chunks) < max_chunks:
        # запас на n_backfill: часть найденного совпадёт с уже взятыми чанками графа
        for h in index.search(question, k=max_chunks + n_backfill, as_of=as_of, query_vec=q):
            if len(chunks) == max_chunks:
                break
            if is_near_duplicate(h, taken):
                continue
            chunks.append({**h, "via": "vector"})
            taken.append(chunk_lines(h))

    retrieved = list(dict.fromkeys(graph["meetings"] + [c["meeting_id"] for c in chunks]))
    return {"graph_context": graph["context"], "graph_items": graph["items"], "graph_fallback": graph["fallback"],
            "chunks": chunks, "n_backfill": n_backfill, "n_vector": len(chunks) - n_backfill,
            "retrieved_meetings": retrieved}
