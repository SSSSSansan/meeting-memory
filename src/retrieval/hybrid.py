"""РОЛЬ C. Гибридный режим (C4): граф даёт навигацию, векторный поиск — полноту.

    from src.retrieval.hybrid import retrieve_hybrid
    r = retrieve_hybrid("Как менялся срок релиза Zhol?", index, as_of="2026-10-01")
    r["graph_context"], r["chunks"], r["retrieved_meetings"]

Бюджет тот же, что у baseline и graph: всего CONTEXT_K элементов (пункт графа с репликами вокруг него или чанк
транскрипта = один элемент), иначе гибрид выигрывал бы объёмом. Две части:
  1. Пункты графа — первые HYBRID_GRAPH_K из обхода роли B (src.graph.traverse, план B6): тот же порядок важности,
     что у режима graph, цепочки замен целиком, со статусами на as_of, связями и репликами ±45 с вокруг пункта
     (третий слой обхода — то, что экстракт мог не поймать).
  2. Обычный поиск baseline (as_of, буст проекта, без почти-дублей) добивает до CONTEXT_K элементов. Чанк, чьи реплики
     уже пришли с пунктом графа, места не занимает (сравнение по общим репликам, как почти-дубли в baseline).
"""
from __future__ import annotations

from typing import Optional

import networkx as nx

from src.graph.traverse import traverse
from src.retrieval.baseline import CONTEXT_K, BaselineIndex, chunk_lines, is_near_duplicate
from src.retrieval.graph_rag import get_aliases, get_graph, graph_context

HYBRID_GRAPH_K = 5         # половина бюджета навигации графа (пункт уже с репликами), половина — поиску
HYBRID_CHUNKS = CONTEXT_K  # верхняя граница чанков (если граф ничего не нашёл); её читает src/eval/estimate.py


def graph_part(question: str, depth: int, G: Optional[nx.MultiDiGraph], as_of: Optional[str],
               max_items: int = HYBRID_GRAPH_K) -> dict:
    """Пункты графа для гибрида: обход роли B, обрезка и текст — как у режима graph (graph_rag.graph_context)."""
    aliases = None
    if G is None:
        G, aliases = get_graph(), get_aliases()
    r = traverse(question, G=G, aliases=aliases, as_of=as_of, depth=depth, with_text=False, k=max_items)
    sub = graph_context(G, question, r, as_of, max_items)
    meetings = list(dict.fromkeys(G.nodes[n]["meeting_id"] for n in sub["items"] if G.nodes[n].get("meeting_id")))
    lines = [{(x["meeting_id"], line["ts"], line["text"]) for line in x["lines"]}
             for x in sub["raw"].values() if x.get("lines")]
    return {"items": sub["items"], "context": sub["context"], "meetings": meetings, "fallback": sub["fallback"],
            "n_found": sub["n_found"], "raw_lines": lines}


def retrieve_hybrid(question: str, index: BaselineIndex, *, depth: int = 2, G: Optional[nx.MultiDiGraph] = None,
                    as_of: Optional[str] = None, context_k: int = CONTEXT_K,
                    graph_k: int = HYBRID_GRAPH_K) -> dict:
    """Возвращает {graph_context, graph_items, chunks, n_graph_text, n_vector, retrieved_meetings}.

    Пунктов графа + чанков всего не больше context_k (тот же бюджет, что у baseline и graph).
    chunks — в формате baseline (для answerer.format_context), у каждого via="vector".
    retrieved_meetings — встречи пунктов графа, затем встречи чанков, без повторов.
    """
    graph = graph_part(question, depth, G, as_of, max_items=min(graph_k, context_k))
    max_chunks = context_k - len(graph["items"])
    chunks: list[dict] = []
    taken: list[set] = list(graph["raw_lines"])  # реплики, уже пришедшие с пунктами графа
    if index.chunks and max_chunks > 0:
        # запас: часть найденного совпадёт с репликами пунктов графа
        for h in index.search(question, k=max_chunks + len(graph["items"]), as_of=as_of):
            if len(chunks) == max_chunks:
                break
            if is_near_duplicate(h, taken):
                continue
            chunks.append({**h, "via": "vector"})
            taken.append(chunk_lines(h))

    retrieved = list(dict.fromkeys(graph["meetings"] + [c["meeting_id"] for c in chunks]))
    return {"graph_context": graph["context"], "graph_items": graph["items"], "graph_fallback": graph["fallback"],
            "chunks": chunks, "n_graph_text": len(graph["raw_lines"]), "n_vector": len(chunks),
            "retrieved_meetings": retrieved}
