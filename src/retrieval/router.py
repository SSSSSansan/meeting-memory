"""РОЛЬ C. Роутер режимов для mode="auto": какой ветке отдать вопрос — baseline или graph.

    from src.retrieval.router import route
    route("Как менялся срок релиза Zhol?")   # Route(mode="graph", qtype="T2", reason="...")

Правила повторяют рабочую гипотезу проекта (раздел 2 контекста): граф нужен на вопросах через
несколько встреч и проектов (T2 эволюция, T3 кросс-проект, T4 конфликты, T5 открытое),
а простой факт из одной встречи (T1) дешевле и не хуже отвечает baseline.
Правила, а не LLM-классификатор: ноль токенов на маршрутизацию и решение видно глазами.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import networkx as nx

from src.retrieval.graph_rag import find_entities, get_graph, last_n_meetings
from src.utils.text import normalize

# Основы слов (ru + kk). Ищутся в начале слова: «менял» ловит «менялся», «менялись».
_PATTERNS = {
    "T2": ["менял", "меняет", "изменил", "изменен", "изменён", "истори", "хронолог", "эволюц",
           "перенос", "перенес", "перенёс", "сдвиг", "сдвинул", "сначала", "раньше", "как развивал",
           "өзгер", "ауыстыр"],
    "T4": ["конфликт", "противореч", "несовмест", "пересека", "загрузк", "перегруж", "одновремен",
           "қайшы", "сәйкес емес"],
    "T5": ["открыт", "незакрыт", "не закрыт", "не сделан", "не выполн", "осталось", "висит", "просроч",
           "ещё не", "еще не", "забыл", "жабылмаған", "орындалмаған", "ашық"],
}
_CROSS = re.compile(r"на встреч\w*\s+(про|по|о)\b", re.I)


@dataclass
class Route:
    mode: str              # "baseline" | "graph"
    qtype: str             # "T1".."T5" — как роутер понял вопрос
    reason: str


def _has(text: str, stems: list[str]) -> Optional[str]:
    for stem in stems:
        if re.search(rf"(?<!\w){re.escape(stem)}", text):
            return stem
    return None


def route(question: str, G: Optional[nx.MultiDiGraph] = None) -> Route:
    G = G if G is not None else get_graph()
    text = normalize(question)
    entities = find_entities(G, question)
    projects = [e for e in entities if e.startswith("project:")]

    for qtype in ("T4", "T2", "T5"):  # T4 первым: «загрузка изменилась» — это про конфликт людей
        stem = _has(text, _PATTERNS[qtype])
        if stem:
            return Route("graph", qtype, f"признак {qtype}: «{stem}»")
    if len(projects) >= 2 or (projects and _CROSS.search(question)):
        return Route("graph", "T3", "вопрос про несколько проектов или про проект на чужих встречах")
    n = last_n_meetings(question)
    if n and n > 1:
        return Route("graph", "T2", f"охват нескольких встреч: последние {n}")
    return Route("baseline", "T1", "признаков вопроса через несколько встреч нет — факт из одной встречи")
