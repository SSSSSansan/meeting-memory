"""Контрольный режим bm25 (A5): поиск по словам по ТЕМ ЖЕ чанкам, что у baseline (индекс header-v2), без эмбеддингов.

    from src.retrieval.bm25 import bm25_search
    bm25_search(index, "Когда релиз Zhol?", k=CONTEXT_K, as_of="2026-09-01")

Зачем: отделить вклад эмбеддингов от вклада «правильных чанков». Если bm25 ≈ baseline, выигрыш baseline не в векторах.
BM25 Okapi (k1=1.5, b=0.75), idf = ln(1 + (N - df + 0.5) / (df + 0.5)) — неотрицательный вариант Lucene.
N и df считаются только по чанкам встреч не позже as_of: статистика корпуса тоже не должна видеть будущее.
Документ чанка — то же, что несёт заголовок header-v2 (название встречи, проекты, говорящие) плюс реплики; таймкоды,
id встречи и дата в слова не идут — это шум для поиска по словам. Буст проекта и свежести baseline здесь не применяются:
контроль — чистый BM25. Почти-дубли (перекрытие окон) убираются, как в baseline, иначе k мест займут соседние окна.
"""
from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from typing import Optional

from src.retrieval.baseline import CONTEXT_K, BaselineIndex, chunk_lines, is_near_duplicate

K1 = 1.5
B = 0.75

# Та же свёртка казахских букв, что scripts/build_gold.py::KK_FOLD (тест следит за совпадением).
KK_FOLD = {
    ord("қ"): "к", ord("ғ"): "г", ord("ә"): "а", ord("ң"): "н", ord("ө"): "о",
    ord("ұ"): "у", ord("ү"): "у", ord("һ"): "х", ord("і"): "и",
}
_NOISE = re.compile(r"\[[^\]]*\]")    # [шум], [неразборчиво] — как в build_gold.normalize_quote
_TOKEN = re.compile(r"[^\W_]+")       # буквы и цифры


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFC", text).lower()
    text = _NOISE.sub(" ", text).translate(KK_FOLD)
    return _TOKEN.findall(text)


def chunk_document(chunk: dict) -> str:
    parts = [chunk.get("title", ""), " ".join(chunk.get("projects", [])), " ".join(chunk.get("speakers", []))]
    parts += [s["text"] for s in chunk.get("segments", [])]
    return "\n".join(parts)


class BM25:
    """Счётчики слов по чанкам индекса; пересчитываются, только если в индекс добавили встречи."""

    def __init__(self, index: BaselineIndex) -> None:
        self.index = index
        self._n = -1
        self._tf: list[Counter] = []
        self._len: list[int] = []

    def _sync(self) -> None:
        if self._n == len(self.index.chunks):
            return
        self._tf = [Counter(tokenize(chunk_document(c))) for c in self.index.chunks]
        self._len = [sum(tf.values()) for tf in self._tf]
        self._n = len(self.index.chunks)

    def scores(self, question: str, as_of: Optional[str] = None) -> list[tuple[int, float]]:
        """(номер чанка, score) для чанков встреч не позже as_of с ненулевым score, по убыванию."""
        self._sync()
        allowed = [i for i, c in enumerate(self.index.chunks) if as_of is None or c["date"] <= as_of]
        terms = list(dict.fromkeys(tokenize(question)))
        if not allowed or not terms:
            return []
        n = len(allowed)
        avgdl = sum(self._len[i] for i in allowed) / n or 1.0
        idf = {}
        for t in terms:
            df = sum(1 for i in allowed if t in self._tf[i])
            idf[t] = math.log(1 + (n - df + 0.5) / (df + 0.5))
        out = []
        for i in allowed:
            tf, dl = self._tf[i], self._len[i]
            s = sum(idf[t] * tf[t] * (K1 + 1) / (tf[t] + K1 * (1 - B + B * dl / avgdl)) for t in terms if t in tf)
            if s > 0:
                out.append((i, s))
        out.sort(key=lambda p: -p[1])   # sort стабильный: при равном score — порядок индекса
        return out

    def search(self, question: str, k: int = CONTEXT_K, as_of: Optional[str] = None) -> list[dict]:
        hits: list[dict] = []
        taken: list[set] = []
        for i, score in self.scores(question, as_of):
            c = self.index.chunks[i]
            if is_near_duplicate(c, taken):
                continue
            taken.append(chunk_lines(c))
            hits.append({**c, "score": score})
            if len(hits) == k:
                break
        return hits


_cache: dict[int, BM25] = {}


def bm25_search(index: BaselineIndex, question: str, k: int = CONTEXT_K, as_of: Optional[str] = None) -> list[dict]:
    """До k чанков в формате baseline (для answerer.format_context), у каждого score BM25."""
    bm = _cache.get(id(index))
    if bm is None or bm.index is not index:
        bm = _cache[id(index)] = BM25(index)
    return bm.search(question, k=k, as_of=as_of)
