"""РОЛЬ C. Baseline RAG: чанки сырых транскриптов → эмбеддинги → top-k по косинусу.

    python -m src.retrieval.baseline "Когда релиз Zhol?"          # индекс по data/transcripts, top-5

    from src.retrieval.baseline import BaselineIndex
    index = BaselineIndex()
    index.add(transcripts)              # батч встреч; уже проиндексированные пропускаются
    index.search("Когда релиз Zhol?", k=5)
    index.save(DEFAULT_INDEX_DIR)

Чанк — окно из нескольких реплик с перекрытием: реплика-ответ («хорошо, но тогда…»)
без предыдущей реплики теряет смысл и для поиска, и для модели.
Формат чанка:
    {"chunk_id": "c_m001_01", "meeting_id": "m_001", "date": "...", "title": "...",
     "text": "[00:04:12] Имя: текст\\n...",                  # уходит на эмбеддинг
     "segments": [{"ts", "speaker", "text", "start_s"}, ...]}  # по ним ищется точный ts цитаты
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

from src.llm_client import embed
from src.schemas import Transcript

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INDEX_DIR = ROOT / "runs" / "index" / "baseline"

WINDOW = 3            # реплик в чанке
STEP = 2              # сдвиг окна; WINDOW - STEP реплик попадают в два соседних чанка
MAX_UNIT_WORDS = 80   # длинные реплики (ASR без диаризации) иначе размывают вектор чанка
EMBED_BATCH = 64      # лимит размера одного запроса к API эмбеддингов

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def _pieces(text: str, max_words: int) -> list[str]:
    """Режет длинную реплику по предложениям; одно слишком длинное предложение — по словам.

    Пробелы внутри куска могут схлопнуться, поэтому кусок совпадает с исходным текстом
    после src.utils.text.normalize — так же, как проверяются цитаты.
    """
    if len(text.split()) <= max_words:
        return [text]
    pieces: list[str] = []
    cur: list[str] = []
    for sentence in _SENTENCE_END.split(text.strip()):
        words = sentence.split()
        while len(words) > max_words:
            if cur:
                pieces.append(" ".join(cur))
                cur = []
            pieces.append(" ".join(words[:max_words]))
            words = words[max_words:]
        if cur and len(cur) + len(words) > max_words:
            pieces.append(" ".join(cur))
            cur = []
        cur += words
    if cur:
        pieces.append(" ".join(cur))
    return pieces


def chunk_transcript(t: Transcript, window: int = WINDOW, step: int = STEP,
                     max_unit_words: int = MAX_UNIT_WORDS) -> list[dict]:
    if not 0 < step <= window:
        raise ValueError(f"нужно 0 < step <= window, получено step={step}, window={window}")
    units = [{"ts": s.ts, "speaker": s.speaker, "text": piece, "start_s": s.start_s}
             for s in t.segments for piece in _pieces(s.text, max_unit_words)]
    sid = t.meeting_id.replace("_", "")  # m_001 -> m001, как в остальных id проекта
    chunks: list[dict] = []
    for start in range(0, len(units), step):
        segs = units[start:start + window]
        chunks.append({
            "chunk_id": f"c_{sid}_{len(chunks) + 1:02d}",
            "meeting_id": t.meeting_id,
            "date": t.date,
            "title": t.title,
            "text": "\n".join(f"[{u['ts']}] {u['speaker']}: {u['text']}" for u in segs),
            "segments": segs,
        })
        if start + window >= len(units):  # окно дошло до конца: следующий чанк был бы хвостом-дублем
            break
    return chunks


def _normalize_rows(m: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    return m / np.where(norms == 0, 1.0, norms)


def _embed_all(texts: list[str]) -> np.ndarray:
    vectors: list[list[float]] = []
    for i in range(0, len(texts), EMBED_BATCH):
        vectors += embed(texts[i:i + EMBED_BATCH])
    return np.array(vectors, dtype=np.float32)


def _embed_meta() -> dict:
    # Векторы разных моделей несравнимы: индекс, собранный на stub, бесполезен для bge-m3.
    return {"backend": os.environ.get("LLM_BACKEND", "stub"),
            "embed_model": os.environ.get("EMBED_MODEL", "")}


class BaselineIndex:
    def __init__(self) -> None:
        self.chunks: list[dict] = []
        self.meta = _embed_meta()
        self._vectors = np.zeros((0, 0), dtype=np.float32)  # строки нормированы: косинус = скалярное произведение

    @property
    def meeting_ids(self) -> set[str]:
        return {c["meeting_id"] for c in self.chunks}

    def add(self, transcripts: Iterable[Transcript], **chunk_kwargs) -> int:
        """Добавляет батч встреч, возвращает число новых чанков. Повторный запуск ничего не дублирует."""
        known = self.meeting_ids
        new: list[dict] = []
        for t in transcripts:
            if t.meeting_id in known:
                continue
            known.add(t.meeting_id)
            new += chunk_transcript(t, **chunk_kwargs)
        if not new:
            return 0
        vectors = _normalize_rows(_embed_all([c["text"] for c in new]))
        self._vectors = vectors if not self.chunks else np.vstack([self._vectors, vectors])
        self.chunks += new
        return len(new)

    def search(self, question: str, k: int = 5) -> list[dict]:
        """Top-k чанков по косинусу; к каждому добавлено поле score."""
        if not self.chunks:
            return []
        q = _normalize_rows(np.array(embed([question]), dtype=np.float32))[0]
        scores = self._vectors @ q
        top = np.argsort(-scores, kind="stable")[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]

    def save(self, path: Path = DEFAULT_INDEX_DIR) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self._vectors)
        (path / "chunks.json").write_text(
            json.dumps({"meta": self.meta, "chunks": self.chunks}, ensure_ascii=False, indent=1),
            encoding="utf-8")

    @classmethod
    def load(cls, path: Path = DEFAULT_INDEX_DIR) -> "BaselineIndex":
        path = Path(path)
        data = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
        if data["meta"] != _embed_meta():
            raise ValueError(f"индекс в {path} собран для {data['meta']}, а сейчас {_embed_meta()}: "
                             "пересоберите его")
        index = cls()
        index.chunks = data["chunks"]
        index._vectors = np.load(path / "vectors.npy")
        return index


def load_transcripts(folder: Path) -> list[Transcript]:
    return [Transcript.model_validate_json(f.read_text(encoding="utf-8"))
            for f in sorted(Path(folder).glob("*.json"))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=5)
    ap.add_argument("--transcripts", default="data/transcripts")
    a = ap.parse_args()

    index = BaselineIndex()
    n = index.add(load_transcripts(ROOT / a.transcripts))
    print(f"Встреч: {len(index.meeting_ids)}, чанков: {n}, backend: {index.meta['backend']}\n")
    for hit in index.search(a.question, k=a.k):
        print(f"{hit['score']:.3f}  {hit['chunk_id']}  {hit['date']}  {hit['title']}")
        print("   " + hit["text"].replace("\n", "\n   ") + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
