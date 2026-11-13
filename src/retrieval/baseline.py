"""РОЛЬ C. Baseline RAG: чанки сырых транскриптов → эмбеддинги → поиск по косинусу.

    python -m src.retrieval.baseline "Когда релиз Zhol?" --as-of 2026-09-01   # индекс по data/transcripts

    from src.retrieval.baseline import BaselineIndex
    index = BaselineIndex()
    index.add(transcripts)              # батч встреч; уже проиндексированные пропускаются
    index.search("Когда релиз Zhol?", k=10, as_of="2026-09-01")
    index.save(DEFAULT_INDEX_DIR)

Чанк — окно из нескольких реплик с перекрытием: реплика-ответ («хорошо, но тогда…»)
без предыдущей реплики теряет смысл и для поиска, и для модели.
Формат чанка:
    {"chunk_id": "c_m001_01", "meeting_id": "m_001", "date": "...", "title": "...",
     "projects": ["Zhol", "Boost"],                          # проекты всей встречи, не только этих реплик
     "speakers": ["Амина", "Нұралы"],
     "header": "[m_001 | 2026-08-03 | «...» | проекты: Zhol, Boost | Амина, Нұралы | 00:04:12]",
     "text": "[00:04:12] Имя: текст\\n...",                   # на эмбеддинг уходит header + text
     "segments": [{"ts", "speaker", "text", "start_s"}, ...]}  # по ним ищется точный ts цитаты

Заголовок нужен потому, что в реплике «давайте перенесём на двадцатое» нет ни проекта, ни даты:
без него вопрос про проект такую реплику не находит, и baseline проигрывает графу нечестно.

Поиск (search): жёсткий фильтр только по дате встречи ≤ as_of → широкий top-pool по косинусу →
мягкая пересортировка (совпадение проекта, немного свежесть) → без почти-дублей → первые k.
По проекту жёстко не фильтруем: встреча «Zhol» может 10 минут обсуждать Qamqor.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from src.llm_client import embed
from src.resolve.entities import _CYR  # одна таблица транслитерации на проект, с казахскими буквами
from src.schemas import Transcript

ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INDEX_DIR = ROOT / "runs" / "index" / "baseline"

WINDOW = 3            # реплик в чанке
STEP = 2              # сдвиг окна; WINDOW - STEP реплик попадают в два соседних чанка
MAX_UNIT_WORDS = 80   # длинные реплики (ASR без диаризации) иначе размывают вектор чанка
EMBED_BATCH = 64      # лимит размера одного запроса к API эмбеддингов
CHUNK_FORMAT = "header-v1"  # что именно эмбеддится; индекс другого формата не грузится

CONTEXT_K = 10         # сколько чанков уходит в контекст (план: 8–12)
POOL_K = 40            # кандидатов до пересортировки: узкий top-5 теряет нужное ещё до буста
PROJECT_BOOST = 0.05   # порядка разницы косинусов между соседними релевантными чанками
RECENCY_BOOST = 0.02   # слабее проекта: на вопросах про историю старые решения тоже нужны
NEAR_DUP_SHARE = 0.67  # доля общих реплик, при которой чанк считается повтором уже выбранного
BODY_MIN_MENTIONS = 2  # «жол» по-казахски ещё и «дорога»: одно упоминание в тексте — не проект встречи

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_WORD = re.compile(r"\w+")


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


_SPELLING = (("dzh", "zh"), ("j", "zh"), ("q", "k"), ("w", "u"), ("v", "u"), ("oo", "u"), ("ou", "u"),
             ("y", "i"), ("ii", "i"))


@lru_cache(maxsize=100_000)
def _key(word: str) -> str:
    """Транслитерация в латиницу без схлопывания гласных.

    Ключ резолвера (entities.key) сводит о/е к «а» — для сравнения двух имён это хорошо, а при поиске
    по всему тексту казахское «жалпы» («в целом») становится «Жол». Здесь гласные сохраняются.
    """
    s = "".join(_CYR.get(ch, ch) for ch in unicodedata.normalize("NFC", word.lower()))
    for a, b in _SPELLING:
        s = s.replace(a, b)
    return s


def mentioned_projects(text: str, projects: list[str], min_mentions: int = 1) -> list[str]:
    """Проекты из реестра, названные в тексте не меньше min_mentions раз; частые первыми.

    Жол / Zhol / Джол и Қамқор / Камкор / Qamqor дают один ключ. Слово может быть длиннее
    названия на падежное окончание: «Жолу», «Қамқорға».
    """
    keys = {p: k for p in projects if len(k := _key(p)) >= 3}
    counts: Counter[str] = Counter()
    for word in _WORD.findall(text):
        kw = _key(word)
        for p, kp in keys.items():
            if kw.startswith(kp) and len(kw) - len(kp) <= 3:
                counts[p] += 1
    return [p for p, n in counts.most_common() if n >= min_mentions]


def meeting_projects(t: Transcript, projects: list[str]) -> list[str]:
    """Проекты всей встречи: из названия — сразу, из текста — от BODY_MIN_MENTIONS упоминаний."""
    in_title = mentioned_projects(t.title, projects)
    in_body = mentioned_projects(t.full_text(), projects, BODY_MIN_MENTIONS)
    return in_title + [p for p in in_body if p not in in_title]


def default_projects() -> list[str]:
    """Реестр проектов — узлы Project графа роли B (тот же граф, что видит режим graph).

    Не data/gold/world.yaml: это эталон, системе его видеть нельзя.
    """
    from src.retrieval.graph_rag import get_graph  # локально: baseline должен жить и без графа

    try:
        G = get_graph()
    except Exception:  # графа нет — заголовок без проектов, поиск всё равно работает
        return []
    return sorted(d["label"] for _, d in G.nodes(data=True) if d.get("type") == "Project" and d.get("label"))


def chunk_header(meeting_id: str, date: str, title: str, projects: list[str], speakers: list[str],
                 ts: str) -> str:
    return (f"[{meeting_id} | {date} | «{title}» | проекты: {', '.join(projects) or '—'} | "
            f"{', '.join(speakers) or '—'} | {ts}]")


def embed_text(chunk: dict) -> str:
    return f"{chunk['header']}\n{chunk['text']}" if chunk.get("header") else chunk["text"]


def chunk_transcript(t: Transcript, window: int = WINDOW, step: int = STEP,
                     max_unit_words: int = MAX_UNIT_WORDS, projects: Optional[list[str]] = None) -> list[dict]:
    """projects — реестр проектов; по нему в заголовок пишутся проекты всей встречи."""
    if not 0 < step <= window:
        raise ValueError(f"нужно 0 < step <= window, получено step={step}, window={window}")
    units = [{"ts": s.ts, "speaker": s.speaker, "text": piece, "start_s": s.start_s}
             for s in t.segments for piece in _pieces(s.text, max_unit_words)]
    sid = t.meeting_id.replace("_", "")  # m_001 -> m001, как в остальных id проекта
    meeting_proj = meeting_projects(t, projects or [])
    chunks: list[dict] = []
    for start in range(0, len(units), step):
        segs = units[start:start + window]
        speakers = list(dict.fromkeys(u["speaker"] for u in segs))
        chunks.append({
            "chunk_id": f"c_{sid}_{len(chunks) + 1:02d}",
            "meeting_id": t.meeting_id,
            "date": t.date,
            "title": t.title,
            "projects": meeting_proj,
            "speakers": speakers,
            "header": chunk_header(t.meeting_id, t.date, t.title, meeting_proj, speakers, segs[0]["ts"]),
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
        vectors += embed(texts[i:i + EMBED_BATCH], purpose="embed_index")
    return np.array(vectors, dtype=np.float32)


def _embed_meta() -> dict:
    # Векторы разных моделей несравнимы: индекс, собранный на stub, бесполезен для bge-m3.
    # Формат чанка тоже: по индексу без заголовков вопрос «про проект» ищет хуже, смешивать нельзя.
    return {"backend": os.environ.get("LLM_BACKEND", "stub"),
            "embed_model": os.environ.get("EMBED_MODEL", ""),
            "chunk_format": CHUNK_FORMAT}


def chunk_lines(chunk: dict) -> set:
    """Реплики чанка — по ним чанки сравниваются на почти-дубли (см. NEAR_DUP_SHARE)."""
    return {(chunk["meeting_id"], s["ts"], s["text"]) for s in chunk["segments"]}


def is_near_duplicate(chunk: dict, taken: list[set]) -> bool:
    lines = chunk_lines(chunk)
    return any(len(lines & prev) >= NEAR_DUP_SHARE * len(lines) for prev in taken)


class BaselineIndex:
    def __init__(self, projects: Optional[list[str]] = None) -> None:
        """projects — реестр проектов для заголовков и буста; по умолчанию — проекты графа."""
        self.chunks: list[dict] = []
        self.meta = _embed_meta()
        self._projects = projects
        self._vectors = np.zeros((0, 0), dtype=np.float32)  # строки нормированы: косинус = скалярное произведение
        self._row: Optional[dict[str, int]] = None          # chunk_id → строка _vectors, строится лениво

    @property
    def projects(self) -> list[str]:
        if self._projects is None:
            self._projects = default_projects()
        return self._projects

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
            new += chunk_transcript(t, projects=self.projects, **chunk_kwargs)
        if not new:
            return 0
        vectors = _normalize_rows(_embed_all([embed_text(c) for c in new]))
        self._vectors = vectors if not self.chunks else np.vstack([self._vectors, vectors])
        self.chunks += new
        return len(new)

    def query_vector(self, question: str) -> np.ndarray:
        return _normalize_rows(np.array(embed([question], purpose="embed_query"), dtype=np.float32))[0]

    def similarity(self, chunk_id: str, q: np.ndarray) -> float:
        """Косинус чанка с уже посчитанным вектором вопроса (гибрид ранжирует им backfill графа)."""
        if self._row is None or len(self._row) != len(self.chunks):
            self._row = {c["chunk_id"]: i for i, c in enumerate(self.chunks)}
        return float(self._vectors[self._row[chunk_id]] @ q)

    def search(self, question: str, k: int = CONTEXT_K, as_of: Optional[str] = None,
               pool: int = POOL_K, query_vec: Optional[np.ndarray] = None) -> list[dict]:
        """До k чанков встреч не позже as_of (ISO-дата). У каждого sim — косинус, score — после буста.

        Без as_of в eval на вопрос «что решили к марту» утекает будущее и метрики завышены.
        query_vec — вектор вопроса из query_vector(), чтобы не эмбеддить вопрос второй раз.
        """
        if not self.chunks:
            return []
        allowed = np.array([i for i, c in enumerate(self.chunks) if as_of is None or c["date"] <= as_of],
                           dtype=int)
        if allowed.size == 0:
            return []
        q = self.query_vector(question) if query_vec is None else query_vec
        sims = self._vectors[allowed] @ q
        top = np.argsort(-sims, kind="stable")[:max(pool, k)]
        cands = [(int(allowed[j]), float(sims[j])) for j in top]

        q_projects = set(mentioned_projects(question, self.projects))
        dates = sorted({self.chunks[i]["date"] for i, _ in cands})
        recency = {d: n / (len(dates) - 1) if len(dates) > 1 else 0.0 for n, d in enumerate(dates)}
        scored = []
        for i, sim in cands:
            c = self.chunks[i]
            boost = PROJECT_BOOST if q_projects & set(c.get("projects", [])) else 0.0
            scored.append((sim + boost + RECENCY_BOOST * recency[c["date"]], sim, i))
        scored.sort(key=lambda t: -t[0])  # sort стабильный: при равном score остаётся порядок косинуса

        out: list[dict] = []
        taken: list[set] = []
        for score, sim, i in scored:
            c = self.chunks[i]
            if is_near_duplicate(c, taken):
                continue
            taken.append(chunk_lines(c))
            out.append({**c, "sim": sim, "score": score})
            if len(out) == k:
                break
        return out

    def save(self, path: Path = DEFAULT_INDEX_DIR) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self._vectors)
        (path / "chunks.json").write_text(
            json.dumps({"meta": self.meta, "projects": self.projects, "chunks": self.chunks},
                       ensure_ascii=False, indent=1),
            encoding="utf-8")

    @classmethod
    def load(cls, path: Path = DEFAULT_INDEX_DIR) -> "BaselineIndex":
        path = Path(path)
        data = json.loads((path / "chunks.json").read_text(encoding="utf-8"))
        if data["meta"] != _embed_meta():
            raise ValueError(f"индекс в {path} собран для {data['meta']}, а сейчас {_embed_meta()}: "
                             "пересоберите его")
        index = cls(projects=data.get("projects"))
        index.chunks = data["chunks"]
        index._vectors = np.load(path / "vectors.npy")
        return index


def load_transcripts(folder: Path) -> list[Transcript]:
    return [Transcript.model_validate_json(f.read_text(encoding="utf-8"))
            for f in sorted(Path(folder).glob("*.json"))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("-k", type=int, default=CONTEXT_K)
    ap.add_argument("--as-of", default=None, help="ISO-дата: только встречи не позже неё")
    ap.add_argument("--transcripts", default="data/transcripts")
    a = ap.parse_args()

    index = BaselineIndex()
    n = index.add(load_transcripts(ROOT / a.transcripts))
    print(f"Встреч: {len(index.meeting_ids)}, чанков: {n}, backend: {index.meta['backend']}\n")
    for hit in index.search(a.question, k=a.k, as_of=a.as_of):
        print(f"{hit['score']:.3f} (cos {hit['sim']:.3f})  {hit['chunk_id']}  {hit['header']}")
        print("   " + hit["text"].replace("\n", "\n   ") + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
