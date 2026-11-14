"""РОЛЬ C. Функции для UI и eval — вся логика здесь, в app/ только отображение.

    from src import api
    pairs = api.pair_scribal_files({"m_001.transcript.json": b"...", "m_001.meeting.json": b"..."})
    transcripts = [api.load_scribal(*p) for p in pairs]
    for done, total in api.index_batches(transcripts): ...     # батчами по 5, индекс сохраняется после каждого
    api.ask("Когда релиз Zhol?")                              # → src.schemas.Answer
    api.get_timeline("Zhol")                                  # пункты проекта по времени, статусы и связи
    api.get_graph("Zhol")                                     # узлы и рёбра для рисунка графа

Загрузка — пара файлов Scribal на встречу: `<meeting_id>.transcript.json` (raw) и
`<meeting_id>.meeting.json` (ответ GET /meetings/{id}), как в data/raw/.
"""
from __future__ import annotations

import json
import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Iterator, Optional

from src.answer.answerer import answer
from src.ingest.scribal_adapter import convert
from src.ingest.speakers import apply_speakers
from src.retrieval.baseline import CONTEXT_K, DEFAULT_INDEX_DIR, ROOT, BaselineIndex
from src.resolve.entities import resolve_projects
from src.retrieval.graph_rag import (B_GRAPH_PATH, GRAPH_PATH, SAMPLE_GRAPH_PATH, default_graph_path, get_aliases,
                                     use_graph_path)
from src.retrieval.graph_rag import get_graph as _load_graph
from src.retrieval.router import Route
from src.retrieval.router import route as _route
from src.schemas import Answer, ScribalMeeting, ScribalTranscript, Transcript

BATCH_SIZE = 5  # решение куратора: встречи обрабатываются батчами по 5
DEMO_DIR = ROOT / "data" / "raw"

_SCRIBAL_NAME = re.compile(r"^(?P<mid>.+)\.(?P<kind>transcript|meeting)\.json$")

_index: Optional[BaselineIndex] = None
_index_dir: Path = DEFAULT_INDEX_DIR


def pair_scribal_files(files: dict[str, bytes]) -> tuple[list[tuple[str, bytes, bytes]], list[str]]:
    """Имя файла → содержимое. Возвращает ([(meeting_id, transcript, meeting)], [ошибки])."""
    found: dict[str, dict[str, bytes]] = {}
    errors = []
    for name, content in files.items():
        m = _SCRIBAL_NAME.match(Path(name).name)
        if not m:
            errors.append(f"{name}: ожидается <id>.transcript.json или <id>.meeting.json")
            continue
        found.setdefault(m["mid"], {})[m["kind"]] = content
    pairs = []
    for mid, kinds in sorted(found.items()):
        missing = {"transcript", "meeting"} - kinds.keys()
        if missing:
            errors.append(f"{mid}: не хватает файла {', '.join(f'{mid}.{k}.json' for k in sorted(missing))}")
        else:
            pairs.append((mid, kinds["transcript"], kinds["meeting"]))
    return pairs, errors


def load_scribal(meeting_id: str, transcript_json: bytes, meeting_json: bytes,
                 source: str = "scribal", speakers: bool = True) -> Transcript:
    """Scribal → канонический транскрипт. speakers=True — имена вместо SPEAKER_XX там, где эвристика
    роли B уверена (src/ingest/speakers.py). Граф строится с теми же именами, поэтому и индекс baseline
    должен их видеть: иначе на вопросах «кто…» baseline проигрывает не из-за архитектуры."""
    raw = ScribalTranscript.model_validate(json.loads(transcript_json))
    meta = ScribalMeeting.model_validate(json.loads(meeting_json))
    t = convert(raw, meta, meeting_id, source)
    return apply_speakers(t) if speakers else t


def demo_files() -> dict[str, bytes]:
    """Синтетические встречи роли A из data/raw/ — те же файлы, что пользователь загрузил бы руками."""
    return {p.name: p.read_bytes() for p in sorted(DEMO_DIR.glob("*.json"))}


def use_index_dir(path: Path) -> None:
    """Куда сохранять индекс (тесты подставляют временную папку)."""
    global _index, _index_dir
    _index, _index_dir = None, Path(path)


def get_index() -> BaselineIndex:
    """Индекс с диска; если его нет или он собран другой моделью эмбеддингов — пустой."""
    global _index
    if _index is None:
        try:
            _index = BaselineIndex.load(_index_dir)
        except (FileNotFoundError, ValueError):
            _index = BaselineIndex()
    return _index


def indexed_meetings() -> list[dict]:
    seen: dict[str, dict] = {}
    for c in get_index().chunks:
        m = seen.setdefault(c["meeting_id"], {"meeting_id": c["meeting_id"], "date": c["date"],
                                              "title": c["title"], "chunks": 0})
        m["chunks"] += 1
    return sorted(seen.values(), key=lambda m: (m["date"], m["meeting_id"]))


def index_batches(transcripts: list[Transcript], batch_size: int = BATCH_SIZE) -> Iterator[tuple[int, int]]:
    """Индексирует по batch_size встреч и сохраняет индекс после каждого батча.

    Отдаёт (обработано встреч, всего) после каждого батча — для прогресса в UI.
    Уже проиндексированные встречи пропускаются, поэтому прерванную загрузку можно повторить.
    """
    index = get_index()
    for i in range(0, len(transcripts), batch_size):
        if index.add(transcripts[i:i + batch_size]):
            index.save(_index_dir)
        yield min(i + batch_size, len(transcripts)), len(transcripts)


def reset_index() -> None:
    global _index
    _index = BaselineIndex()
    for name in ("chunks.json", "vectors.npy"):
        (_index_dir / name).unlink(missing_ok=True)


def ask(question: str, mode: str = "baseline", k: int = CONTEXT_K, depth: int = 2,
        as_of: Optional[str] = None) -> Answer:
    """Одна точка для UI и eval. baseline ищет по индексу встреч, загруженных в UI;
    graph — по графу роли B (graph_info()); hybrid — граф + реплики вокруг его пунктов + поиск по индексу;
    auto — ветку выбирает роутер (см. route()). Контроли eval (A5): bm25 — поиск по словам по тем же чанкам;
    longctx — все встречи индекса до as_of целиком, без поиска и без k.
    as_of — ISO-дата вопроса: все ветки видят только встречи не позже неё (для eval обязательно).
    Какой веткой реально ответили — Answer.mode; какие встречи ушли в контекст — Answer.retrieved_meetings."""
    return answer(question, mode=mode, k=k, depth=depth, index=get_index(), as_of=as_of)


def route(question: str) -> Route:
    """Какую ветку выберет auto и почему — без вызова LLM, для подписи в UI."""
    return _route(question)


def graph_info() -> dict:
    """Какой граф читает режим graph и что в нём есть — чтобы в UI было видно расхождение с индексом."""
    path = default_graph_path()
    G = _load_graph(path)
    meetings = sorted({d["meeting_id"] for _, d in G.nodes(data=True) if d.get("type") == "Meeting"})
    return {"path": path.relative_to(ROOT).as_posix(), "is_sample": path == SAMPLE_GRAPH_PATH, "meetings": meetings,
            "aliases": len(get_aliases(path))}


# ---------- Граф для UI: таймлайн проекта и рисунок ----------

ITEM_TYPES = ("Decision", "Task", "OpenQuestion")
LINK_TYPES = ("SUPERSEDES", "CONTRADICTS", "DEPENDS_ON", "REFINES", "SAME_TOPIC")


def list_projects() -> list[str]:
    G = _load_graph()
    return sorted(d["label"] for _, d in G.nodes(data=True) if d.get("type") == "Project")


def _project_node(G, project: str) -> Optional[str]:
    """Узел проекта по названию или его варианту: «zhol», «Жол», вариант из aliases.json."""
    labels = {d["label"]: n for n, d in G.nodes(data=True) if d.get("type") == "Project"}
    wanted = project.strip()
    for label, n in labels.items():
        if label.casefold() == wanted.casefold():
            return n
    canon = get_aliases().get(wanted)
    if canon in labels:
        return labels[canon]
    canon = resolve_projects([wanted], sorted(labels)).get(wanted)
    return labels.get(canon)


def _links(G, n: str) -> list[dict]:
    def link(e: dict, direction: str, other: str) -> dict:
        d = G.nodes[other]  # о связанном пункте — то, что понятно человеку, а не только id
        return {"type": e["type"], "direction": direction, "other": other, "rationale": e.get("rationale", ""),
                "other_statement": d.get("statement", ""), "other_date": d.get("date", ""),
                "other_meeting": d.get("meeting_id", "")}

    out = [link(e, "out", v) for _, v, e in G.out_edges(n, data=True) if e["type"] in LINK_TYPES]
    out += [link(e, "in", u) for u, _, e in G.in_edges(n, data=True) if e["type"] in LINK_TYPES]
    return out


def get_timeline(project: str) -> list[dict]:
    """Решения, задачи и открытые вопросы проекта по времени — для таймлайна в UI.

    Пункт попадает в проект по своему полю project, а не по названию встречи, поэтому здесь есть
    и то, что обсуждали на встречах про другие проекты. Статусы вычислены ролью B при сборке графа.
    Неизвестный проект — пустой список.
    """
    G = _load_graph()
    pid = _project_node(G, project)
    if pid is None:
        return []
    items = [u for u, _, e in G.in_edges(pid, data=True)
             if e["type"] == "BELONGS_TO" and G.nodes[u].get("type") in ITEM_TYPES]
    out = []
    for n in sorted(items, key=lambda n: (G.nodes[n].get("date", ""), G.nodes[n].get("ts", ""), n)):
        d = G.nodes[n]
        meeting = G.nodes.get(f"meeting:{d.get('meeting_id')}", {})
        out.append({"id": n, "kind": d["type"], "date": d.get("date", ""), "meeting_id": d.get("meeting_id", ""),
                    "meeting_title": meeting.get("label", ""), "ts": d.get("ts", ""),
                    "statement": d.get("statement", ""), "quote": d.get("quote", ""),
                    "status": d.get("status", ""), "in_conflict": bool(d.get("in_conflict")),
                    "owner": d.get("owner"), "due": d.get("due"), "made_by": d.get("made_by"),
                    "links": _links(G, n)})
    return out


def get_graph(project: Optional[str] = None) -> dict:
    """Узлы и рёбра для рисунка. Без проекта — весь граф без рёбер «участвовал во встрече»
    (их больше всех остальных вместе, рисунок превращается в кашу). С проектом — проект, его пункты,
    связанные с ними решения, встречи, где пункты прозвучали, и исполнители задач."""
    G = _load_graph()
    if project is None:
        keep = set(G.nodes)
    else:
        pid = _project_node(G, project)
        if pid is None:
            return {"nodes": [], "edges": []}
        items = {u for u, _, e in G.in_edges(pid, data=True) if e["type"] == "BELONGS_TO"}
        items |= {link["other"] for n in list(items) for link in _links(G, n)}
        keep = {pid} | items
        for n in items:
            keep |= {v for _, v, e in G.out_edges(n, data=True) if e["type"] in ("STATED_IN", "ASSIGNED_TO")}
    return _graph_of(G, keep)


def _graph_of(G, keep: set[str]) -> dict:
    nodes = [{"id": n, **{k: v for k, v in G.nodes[n].items() if k != "participants"}} for n in sorted(keep)]
    edges = [{"source": u, "target": v, **e} for u, v, e in G.edges(data=True)
             if u in keep and v in keep and e["type"] != "PARTICIPATED_IN"]
    return {"nodes": nodes, "edges": edges}


def context_subgraph(question: str, depth: int = 2, as_of: Optional[str] = None) -> dict:
    """Что из графа уходит в контекст на этот вопрос (режимы graph и hybrid) — для подсветки на рисунке.

    Тот же retrieve_subgraph, что и при ответе: детерминированно и без LLM, поэтому пересчёт совпадает с
    контекстом ответа. {"items": [id пунктов], "entities": [id стартовых сущностей], "fallback": bool}.
    """
    from src.retrieval.graph_rag import retrieve_subgraph

    sub = retrieve_subgraph(question, depth=depth, G=_load_graph(), as_of=as_of)
    return {"items": sub["items"], "entities": sub["entities"], "fallback": sub["fallback"]}


# ---------- Демо-UI (C7): выбор графа, конфликты и замены, источник цитаты, рисунок ----------

GRAPH_LABELS = {B_GRAPH_PATH: "полный: все встречи графа B (data/graph_raw)",
                GRAPH_PATH: "для eval: встречи до m_008 (data/graph)",
                SAMPLE_GRAPH_PATH: "заглушка: старый сюжет Атлас/Горизонт (data/sample)"}


def available_graphs() -> list[dict]:
    """Графы, которые есть на диске, — для выбора в UI. Полный граф B первым: для демо он нужнее."""
    out = []
    for path in (p for p in GRAPH_LABELS if p.exists()):
        G = _load_graph(path)
        meetings = sorted({d["meeting_id"] for _, d in G.nodes(data=True) if d.get("type") == "Meeting"})
        out.append({"path": path, "label": GRAPH_LABELS[path], "meetings": meetings})
    return out


def demo_default_graph() -> Optional[Path]:
    """Граф, выбранный в UI по умолчанию: MEETING_GRAPH, если задана, иначе самый полный из available_graphs()."""
    env = os.environ.get("MEETING_GRAPH")
    if env:
        return default_graph_path()
    graphs = available_graphs()
    return graphs[0]["path"] if graphs else None


def use_graph(path: Optional[Path]) -> None:
    """Какой граф читают режим graph, таймлайн и рисунок; None — граф по умолчанию (тот же, что у eval)."""
    use_graph_path(path)


_ITEM_ID = re.compile(r"\b([dtq])_m(\d{3})_\d{2}\b")
_ITEM_WORD = {"d": "решение", "t": "задача", "q": "вопрос"}


def readable_ids(text: str) -> str:
    """«согласно решению d_m007_01» → «согласно решению (m_007, 2026-08-31)»: в графовом контексте
    у пунктов есть внутренние id, и модель иногда их цитирует; руководителю нужны встреча и дата."""
    G = _load_graph()

    def repl(m: re.Match) -> str:
        d = G.nodes.get(m.group(0), {})
        mid = d.get("meeting_id") or f"m_{m.group(2)}"
        return f"({mid}, {d['date']})" if d.get("date") else f"({_ITEM_WORD[m.group(1)]} {mid})"

    return _ITEM_ID.sub(repl, text)


def overview() -> dict:
    """Счётчики для шапки UI: что загружено в индекс и что есть в выбранном графе."""
    G = _load_graph()
    kinds = [d.get("type") for _, d in G.nodes(data=True)]
    changes = get_changes()
    open_tasks = sum(1 for _, d in G.nodes(data=True) if d.get("type") == "Task" and d.get("status") == "open")
    return {"meetings_indexed": len(indexed_meetings()), "graph_meetings": kinds.count("Meeting"),
            "projects": kinds.count("Project"), "decisions": kinds.count("Decision"), "tasks": kinds.count("Task"),
            "open_tasks": open_tasks,
            "open_questions": kinds.count("OpenQuestion"),
            "conflicts": len(changes["conflicts"]), "replacements": len(changes["replacements"])}


def _card(G, n: str) -> dict:
    d = G.nodes[n]
    meeting = G.nodes.get(f"meeting:{d.get('meeting_id')}", {})
    return {"id": n, "project": d.get("project", ""), "statement": d.get("statement", ""),
            "date": d.get("date", ""), "meeting_id": d.get("meeting_id", ""), "ts": d.get("ts", ""),
            "meeting_title": meeting.get("label", ""), "quote": d.get("quote", ""), "status": d.get("status", "")}


def get_changes(project: Optional[str] = None) -> dict:
    """Конфликты и замены решений — то, ради чего нужен граф. Без проекта — по всему графу."""
    G = _load_graph()
    label = None
    if project:
        pid = _project_node(G, project)
        if pid is None:
            return {"conflicts": [], "replacements": []}
        label = G.nodes[pid]["label"]
    conflicts, replacements = [], []
    for u, v, e in G.edges(data=True):
        if label and label not in (G.nodes[u].get("project"), G.nodes[v].get("project")):
            continue
        if e["type"] == "CONTRADICTS":
            a, b = sorted((_card(G, u), _card(G, v)), key=lambda c: (c["date"], c["id"]))  # раньше принятое — первым
            conflicts.append({"a": a, "b": b, "rationale": e.get("rationale", "")})
        elif e["type"] == "SUPERSEDES":
            replacements.append({"new": _card(G, u), "old": _card(G, v), "rationale": e.get("rationale", "")})
    conflicts.sort(key=lambda c: (c["a"]["date"], c["a"]["id"]))
    replacements.sort(key=lambda r: (r["new"]["date"], r["new"]["id"]))
    return {"conflicts": conflicts, "replacements": replacements}


@lru_cache(maxsize=32)
def _transcript(meeting_id: str, mtime: float) -> Optional[Transcript]:
    paths = [DEMO_DIR / f"{meeting_id}.{kind}.json" for kind in ("transcript", "meeting")]
    if not all(p.exists() for p in paths):
        return None
    return load_scribal(meeting_id, *(p.read_bytes() for p in paths))


def get_source(meeting_id: str, ts: str, quote: str, around: int = 1) -> Optional[dict]:
    """Реплика, из которой взята цитата, и соседние реплики — чтобы показать цитату в контексте.

    highlight — (начало, конец) цитаты в тексте реплики; совпадение без учёта регистра и пробелов.
    Реплика ищется по таймкоду, а если цитаты там нет — по тексту всей встречи. Встречи нет — None.
    """
    path = DEMO_DIR / f"{meeting_id}.transcript.json"
    t = _transcript(meeting_id, path.stat().st_mtime) if path.exists() else None
    if t is None:
        return None
    words = re.findall(r"\w+", quote)
    pattern = re.compile(r"\W+".join(map(re.escape, words)), re.I) if words else None

    def span(i: int) -> Optional[tuple[int, int]]:
        m = pattern.search(t.segments[i].text) if pattern else None
        return (m.start(), m.end()) if m else None

    at_ts = [i for i, s in enumerate(t.segments) if s.ts == ts]
    found = [(i, span(i)) for i in at_ts + list(range(len(t.segments)))]
    i, hl = next(((i, h) for i, h in found if h), (at_ts[0], None) if at_ts else (None, None))
    if i is None:
        return None

    def seg(j: int) -> dict:
        s = t.segments[j]
        return {"ts": s.ts, "speaker": s.speaker, "text": s.text}

    return {"meeting_id": meeting_id, "title": t.title, "date": t.date, **seg(i), "highlight": hl,
            "before": [seg(j) for j in range(max(0, i - around), i)],
            "after": [seg(j) for j in range(i + 1, min(len(t.segments), i + 1 + around))]}


_NODE_STYLE = {"Project": 'shape=box style="rounded,filled" fillcolor="#E1F5EE" color="#0F6E56"',
               "Meeting": 'shape=note style=filled fillcolor="#F1EFE8" color="#888780"',
               "Person": 'shape=ellipse style=filled fillcolor="#EEEDFE" color="#534AB7"',
               "Decision": 'shape=box style="rounded,filled" fillcolor="#E6F1FB" color="#185FA5"',
               "Task": 'shape=box style="rounded,filled" fillcolor="#FAEEDA" color="#854F0B"',
               "OpenQuestion": 'shape=box style="rounded,filled,dashed" fillcolor="#FBEAF0" color="#993556"'}
_EDGE_STYLE = {"SUPERSEDES": 'color="#5F5E5A" style=bold label="заменяет"',
               "CONTRADICTS": 'color="#E24B4A" style=bold penwidth=2 dir=both label="конфликт"',
               "REFINES": 'color="#185FA5" style=dashed label="уточняет"',
               "DEPENDS_ON": 'color="#854F0B" style=dashed label="зависит от"',
               "SAME_TOPIC": 'color="#B4B2A9" style=dotted label="та же тема"',
               "BELONGS_TO": 'color="#B4B2A9"', "STATED_IN": 'color="#D3D1C7" style=dotted',
               "ASSIGNED_TO": 'color="#7F77DD" style=dashed', "DISCUSSES": 'color="#D3D1C7" style=dotted'}


_FADED_NODE = ' fillcolor="#FAFAF8" color="#E0DED6" fontcolor="#B4B2A9" penwidth=1'
_FADED_EDGE = 'color="#E8E6DF" fontcolor="#D3D1C7"'
_CONTEXT_NODE = ' penwidth=3'


def graph_dot(project: Optional[str] = None, max_label: int = 38, highlight: Optional[dict] = None,
              focus: bool = False) -> str:
    """Граф проекта (или весь) на языке DOT для st.graphviz_chart. Заменённые решения серые
    и с пометкой, решения в конфликте — с красной рамкой, связь «конфликт» — красная.

    highlight — результат context_subgraph(): пункты, ушедшие в контекст, и стартовые сущности обведены
    жирно, остальное приглушено. focus=True — рисуются только они и проекты этих пунктов.
    """
    lit: set[str] = set()
    if highlight:
        lit = set(highlight["items"]) | set(highlight["entities"])
    if focus and lit:
        G = _load_graph()
        lit &= set(G.nodes)
        keep = lit | {v for n in lit for _, v, e in G.out_edges(n, data=True) if e["type"] == "BELONGS_TO"}
        g = _graph_of(G, keep)
    else:
        g = get_graph(project)

    def esc(text: str) -> str:
        text = text if len(text) <= max_label else text[:max_label - 1] + "…"
        return text.replace("\\", "\\\\").replace('"', '\\"')

    lines = ["digraph G {", '  rankdir=LR; bgcolor="transparent";',
             '  node [fontname="Helvetica" fontsize=10]; edge [fontname="Helvetica" fontsize=9];']
    linked = {e["source"] for e in g["edges"]} | {e["target"] for e in g["edges"]}
    for n in (n for n in g["nodes"] if n["id"] in linked):  # люди без задач — одинокие кружки, только шум
        style = _NODE_STYLE.get(n["type"], "")
        label = n.get("label", n["id"])
        if n["type"] in ITEM_TYPES:
            label = f"{n.get('date', '')} · {label}"
            if n.get("status") == "superseded":
                style += ' fillcolor="#F1EFE8" fontcolor="#888780"'
                label = f"[заменено] {label}"
            if n.get("in_conflict"):
                style += ' color="#E24B4A" penwidth=2'
        if lit:
            style += _CONTEXT_NODE if n["id"] in lit else ("" if focus else _FADED_NODE)
        lines.append(f'  "{esc(n["id"])}" [label="{esc(label)}" {style}];')
    for e in g["edges"]:
        style = _EDGE_STYLE.get(e["type"], "")
        if lit and not focus and not (e["source"] in lit and e["target"] in lit):
            style = _FADED_EDGE
        lines.append(f'  "{esc(e["source"])}" -> "{esc(e["target"])}" [{style}];')
    lines.append("}")
    return "\n".join(lines)


# ---------- Метрики (вкладка «Метрики») ----------
# Только чтение results/eval_<run_id>.json роли A: агрегаты и интервалы считает харнес (src/eval/results_v2.py),
# здесь они лишь раскладываются по таблицам. Пересчитывать бутстрап в UI нельзя: цифры разошлись бы с отчётом.

RESULTS_DIR = ROOT / "results"
EVAL_FIXTURE = ROOT / "tests" / "fixtures" / "eval_fixture.json"
QUESTION_TYPES = ("T1", "T2", "T3", "T4", "T5", "all")
QUALITY_METRICS = {"fact_f1": "F1 по фактам", "fact_recall": "полнота по фактам",
                   "fact_precision": "точность по фактам", "citation_recall": "полнота по встречам-источникам",
                   "retrieval_recall": "полнота поиска (до генерации)"}
SOURCE_METRICS = {"citation_recall": "полнота по источникам", "citation_precision": "точность по источникам",
                  "quote_validity": "цитаты найдены в транскрипте", "retrieval_recall": "полнота поиска"}
# Файлы для плашек над таблицами; формат описан в memory.md (2026-11-13, C5).
EXTRACTION_QUALITY = "extraction_quality.json"   # B: {"hallucination_rate", "n_meetings"}
INDEXING_COST = "indexing_cost.json"             # B: {"full_corpus": {"meetings", "cost_usd"}}
JUDGE_AGREEMENT = "judge_agreement.json"         # A: {"accuracy", "kappa", "n"}


def _read_json(path: Path) -> Optional[dict]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _run_info(path: Path, data: dict) -> dict:
    cfg = data.get("config") or {}
    fixture = bool(cfg.get("fixture"))
    label = f"{data.get('run_id', path.stem)} · {cfg.get('split', '?')} · {cfg.get('n_questions', '?')} вопросов"
    return {"path": str(path), "run_id": data.get("run_id", path.stem), "timestamp": data.get("timestamp") or "",
            "split": cfg.get("split"), "modes": cfg.get("modes") or [], "fixture": fixture,
            "label": label + (" · ФИКСТУРА, числа выдуманы" if fixture else "")}


def eval_runs(results_dir: Optional[Path] = None) -> list[dict]:
    """Прогоны eval из results/, свежие первыми. Если настоящих нет — фикстура роли A с пометкой."""
    runs = []
    for p in Path(results_dir or RESULTS_DIR).glob("eval_*.json"):
        data = _read_json(p)
        if data and "aggregate" in data:
            runs.append(_run_info(p, data))
    runs.sort(key=lambda r: (r["timestamp"], r["run_id"]), reverse=True)
    if not runs and EVAL_FIXTURE.exists():
        data = _read_json(EVAL_FIXTURE)
        if data:
            runs.append(_run_info(EVAL_FIXTURE, data))
    return runs


def load_eval(path: str | Path) -> dict:
    data = _read_json(Path(path))
    if data is None or "aggregate" not in data:
        raise ValueError(f"{path}: не json-результат eval (нет aggregate)")
    return data


def fmt_interval(cell: Optional[dict], spec: str = ".2f") -> str:
    """{mean, ci_low, ci_high, n, note} → «0.61 [0.33; 0.85] n=8». «—» — не измерено (null), а не ноль."""
    if not cell or cell.get("mean") is None:
        return "—"
    out = format(cell["mean"], spec)
    if cell.get("ci_low") is not None and cell.get("ci_high") is not None:
        out += f" [{format(cell['ci_low'], spec)}; {format(cell['ci_high'], spec)}]"
    return out + f" n={cell.get('n', '?')}" + (" *" if cell.get("note") else "")


def _fmt_diff(cell: Optional[dict]) -> str:
    if not cell or cell.get("diff_mean") is None:
        return "—"
    return f"{cell['diff_mean']:+.2f} [{cell['ci_low']:+.2f}; {cell['ci_high']:+.2f}]"


def _modes(result: dict, section: dict) -> list[str]:
    return (result.get("config") or {}).get("modes") or list(section)


def _type_title(t: str) -> str:
    return "все типы" if t == "all" else t


def quality_table(result: dict, metric: str = "fact_f1") -> dict:
    """Главная таблица: режимы по строкам, типы вопросов по колонкам, в ячейке среднее [95% интервал].

    rows — по режиму; paired — разность graph − baseline по типам с выводом бутстрапа; measured — есть ли хоть
    одно значение (без --judge-facts факты не измерены, и это надо сказать словами, а не нулями).
    """
    agg = result["aggregate"]
    by = agg.get("by_mode_type") or {}
    present = {t for groups in by.values() for t in groups}
    types = [t for t in QUESTION_TYPES if t in present]
    rows, measured, notes = [], False, set()
    for mode in _modes(result, by):
        row = {"режим": mode}
        for t in types:
            cell = (by.get(mode) or {}).get(t, {}).get(metric)
            measured |= bool(cell and cell.get("mean") is not None)
            if cell and cell.get("note"):
                notes.add(cell["note"])
            row[_type_title(t)] = fmt_interval(cell)
        rows.append(row)
    paired = []
    for t in types:
        cell = (agg.get("paired") or {}).get(t, {}).get(metric)
        if cell:
            paired.append({"тип": _type_title(t), "graph − baseline": _fmt_diff(cell),
                           "пар вопросов": cell.get("n"), "вывод": cell.get("verdict") or ""})
    return {"rows": rows, "paired": paired, "measured": measured, "notes": sorted(notes)}


def sources_table(result: dict, qtype: str = "all") -> list[dict]:
    by = result["aggregate"].get("by_mode_type") or {}
    return [{"режим": m, **{title: fmt_interval((by.get(m) or {}).get(qtype, {}).get(k))
                            for k, title in SOURCE_METRICS.items()}} for m in _modes(result, by)]


def cost_latency(result: dict) -> list[dict]:
    """Стоимость и задержка ответа по режимам.

    Если харнес положил aggregate.cost_latency[mode] = {cost_usd, latency_ms} (ячейки с mean) — берём его.
    Иначе простое среднее по ответам со статусом ok: это арифметика над записанными числами, не оценка качества,
    интервал к ней не заявляется. Токены промпта — aggregate.by_mode_type[mode]["all"]["context_tokens"].
    """
    agg = result["aggregate"]
    by = agg.get("by_mode_type") or {}
    given = agg.get("cost_latency") or {}
    out = []
    for m in _modes(result, by):
        if m in given:
            cost = (given[m].get("cost_usd") or {}).get("mean")
            lat = (given[m].get("latency_ms") or {}).get("mean")
            n = (given[m].get("cost_usd") or {}).get("n")
        else:
            rows = [r for r in result.get("per_question") or [] if r.get("mode") == m and r.get("status") == "ok"]
            costs = [r["usage"]["cost_usd"] for r in rows if (r.get("usage") or {}).get("cost_usd") is not None]
            lats = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
            cost = sum(costs) / len(costs) if costs else None
            lat = sum(lats) / len(lats) if lats else None
            n = len(rows)
        ctx = ((by.get(m) or {}).get("all") or {}).get("context_tokens") or {}
        out.append({"режим": m, "cost_usd": cost, "latency_ms": lat, "context_tokens": ctx.get("mean"), "n": n})
    return out


def lang_table(result: dict, metric: str = "fact_f1") -> list[dict]:
    by = result["aggregate"].get("by_mode_lang") or {}
    modes = _modes(result, by)
    langs = sorted({g for m in modes for g in (by.get(m) or {})})
    return [{"язык": g, **{m: fmt_interval((by.get(m) or {}).get(g, {}).get(metric)) for m in modes}}
            for g in langs]


def abstention_table(result: dict) -> list[dict]:
    ab = result["aggregate"].get("abstention") or {}
    titles = {"abstain_rate_unanswerable": "честные отказы, когда ответа нет в данных (выше лучше)",
              "false_abstain_rate": "ложные отказы, когда ответ есть (ниже лучше)"}
    return [{"метрика": title, **{m: fmt_interval((ab.get(m) or {}).get(k)) for m in _modes(result, ab)}}
            for k, title in titles.items()]


def metric_cards(results_dir: Optional[Path] = None) -> list[dict]:
    """Плашки над таблицами: {title, value | None, hint}. None — файла ещё нет, в hint сказано, чей он."""
    folder = Path(results_dir or RESULTS_DIR)
    ext = _read_json(folder / EXTRACTION_QUALITY) or {}
    cost = (_read_json(folder / INDEXING_COST) or {}).get("full_corpus") or {}
    judge = _read_json(folder / JUDGE_AGREEMENT) or {}
    hal, idx = ext.get("hallucination_rate"), cost.get("cost_usd")
    acc, kappa = judge.get("accuracy"), judge.get("kappa")
    return [
        {"title": "Выдуманные цитаты извлечения", "value": None if hal is None else f"{hal:.1%}",
         "hint": (f"по {ext.get('n_meetings', '?')} встречам" if hal is not None
                  else f"нет results/{EXTRACTION_QUALITY} (роль B, validate_extraction)")},
        {"title": "Индексация графа, весь корпус", "value": None if idx is None else f"${idx:.3f}",
         "hint": (f"{cost.get('meetings', '?')} встреч: извлечение и связи" if idx is not None
                  else f"нет results/{INDEXING_COST} (роль B, B8)")},
        {"title": "Судья совпадает с человеком", "value": None if acc is None else f"{acc:.0%}",
         "hint": (f"каппа {kappa:.2f}, n={judge.get('n', '?')}" if kappa is not None
                  else f"нет results/{JUDGE_AGREEMENT} (роль A, A8)")},
    ]
