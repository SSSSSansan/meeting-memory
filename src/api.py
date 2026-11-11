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
from src.retrieval.baseline import DEFAULT_INDEX_DIR, ROOT, BaselineIndex
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
                 source: str = "scribal") -> Transcript:
    raw = ScribalTranscript.model_validate(json.loads(transcript_json))
    meta = ScribalMeeting.model_validate(json.loads(meeting_json))
    return convert(raw, meta, meeting_id, source)


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


def ask(question: str, mode: str = "baseline", k: int = 5, depth: int = 2) -> Answer:
    """Одна точка для UI и eval. baseline ищет по индексу встреч, загруженных в UI;
    graph — по графу роли B (graph_info()); auto — ветку выбирает роутер (см. route()).
    Какой веткой реально ответили — Answer.mode."""
    return answer(question, mode=mode, k=k, depth=depth, index=get_index())


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
    nodes = [{"id": n, **{k: v for k, v in G.nodes[n].items() if k != "participants"}} for n in sorted(keep)]
    edges = [{"source": u, "target": v, **e} for u, v, e in G.edges(data=True)
             if u in keep and v in keep and e["type"] != "PARTICIPATED_IN"]
    return {"nodes": nodes, "edges": edges}


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
    # имена говорящих — где эвристика роли B уверена; иначе остаётся SPEAKER_XX
    return apply_speakers(load_scribal(meeting_id, *(p.read_bytes() for p in paths)))


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


def graph_dot(project: Optional[str] = None, max_label: int = 38) -> str:
    """Граф проекта (или весь) на языке DOT для st.graphviz_chart. Заменённые решения серые
    и с пометкой, решения в конфликте — с красной рамкой, связь «конфликт» — красная."""
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
        lines.append(f'  "{esc(n["id"])}" [label="{esc(label)}" {style}];')
    for e in g["edges"]:
        lines.append(f'  "{esc(e["source"])}" -> "{esc(e["target"])}" [{_EDGE_STYLE.get(e["type"], "")}];')
    lines.append("}")
    return "\n".join(lines)
