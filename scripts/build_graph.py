"""B6. Сборка графа из data/raw + извлечения: адаптер → имена говорящих → resolve → граф.

    python scripts/build_graph.py [--extracted data/extracted_llm_raw] [--out data/graph_raw/graph.json]
    python scripts/build_graph.py --meetings m_001 m_002 ... m_008 --out data/graph/graph.json
    python scripts/build_graph.py --links data/graph/links.json   # подтянуть связи B5 (SUPERSEDES/...)
Транскрипты кладёт в data/transcripts_raw/ (не трогает data/transcripts/). Алиасы пишутся рядом с --out
(«<папка out>/aliases.json»), поэтому обычный вызов не трогает data/graph_raw/aliases.json при другом --out.
--meetings ограничивает и исходные транскрипты, и файлы извлечения этим списком meeting_id (для графа без
утечки будущих встреч, например для эталона роли A по as_of m_008). --links — результат `src.graph.link`
(контракт 4.3); решения вне выбранных --meetings в связях просто не найдутся в графе и будут пропущены.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.graph.builder import build_graph, save_graph  # noqa: E402
from src.ingest.scribal_adapter import convert_files  # noqa: E402
from src.ingest.speakers import apply_speakers  # noqa: E402
from src.resolve.entities import resolve_people, resolve_projects  # noqa: E402
from src.schemas import Extraction, Link  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--extracted", default="data/extracted_llm_raw")
ap.add_argument("--out", default="data/graph_raw/graph.json")
ap.add_argument("--meetings", nargs="*", default=None, help="ограничить списком meeting_id, например m_001 m_002")
ap.add_argument("--links", default=None, help="путь к links.json (контракт 4.3), опционально")
a = ap.parse_args()
allowed = set(a.meetings) if a.meetings else None
links = [Link.model_validate(ln) for ln in json.loads((ROOT / a.links).read_text(encoding="utf-8"))] if a.links else []

tr_dir = ROOT / "data/transcripts_raw"
tr_dir.mkdir(exist_ok=True)
people, projects, known, known_projects = [], [], [], []
for f in sorted((ROOT / "data/raw").glob("*.transcript.json")):
    mid = f.name.split(".")[0]
    if allowed is not None and mid not in allowed:
        continue
    t = apply_speakers(convert_files(f, f.with_name(f"{mid}.meeting.json"), mid))
    (tr_dir / f"{mid}.json").write_text(t.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")
    known += t.participants
    title_project = t.title.split(":")[0].strip()
    if title_project:
        known_projects.append(title_project)
for f in sorted((ROOT / a.extracted).glob("*.json")):
    if allowed is not None and f.stem not in allowed:
        continue
    e = Extraction.model_validate_json(f.read_text(encoding="utf-8"))
    items = [*e.decisions, *e.tasks, *e.open_questions]
    projects += [*e.projects_discussed, *(i.project for i in items)]
    people += [i.owner for i in e.tasks if i.owner] + [i.made_by for i in e.decisions if i.made_by]

aliases = {**resolve_projects(projects, sorted(set(known_projects))),
           **resolve_people(people, sorted(set(known)))}
aliases = {k: v for k, v in aliases.items() if k != v}
out_path = ROOT / a.out
out_path.parent.mkdir(parents=True, exist_ok=True)
(out_path.parent / "aliases.json").write_text(json.dumps(aliases, ensure_ascii=False, indent=2), encoding="utf-8")
extracted_dir = ROOT / a.extracted
if allowed is not None:
    # build_graph сам глобит *.json по папке — фильтруем через временную подпапку с нужными файлами
    extracted_dir = out_path.parent / "_extracted_filtered"
    extracted_dir.mkdir(exist_ok=True)
    for old in extracted_dir.glob("*.json"):
        old.unlink()
    for f in sorted((ROOT / a.extracted).glob("*.json")):
        if f.stem in allowed:
            (extracted_dir / f.name).write_text(f.read_text(encoding="utf-8"), encoding="utf-8")
G = build_graph(extracted_dir, tr_dir, links, aliases)
save_graph(G, out_path)
if allowed is not None:
    import shutil
    shutil.rmtree(extracted_dir)
types = {}
for _, d in G.nodes(data=True):
    types[d["type"]] = types.get(d["type"], 0) + 1
print(dict(sorted(types.items())), "алиасов:", len(aliases), "→", a.out)
