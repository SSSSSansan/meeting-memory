"""B6. Сборка графа из data/raw + извлечения: адаптер → имена говорящих → resolve → граф.

    python scripts/build_graph.py [--extracted data/extracted_llm_raw] [--out data/graph_raw/graph.json]
Транскрипты кладёт в data/transcripts_raw/ (не трогает data/transcripts/), алиасы — в data/graph_raw/aliases.json.
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
from src.schemas import Extraction  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--extracted", default="data/extracted_llm_raw")
ap.add_argument("--out", default="data/graph_raw/graph.json")
a = ap.parse_args()

tr_dir = ROOT / "data/transcripts_raw"
tr_dir.mkdir(exist_ok=True)
people, projects, known = [], [], []
for f in sorted((ROOT / "data/raw").glob("*.transcript.json")):
    mid = f.name.split(".")[0]
    t = apply_speakers(convert_files(f, f.with_name(f"{mid}.meeting.json"), mid))
    (tr_dir / f"{mid}.json").write_text(t.model_dump_json(indent=2, exclude_none=True), encoding="utf-8")
    known += t.participants
for f in sorted((ROOT / a.extracted).glob("*.json")):
    e = Extraction.model_validate_json(f.read_text(encoding="utf-8"))
    items = [*e.decisions, *e.tasks, *e.open_questions]
    projects += [*e.projects_discussed, *(i.project for i in items)]
    people += [i.owner for i in e.tasks if i.owner] + [i.made_by for i in e.decisions if i.made_by]

aliases = {**resolve_projects(projects), **resolve_people(people, sorted(set(known)))}
aliases = {k: v for k, v in aliases.items() if k != v}
(ROOT / "data/graph_raw").mkdir(exist_ok=True)
(ROOT / "data/graph_raw/aliases.json").write_text(json.dumps(aliases, ensure_ascii=False, indent=2), encoding="utf-8")
G = build_graph(ROOT / a.extracted, tr_dir, [], aliases)
save_graph(G, ROOT / a.out)
types = {}
for _, d in G.nodes(data=True):
    types[d["type"]] = types.get(d["type"], 0) + 1
print(dict(sorted(types.items())), "алиасов:", len(aliases), "→", a.out)
