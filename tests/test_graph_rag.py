from pathlib import Path

import pytest

import src.answer.answerer as ans
from src.graph.builder import _load_links, build_graph
from src.graph.queries import load_graph
from src.retrieval.graph_rag import (_stem_match, find_entities, last_n_meetings,
                                     retrieve_subgraph)
from src.schemas import Link

ROOT = Path(__file__).resolve().parent.parent


def _graph(extra_links=()):
    links = _load_links(ROOT / "data/sample/links.json") + list(extra_links)
    return build_graph(ROOT / "data/extracted", ROOT / "data/transcripts", links)


CROSS_CONFLICT = Link.model_validate({"from": "d_m002_02", "to": "d_m002_01", "type": "contradicts",
                                      "confidence": 0.8, "rationale": "Айгерим нужна и на релизе Атласа"})


@pytest.mark.parametrize("label, word, ok", [
    ("атлас", "атласу", True), ("горизонт", "горизонту", True), ("айгерим", "айгерим", True),
    ("нурланова", "нурлановой", True), ("атлас", "атом", False), ("ким", "кимберли", False),
    ("ким", "ким", True), ("горизонт", "гор", False),
])
def test_stem_match(label, word, ok):
    assert _stem_match(label, word) is ok


def test_entities_from_inflected_question():
    G = _graph()
    assert find_entities(G, "Что решили по Атласу?") == ["project:Атлас"]
    assert find_entities(G, "Что решали по Горизонту на встречах про Атлас?") == ["project:Атлас", "project:Горизонт"]
    assert find_entities(G, "Сколько задач у Айгерим?") == ["person:Айгерим Нурланова"]
    assert find_entities(G, "Есть ли конфликты по загрузке людей?") == []


@pytest.mark.parametrize("q, n", [("за последние три встречи", 3), ("за последние 2 встречи", 2),
                                  ("последних двух встреч", 2), ("за всё время", None)])
def test_last_n_meetings(q, n):
    assert last_n_meetings(q) == n


def test_t2_evolution_is_chronological_with_supersedes():
    sub = retrieve_subgraph("Как менялся срок релиза Атласа?", G=_graph())
    ctx = sub["context"]
    assert ctx.index("d_m001_01") < ctx.index("d_m002_01")
    assert "ЗАМЕНЕНО" in ctx and "заменено решением d_m002_01" in ctx and "заменяет d_m001_01" in ctx
    assert "[m_001 00:04:12] цитата: перенесём релиз на двадцатое апреля" in ctx


def test_t3_cross_project_items_come_from_meetings_about_other_project():
    sub = retrieve_subgraph("Что решали по Горизонту?", G=_graph())
    assert set(sub["items"]) == {"q_m001_01", "d_m002_02"}
    assert "«Еженедельный статус — Атлас»" in sub["context"]   # встреча называется про Атлас


def test_t5_open_items_are_marked():
    ctx = retrieve_subgraph("Что по Атласу ещё не закрыто?", G=_graph())["context"]
    assert "Задача t_m001_01" in ctx and "[открыта; исполнитель: Айгерим Нурланова; срок: 2026-02-10]" in ctx
    assert "[выполнена; исполнитель: Айгерим Нурланова]" in ctx


def test_depth_controls_cross_project_links():
    G = _graph([CROSS_CONFLICT])
    assert "d_m002_02" not in retrieve_subgraph("Что по Атласу?", depth=1, G=G)["items"]
    sub = retrieve_subgraph("Что по Атласу?", depth=2, G=G)
    assert "d_m002_02" in sub["items"]
    assert "В КОНФЛИКТЕ" in sub["context"] and "Айгерим нужна и на релизе Атласа" in sub["context"]
    with pytest.raises(ValueError):
        retrieve_subgraph("Что по Атласу?", depth=0, G=G)


def test_traversal_does_not_leak_through_meetings_or_other_projects():
    # обе встречи обсуждали и Атлас, но без связей между решениями Атлас в подграф не попадает
    assert set(retrieve_subgraph("Что по Горизонту?", depth=3, G=_graph())["items"]) == {"q_m001_01", "d_m002_02"}


def test_person_question_finds_assigned_and_mentioned_items():
    items = retrieve_subgraph("Какая нагрузка у Айгерим?", G=_graph())["items"]
    assert set(items) == {"t_m001_01", "t_m002_01", "d_m002_02"}


def test_last_meetings_filter():
    sub = retrieve_subgraph("Что решили по Атласу за последние 1 встречи?", G=_graph())
    assert sub["last_meetings"] == 1 and all(i.startswith(("d_m002", "t_m002")) for i in sub["items"])
    assert "этого пункта нет в контексте" in sub["context"]


def test_no_entities_falls_back_with_conflicts_first():
    sub = retrieve_subgraph("Есть ли конфликты по загрузке людей?", G=_graph([CROSS_CONFLICT]))
    assert sub["fallback"] and sub["entities"] == []
    assert {"d_m002_01", "d_m002_02"} <= set(sub["items"])


def test_graph_context_quotes_pass_answer_verification():
    ctx = retrieve_subgraph("Как менялся срок релиза Атласа?", G=_graph())["context"]
    cits = [ans.Citation(meeting_id="m_002", ts="00:01:05", quote="Переносим релиз Атласа на пятое мая"),
            ans.Citation(meeting_id="m_002", ts="00:01:05", quote="Релиз Атласа переносится на 5 мая")]
    good, bad, _ = ans._verify(cits, ctx)
    assert [c.quote for c in good] == ["Переносим релиз Атласа на пятое мая"]
    assert [c.quote for c in bad] == ["Релиз Атласа переносится на 5 мая"]   # «Суть» — пересказ, не источник


def test_committed_sample_graph_works():
    sub = retrieve_subgraph("Как менялся срок релиза Атласа?", G=load_graph(ROOT / "data/sample/graph.json"))
    assert sub["items"][:1] == ["d_m001_01"]


# ---------- граф роли B и варианты имён ----------

import src.retrieval.graph_rag as gr  # noqa: E402
from src.retrieval.graph_rag import default_graph_path, get_aliases  # noqa: E402

B_GRAPH = ROOT / "data/graph_raw/graph.json"
needs_b_graph = pytest.mark.skipif(not B_GRAPH.exists(), reason="граф роли B ещё не собран")


def test_default_graph_path_order(monkeypatch, tmp_path):
    monkeypatch.delenv("MEETING_GRAPH", raising=False)
    contract, b_out, sample = tmp_path / "graph/graph.json", tmp_path / "graph_raw/graph.json", tmp_path / "s.json"
    monkeypatch.setattr(gr, "GRAPH_CANDIDATES", [contract, b_out, sample])
    sample.write_text("{}")
    assert default_graph_path() == sample
    b_out.parent.mkdir()
    b_out.write_text("{}")
    assert default_graph_path() == b_out
    contract.parent.mkdir()
    contract.write_text("{}")
    assert default_graph_path() == contract
    monkeypatch.setenv("MEETING_GRAPH", "data/sample/graph.json")
    assert default_graph_path() == ROOT / "data/sample/graph.json"


def test_aliases_are_read_next_to_graph(tmp_path):
    (tmp_path / "aliases.json").write_text('{"Коля": "Николай"}', encoding="utf-8")
    assert get_aliases(tmp_path / "graph.json") == {"Коля": "Николай"}
    assert get_aliases(tmp_path / "other" / "graph.json") == {}


def test_alias_variant_is_found_with_inflection():
    G = _graph()
    assert find_entities(G, "Что поручили Айке?", {"Айка": "Айгерим Нурланова"}) == ["person:Айгерим Нурланова"]
    assert find_entities(G, "Что поручили Айке?") == []


def test_sentence_initial_word_is_not_fuzzy_matched():
    G = _graph()
    assert find_entities(G, "Горизонт: что решили?") == ["project:Горизонт"]   # точное имя в начале — по основе
    assert gr._proper_nouns("Жалко, что Жол не успели. Коля сказал") == ["Жол"]


@needs_b_graph
@pytest.mark.parametrize("question, entity", [
    ("Что решили по Жол?", "project:Zhol"),                 # кириллица → Zhol через resolve роли B
    ("Что решили по проекту Жолға?", "project:Zhol"),       # казахское окончание
    ("Что по Камкор ещё не закрыто?", "project:Qamqor"),
    # алиас из aliases.json + склонение (на синтетических данных) покрыт
    # test_alias_variant_is_found_with_inflection — здесь убрано, т.к. extraction больше не
    # пишет диминутив «Коля» ни в одной встрече, aliases.json на реальном графе теперь {}
    ("Что поручили Нуралы?", "person:Нұралы"),              # без казахской буквы
])
def test_role_b_graph_variants(question, entity):
    path = B_GRAPH
    assert entity in find_entities(load_graph(path), question, get_aliases(path))


@needs_b_graph
def test_role_b_graph_context_passes_citation_check():
    sub = retrieve_subgraph("Как менялся срок релиза Zhol?", G=load_graph(B_GRAPH))
    assert sub["entities"] == ["project:Zhol"] and sub["items"]
    first = sub["items"][0]
    node = load_graph(B_GRAPH).nodes[first]
    good, bad, _ = ans._verify([ans.Citation(meeting_id=node["meeting_id"], ts=node["ts"], quote=node["quote"])],
                               sub["context"])
    assert len(good) == 1 and bad == []
