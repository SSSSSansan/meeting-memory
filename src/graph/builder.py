"""РОЛЬ B. Сборка графа NetworkX и классификация связей между решениями.

TODO (день 3–4):
  - узлы Meeting/Project/Person/Decision/Task/OpenQuestion (контекст, 4.4)
  - кандидаты на связи: эмбеддинг-поиск по решениям того же и смежных проектов
  - классификация типа связи (промпт 11.2), порог LINK_CONFIDENCE_THRESHOLD
  - статус решения: superseded, если есть входящее ребро SUPERSEDES, иначе active
  - сериализация: data/graph/graph.json (networkx.node_link_data)
"""
import networkx as nx


def build_graph(extraction_dir: str) -> nx.MultiDiGraph:
    raise NotImplementedError("Роль B")
