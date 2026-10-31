"""РОЛЬ C. GraphRAG: вопрос → сущности → обход графа → подграф в контекст.

TODO (день 3): читает data/graph/graph.json (его пишет роль B).
До готовности графа тестируйте на заглушке из data/extracted/.
"""


def retrieve_subgraph(question: str, depth: int = 2) -> dict:
    raise NotImplementedError("Роль C")
