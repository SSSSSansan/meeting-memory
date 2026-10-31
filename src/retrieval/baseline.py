"""РОЛЬ C. Baseline RAG: чанки сырых транскриптов → эмбеддинги → top-k.

TODO (день 1): чанкование с таймкодами и meeting_id, индекс, search(question, k).
"""


def search(question: str, k: int = 5) -> list[dict]:
    raise NotImplementedError("Роль C")
