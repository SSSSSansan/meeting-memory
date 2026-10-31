"""РОЛЬ C. Генерация ответа с цитатами. ОДИН промпт для baseline и graph (раздел 11.3).

Возвращает src.schemas.Answer — единый формат для обеих веток.
"""
from src.schemas import Answer


def answer(question: str, mode: str = "graph") -> Answer:
    raise NotImplementedError("Роль C")
