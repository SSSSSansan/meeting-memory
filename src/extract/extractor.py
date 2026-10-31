"""РОЛЬ B. Извлечение решений/задач/вопросов из транскрипта через LLM.

TODO (день 1–2):
  - промпт из docs/K02_MEETING_MEMORY_CONTEXT.md, раздел 11.1
  - llm_client.complete(..., json_schema=...) → Extraction.model_validate_json
  - retry при невалидном JSON с показом ошибки модели
  - id формата <тип>_<meeting_id>_<NN>, детерминированно
  - сохранить в data/extracted/<meeting_id>.json
"""
from src.schemas import Extraction, Transcript


def extract(transcript: Transcript) -> Extraction:
    raise NotImplementedError("Роль B: реализовать извлечение")
