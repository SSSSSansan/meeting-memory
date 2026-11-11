import pytest


@pytest.fixture(autouse=True)
def _never_call_paid_model(monkeypatch):
    # llm_client читает .env при импорте: с LLM_BACKEND=openai в .env тесты без своей фикстуры
    # ходили в OpenAI и тратили бюджет. Тесты, которым нужен другой backend, ставят его сами.
    monkeypatch.setenv("LLM_BACKEND", "stub")
