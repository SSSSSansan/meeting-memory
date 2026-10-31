# Meeting Memory (K02)

Система памяти встреч: из транскриптов извлекает решения, задачи и договорённости,
связывает их между встречами и проектами и отвечает на вопросы со ссылками на источники.
Главный результат — честное сравнение **baseline RAG** и **GraphRAG** на одних данных.

Полный контекст проекта, схемы, промпты, метрики и план по дням:
[`docs/K02_MEETING_MEMORY_CONTEXT.md`](docs/K02_MEETING_MEMORY_CONTEXT.md).
**Если работаете с LLM — отдайте ей этот файл целиком и назовите свою роль.**

## Быстрый старт

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # LLM_BACKEND=stub работает без ключей
python -m pytest -q           # тесты каркаса
python -m scripts.validate_extraction
```

## Роли

| Роль | Кто | Папки |
|---|---|---|
| A. Данные и оценка | _имя_ | `data/transcripts`, `data/gold`, `src/eval` |
| B. Извлечение и граф | _имя_ | `src/extract`, `src/resolve`, `src/graph` |
| C. Поиск, ответы, приложение | _имя_ | `src/retrieval`, `src/answer`, `app` |

Общий контракт — `src/schemas.py`. Менять только по согласованию всех троих.
Вклад каждого фиксируется в этом разделе к сдаче.

## LLM backend

Весь код ходит в модель только через `src/llm_client.py`.
`LLM_BACKEND=stub|openai|local` в `.env`. Расходы ограничены `BUDGET_USD`.
Ключи хранить только в `.env` (он в `.gitignore`).

## Работа в git

- Ветка на человека: `feat/role-a-data`, `feat/role-b-graph`, `feat/role-c-retrieval`.
- В `main` через pull request, коммиты маленькие и частые.
- `src/schemas.py` и `data/gold/what_is_a_decision.md` правятся только отдельным PR с одобрением всех.
