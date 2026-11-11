# Meeting Memory (K02)

Система памяти встреч: из транскриптов извлекает решения, задачи и договорённости,
связывает их между встречами и проектами и отвечает на вопросы со ссылками на источники.
Главный результат — честное сравнение **baseline RAG** и **GraphRAG** на одних данных.

Полный контекст проекта, схемы, промпты, метрики и план по дням:
[`docs/K02_MEETING_MEMORY_CONTEXT.md`](docs/K02_MEETING_MEMORY_CONTEXT.md).
**Если работаете с LLM — отдайте ей этот файл целиком и назовите свою роль.**

## Быстрый старт

```bash
python3 -m venv .venv && source .venv/bin/activate     # Windows: python -m venv .venv; .venv\Scripts\activate
pip install -r requirements.txt ruff
cp .env.example .env          # LLM_BACKEND=stub работает без ключей
ruff check .                  # линтер (настройки — pyproject.toml)
python -m pytest -q           # тесты
python -m scripts.validate_extraction
streamlit run app/streamlit_app.py
```

CI (`.github/workflows/ci.yml`) гоняет `ruff check .` и `pytest -q` на Python 3.11 и 3.13 для каждого PR и push в `main`.
Перед PR запустите обе команды локально — красный CI блокирует слияние только по договорённости, не технически.

## Роли

| Роль | Кто | Папки |
|---|---|---|
| A. Данные и оценка | _имя_ | `data/transcripts`, `data/gold`, `src/eval` |
| B. Извлечение и граф | _имя_ | `src/extract`, `src/resolve`, `src/graph` |
| C. Поиск, ответы, приложение | _имя_ | `src/retrieval`, `src/answer`, `app`, `src/api.py`, `src/llm_client.py`, `scripts/cost_guard.py` |

Общий контракт — `src/schemas.py`. Менять только по согласованию всех троих.
Вклад каждого фиксируется в этом разделе к сдаче.

### Вклад роли C (на 7 ноября)

- **Baseline RAG** (C1, C2): чанки-окна из реплик с перекрытием, векторный индекс с инкрементальным добавлением батчей, ответ в едином формате 4.6.
- **Проверка цитат** (C3): каждая цитата ответа сверяется с источником дословно; один повтор с обратной связью, неподтверждённые цитаты удаляются с пометкой; доля невалидных цитат — в `runs/answer_report.jsonl`.
- **GraphRAG** (C4): сущности из вопроса → обход графа на `depth` шагов → контекст с хронологией, статусами и связями решений; тот же промпт и та же проверка цитат, что у baseline.
- **Роутер** (C5): `mode=auto` — граф для вопросов через несколько встреч, baseline для фактов; правила для ru и kk, без LLM.
- **Streamlit** (C6): загрузка файлов Scribal, индексация батчами по 5, вопросы в режимах auto / baseline / graph.
- **LLM-клиент и бюджет** (P1, P2): один вход в модель, кэш, лимит расходов, логи токенов и стоимости.
- **CI** (T0.3): линтер и тесты на каждый PR.

Подробный статус задач — в [`TEAM.md`](TEAM.md), решения и известные проблемы — в [`memory.md`](memory.md).

## LLM backend

Весь код ходит в модель только через `src/llm_client.py`.
`LLM_BACKEND=stub|openai|local` в `.env`. Расходы ограничены `BUDGET_USD`.
Ключи хранить только в `.env` (он в `.gitignore`).

| Переменная | Значение по умолчанию | Зачем |
|---|---|---|
| `LLM_BACKEND` | `stub` | `stub` — без модели и денег; `openai`; `local` — OpenAI-совместимый сервер (vLLM, Ollama) |
| `LLM_MODEL`, `EMBED_MODEL` | — | модели чата и эмбеддингов |
| `LOCAL_BASE_URL`, `LOCAL_API_KEY` | `http://localhost:8000/v1` | адрес локального сервера |
| `PRICE_IN_PER_M`, `PRICE_OUT_PER_M`, `PRICE_EMBED_PER_M` | `0` | цены за 1M токенов — брать актуальные у провайдера, иначе учёт расходов врёт |
| `BUDGET_USD` | `3.0` | жёсткий лимит: при превышении вызовы падают с `BudgetExceeded` |
| `LLM_MAX_RETRIES`, `LLM_TIMEOUT_S` | `3`, `120` | повторы при сбоях сети, 429 и 5xx |
| `MEETING_GRAPH` | — | какой граф читает режим graph; без неё — первый существующий из `data/graph/graph.json`, `data/graph_raw/graph.json`, `data/sample/graph.json` |

Каждый вызов модели и эмбеддингов пишется в `runs/calls.jsonl` (`purpose`, токены, `cost_usd`, `spent_usd`, задержка; ответы из кэша — с `spent_usd = 0`).
Потрачено и остаток бюджета: `python -m scripts.cost_guard --status`.

## Интерфейсы роли C

UI и оценка ходят только сюда; логики в `app/` нет.

```python
from src import api

pairs, errors = api.pair_scribal_files({"m_001.transcript.json": b"...", "m_001.meeting.json": b"..."})
transcripts = [api.load_scribal(*p) for p in pairs]       # Scribal raw + meta → Transcript (адаптер роли B)
for done, total in api.index_batches(transcripts): ...    # батчи по 5, индекс сохраняется после каждого
api.ask("Как менялся срок релиза Zhol?", mode="auto")     # → src.schemas.Answer (формат 4.6)
api.route("Что по Qamqor ещё не закрыто?")                # какую ветку выберет auto и почему
api.graph_info()                                           # какой граф читает режим graph
api.list_projects()                                        # проекты в графе
api.get_timeline("Жол")                                   # пункты проекта по времени: статус, связи, цитата
api.get_graph("Zhol")                                      # {"nodes", "edges"} для рисунка; без проекта — весь граф
```

| Функция | Что делает |
|---|---|
| `src.llm_client.complete(messages, *, json_schema=None, temperature=0.0, max_tokens=1024, purpose="generic") -> LLMResponse` | один вызов чат-модели: кэш, бюджет, лог |
| `src.llm_client.embed(texts, *, purpose="embed") -> list[list[float]]` | эмбеддинги |
| `src.answer.answerer.answer(question, mode="baseline", k=5, depth=2) -> Answer` | ответ с проверенными цитатами; `mode`: `baseline` / `graph` / `auto` |
| `src.retrieval.baseline.BaselineIndex` | `add(transcripts)`, `search(question, k)`, `save()` / `load()` |
| `src.retrieval.graph_rag.retrieve_subgraph(question, depth=2) -> dict` | подграф по вопросу и контекст для модели |
| `src.retrieval.router.route(question) -> Route` | выбор ветки для `auto` |
| `src.api.get_timeline(project) -> list[dict]` | решения, задачи и открытые вопросы проекта по времени; у каждого `status`, `in_conflict`, `links` (SUPERSEDES, CONTRADICTS, …), цитата и встреча |
| `src.api.get_graph(project=None) -> dict` | узлы и рёбра графа для рисунка (без рёбер «участвовал во встрече») |

Командная строка:

```bash
python -m src.answer.answerer "Как менялся срок релиза Zhol?" --mode auto
python -m src.retrieval.graph_rag "Что по Камкор ещё не закрыто?" --depth 2
python -m src.retrieval.baseline "Кто делает миграцию?" -k 5
```

Режим graph читает граф роли B (`data/graph_raw/graph.json`) и его `aliases.json`: вопрос можно писать с другим написанием имён («Жол», «Камкор», «Коля»). Ограничения сейчас: в графе B ещё нет связей между решениями (B5); на `LLM_BACKEND=stub` поиск и ответы по смыслу случайны.

## Работа в git

Правила — в [`GIT.md`](GIT.md): ветка `<a|b|c>/<описание>`, PR в `main`, в заголовке PR помечаем критичные файлы.
Остальные файлы для команды: [`TEAM.md`](TEAM.md) (задачи), [`CLAUDE.md`](CLAUDE.md), [`SKILLS.md`](SKILLS.md), [`TESTING.md`](TESTING.md), [`memory.md`](memory.md).
