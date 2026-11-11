# Meeting Memory (K02)

Память встреч для Scribal: из транскриптов встреч система извлекает решения, задачи и открытые вопросы,
связывает их между встречами и проектами и отвечает на вопросы руководителя со ссылками на встречу,
таймкод и дословную цитату.

Главный результат проекта — **честное сравнение двух архитектур на одних данных**:

- **baseline RAG** — поиск по фрагментам транскриптов, модель отвечает по найденному;
- **GraphRAG** — модель заранее извлекает решения и задачи, из них строится граф со связями
  «заменяет» и «противоречит», поиск идёт по графу.

Обе ветки отвечают одним промптом и проходят одну проверку цитат, поэтому разница в качестве — от поиска,
а не от формулировок. Полный контекст задания, схемы и план — [`docs/K02_MEETING_MEMORY_CONTEXT.md`](docs/K02_MEETING_MEMORY_CONTEXT.md).

```
data/raw/*.json  (Scribal: transcript + meeting)
   │
   ├─► чанки реплик → эмбеддинги → индекс ───────────────► baseline ─┐
   │                                                                 ├─► один промпт → проверка цитат → ответ
   └─► извлечение (LLM) → resolve → связи (LLM) → граф ──► graph ────┘        (auto выбирает ветку сам)
```

## Запуск демо с чистого клона (около 10 минут)

Нужны Python 3.11+ и git. Ключи не нужны: по умолчанию `LLM_BACKEND=stub`, всё работает без модели и денег,
но ответы и поиск по смыслу случайны — так проверяется сам интерфейс.

```bash
git clone https://github.com/SSSSSansan/meeting-memory.git
cd meeting-memory
python -m venv .venv
```

```bash
source .venv/bin/activate          # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt    # 3–8 минут: streamlit тянет много пакетов
cp .env.example .env               # Windows: copy .env.example .env
streamlit run app/streamlit_app.py
```

Откроется http://localhost:8501. Что показать:

1. **Боковая панель → «Демо-встречи».** Индексируются 14 синтетических встреч из `data/raw/` батчами по 5.
   Ниже — какой граф читать; по умолчанию полный граф (10 встреч).
2. **«Вопросы».** Режим `auto` сам выбирает ветку и пишет почему; `сравнить` — baseline и graph рядом
   на один вопрос. У каждой цитаты — «Показать в транскрипте» с подсветкой. Внизу токены, стоимость и время.
3. **«Таймлайн проекта».** Решения и задачи проекта по дате: заменённые зачёркнуты, конфликты красные.
4. **«Граф».** Рисунок проекта: связи «заменяет», «уточняет», «конфликт».
5. **«Конфликты и замены».** Например, Николай в одни даты поставлен и на Squad (m_007), и на хотфикс Zhol (m_008).

Вопросы для показа и ожидаемые ответы — [`data/gold/scenarios.md`](data/gold/scenarios.md) и `data/gold/questions.json`.

### С настоящей моделью

В `.env`: `LLM_BACKEND=openai`, `OPENAI_API_KEY=…`, модели и **актуальные цены** (`PRICE_IN_PER_M`,
`PRICE_OUT_PER_M`, `PRICE_EMBED_PER_M`) — по ним `cost_guard` считает траты и останавливает всё при `BUDGET_USD`.
Один вопрос на gpt-4o-mini стоит доли цента. Сколько уже потрачено: `python -m scripts.cost_guard --status`.
После смены модели эмбеддингов нажмите «Очистить индекс» и снова «Демо-встречи»: индекс другой модели не загрузится.

## Данные

Реальные транскрипты клиентов недоступны, поэтому данные синтетические, в формате выгрузки Scribal.

| Где | Что | Вход системы? |
|---|---|---|
| `data/raw/` | 14 встреч m_001–m_015 (m_014 исключена) по проектам Zhol, Squad, Qamqor, Boost; 8 из 14 — с заметной казахской речью | да |
| `data/noisy/` | те же встречи с шумом ASR: 3 уровня × с метками говорящих и без | да (срез устойчивости) |
| `data/extracted_llm_raw/` | извлечение роли B (gpt-4o) | промежуточный результат |
| `data/graph_raw/graph.json` | полный граф: 10 встреч, решения, задачи, связи | для демо |
| `data/graph/graph.json`, `links.json` | граф по встречам до m_008 и связи решений | для eval |
| `data/gold/` | сюжет `world.yaml`, эталон `decisions.json`, вопросы `questions.json`, сценарии | **нет** — только оценка |

Эталон никогда не подаётся системе: это ответы, по которым её проверяют.

## Полный конвейер

Готовые результаты лежат в репозитории, поэтому для демо ничего пересобирать не нужно. Шаги ниже вызывают
модель и **стоят денег** (кроме проверок и `--dry-run`); перед платным запуском — `cost_guard --status`.

```bash
python -m src.extract.run --raw --out data/extracted_llm_raw             # B: извлечение по data/raw
python -m scripts.validate_extraction data/extracted_llm_raw --raw       # B: схема, дословность цитат, hallucination rate
python scripts/compare_gold.py data/extracted_llm_raw                    # B: сверка с эталоном
python -m src.graph.link                                                 # B: связи решений → data/graph/links.json
python scripts/build_graph.py --links data/graph/links.json              # B: граф → data/graph_raw/graph.json
python -m src.eval.run --mode both --split test --as-of m_015 --dry-run  # A: план и цена прогона, без вызовов
python -m src.eval.run --mode both --split test --as-of m_015 --repeats 3 --judge --confirm-paid   # A: сравнение
```

Eval держит «что известно на дату вопроса» (`as_of`): baseline индексирует только встречи до неё, graph запускается,
только если в графе ровно эти встречи, иначе в таблице честная заглушка с причиной. Итог — `results/summary_<дата>.md`
(точность по типам вопросов T1–T5, стоимость, задержка).

Отдельные инструменты для отладки:

```bash
python -m src.answer.answerer "Как менялся срок релиза Zhol?" --mode auto
python -m src.retrieval.graph_rag "Что по Камкор ещё не закрыто?" --depth 2
python -m src.retrieval.baseline "Кто делает миграцию?" -k 5
```

## Проверки

```bash
pip install ruff
ruff check .          # настройки — pyproject.toml
python -m pytest -q   # без сети и ключей
```

CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) запускает обе команды на Python 3.11 и 3.13 для каждого PR
и push в `main`.

## Почему NetworkX, а не Neo4j

- **Объём.** 14 встреч — это десятки решений и задач и пара сотен рёбер. Обход такого графа в памяти занимает
  миллисекунды, серверная графовая БД здесь ничего не ускоряет.
- **Ноль инфраструктуры.** NetworkX — библиотека: у всех троих и у проверяющего граф поднимается одним `pip install`,
  без сервера, пользователей и портов. Neo4j добавил бы день на настройку и точку отказа на демо.
- **Граф — файл в репозитории.** `graph.json` (`networkx.node_link_data`) коммитится, его видно в PR и можно сравнить
  между прогонами; eval воспроизводим с любого клона.
- **Заказчик сам рекомендует NetworkX** для этого объёма.

Когда переходить на Neo4j: тысячи встреч, несколько одновременных пользователей и запись в граф онлайн.
Код читает граф только через `src/graph/queries.py` и `src/retrieval/graph_rag.py`, поэтому замена хранилища
затронет их, а не UI и eval. LangGraph тоже не используется: конвейер линейный, а нам нужен точный контроль
над промптами и подсчётом токенов.

## Роли и вклад

| Роль | Кто | Зона |
|---|---|---|
| A. Данные и оценка | _имя_ | `data/raw`, `data/noisy`, `data/gold`, `src/eval`, скрипты данных и судьи |
| B. Извлечение и граф | _имя_ | `src/extract`, `src/resolve`, `src/graph`, `src/ingest`, `scripts/build_graph.py`, `scripts/validate_extraction.py` |
| C. Поиск, ответы, приложение | _имя_ | `src/retrieval`, `src/answer`, `app`, `src/api.py`, `src/llm_client.py`, `scripts/cost_guard.py`, CI, README |

Общий контракт — `src/schemas.py`, меняется только по согласованию всех. Статус каждой задачи — [`TEAM.md`](TEAM.md),
решения и известные проблемы — [`memory.md`](memory.md).

**A — данные и оценка.** Сюжет на 4 проекта и 7 человек с заложенными ситуациями (`world.yaml`): перенос срока,
замена и отмена решения, конфликт загрузки, забытая задача, смешанная речь. 14 встреч в формате Scribal и слой шума ASR.
Эталон решений и задач с цитатами, 35 вопросов T1–T5 (6 на казахском). Eval с `as_of`, повторами и срезами по типам
и языку; LLM-судья и лист калибровки по ручной разметке.

**B — извлечение и граф.** Адаптер Scribal, восстановление имён говорящих, единая нормализация текста для
казахских букв. Извлечение с проверкой дословности цитат и повтором: 0 % выдуманных цитат на 10 встречах,
найдено 21 из 22 событий эталона (gpt-4o). Склейка вариантов имён и проектов, классификация связей решений,
сборка графа с вычисляемыми статусами.

**C — поиск, ответы, приложение.**
- Baseline: окна реплик с перекрытием, векторный индекс с добавлением батчей по 5.
- Ответ в едином формате для обеих веток; каждая цитата проверяется на дословность, при ошибке — один повтор
  с перечнем плохих цитат, неподтверждённые удаляются с пометкой.
- GraphRAG: проекты и люди из вопроса (с вариантами написания через resolve B) → обход графа → контекст
  с хронологией, статусами и связями.
- Роутер `auto` по признакам T1–T5 на русском и казахском, без LLM.
- Streamlit: загрузка, сравнение режимов, источники с подсветкой, таймлайн, рисунок графа, конфликты и замены.
- LLM-клиент с кэшем, бюджетом и логом токенов, стоимости и задержки; CI.

## LLM backend и настройки

Весь код ходит в модель только через `src/llm_client.py`; ключи — только в `.env` (он в `.gitignore`).

| Переменная | По умолчанию | Зачем |
|---|---|---|
| `LLM_BACKEND` | `stub` | `stub` — без модели и денег; `openai`; `local` — OpenAI-совместимый сервер (vLLM, Ollama) |
| `LLM_MODEL`, `EMBED_MODEL` | `gpt-4o-mini`, `text-embedding-3-small` | модели ответа и эмбеддингов |
| `OPENAI_API_KEY` | — | ключ для `openai` |
| `LOCAL_BASE_URL`, `LOCAL_API_KEY` | `http://localhost:8000/v1` | адрес локального сервера |
| `PRICE_IN_PER_M`, `PRICE_OUT_PER_M`, `PRICE_EMBED_PER_M` | `0` | цены за 1M токенов; без них учёт трат показывает ноль |
| `JUDGE_MODEL`, `JUDGE_PRICE_IN_PER_M`, `JUDGE_PRICE_OUT_PER_M` | как у `LLM_MODEL` | модель и цены LLM-судьи в eval |
| `BUDGET_USD` | `3.0` | жёсткий лимит: при превышении вызовы падают с `BudgetExceeded` |
| `LLM_MAX_RETRIES`, `LLM_TIMEOUT_S` | `3`, `120` | повторы при сбоях сети, 429 и 5xx |
| `MEETING_GRAPH` | — | какой граф читает режим graph; без неё — первый из `data/graph/graph.json`, `data/graph_raw/graph.json`, `data/sample/graph.json`. В UI граф выбирается в боковой панели |

Логи: каждый вызов модели и эмбеддингов — `runs/calls.jsonl` (`purpose`, токены, `cost_usd`, `spent_usd`, задержка;
у ответов из кэша `spent_usd = 0`); каждый вопрос — `runs/answer_report.jsonl` (режим, маршрут `auto`, цитаты
до и после проверки, токены, `latency_ms`, `retrieval_ms`, `llm_ms`, `from_cache`).

## Интерфейсы для интеграции

UI и eval ходят только в `src/api.py`; логики в `app/` нет.

```python
from src import api

pairs, errors = api.pair_scribal_files({"m_001.transcript.json": b"...", "m_001.meeting.json": b"..."})
transcripts = [api.load_scribal(*p) for p in pairs]       # Scribal raw + meta → Transcript (адаптер роли B)
for done, total in api.index_batches(transcripts): ...    # батчи по 5, индекс сохраняется после каждого
api.ask("Как менялся срок релиза Zhol?", mode="auto")     # → src.schemas.Answer: answer, citations, mode, usage, latency_ms
api.route("Что по Qamqor ещё не закрыто?")                # какую ветку выберет auto и почему
api.get_timeline("Жол")                                   # пункты проекта по времени: статус, связи, цитата
api.get_changes()                                          # конфликты и замены решений
api.get_source("m_006", "00:06:02", "релиз Zhol переносится")  # реплика с цитатой и соседние, подсветка
api.available_graphs(); api.use_graph(path)                # выбор графа для graph, таймлайна и рисунка
```

| Функция | Что делает |
|---|---|
| `src.llm_client.complete(messages, *, json_schema=None, temperature=0.0, max_tokens=1024, purpose="generic") -> LLMResponse` | один вызов чат-модели: кэш, бюджет, лог |
| `src.llm_client.embed(texts, *, purpose="embed") -> list[list[float]]` | эмбеддинги |
| `src.answer.answerer.answer(question, mode="baseline", k=5, depth=2) -> Answer` | ответ с проверенными цитатами; `mode`: `baseline` / `graph` / `auto` |
| `src.retrieval.baseline.BaselineIndex` | `add(transcripts)`, `search(question, k)`, `save()` / `load()` |
| `src.retrieval.graph_rag.retrieve_subgraph(question, depth=2) -> dict` | подграф по вопросу и контекст для модели |
| `src.retrieval.router.route(question) -> Route` | выбор ветки для `auto` |
| `src.api.list_projects()`, `graph_info()` | проекты в графе; какой граф читается и какие в нём встречи |
| `src.api.get_timeline(project) -> list[dict]` | решения, задачи и открытые вопросы проекта по времени; у каждого `status`, `in_conflict`, `links`, цитата и встреча |
| `src.api.get_graph(project=None) -> dict` | узлы и рёбра графа для рисунка (без рёбер «участвовал во встрече») |
| `src.api.get_changes(project=None) -> dict` | `conflicts` (пары решений и пояснение) и `replacements` (старое → новое) |
| `src.api.get_source(meeting_id, ts, quote) -> dict` | реплика с цитатой из `data/raw`, `highlight` — позиция цитаты, соседние реплики |
| `src.api.graph_dot(project=None) -> str` | граф в DOT для `st.graphviz_chart`: заменённые серые, конфликт красный |

## Ограничения

- **Данные синтетические.** Реальных транскриптов клиентов нет; синтетика написана с шумом ASR, смешанной речью
  и заложенными ловушками, но качество на живых встречах может отличаться.
- **Цифры сравнения baseline и GraphRAG появятся после платного прогона eval** (`src.eval.run … --judge`); до него
  в `results/` итоговой таблицы нет. На `stub` ответы и поиск по смыслу случайны.
- **Граф покрывает 10 из 14 встреч**, граф для eval — встречи до m_008. Baseline в демо видит все 14 встреч —
  при сравнении режимов в UI это надо учитывать.
- **Локальная модель не развёрнута** (vLLM и `bge-m3` на GPU): поиск между казахскими и русскими фрагментами
  проверен только на синтетических тестах, backend `local` — только на подменённом клиенте.
- **Имена говорящих** восстанавливаются эвристикой: там, где она не уверена, в источниках остаётся `SPEAKER_XX`.
- **Пакетного конвейера одной командой нет** (`src.pipeline` из плана): шаги запускаются по отдельности, см. «Полный конвейер».
- **Роутер `auto` — правила по словам вопроса**, его точность посчитает eval; сущности в вопросе находятся по
  названиям узлов графа и вариантам из resolve.
- **UI рассчитан на одного пользователя**: выбор графа общий для процесса Streamlit.

## Работа в git

Правила — [`GIT.md`](GIT.md): ветка `<a|b|c>/<описание>`, PR в `main`, критичные общие файлы помечаются в заголовке PR.
Остальное для команды: [`TEAM.md`](TEAM.md), [`CLAUDE.md`](CLAUDE.md), [`SKILLS.md`](SKILLS.md), [`TESTING.md`](TESTING.md), [`memory.md`](memory.md).
