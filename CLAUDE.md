# CLAUDE.md — K02 Meeting Memory

Инструкция для Claude Code в этом репозитории. Читай целиком в начале сессии, затем `memory.md`.

Полный контекст задания, архитектура и промпты: `docs/K02_MEETING_MEMORY_CONTEXT.md`. Роли и задачи: `TEAM.md`. Git: `GIT.md`. Skills: `SKILLS.md`. Тесты: `TESTING.md`.

## Что строим

Сравнение **baseline RAG** и **GraphRAG** на транскриптах встреч (клиент — Scribal). Результат — не «система», а **честная таблица метрик** по типам вопросов T1–T5 + стоимость + задержка. Отрицательный результат по GraphRAG с цифрами засчитывается.

Поток: `Scribal JSON → ingest → extract (LLM) → resolve → link (LLM) → граф NetworkX` и параллельно `чанки → эмбеддинги → baseline`. Оба режима отвечают через один интерфейс (контракт 4.6).

## Первое действие в сессии

1. Прочитай `memory.md` (решения, грабли, статус).
2. Спроси или определи, **какая роль** (A/B/C) — работай только в её папках (см. таблицу ниже).
3. Перед завершением — допиши в `memory.md` (протокол в `SKILLS.md`).

## Роли и владение папками

| Роль | Зона | Папки |
|---|---|---|
| A — Данные и оценка | транскрипты, gold, eval, отчёт | `data/transcripts/`, `data/gold/`, `src/eval/`, `scripts/generate_transcripts.py`, `scripts/asr_*`, `REPORT.md` |
| B — Извлечение и граф | extract, resolve, link, graph | `src/extract/`, `src/resolve/`, `src/graph/`, `src/ingest/`, `src/pipeline.py`, `src/utils/text.py`, `scripts/validate_extraction.py` |
| C — Поиск, ответы, UI | baseline, GraphRAG, answer, Streamlit | `src/retrieval/`, `src/answer/`, `app/`, `src/llm_client.py`, `src/api.py`, `scripts/cost_guard.py`, `README.md` |

Общие файлы (`src/schemas.py`, `.env.example`, `requirements.txt`) меняются **только через PR с ревью всех ролей** (см. `GIT.md`). Чужую папку не правь — оставь issue/комментарий владельцу.

## Жёсткие правила

0. **OpenAI (платный backend) — только после явного «да» от пользователя в чате.** По умолчанию `stub`/`local`. Перед каждым запуском с `LLM_BACKEND=openai` спросить: что запускаем и примерно сколько стоит.

1. **Контракты не менять молча.** Схемы 4.1–4.6 из контекста + Scribal-формат ниже. Любое изменение — только *аддитивное* (новые опциональные поля), через PR в `src/schemas.py` и запись в `memory.md`.
2. **В LLM ходим только через `src/llm_client.py`** (`complete`, `embed`). Никаких прямых `openai.*` и `requests` к vLLM в других файлах. Backend: `LLM_BACKEND=openai|local`.
3. **`source.quote` — дословная подстрока транскрипта.** Перефразированная цитата = невалидная запись. Проверка через `src/utils/text.py::normalize` (NFC, lowercase, схлопнуть пробелы) — одна функция на всех, включая казахские буквы `ә ғ қ ң ө ұ ү һ і`.
4. **Не выдумывай решения.** Пустой список лучше выдуманной записи. Ответ без опоры на контекст = «в контексте нет».
5. **ID детерминированные:** `<тип>_<meeting_id>_<NN>`. Прогоны воспроизводимы, `temperature=0`.
6. **Каждый вызов LLM логируется** в `runs/*.jsonl` (prompt/completion tokens, $, latency, mode, stage) и идёт через `cost_guard`. Кэш по хэшу промпта — `runs/cache/`.
7. **Секреты** — только `.env` (в `.gitignore`). Presigned URL из Scribal API живут 1 час и содержат подпись — не логировать и не коммитить.
8. **Реальные данные Scribal не коммитить.** В репо — только синтетика и свои записи, обезличенные.
9. **Казахский — не опция.** 20–30% встреч у клиентов содержат казахскую речь (часто смешанную). Система должна полностью работать на казахском: эмбеддинги `bge-m3`, нормализация NFC, отдельный срез в eval (`ru` / `mixed` / `kk`).
10. **Обработка батчами по 5 встреч** (решение куратора). Индексация идемпотентна и инкрементальна: новый батч добавляется в существующий граф, без полной пересборки.
11. **Не режем:** eval-harness и сравнение с baseline. Режем в порядке: 15→10 встреч, 60→40 вопросов, граф-визуализация→таблица.

## Scribal I/O (реальный формат от куратора)

**Сырой транскрипт** (файл `transcript.json` по presigned URL):

```json
{
  "text": "полный текст",
  "language": "ru",
  "detected_language": "rus",
  "segments": [
    {"start": 0.42, "end": 6.81, "speaker": "SPEAKER_00",
     "text": "По проекту X договорились закрыть до конца октября.",
     "words": [{"word": "По", "start": 0.42, "end": 0.55, "score": 0.97}]}
  ]
}
```

**Метаданные встречи:** `GET /api/v1/meetings/{public_id}` → `public_id`, `title`, `language`, `started_at`/`ended_at` (ISO, +05:00), `participants[].user.full_name`, `departments`, `status` (`READY`), `has_transcript`, `speaker_names` (**может быть `null`**), `actual_duration_seconds`.
**Транскрипт:** `GET /api/v1/meetings/{public_id}/transcript` → `{"url": "<presigned s3>"}` (TTL 1 ч).

Что это значит для нас:
- `speaker` — метки `SPEAKER_00`, не имена. Имена берём из `speaker_names`, иначе маппинг по `participants` + эвристика/LLM (роль B, preprocessing). Хранить `speaker_raw` и `speaker` отдельно.
- Время — секунды (float). Канонический `ts` = `HH:MM:SS` от `start`; `start_s`/`end_s` храним дополнительно.
- Дата встречи = `started_at` (для резолва «к пятнице»). Язык — `language`/`detected_language`, на уровне сегмента может быть смешан.
- `words[].score` — уверенность ASR; можно использовать для понижения `confidence` на шумных сегментах.

Адаптер `src/ingest/scribal_adapter.py` превращает (raw JSON + meeting meta) в канонический `data/transcripts/m_XXX.json` (контракт 4.1) с аддитивными полями: `language`, `source` (`synthetic|own_asr|scribal`), `started_at`, и в сегментах — `start_s`, `end_s`, `speaker_raw`, `lang`. Вход проекта = Scribal JSON, а не «своя схема».

## Команды

```bash
pip install -r requirements.txt
cp .env.example .env                      # LLM_BACKEND, OPENAI_API_KEY, BUDGET_USD, VLLM_BASE_URL
python -m src.pipeline --input data/raw --batch-size 5     # ingest→extract→resolve→link→graph, батчами
python scripts/validate_extraction.py                      # схема + цитаты + hallucination rate
python -m src.eval.run --mode both --repeats 3             # baseline vs graph
streamlit run app/streamlit_app.py
pytest -q                                                  # см. TESTING.md
```

## Стиль

Python 3.11, type hints, pydantic v2, `ruff` + `black`. Комментарии — только «почему», не «что». Тексты промптов — в `src/**/prompts/*.md` или константах, не размазаны по коду. Язык кода и коммитов — английский; документация и промпты — русский.
