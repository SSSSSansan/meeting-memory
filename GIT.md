# GIT.md — работа с репозиторием

Репо: https://github.com/SSSSSansan/meeting-memory

## Ветки

- `main` — всегда запускается. Прямых пушей нет, только PR.
- **Имя ветки всегда начинается с буквы роли и слэша:** `<роль>/<описание>`, роль — `a`, `b` или `c`. Примеры: `a/transcripts-batch1`, `b/extract-prompt`, `c/fix-ux-error`.
- Ветки контрактов: `contract/<что>` — единственный тип веток, который трогает `src/schemas.py`.
- **Ветку регулярно обновляем из `main`** (минимум раз в день и перед каждым PR): `git fetch && git merge origin/main` (или `git pull --rebase origin main`). Так не теряем актуальное состояние и не копим конфликты.

## Коммиты

Формат: `<scope>: <что сделано>` — `extract: add retry on invalid json`, `data: add m_004 cross-project`, `eval: calibrate judge on 30 samples`.
Scope = `data | extract | resolve | graph | retrieval | answer | app | eval | llm | ingest | pipeline | docs | contract | chore`.
Один коммит — одна мысль. Не коммитим `.env`, `runs/cache/`, `runs/*.jsonl`, реальные выгрузки Scribal, presigned URL.

## Pull Request

1. Размер PR не ограничен: можно делать фичи крупными кусками. В заголовке — задачный ID из `TEAM.md`: `[B4] resolve: entity clustering`.
2. В описании: что сделано, как проверить (команда), какие файлы чужих ролей затронуты (должно быть «нет»).
3. Перед PR: `ruff check . && pytest -q` зелёные, ветка обновлена из `main`.
4. **Если затронуты критичные общие файлы** (`src/schemas.py`, `.env.example`, `requirements.txt`, формат данных) — **пишем это в заголовке PR**, например: `[B4] resolve: entity clustering [CRITICAL: schemas.py, requirements.txt]`. Так по названию сразу видно, что PR трогает общее.
5. Merge: squash. Ветку удалить.
6. В `memory.md` дописать, если PR принёс решение, которое влияет на других.

## Изменение контракта (процесс)

1. Ветка `contract/<что>`, правка только аддитивная (новое опциональное поле).
2. PR с пометкой `[CRITICAL: schemas.py]` в заголовке и описанием: кого касается, что сломается, миграция.
3. Остальные участники подтверждают в чате/PR. Запись в `memory.md` → раздел «Контракты».
4. После merge владельцы затронутых модулей обновляют код в течение дня.

## Данные в git

- `data/transcripts/`, `data/gold/`, `data/extracted/`, `data/graph/` — коммитим (синтетика, небольшие JSON).
- `data/raw/` (выгрузки Scribal и записи с ASR) — в `.gitignore`; в репо попадает только обезличенный результат адаптера.
- Большие модели/индексы/эмбеддинги — не коммитим; воспроизводятся командой из README.

## Тег сдачи

`v1.0-submission` на `main` в день 7 после прогона «чистый клон → README → демо-сценарий».

## .gitignore (минимум)

```
.env
runs/cache/
runs/*.jsonl
runs/budget.json
data/raw/
__pycache__/
.venv/
*.faiss
.chroma/
```

## Конфликты

`git pull --rebase origin main` перед PR. Если конфликт в чужом файле — не решай «своей версией», пиши владельцу.
