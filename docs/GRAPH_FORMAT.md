# Формат графа (контракт между ролью B и ролью C)

Файл: `data/sample/graph.json` — заглушка из ручных примеров. Настоящий граф: `data/graph/graph.json`
(тот же формат). Хранится как `networkx.node_link_data`, читается через `src/graph/queries.py`.

```python
from src.graph.queries import load_graph, project_decisions, decision_history, conflicts, open_tasks, project_meetings
G = load_graph("data/sample/graph.json")
project_decisions(G, "Атлас", last_meetings=3)   # решения по проекту за N последних встреч
decision_history(G, "d_m002_01")                 # как менялось решение: от раннего к позднему
conflicts(G)                                      # противоречия
open_tasks(G, "Атлас")                            # ещё не закрытые задачи
```

## Узлы (`type`)

| type | id | Основные атрибуты |
|---|---|---|
| Meeting | `meeting:m_001` | `meeting_id`, `label` (название), `date`, `participants` |
| Project | `project:Атлас` | `label` |
| Person | `person:Руслан Ким` | `label` |
| Decision | `d_m001_01` | `statement`, `project`, `meeting_id`, `date`, `ts`, `quote`, `confidence`, `made_by`, **`status`** (`active`/`superseded`), `in_conflict` |
| Task | `t_m001_01` | `statement`, `project`, `meeting_id`, `date`, `ts`, `quote`, `owner`, `due`, `status` (`open`/`done`/`cancelled`) |
| OpenQuestion | `q_m001_01` | `statement`, `project`, `meeting_id`, `date`, `ts`, `quote` |

`quote` + `meeting_id` + `ts` — это источник для цитаты в ответе. Цитата дословная (проверяется автоматически).

## Рёбра (`type`)

| type | откуда → куда |
|---|---|
| BELONGS_TO | Decision / Task / OpenQuestion → Project (проект пункта, а не название встречи) |
| STATED_IN | Decision / Task / OpenQuestion → Meeting |
| DISCUSSES | Meeting → Project (все проекты, обсуждённые на встрече) |
| PARTICIPATED_IN | Person → Meeting |
| ASSIGNED_TO | Task → Person |
| SUPERSEDES | **новое** решение → **старое** (оно заменяет старое) |
| CONTRADICTS | Decision ↔ Decision (направление не важно) |
| DEPENDS_ON | Decision → Decision, от которого зависит |
| REFINES | Decision → Decision, которое уточняет |
| SAME_TOPIC | Decision ↔ Decision, про одно, логической связи нет |

У связей между решениями есть `confidence` и `rationale` (одно предложение — почему связаны).
Связи с уверенностью ниже 0.6 в граф не попадают.

## Важно знать

- Статус решения **вычисляется**: `superseded`, если на него есть входящее SUPERSEDES.
- Сейчас названия проектов и людей не склеены (шаг «склейка сущностей» ещё впереди): «Атлас» и «проект Атлас» могут быть разными узлами.
- В заглушке одна связь (SUPERSEDES). Противоречий и зависимостей в ней нет — для тестов их строят вручную (см. `tests/test_graph.py`).
- Связь «закрытая задача ↔ ранее выданная задача» пока не моделируется (появится на шаге связей).
