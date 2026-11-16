# Четыре сценария на Cypher (B7)

Те же четыре сценария, что в замере движков ([GRAPH_ENGINES.md](GRAPH_ENGINES.md)): на NetworkX они реализованы обходом в Python, здесь — запросом. Neo4j используется только для замера и показа в браузере, ответы пользователям выдаёт NetworkX.

## Запуск

```bash
docker compose -f scripts/neo4j/docker-compose.yml up -d      # Neo4j 5.26, порты 7474 (браузер) и 7687 (bolt)
python scripts/neo4j/bench_engines.py --engine both --scales 1  # заливает граф и сверяет результат с NetworkX
docker compose -f scripts/neo4j/docker-compose.yml down -v     # остановить и стереть том
```

Браузер: http://localhost:7474, пользователь `neo4j`, пароль `k02-local-pass` (локальный, данные синтетические). Граф берётся из `data/graph/graph.json` (14 встреч, 64 узла, 172 ребра).

## Модель данных в Neo4j

Каждый узел имеет метку `Node` и свойства `nid` (идентификатор из `graph.json`), `type` (`Decision`, `Task`, `OpenQuestion`, `Closure`, `Meeting`, `Project`, `Person`), `project`, `date`, `status`, `due`; решения дополнительно получают метку `Decision`. Тип ребра совпадает с типом в `graph.json` (`SUPERSEDES`, `CONTRADICTS`, `CLOSES`, `ASSIGNED_TO`, `BELONGS_TO` и так далее). Индексы: по `Node(nid)`, `Decision(project)` и `Node(type, status)`.

## Запросы

Это ровно те строки, которые выполняет `scripts/neo4j/bench_engines.py` (словарь `CYPHER`); параметры в замере: решение `d_m001_01`, человек `person:Николай`, дата `2026-10-26`.

**S1. История решения** (цепочка замен в обе стороны, путь переменной длины):

```cypher
MATCH (a:Node {nid:$id})-[:SUPERSEDES*0..]-(m) RETURN DISTINCT m.nid AS r ORDER BY r
```

**S2. Кросс-проектный** (в каких проектах человеку что поручено):

```cypher
MATCH (t)-[:ASSIGNED_TO]->(p:Node {nid:$p}), (t)-[:BELONGS_TO]->(pr)
RETURN pr.nid + '|' + t.nid AS r ORDER BY r
```

**S3. Противоречия** (все пары `CONTRADICTS`, без фильтра по статусу):

```cypher
MATCH (a)-[:CONTRADICTS]->(b) RETURN a.nid + '->' + b.nid AS r ORDER BY r
```

**S4. Открытые и просроченные задачи на дату**:

```cypher
MATCH (t:Node {type:'Task', status:'open'})
RETURN t.nid + '|' + CASE WHEN t.due <> '' AND t.due < $t THEN 'overdue' ELSE 'open' END AS r ORDER BY r
```

## Что важно знать

- S2 и S3 упрощены относительно формулировок плана: S2 отвечает на «в каких проектах человеку что поручено», S3 возвращает все пары `CONTRADICTS` независимо от статуса решений.
- S4 берёт задачи со статусом `open` из `graph.json` (закрытые через `CLOSES` в графе уже имеют статус `done`, их в выдаче нет) и делит их на просроченные и нет по сроку. Правило «просрочено и не упоминалось на поздних встречах» из `src/graph/status.py` в Cypher не переносилось.
- Результаты всех трёх реализаций (Python без индекса, Python со словарями, Cypher) совпали на всех сценариях и масштабах ×1…×1000 (`results/graph_engines.json`, поле `agree`).

## Для показа в браузере

Для рисунка графа вместо таблицы возвращайте путь, а не `nid`. Эти варианты предназначены для показа, в замере не участвуют и в этой сессии не запускались:

```cypher
MATCH p=(a:Node {nid:'d_m001_01'})-[:SUPERSEDES*0..]-(m) RETURN p
MATCH p=()-[:CONTRADICTS]->() RETURN p
```
