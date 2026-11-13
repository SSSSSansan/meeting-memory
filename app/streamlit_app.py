"""РОЛЬ C. Демо-приложение. Запуск: streamlit run app/streamlit_app.py

Только отображение: загрузка, индексация, ответы, таймлайн и граф — в src/api.py.
Шапка со счётчиками; вкладки: вопросы (примеры, режимы и сравнение, источники с подсветкой),
таймлайн проекта, рисунок графа, конфликты и замены решений, метрики прогона eval (только чтение results/).
"""
import html
import os
import sys
from pathlib import Path

# streamlit кладёт в sys.path папку app/, а не корень проекта — без этого не импортируется src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from src import api  # noqa: E402

st.set_page_config(page_title="Meeting Memory", page_icon=":material/hub:", layout="wide")
st.markdown("""<style>
mark {background: #FDE68A; padding: 0 .15em; border-radius: .2em;}
[data-testid="stMetricValue"] {font-size: 1.9rem;}
</style>""", unsafe_allow_html=True)

backend = os.environ.get("LLM_BACKEND", "stub")
model = os.environ.get("LLM_MODEL", "")

KIND = {"Decision": "Решение", "Task": "Задача", "OpenQuestion": "Открытый вопрос"}
STATUS = {"active": ":green[действует]", "superseded": ":gray[заменено]", "open": ":orange[открыта]",
          "done": ":green[выполнена]", "cancelled": ":gray[отменена]"}
LINK = {("SUPERSEDES", "out"): "заменяет", ("SUPERSEDES", "in"): "заменено решением",
        ("CONTRADICTS", "out"): ":red[конфликтует с]", ("CONTRADICTS", "in"): ":red[конфликтует с]",
        ("REFINES", "out"): "уточняет", ("REFINES", "in"): "уточнено решением",
        ("DEPENDS_ON", "out"): "зависит от", ("DEPENDS_ON", "in"): "от него зависит",
        ("SAME_TOPIC", "out"): "на ту же тему, что", ("SAME_TOPIC", "in"): "на ту же тему, что"}
MODE_BADGE = {"graph": ("GraphRAG", "violet"), "baseline": ("Baseline RAG", "blue"),
              "hybrid": ("Hybrid: граф + реплики", "green")}
COMPARE = ("baseline", "graph", "hybrid")
EXAMPLES = [(":material/fact_check:", "Факт из встречи", "Кто делает экран статистики и к какому сроку?"),
            (":material/history:", "Как менялось решение", "Как менялся срок релиза Zhol?"),
            (":material/warning:", "Конфликты", "Есть ли конфликты по загрузке людей?"),
            (":material/block:", "Отмена решения", "Что решили по push-уведомлениям в Zhol?"),
            (":material/translate:", "Вопрос на казахском", "Boost-ты қай проект үшін бірінші жасайды?"),
            (":material/help:", "Ответа нет в данных", "Какой бюджет выделен на Boost?")]


# ---------- Боковая панель: встречи для baseline и граф для graph ----------

with st.sidebar:
    st.header("Данные")
    files = None
    if st.button("Загрузить демо-встречи", type="primary", width="stretch",
                 help="14 синтетических встреч Scribal из data/raw"):
        files = api.demo_files()
    with st.expander("Свои файлы Scribal"):
        uploaded = st.file_uploader("<id>.transcript.json и <id>.meeting.json", type="json",
                                    accept_multiple_files=True)
        if st.button("Индексировать", disabled=not uploaded, width="stretch"):
            files = {f.name: f.getvalue() for f in uploaded}

    if files is not None:
        pairs, errors = api.pair_scribal_files(files)
        for e in errors:
            st.error(e)
        transcripts = []
        for mid, transcript_json, meeting_json in pairs:
            try:
                transcripts.append(api.load_scribal(mid, transcript_json, meeting_json))
            except ValueError as e:
                st.error(f"{mid}: файл не в формате Scribal — {e}")
        if transcripts:
            bar = st.progress(0.0, text="Индексация…")
            for done, total in api.index_batches(transcripts):
                bar.progress(done / total, text=f"Проиндексировано встреч: {done} из {total}")
            st.success(f"Готово: {len(transcripts)} встреч(и)")

    meetings = api.indexed_meetings()
    st.subheader(f"В индексе: {len(meetings)}")
    if meetings:
        with st.expander("Список встреч"):
            st.dataframe([{"встреча": m["meeting_id"], "дата": m["date"], "название": m["title"]}
                          for m in meetings], hide_index=True, width="stretch")
        if st.button("Очистить индекс", width="stretch"):
            api.reset_index()
            st.session_state.pop("last", None)
            st.rerun()

    st.header("Граф")
    graphs = api.available_graphs()
    default = api.demo_default_graph()
    paths = [g["path"] for g in graphs]
    chosen_graph = st.selectbox("Какой граф читать", paths, index=paths.index(default) if default in paths else 0,
                                format_func=lambda p: next(g["label"] for g in graphs if g["path"] == p))
    api.use_graph(chosen_graph)
    info = api.graph_info()
    st.caption(f"Граф: {info['path']} — {len(info['meetings'])} встреч: {', '.join(info['meetings'])}")
    st.caption(f"LLM backend: {backend}" + (f" · {model}" if backend != "stub" else ""))

titles = {m["meeting_id"]: f"{m['title']}, {m['date']}" for m in meetings}


# ---------- Шапка ----------

left, right = st.columns([5, 1], vertical_alignment="center")
with left:
    st.title("Meeting Memory")
    st.markdown("Память встреч для **Scribal**: решения, задачи и их изменения между встречами — "
                "каждый ответ со ссылкой на встречу, таймкод и дословную цитату.")
with right:
    if backend == "stub":
        st.badge("stub — без модели", icon=":material/science:", color="orange")
    else:
        st.badge(f"{backend} · {model}", icon=":material/bolt:", color="green")

if backend == "stub":
    st.warning("LLM_BACKEND=stub: поиск и ответы — заглушка, по смыслу они случайны. "
               "Подключите модель в .env.")

ov = api.overview()
for col, (label, value, hint) in zip(st.columns(5), [
        ("Встреч в индексе", ov["meetings_indexed"], "для поиска по репликам (baseline)"),
        ("Решений в графе", ov["decisions"], f"граф по {ov['graph_meetings']} встречам"),
        ("Открытых задач", ov["open_tasks"], f"всего задач: {ov['tasks']}"),
        ("Конфликтов", ov["conflicts"], "решения, которые противоречат друг другу"),
        ("Замен решений", ov["replacements"], "решения, изменённые на поздних встречах")]):
    col.metric(label, value, help=hint, border=True)

if not meetings:
    st.info("Нажмите «Загрузить демо-встречи» слева — 14 встреч проиндексируются за полминуты.",
            icon=":material/arrow_back:")


# ---------- Отрисовка ----------

def highlighted(text: str, span) -> str:
    """Текст реплики с подсвеченной цитатой; HTML экранирован, кроме самой подсветки."""
    if not span:
        return html.escape(text)
    a, b = span
    return f"{html.escape(text[:a])}<mark>{html.escape(text[a:b])}</mark>{html.escape(text[b:])}"


def show_sources(result) -> None:
    if not result.citations:
        return
    st.markdown(f"**Источники · {len(result.citations)}**")
    for c in result.citations:
        src = api.get_source(c.meeting_id, c.ts, c.quote)
        title = src["title"] + ", " + src["date"] if src else titles.get(c.meeting_id, "")
        st.markdown(f"**{c.meeting_id} · {c.ts}** ({title})  \n«{c.quote}»")
        if src is None:
            st.caption("Транскрипта этой встречи нет в data/raw — показать контекст нельзя.")
            continue
        with st.expander("Показать в транскрипте"):
            for s in src["before"]:
                st.caption(f"[{s['ts']}] {s['speaker']}: {s['text']}")
            line = highlighted(src["text"], src["highlight"])
            st.markdown(f"**[{src['ts']}] {html.escape(src['speaker'])}:** {line}", unsafe_allow_html=True)
            for s in src["after"]:
                st.caption(f"[{s['ts']}] {s['speaker']}: {s['text']}")


def show_context_graph(context: dict) -> None:
    """Пункты графа, ушедшие в контекст ответа, с их проектами — видно, как сработал поиск по графу."""
    n = len(context["items"])
    how = ("сущностей в вопросе нет — взяты конфликты и действующие пункты" if context["fallback"]
           else "от сущностей вопроса: " + ", ".join(e.split(":", 1)[-1] for e in context["entities"]))
    with st.expander(f"Из графа в контекст: {n} пунктов", icon=":material/hub:"):
        st.caption(how + ". Жирная рамка — ушло в контекст.")
        if n:
            st.graphviz_chart(api.graph_dot(highlight=context, focus=True), width="stretch")


def show_answer(result, chosen=None, context=None) -> None:
    with st.container(border=True):
        name, color = MODE_BADGE[result.mode]
        st.badge(name, color=color)
        if chosen:
            fallback = "; в графе по вопросу ничего нет" if chosen.mode != result.mode else ""
            st.caption(f"auto → {result.mode} ({chosen.qtype}): {chosen.reason}{fallback}")
        st.markdown(api.readable_ids(result.answer))
        show_sources(result)
        if context and result.mode in ("graph", "hybrid"):
            show_context_graph(context)
        st.caption(f"Токены: {result.usage.prompt_tokens} + {result.usage.completion_tokens} · "
                   f"${result.usage.cost_usd:.5f} · {result.latency_ms} мс")


def item_line(i: dict) -> str:
    text = f"~~{i['statement']}~~" if i["status"] == "superseded" else i["statement"]
    marks = [STATUS.get(i["status"], "")] + ([":red[в конфликте]"] if i["in_conflict"] else [])
    who = i["owner"] or i["made_by"]
    extra = (f" · {who}" if who else "") + (f" · срок {i['due']}" if i["due"] else "")
    return f"**{KIND[i['kind']]}** {' '.join(m for m in marks if m)} — {text}{extra}"


def pick_example(question: str) -> None:
    st.session_state.q = question
    st.session_state.go = True


# ---------- Вкладки ----------

tab_qa, tab_timeline, tab_graph, tab_changes, tab_metrics = st.tabs(
    [":material/forum: Вопросы", ":material/timeline: Таймлайн проекта", ":material/hub: Граф",
     ":material/warning: Конфликты и замены", ":material/leaderboard: Метрики"])

with tab_qa:
    st.markdown("**Примеры вопросов** — нажмите, чтобы спросить")
    cols = st.columns(3)
    for n, (icon, label, q) in enumerate(EXAMPLES):
        cols[n % 3].button(label, key=f"ex_{n}", icon=icon, help=q, width="stretch",
                           on_click=pick_example, args=(q,))

    mode = st.radio("Режим", ["auto", "baseline", "graph", "hybrid", "сравнить"], horizontal=True,
                    help="baseline — поиск по репликам загруженных встреч; graph — по графу решений и задач; "
                         "hybrid — граф и реплики вокруг его пунктов плюс поиск по смыслу; "
                         "auto — граф для вопросов через несколько встреч, baseline для простых фактов; "
                         "сравнить — baseline, graph и hybrid на один вопрос рядом")
    question = st.text_input("Вопрос", key="q", placeholder="Например: как менялся срок релиза Zhol?")
    asked = st.button("Спросить", type="primary", disabled=not question.strip())
    if (asked or st.session_state.pop("go", False)) and question.strip():
        chosen = api.route(question) if mode == "auto" else None
        needs_index = mode in ("baseline", "hybrid", "сравнить") or (chosen and chosen.mode == "baseline")
        if needs_index and not meetings:
            st.session_state.last = None
            st.info("Индекс пуст: загрузите встречи слева.")
        else:
            modes = list(COMPARE) if mode == "сравнить" else [mode]
            with st.spinner("Ищу и отвечаю…"):
                results = [api.ask(question, mode=m) for m in modes]
            # тот же детерминированный обход, что внутри ответа: для подсветки на рисунке графа
            context = (api.context_subgraph(question) if any(r.mode in ("graph", "hybrid") for r in results)
                       else None)
            st.session_state.last = {"question": question, "mode": mode, "results": results, "chosen": chosen,
                                     "context": context}

    last = st.session_state.get("last")
    if last:  # ответ живёт в сессии: переключение вкладок и выбор проекта его не стирают
        st.markdown(f"#### {last['question']}")
        if last["mode"] == "сравнить":
            for col, m, result in zip(st.columns(len(COMPARE)), COMPARE, last["results"]):
                with col:
                    st.subheader(m)
                    show_answer(result, context=last.get("context"))
        else:
            show_answer(last["results"][0], last["chosen"], context=last.get("context"))

projects = api.list_projects()
# по умолчанию — проект, где больше всего пунктов: на нём видны и замены, и конфликты
busiest = max(range(len(projects)), key=lambda n: len(api.get_timeline(projects[n]))) if projects else 0

with tab_timeline:
    if not projects:
        st.info("В графе нет проектов.")
    else:
        project = st.selectbox("Проект", projects, index=busiest, key="timeline_project")
        items = api.get_timeline(project)
        st.caption(f"{len(items)} пунктов по дате. В проект попадает и то, что обсуждали на встречах "
                   "про другие проекты: важно поле проекта у пункта, а не название встречи.")
        for i in items:
            with st.container(border=True):
                when, what = st.columns([1, 4])
                when.markdown(f"**{i['date']}**  \n{i['meeting_id']} · {i['ts']}")
                when.caption(i["meeting_title"])
                what.markdown(item_line(i))
                for link in i["links"]:
                    what.caption(f"{LINK.get((link['type'], link['direction']), link['type'])} "
                                 f"«{link['other_statement']}» ({link['other_meeting']}, {link['other_date']})",
                                 help=link["rationale"] or None)
                what.caption(f"«{i['quote']}»")

with tab_graph:
    scope = st.selectbox("Что показать", ["все проекты"] + projects, index=busiest + 1 if projects else 0,
                         key="graph_project")
    last = st.session_state.get("last") or {}
    context = last.get("context")
    lit = None
    if context:
        if st.toggle(f"Подсветить, что ушло в контекст последнего ответа: «{last['question']}»", value=True,
                     key="graph_highlight"):
            lit = context
            st.caption(f"Жирная рамка — {len(context['items'])} пунктов из контекста ответа, "
                       "остальное приглушено.")
    st.graphviz_chart(api.graph_dot(None if scope == "все проекты" else scope, highlight=lit), width="stretch")
    st.caption("Синие — решения, жёлтые — задачи, розовые — открытые вопросы, зелёные — проекты, "
               "серые с пометкой [заменено] — заменённые решения, красная рамка и красная стрелка — конфликт.")

with tab_changes:
    scope = st.selectbox("Проект", ["все проекты"] + projects, key="changes_project")
    changes = api.get_changes(None if scope == "все проекты" else scope)
    st.subheader(f"Конфликты: {len(changes['conflicts'])}")
    if not changes["conflicts"]:
        st.caption("Конфликтов в графе нет.")
    for c in changes["conflicts"]:
        with st.container(border=True):
            st.markdown(f":red[**Конфликт**] — {c['rationale']}")
            for side in (c["a"], c["b"]):
                st.markdown(f"- **{side['date']}** {side['meeting_id']} «{side['meeting_title']}» · {side['ts']}: "
                            f"{side['statement']}  \n  «{side['quote']}»")
    st.subheader(f"Замены решений: {len(changes['replacements'])}")
    if not changes["replacements"]:
        st.caption("Заменённых решений в графе нет.")
    for r in changes["replacements"]:
        with st.container(border=True):
            st.markdown(f"~~{r['old']['statement']}~~ ({r['old']['date']}, {r['old']['meeting_id']})  \n"
                        f"→ **{r['new']['statement']}** ({r['new']['date']}, {r['new']['meeting_id']})")
            if r["rationale"]:
                st.caption(r["rationale"])

with tab_metrics:
    runs = api.eval_runs()
    if not runs:
        st.info("В results/ нет eval_*.json. Прогон: `python -m src.eval.run --mode both --judge-facts …`")
    else:
        run = st.selectbox("Прогон", runs, format_func=lambda r: r["label"], key="metrics_run")
        result = api.load_eval(run["path"])
        cfg = result.get("config") or {}
        st.caption(f"{result.get('timestamp', '')} · модель {cfg.get('model')} · эмбеддинги {cfg.get('embed_model')}"
                   f" · судья {cfg.get('judge_model')} ({cfg.get('JUDGE_FACTS_PROMPT_VERSION')}) · "
                   f"split {cfg.get('split')} · вопросов {cfg.get('n_questions')} · as_of {cfg.get('as_of')} · "
                   f"commit {cfg.get('commit')}")
        if cfg.get("fixture"):
            st.warning("Фикстура роли A: числа выдуманы, это не результат системы. "
                       "Вкладка покажет настоящий прогон, как только в results/ появится eval_*.json.",
                       icon=":material/science:")
        if cfg.get("split") == "dev":
            st.info("Split dev: разбивка по типам разведочная — на каждый тип по несколько вопросов. "
                    "Итоговые утверждения — по прогону на test.", icon=":material/info:")

        for col, card in zip(st.columns(3), api.metric_cards()):
            col.metric(card["title"], card["value"] or "—", help=card["hint"], border=True)
            if card["value"] is None:
                col.caption(card["hint"])

        st.subheader("Качество ответа по типам вопросов")
        metric = st.radio("Метрика", list(api.QUALITY_METRICS), horizontal=True, key="metrics_metric",
                          format_func=lambda k: api.QUALITY_METRICS[k])
        table = api.quality_table(result, metric)
        if not table["measured"]:
            st.info(f"«{api.QUALITY_METRICS[metric]}» в этом прогоне не измерена (нет значений, а не нули). "
                    "Факты считает судья: прогон с --judge-facts.")
        st.dataframe(table["rows"], hide_index=True, width="stretch")
        st.caption("Ячейка: среднее [95% интервал] n — число вопросов. Интервал — бутстрап по вопросам: "
                   "показывает зависимость от набора вопросов, а не разброс повторов. «—» — не измерено."
                   + (" * — " + "; ".join(table["notes"]) if table["notes"] else ""))
        if table["paired"]:
            st.markdown("**Разность graph − baseline** (парный бутстрап: одни и те же вопросы в обоих режимах)")
            st.dataframe(table["paired"], hide_index=True, width="stretch")

        st.subheader("Источники и поиск")
        st.dataframe(api.sources_table(result), hide_index=True, width="stretch")
        st.caption("По встречам-источникам из эталона, без модели. Полнота поиска — эталонные встречи в контексте "
                   "ещё до генерации: отделяет «поиск не нашёл» от «модель не использовала».")

        st.subheader("Стоимость и задержка ответа")
        cl = api.cost_latency(result)
        left, mid, right = st.columns(3)
        left.markdown("**$ за ответ**")
        left.bar_chart({r["режим"]: r["cost_usd"] or 0 for r in cl}, horizontal=True, height=160)
        mid.markdown("**Задержка, мс**")
        mid.bar_chart({r["режим"]: r["latency_ms"] or 0 for r in cl}, horizontal=True, height=160)
        right.markdown("**Токены промпта**")
        right.bar_chart({r["режим"]: r["context_tokens"] or 0 for r in cl}, horizontal=True, height=160)
        st.caption("Средние по ответам со статусом ok"
                   + ", n = " + ", ".join(f"{r['режим']}: {r['n']}" for r in cl)
                   + ". Токены промпта — контекст, системный промпт и вопрос вместе.")

        with st.expander("Срез по языку вопроса"):
            st.dataframe(api.lang_table(result, metric), hide_index=True, width="stretch")
        with st.expander("Отказы отвечать"):
            st.dataframe(api.abstention_table(result), hide_index=True, width="stretch")
