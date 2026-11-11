"""РОЛЬ C. Демо-приложение. Запуск: streamlit run app/streamlit_app.py

Только отображение: загрузка, индексация, ответы, таймлайн и граф — в src/api.py.
Вкладки: вопросы (режимы и сравнение, источники с подсветкой), таймлайн проекта, рисунок графа,
конфликты и замены решений.
"""
import html
import os
import sys
from pathlib import Path

# streamlit кладёт в sys.path папку app/, а не корень проекта — без этого не импортируется src
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st  # noqa: E402

from src import api  # noqa: E402

st.set_page_config(page_title="Meeting Memory", layout="wide")
st.title("Meeting Memory")

backend = os.environ.get("LLM_BACKEND", "stub")
if backend == "stub":
    st.warning("LLM_BACKEND=stub: поиск и ответы — заглушка, по смыслу они случайны. "
               "Подключите модель в .env.")

KIND = {"Decision": "Решение", "Task": "Задача", "OpenQuestion": "Открытый вопрос"}
STATUS = {"active": ":green[действует]", "superseded": ":gray[заменено]", "open": ":orange[открыта]",
          "done": ":green[выполнена]", "cancelled": ":gray[отменена]"}
LINK = {("SUPERSEDES", "out"): "заменяет", ("SUPERSEDES", "in"): "заменено решением",
        ("CONTRADICTS", "out"): ":red[конфликтует с]", ("CONTRADICTS", "in"): ":red[конфликтует с]",
        ("REFINES", "out"): "уточняет", ("REFINES", "in"): "уточнено решением",
        ("DEPENDS_ON", "out"): "зависит от", ("DEPENDS_ON", "in"): "от него зависит",
        ("SAME_TOPIC", "out"): "на ту же тему, что", ("SAME_TOPIC", "in"): "на ту же тему, что"}


# ---------- Боковая панель: встречи для baseline и граф для graph ----------

with st.sidebar:
    st.header("Встречи")
    uploaded = st.file_uploader("Файлы Scribal: <id>.transcript.json и <id>.meeting.json",
                                type="json", accept_multiple_files=True)
    col_upload, col_demo = st.columns(2)
    files = None
    if col_upload.button("Индексировать", disabled=not uploaded, width="stretch"):
        files = {f.name: f.getvalue() for f in uploaded}
    if col_demo.button("Демо-встречи", help="data/raw — синтетика роли A", width="stretch"):
        files = api.demo_files()

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
        st.dataframe([{"встреча": m["meeting_id"], "дата": m["date"], "название": m["title"],
                       "чанков": m["chunks"]} for m in meetings],
                     hide_index=True, width="stretch")
        if st.button("Очистить индекс"):
            api.reset_index()
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
    st.caption(f"LLM backend: {backend}")

titles = {m["meeting_id"]: f"{m['title']}, {m['date']}" for m in meetings}


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
    st.markdown("**Источники**")
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


def show_answer(result, chosen=None) -> None:
    if chosen:
        fallback = "; в графе по вопросу ничего нет" if chosen.mode != result.mode else ""
        st.caption(f"auto → {result.mode} ({chosen.qtype}): {chosen.reason}{fallback}")
    st.markdown(result.answer)
    show_sources(result)
    st.caption(f"Токены: {result.usage.prompt_tokens} + {result.usage.completion_tokens} · "
               f"${result.usage.cost_usd:.5f} · {result.latency_ms} мс")


def item_line(i: dict) -> str:
    text = f"~~{i['statement']}~~" if i["status"] == "superseded" else i["statement"]
    marks = [STATUS.get(i["status"], "")] + ([":red[в конфликте]"] if i["in_conflict"] else [])
    who = i["owner"] or i["made_by"]
    extra = (f" · {who}" if who else "") + (f" · срок {i['due']}" if i["due"] else "")
    return f"**{KIND[i['kind']]}** {' '.join(m for m in marks if m)} — {text}{extra}"


# ---------- Вкладки ----------

tab_qa, tab_timeline, tab_graph, tab_changes = st.tabs(
    ["Вопросы", "Таймлайн проекта", "Граф", "Конфликты и замены"])

with tab_qa:
    mode = st.radio("Режим", ["auto", "baseline", "graph", "сравнить"], horizontal=True,
                    help="baseline — поиск по репликам загруженных встреч; graph — по графу решений и задач; "
                         "auto — граф для вопросов через несколько встреч, baseline для простых фактов; "
                         "сравнить — baseline и graph на один вопрос рядом")
    question = st.text_input("Вопрос", placeholder="Как менялся срок релиза Zhol?")
    if st.button("Спросить", type="primary", disabled=not question.strip()):
        chosen = api.route(question) if mode == "auto" else None
        needs_index = mode in ("baseline", "сравнить") or (chosen and chosen.mode == "baseline")
        if needs_index and not meetings:
            st.info("Индекс пуст: загрузите встречи слева или нажмите «Демо-встречи».")
        elif mode == "сравнить":
            left, right = st.columns(2)
            for col, m in ((left, "baseline"), (right, "graph")):
                with col:
                    st.subheader(m)
                    with st.spinner(f"{m}: ищу и отвечаю…"):
                        show_answer(api.ask(question, mode=m))
        else:
            with st.spinner("Ищу и отвечаю…"):
                result = api.ask(question, mode=mode)
            show_answer(result, chosen)

projects = api.list_projects()

with tab_timeline:
    if not projects:
        st.info("В графе нет проектов.")
    else:
        project = st.selectbox("Проект", projects, key="timeline_project")
        items = api.get_timeline(project)
        st.caption(f"{len(items)} пунктов по дате. В проект попадает и то, что обсуждали на встречах "
                   "про другие проекты: важно поле проекта у пункта, а не название встречи.")
        for i in items:
            with st.container(border=True):
                st.markdown(f"**{i['date']}** · {i['meeting_id']} «{i['meeting_title']}» · {i['ts']}")
                st.markdown(item_line(i))
                for link in i["links"]:
                    why = f" — {link['rationale']}" if link["rationale"] else ""
                    st.caption(f"{LINK.get((link['type'], link['direction']), link['type'])} {link['other']}{why}")
                st.caption(f"«{i['quote']}»")

with tab_graph:
    scope = st.selectbox("Что показать", ["все проекты"] + projects, key="graph_project")
    st.graphviz_chart(api.graph_dot(None if scope == "все проекты" else scope), width="stretch")
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
