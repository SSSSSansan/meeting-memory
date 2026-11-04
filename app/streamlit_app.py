"""РОЛЬ C. Демо-приложение. Запуск: streamlit run app/streamlit_app.py

Только отображение: загрузка, индексация и ответы — в src/api.py.
TODO (C7): таймлайн решений по проекту, визуализация графа, подсветка источников.
"""
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
    st.caption(f"LLM backend: {backend}")

mode = st.radio("Режим", ["baseline", "graph"], horizontal=True,
                help="baseline — поиск по репликам загруженных встреч; graph — по графу решений и задач")
if mode == "graph":
    g = api.graph_info()
    note = " — это заглушка, настоящий граф строит роль B" if g["is_sample"] else ""
    st.caption(f"Граф: {g['path']} ({len(g['meetings'])} встреч: {', '.join(g['meetings'])}){note}")
question = st.text_input("Вопрос", placeholder="Что решили по Zhol за последние три встречи?")

if st.button("Спросить", type="primary", disabled=not question.strip()):
    if mode == "baseline" and not meetings:
        st.info("Индекс пуст: загрузите встречи слева или нажмите «Демо-встречи».")
    else:
        with st.spinner("Ищу и отвечаю…"):
            result = api.ask(question, mode=mode)
        st.markdown(result.answer)
        titles = {m["meeting_id"]: f"{m['title']}, {m['date']}" for m in meetings}
        if result.citations:
            st.subheader("Источники")
            for c in result.citations:
                st.markdown(f"**{c.meeting_id} · {c.ts}** ({titles.get(c.meeting_id, '')})  \n«{c.quote}»")
        st.caption(f"Токены: {result.usage.prompt_tokens} + {result.usage.completion_tokens} · "
                   f"${result.usage.cost_usd:.5f} · {result.latency_ms} мс")
