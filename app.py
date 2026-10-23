"""Footnote web app: ask about your lectures, get answers with playable, checkable sources.

Run from the Footnote folder:
    streamlit run app.py
"""
import streamlit as st

from footnote import llm
from footnote.index import build, cite, get_collection
from footnote.qa import answer
from footnote.utils import course_dirs, load_course

ALL = "All courses"

st.set_page_config(page_title="Footnote", layout="centered")


@st.cache_data
def courses() -> dict[str, dict]:
    """course id -> its settings from course.json (read once, then cached)."""
    return {d.name: load_course(d) for d in course_dirs()}


def show_source(n: int, hit: dict) -> None:
    """One cited source: its citation as a heading, a video player at the cited second, the passage."""
    meta = hit["metadata"]
    with st.expander(f"[{n}] {cite(meta)}"):
        if meta["source_type"] == "video":
            lecture = courses().get(meta.get("course"), {}).get("lectures", {}).get(meta["lecture"], {})
            if lecture.get("video"):
                st.video(lecture["video"], start_time=int(meta["start"]))
        st.markdown(hit["text"])


def show_answer(msg: dict) -> None:
    st.markdown(msg["text"])
    for n, hit in msg["cited"]:
        show_source(n, hit)
    if msg["invalid"]:
        st.warning(f"The answer cites sources that don't exist: {msg['invalid']}")


# ---------- Sidebar ----------
with st.sidebar:
    st.title("Footnote")
    st.caption("Answers from your lecture videos and notes, citing the exact timestamp or page.")
    names = {cid: info["name"] for cid, info in courses().items()}
    choice = st.selectbox("Course", [ALL, *names], format_func=lambda c: names.get(c, c))
    course = None if choice == ALL else choice
    st.caption(f"Model: `{llm.MODEL}` via {llm.PROVIDER}")
    if st.button("Clear chat"):
        st.session_state.messages = []

# ---------- Make sure there is something to search ----------
if get_collection().count() == 0:
    st.warning("The search index is empty. Build it from the processed lectures (takes a few seconds).")
    if st.button("Build index"):
        with st.spinner("Building the index..."):
            build()
        st.rerun()
    st.stop()

# ---------- Chat ----------
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:          # redraw the conversation so far
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["text"])
        else:
            show_answer(msg)

if question := st.chat_input("Ask about your lectures..."):
    st.session_state.messages.append({"role": "user", "text": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your lectures and notes..."):
                result = answer(question, course)
        except (llm.LLMError, llm.QuotaExhaustedError) as e:
            st.error(f"The model couldn't answer: {e}")
            st.stop()
        msg = {
            "role": "assistant",
            "text": result["answer"],
            "cited": [(n, result["hits"][n - 1]) for n, _ in result["sources"]],
            "invalid": result["invalid"],
        }
        show_answer(msg)
    st.session_state.messages.append(msg)
