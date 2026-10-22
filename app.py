# app.py
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from src.ingestion import process_pdf
from src.router import retrieve_top_chunks, answer_or_route
from src.synthesize import synthesize_answer_stream
from src.web_agent import search_web

st.set_page_config(page_title="Ask Your Documents", page_icon="📖", layout="centered")

# --- Calm, study-room styling ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Lora:wght@500;600&family=Karla:wght@400;500&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"] { font-family: 'Karla', sans-serif; }
    .stApp { background-color: #F5F0E6; }
    h1, h2, h3 { font-family: 'Lora', serif; color: #23262B; font-weight: 600; }
    p, span, label, .stMarkdown { color: #23262B; }

    /* Text input: remove Streamlit's default red focus/validation ring */
    .stTextInput > div > div {
        border: 1px solid #E3DCCB !important;
        border-radius: 6px !important;
        box-shadow: none !important;
        background-color: #FBF8F2 !important;
    }
    .stTextInput > div > div:focus-within {
        border-color: #3F5945 !important;
        box-shadow: 0 0 0 1px #3F5945 !important;
    }
    .stTextInput input {
        color: #23262B !important;
    }

    .answer-card, .answer-card p {
        background-color: #FBF8F2;
        border: 1px solid #E3DCCB;
        border-radius: 8px;
        padding: 1.5rem 1.75rem;
        margin-top: 1rem;
        color: #23262B !important;
    }
    .source-note {
        font-family: 'Lora', serif;
        font-style: italic;
        font-size: 0.85rem;
        color: #6B6F66;
        margin-top: 0.75rem;
    }
    .source-note.local { color: #5C7A5F; }
    .source-note.web { color: #A6702F; }
</style>
""", unsafe_allow_html=True)

st.title("Ask Your Documents")
st.caption("Upload a PDF, or just ask — it'll search your document first, and the web if it needs to.")

if "index" not in st.session_state:
    st.session_state.index = None
    st.session_state.chunks = None
    st.session_state.filename = None

uploaded_file = st.file_uploader("Upload a PDF (optional)", type="pdf")

if uploaded_file is not None and uploaded_file.name != st.session_state.filename:
    with st.spinner("Reading your document..."):
        try:
            index, chunks = process_pdf(uploaded_file)
            st.session_state.index = index
            st.session_state.chunks = chunks
            st.session_state.filename = uploaded_file.name
            st.success(f"Ready — {len(chunks)} sections indexed from {uploaded_file.name}")
        except ValueError as e:
            st.error(str(e))
            st.session_state.index = None

question = st.text_input("Ask a question:")

if question:
    if st.session_state.index is not None:
        # We have a document — check it first.
        with st.spinner("Checking your document..."):
            top_chunks = retrieve_top_chunks(question, st.session_state.index, st.session_state.chunks)
            result = answer_or_route(question, top_chunks)

        if result["sufficient"]:
            st.markdown(f'<div class="answer-card">{result["answer"]}</div>', unsafe_allow_html=True)
            st.markdown('<p class="source-note local">— from your document</p>', unsafe_allow_html=True)
        else:
            with st.spinner("Not in your document — searching the web..."):
                web_result = search_web(question)
            st.markdown('<div class="answer-card">', unsafe_allow_html=True)
            st.write_stream(synthesize_answer_stream(question, [web_result]))
            st.markdown('</div>', unsafe_allow_html=True)
            st.markdown('<p class="source-note web">— from the web</p>', unsafe_allow_html=True)
    else:
        # No document uploaded — skip the router, go straight to web.
        with st.spinner("Searching the web..."):
            web_result = search_web(question)
        st.markdown('<div class="answer-card">', unsafe_allow_html=True)
        st.write_stream(synthesize_answer_stream(question, [web_result]))
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<p class="source-note web">— from the web (no document uploaded)</p>', unsafe_allow_html=True)