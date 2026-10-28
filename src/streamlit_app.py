import asyncio
import io
import uuid
from pathlib import Path

import streamlit as st

from streamlit_backend import send_rag_ingest_event, send_rag_query_event, upload_pdf_to_s3, wait_for_run_output

MAX_PDFS_PER_SESSION = 3
SAMPLE_PDF = Path(__file__).parent / "samples" / "how_this_demo_works.pdf"

st.set_page_config(page_title="RAG PDF Search", page_icon="📄", layout="centered")

# One random id per browser session. Uploads and searches are scoped to it,
# so visitors never see each other's documents.
if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex
    st.session_state.ingested = []
session_id = st.session_state.session_id


def ingest(file) -> None:
    with st.spinner(f"Ingesting {file.name}..."):
        key = upload_pdf_to_s3(file, session_id)
        event_id = asyncio.run(send_rag_ingest_event(key, session_id))
        output = wait_for_run_output(event_id)
    st.session_state.ingested.append(file.name)
    st.success(f"Ingested {file.name} ({output.get('ingested', 0)} chunks)")


st.title("Search your PDFs")
st.caption(
    "Embeddings run locally (bge-small via fastembed) inside AWS Lambda, orchestrated by Inngest. "
    "Your PDFs are private to this tab: the file is deleted right after processing "
    "and its vectors are removed within an hour."
)

remaining = MAX_PDFS_PER_SESSION - len(st.session_state.ingested)
if remaining <= 0:
    st.info(f"Limit of {MAX_PDFS_PER_SESSION} PDFs per session reached.")
else:
    uploaded = st.file_uploader("Upload a PDF (max 10 MB, 50 pages)", type=["pdf"])
    if uploaded is not None and uploaded.name not in st.session_state.ingested:
        try:
            ingest(uploaded)
        except Exception:
            st.error("Couldn't process that PDF. Make sure it's a text PDF under 50 pages.")
    if SAMPLE_PDF.exists() and SAMPLE_PDF.name not in st.session_state.ingested:
        if st.button("No PDF handy? Try a sample"):
            sample = io.BytesIO(SAMPLE_PDF.read_bytes())
            sample.name = SAMPLE_PDF.name
            try:
                ingest(sample)
            except Exception:
                st.error("Couldn't load the sample. Try again in a moment.")

st.divider()
st.subheader("Ask a question")

with st.form("rag_query_form"):
    question = st.text_input("Your question", max_chars=500)
    top_k = st.number_input("How many passages", min_value=1, max_value=10, value=3, step=1)
    submitted = st.form_submit_button("Search", disabled=not st.session_state.ingested)

if submitted and question.strip():
    with st.spinner("Searching..."):
        try:
            event_id = asyncio.run(send_rag_query_event(question.strip(), int(top_k), session_id))
            output = wait_for_run_output(event_id)
        except Exception:
            st.error("Search failed. Try again in a moment.")
            output = {}
    passages = output.get("passages", [])
    if not passages and output:
        st.write("No matching passages.")
    for p in passages:
        # Plain text only, so PDF content can't inject markdown, links or HTML
        st.caption(f"{p.get('source', '')} · score {p.get('score', 0)}")
        st.code(p.get("text", ""), language=None, wrap_lines=True)
elif not st.session_state.ingested:
    st.caption("Upload a PDF or load the sample first.")
