import asyncio
from pathlib import Path
import time

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests
import boto3
#b 
load_dotenv()

s3 = boto3.client("s3")
S3_BUCKET = os.getenv("S3_BUCKET")



st.set_page_config(page_title="RAG Ingest PDF", page_icon="📄", layout="centered")


@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app")

#b
def upload_pdf_to_s3(file) -> str:
    key = f"uploads/{file.name}"
    s3.upload_fileobj(file, S3_BUCKET, key)
    return key

#b
async def send_rag_ingest_event(s3_key: str) -> None:
    client = get_inngest_client()
    await client.send(
        inngest.Event(
            name="rag/ingest_pdf",
            data={
                "s3_key": s3_key,
                "source_id": s3_key.split("/")[-1],
            },
        )
    )


st.title("Upload a PDF to Ingest")
uploaded = st.file_uploader("Choose a PDF", type=["pdf"], accept_multiple_files=False)
url = 'IDK'
if uploaded is not None:
    with st.spinner("Uploading and triggering ingestion..."):
        key = requests.get(f"{url}/api.uploaded_file", files=uploaded)   
        #asyncio.run(send_rag_ingest_event(key))#  call
        asyncio.run(requests.get("f{url}/api/send_rag_inngest_event", data=key)) #idk if u actualy are sup 2 use key here
        time.sleep(0.3)
    st.success(f"Triggered ingestion for: {key.split('/')[-1]}")
    st.caption("You can upload another PDF if you like.")

st.divider()
st.title("Ask a question about your PDFs")


async def send_rag_query_event(question: str, top_k: int) -> None:
    client = get_inngest_client()
    result = await client.send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": question,
                "top_k": top_k,
            },
        )
    )
    #output evt id
    return result[0]



def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 0.5) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = requests.get(f"{url}/api.uploaded_file", data=event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status
            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)


with st.form("rag_query_form"):
    question = st.text_input("Your question")
    top_k = st.number_input("How many chunks to retrieve", min_value=1, max_value=20, value=5, step=1)
    submitted = st.form_submit_button("Ask")

    if submitted and question.strip():
        with st.spinner("Sending event and generating answer..."):
            # Fire-and-forget event to Inngest for observability/workflow
            event_id = asyncio.run(send_rag_query_event(question.strip(), int(top_k)))
            # Poll the local Inngest API for the run's output
            output = wait_for_run_output(event_id)
            answer = output.get("answer", "")
            sources = output.get("sources", [])

        st.subheader("Answer")
        st.write(answer or "(No answer)")
        if sources:
            st.caption("Sources")
            for s in sources:
                st.write(f"- {s}")