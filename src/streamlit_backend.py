import os
import re
import time

import boto3
import inngest
import requests
from dotenv import load_dotenv

load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET")

_s3 = None
def get_s3():
    # On EC2 this uses the instance role, so no access keys live on the box
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3")
    return _s3


def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app")


def safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]", "_", os.path.basename(name))[:100]
    return name if name.lower().endswith(".pdf") else f"{name}.pdf"


def upload_pdf_to_s3(file, session_id: str) -> str:
    key = f"uploads/{session_id}/{safe_filename(file.name)}"
    get_s3().upload_fileobj(file, S3_BUCKET, key, ExtraArgs={"ContentType": "application/pdf"})
    return key


async def send_rag_ingest_event(s3_key: str, session_id: str) -> str:
    result = await get_inngest_client().send(
        inngest.Event(name="rag/ingest_pdf", data={"s3_key": s3_key, "session_id": session_id})
    )
    return result[0]


async def send_rag_query_event(question: str, top_k: int, session_id: str) -> str:
    result = await get_inngest_client().send(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={"question": question, "top_k": top_k, "session_id": session_id},
        )
    )
    return result[0]


def _inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "https://api.inngest.com/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    headers = {"Authorization": f"Bearer {os.getenv('INNGEST_SIGNING_KEY')}"}
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    return resp.json().get("data", [])


def wait_for_run_output(event_id: str, timeout_s: float = 120.0, poll_interval_s: float = 1.0) -> dict:
    start = time.time()
    last_status = None
    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            last_status = run.get("status") or last_status
            if last_status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}
            if last_status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {last_status}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for run output (last status: {last_status})")
        time.sleep(poll_interval_s)
