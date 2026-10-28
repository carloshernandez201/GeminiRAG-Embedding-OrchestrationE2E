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



def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app")

def upload_pdf_to_s3(file) -> str:
    key = f"uploads/{file.name}"
    s3.upload_fileobj(file, S3_BUCKET, key)
    return key

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

def _inngest_api_base() -> str:
    # Local dev server default; configurable via env
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")

#b
def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    headers = {"Authorization": f"Bearer {os.getenv('INNGEST_SIGNING_KEY')}"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])