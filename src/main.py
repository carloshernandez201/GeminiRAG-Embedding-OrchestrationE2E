import logging
from fastapi import FastAPI
import inngest
import inngest.fast_api
from inngest.experimental import ai
from dotenv import load_dotenv
import uuid
import os
import boto3
from data_loader import load_and_chunk_pdf, embed_texts, download_pdf_from_s3
from vector_db import QdrantStorage
from custom_types import RAGQueryResult, RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc
import uvicorn 
from mangum import Mangum
from pathlib import Path
import inngest
from dotenv import load_dotenv
import os
import requests


s3 = boto3.client("s3")
S3_BUCKET = os.getenv("S3_BUCKET")



UNDER_DEV_MESSAGE = "🚧 This feature is under development. Please check back soon."
logger = logging.getLogger("uvicorn")
def get_inngest_client():
    return  inngest.Inngest(
        app_id="rag_app",
        logger=logging.getLogger("uvicorn"),
        serializer=inngest.PydanticSerializer(),
        signing_key=os.getenv("INNGEST_SIGNING_KEY")
    )

inngest_client = get_inngest_client()


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    def _load(ctx: inngest.Context) -> RAGChunkAndSrc:
        s3_key = ctx.event.data["s3_key"]
        source_id = ctx.event.data.get("source_id", s3_key)
        pdf_path = download_pdf_from_s3(s3_key)
        chunks = load_and_chunk_pdf(pdf_path)
        os.remove(pdf_path)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        source_id = chunks_and_src.source_id
        try:
            vecs = embed_texts(chunks)
        except Exception:
            logger.exception("OpenAI embedding failed during ingest")
            return RAGUpsertResult(ingested=0)
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [{"source": source_id, "text": chunks[i]} for i in range(len(chunks))]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", lambda: _load(ctx), output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)

async def upload_s3(ctx):
    

async def rag_query_pdf_ai(ctx: inngest.Context):
    def _search(question: str, top_k: int = 5) -> RAGSearchResult:
        query_vec = embed_texts([question])[0]
        store = QdrantStorage()
        found = store.search(query_vec, top_k)
        return RAGSearchResult(contexts=found["contexts"], sources=found["sources"])

    question = ctx.event.data["question"]
    top_k = int(ctx.event.data.get("top_k", 5))

    def _safe_search(question: str, top_k: int) -> RAGSearchResult:
        try:
            return _search(question, top_k)
        except Exception:
            logger.exception("OpenAI embedding failed during query")
            return RAGSearchResult(contexts=[], sources=[], failed=True)

    under_dev = {"answer": UNDER_DEV_MESSAGE, "sources": [], "num_contexts": 0}

    found = await ctx.step.run("embed-and-search", lambda: _safe_search(question, top_k), output_type=RAGSearchResult)
    if found.failed or not os.getenv("OPENAI_API_KEY"):
        return under_dev

    context_block = "\n\n".join(f"- {c}" for c in found.contexts)
    user_content = (
        "Use the following context to answer the question.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n"
        "Answer concisely using the context above."
    )

    adapter = ai.openai.Adapter(
        auth_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini"
    )

    try:
        res = await ctx.step.ai.infer(
            "llm-answer",
            adapter=adapter,
            body={
                "max_tokens": 1024,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": "You answer questions using only the provided context."},
                    {"role": "user", "content": user_content}
                ]
            }
        )

        answer = res["choices"][0]["message"]["content"].strip()
    except Exception:
        logger.exception("OpenAI answer generation failed")
        return under_dev
    return {"answer": answer, "sources": found.sources, "num_contexts": len(found.contexts)}





def _inngest_api_base() -> str:
    # Local dev server default; configurable via env
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")





app = FastAPI()

inngest.fast_api.serve(app, get_inngest_client(), [rag_ingest_pdf, rag_query_pdf_ai])
'''
@app.get("/api/upload_pdf")
def upload_pdf_to_s3(file) -> str:
    key = f"uploads/{file.name}"
    s3.upload_fileobj(file, S3_BUCKET, key)
    return key'''
'''
@app.get("/api/send_rag_inngest_event")
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
    )'''
@app.get("/api/fetch_runs")
def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    headers = {"Authorization": f"Bearer {os.getenv('INNGEST_SIGNING_KEY')}"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
handler = Mangum(app)