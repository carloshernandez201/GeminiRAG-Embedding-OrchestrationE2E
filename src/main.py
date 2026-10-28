import logging
import os
import re
import time
import uuid

import inngest
import inngest.fast_api
from dotenv import load_dotenv
from fastapi import FastAPI
from mangum import Mangum

from custom_types import RAGChunkAndSrc, RAGSearchResult, RAGUpsertResult
from data_loader import delete_pdf_from_s3, download_pdf_from_s3, embed_query, embed_texts, load_and_chunk_pdf
from vector_db import QdrantStorage

load_dotenv()

logger = logging.getLogger("uvicorn")

# Visitor data is ephemeral: the PDF is deleted from S3 right after ingest and
# its vectors are removed by rag_cleanup once they are older than this.
RETENTION_SECONDS = 60 * 60
SESSION_ID_RE = re.compile(r"^[0-9a-f]{32}$")

def get_inngest_client():
    return  inngest.Inngest(
        app_id="rag_app",
        logger=logging.getLogger("uvicorn"),
        serializer=inngest.PydanticSerializer(),
        signing_key=os.getenv("INNGEST_SIGNING_KEY")
    )

inngest_client = get_inngest_client()


def _session_id(ctx: inngest.Context) -> str:
    session_id = str(ctx.event.data.get("session_id", ""))
    if not SESSION_ID_RE.match(session_id):
        raise inngest.NonRetriableError("Invalid session_id")
    return session_id


@inngest_client.create_function(
    fn_id="RAG: Ingest PDF",
    trigger=inngest.TriggerEvent(event="rag/ingest_pdf")
)
async def rag_ingest_pdf(ctx: inngest.Context):
    session_id = _session_id(ctx)
    s3_key = str(ctx.event.data["s3_key"])
    # Only accept keys inside this session's own upload folder
    if not s3_key.startswith(f"uploads/{session_id}/") or ".." in s3_key:
        raise inngest.NonRetriableError("s3_key outside session folder")
    source_id = os.path.basename(s3_key)

    def _load() -> RAGChunkAndSrc:
        pdf_path = download_pdf_from_s3(s3_key)
        try:
            chunks = load_and_chunk_pdf(pdf_path)
        finally:
            os.remove(pdf_path)
            delete_pdf_from_s3(s3_key)
        return RAGChunkAndSrc(chunks=chunks, source_id=source_id)

    def _upsert(chunks_and_src: RAGChunkAndSrc) -> RAGUpsertResult:
        chunks = chunks_and_src.chunks
        vecs = embed_texts(chunks)
        now = int(time.time())
        ids = [str(uuid.uuid5(uuid.NAMESPACE_URL, f"{session_id}:{source_id}:{i}")) for i in range(len(chunks))]
        payloads = [
            {"source": source_id, "text": chunks[i], "session_id": session_id, "created_at": now}
            for i in range(len(chunks))
        ]
        QdrantStorage().upsert(ids, vecs, payloads)
        return RAGUpsertResult(ingested=len(chunks))

    chunks_and_src = await ctx.step.run("load-and-chunk", _load, output_type=RAGChunkAndSrc)
    ingested = await ctx.step.run("embed-and-upsert", lambda: _upsert(chunks_and_src), output_type=RAGUpsertResult)
    return ingested.model_dump()


@inngest_client.create_function(
    fn_id="RAG: Query PDF",
    trigger=inngest.TriggerEvent(event="rag/query_pdf_ai")
)
async def rag_query_pdf_ai(ctx: inngest.Context):
    session_id = _session_id(ctx)
    question = str(ctx.event.data["question"])[:500]
    top_k = min(max(int(ctx.event.data.get("top_k", 5)), 1), 10)

    def _search() -> RAGSearchResult:
        found = QdrantStorage().search(embed_query(question), session_id, top_k)
        return RAGSearchResult(**found)

    found = await ctx.step.run("embed-and-search", _search, output_type=RAGSearchResult)
    # Retrieval-only: return the best-matching passages instead of an LLM answer
    passages = [
        {"text": c, "source": s, "score": round(sc, 3)}
        for c, s, sc in zip(found.contexts, found.sources, found.scores)
    ]
    return {"passages": passages, "num_contexts": len(passages)}


@inngest_client.create_function(
    fn_id="RAG: Cleanup expired data",
    trigger=inngest.TriggerCron(cron="*/15 * * * *")
)
async def rag_cleanup(ctx: inngest.Context):
    cutoff = int(time.time()) - RETENTION_SECONDS
    await ctx.step.run("delete-old-points", lambda: QdrantStorage().delete_older_than(cutoff))
    return {"deleted_before": cutoff}


app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

inngest.fast_api.serve(app, inngest_client, [rag_ingest_pdf, rag_query_pdf_ai, rag_cleanup])

handler = Mangum(app)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
