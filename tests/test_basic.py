from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from custom_types import RAGChunkAndSrc, RAGSearchResult, RAGUpsertResult
from data_loader import EMBED_DIM, embed_query, embed_texts, load_and_chunk_pdf, split_text
from vector_db import QdrantStorage

SAMPLE = Path(__file__).resolve().parent.parent / "src" / "samples" / "how_this_demo_works.pdf"


def test_imports():
    """The Lambda entrypoint imports without crashing"""
    import main
    assert main.handler is not None


def test_rag_search_result_model():
    result = RAGSearchResult(contexts=["hello world"], sources=["test.pdf"])
    assert result.contexts == ["hello world"]
    assert result.sources == ["test.pdf"]


def test_rag_upsert_result_model():
    assert RAGUpsertResult(ingested=5).ingested == 5


def test_rag_chunk_and_src_model():
    result = RAGChunkAndSrc(chunks=["chunk1", "chunk2"], source_id="test.pdf")
    assert len(result.chunks) == 2
    assert result.source_id == "test.pdf"


def test_split_text_overlaps_and_bounds():
    chunks = split_text("word " * 1000, size=100, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 100 for c in chunks)


def test_rejects_non_pdf(tmp_path):
    fake = tmp_path / "evil.pdf"
    fake.write_bytes(b"<html>not a pdf</html>")
    with pytest.raises(ValueError):
        load_and_chunk_pdf(str(fake))


def test_sample_pdf_chunks():
    chunks = load_and_chunk_pdf(str(SAMPLE))
    assert chunks
    assert "Inngest" in " ".join(chunks)


def test_embedding_dims():
    vecs = embed_texts(["hello", "world"])
    assert len(vecs) == 2 and len(vecs[0]) == EMBED_DIM
    assert len(embed_query("hello")) == EMBED_DIM


def test_search_is_scoped_to_session():
    store = QdrantStorage(client=QdrantClient(":memory:"))
    texts = ["The cat sat on the mat.", "Stocks rose sharply today."]
    store.upsert(["00000000-0000-0000-0000-000000000001", "00000000-0000-0000-0000-000000000002"],
                 embed_texts(texts),
                 [{"text": texts[0], "source": "a.pdf", "session_id": "a" * 32, "created_at": 100},
                  {"text": texts[1], "source": "b.pdf", "session_id": "b" * 32, "created_at": 200}])
    found = store.search(embed_query("Where did the cat sit?"), "a" * 32, top_k=5)
    assert found["sources"] == ["a.pdf"]
    store.delete_older_than(150)
    assert store.search(embed_query("cat"), "a" * 32)["contexts"] == []
    assert store.search(embed_query("stocks"), "b" * 32)["contexts"]
