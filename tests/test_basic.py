import pytest


def test_imports():
    """All our modules import without crashing"""
    from src.vector_db import QdrantStorage
    from src.data_loader import load_and_chunk_pdf, embed_texts
    from src.custom_types import RAGSearchResult, RAGUpsertResult, RAGChunkAndSrc


def test_rag_search_result_model():
    from src.custom_types import RAGSearchResult
    result = RAGSearchResult(contexts=["hello world"], sources=["test.pdf"])
    assert result.contexts == ["hello world"]
    assert result.sources == ["test.pdf"]


def test_rag_upsert_result_model():
    from src.custom_types import RAGUpsertResult
    result = RAGUpsertResult(ingested=5)
    assert result.ingested == 5


def test_rag_chunk_and_src_model():
    from src.custom_types import RAGChunkAndSrc
    result = RAGChunkAndSrc(chunks=["chunk1", "chunk2"], source_id="test.pdf")
    assert len(result.chunks) == 2
    assert result.source_id == "test.pdf"