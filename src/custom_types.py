import pydantic

class RAGChunkAndSrc(pydantic.BaseModel):
    chunks: list[str]
    source_id: str = None

class RAGUpsertResult(pydantic.BaseModel):
    ingested:int

class RAGSearchResult(pydantic.BaseModel):
    contexts: list[str]
    sources: list[str]
    scores: list[float] = []
    failed: bool = False

class RAGQueryResult(pydantic.BaseModel):
    passages: list[dict]
    num_contexts: int
