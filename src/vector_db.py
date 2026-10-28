import os

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    Range,
    VectorParams,
)

from data_loader import EMBED_DIM


class QdrantStorage:
    def __init__(self, collection="docs_bge384", dim=EMBED_DIM, client=None):
        self.client = client or QdrantClient(
            url=os.getenv("QDRANT_CLUSTER_ENDPOINT"),
            port=443,
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=30,
        )
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(
                collection_name=self.collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
            self.client.create_payload_index(self.collection, "session_id", PayloadSchemaType.KEYWORD)
            self.client.create_payload_index(self.collection, "created_at", PayloadSchemaType.INTEGER)

    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(id=ids[i], vector=vectors[i], payload=payloads[i])
            for i in range(len(ids))
        ]
        self.client.upsert(self.collection, points=points)

    def search(self, query_vector, session_id: str, top_k: int = 5):
        # Every search is scoped to one visitor's session
        results = self.client.query_points(
            collection_name=self.collection,
            query=query_vector,
            query_filter=Filter(must=[FieldCondition(key="session_id", match=MatchValue(value=session_id))]),
            with_payload=True,
            limit=top_k,
        ).points

        contexts = []
        sources = []
        scores = []
        for result in results:
            payload = result.payload or {}
            text = payload.get("text", "")
            if text:
                contexts.append(text)
                sources.append(payload.get("source", ""))
                scores.append(float(result.score))
        return {"contexts": contexts, "sources": sources, "scores": scores}

    def delete_older_than(self, ts: int) -> None:
        self.client.delete(
            collection_name=self.collection,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="created_at", range=Range(lt=ts))])
            ),
        )
