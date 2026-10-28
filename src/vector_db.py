from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
from data_loader import EMBED_DIM
import os
class QdrantStorage:
    def __init__(self, url="https://393412cd-c3c8-4629-8ccd-4a3965f60e86.us-east-1-1.aws.cloud.qdrant.io", collection="docs", dim=EMBED_DIM):
        self.client = QdrantClient(url=url, port=443, api_key=os.getenv("QDRANT_API_KEY"), timeout=30)
        self.collection = collection
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(collection_name= self.collection,
                                          vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
                                              )
    def upsert(self, ids, vectors, payloads):
        points = [
            PointStruct(
                id=ids[i],
                vector=vectors[i],
                payload=payloads[i],
            )
            for i in range(len(ids))
        ]
        self.client.upsert(self.collection, points=points)
    def search(self, query_vector, top_k:int = 5):
        results = self.client.query_points(
        collection_name=self.collection,
        query=query_vector,
        with_payload=True,
        limit=top_k).points

        contexts = []
        sources = set()
        for result in results:
            payload = getattr(result, "payload", None) or {}
            text = payload.get("text", "")
            source = payload.get("source", "")
            if text:
                contexts.append(text)
                sources.add(source)
        return {"contexts": contexts, "sources": list(sources)}

