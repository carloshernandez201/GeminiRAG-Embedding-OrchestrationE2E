from dotenv import load_dotenv
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from openai import OpenAI
import os
import boto3

load_dotenv()

import os
import boto3

s3 = boto3.client("s3")
S3_BUCKET = os.getenv("S3_BUCKET")


def download_pdf_from_s3(s3_key: str) -> str:
    local_path = os.path.join("/tmp", os.path.basename(s3_key))
    s3.download_file(S3_BUCKET, s3_key, local_path)
    return local_path

EMBED_MODEL = "text-embedding-3-small"
EMBED_DIM = 1536

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks



_client = None
def get_client() -> OpenAI:
    # Created lazily so the app still starts when OPENAI_API_KEY is missing
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = get_client().embeddings.create(
        model=EMBED_MODEL,
        input=texts,
        dimensions=EMBED_DIM,
    )
    return [d.embedding for d in response.data]
