import os

import boto3
from dotenv import load_dotenv
from fastembed import TextEmbedding
from pypdf import PdfReader

load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET")

# Local embedding model: no API key, runs inside Lambda. The Dockerfile bakes
# the weights into FASTEMBED_CACHE so cold starts never hit the network.
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384
FASTEMBED_CACHE = os.getenv("FASTEMBED_CACHE")

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MAX_PAGES = 50
MAX_CHUNKS = 300

_s3 = None
def get_s3():
    global _s3
    if _s3 is None:
        _s3 = boto3.client("s3")
    return _s3


def download_pdf_from_s3(s3_key: str) -> str:
    local_path = os.path.join("/tmp", os.path.basename(s3_key))
    get_s3().download_file(S3_BUCKET, s3_key, local_path)
    return local_path


def delete_pdf_from_s3(s3_key: str) -> None:
    get_s3().delete_object(Bucket=S3_BUCKET, Key=s3_key)


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    text = " ".join(text.split())
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        # Prefer to break on a sentence or word boundary inside the window
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind(" ", start, end))
            if cut > start + size // 2:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return [c for c in chunks if c]


def load_and_chunk_pdf(path: str) -> list[str]:
    with open(path, "rb") as f:
        if f.read(5) != b"%PDF-":
            raise ValueError("Not a PDF file")
    reader = PdfReader(path)
    if len(reader.pages) > MAX_PAGES:
        raise ValueError(f"PDF has more than {MAX_PAGES} pages")
    chunks = []
    for page in reader.pages:
        chunks.extend(split_text(page.extract_text() or ""))
        if len(chunks) >= MAX_CHUNKS:
            return chunks[:MAX_CHUNKS]
    return chunks


_model = None
def get_model() -> TextEmbedding:
    # Created lazily so importing this module stays cheap
    global _model
    if _model is None:
        _model = TextEmbedding(EMBED_MODEL, cache_dir=FASTEMBED_CACHE)
    return _model


def embed_texts(texts: list[str]) -> list[list[float]]:
    return [v.tolist() for v in get_model().embed(texts)]


def embed_query(text: str) -> list[float]:
    # bge models expect a query prefix; query_embed adds it
    return next(iter(get_model().query_embed(text))).tolist()
