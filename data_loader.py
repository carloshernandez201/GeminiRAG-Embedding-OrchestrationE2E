from dotenv import load_dotenv
from llama_index.readers.file import PDFReader
from llama_index.core.node_parser import SentenceSplitter
from google.cloud import aiplatform
from vertexai.language_models import TextEmbeddingModel

load_dotenv()

# Init Vertex AI ONCE
aiplatform.init(
    project="ragapp-483600",
    location="us-east1",
)

EMBED_MODEL = "textembedding-gecko@003"
EMBED_DIM = 768

splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=200)

def load_and_chunk_pdf(path: str):
    docs = PDFReader().load_data(file=path)
    texts = [d.text for d in docs if getattr(d, "text", None)]
    chunks = []
    for t in texts:
        chunks.extend(splitter.split_text(t))
    return chunks


def embed_texts(texts: list[str]) -> list[list[float]]:
    model = TextEmbeddingModel.from_pretrained(EMBED_MODEL)
    embeddings = model.get_embeddings(texts)
    return [e.values for e in embeddings]
