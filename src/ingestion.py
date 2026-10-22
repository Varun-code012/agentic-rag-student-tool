# src/ingestion.py
import pdfplumber
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

# Loaded once, reused for every PDF — loading this model is the slow part,
# so we don't want to reload it per upload.
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def extract_text_from_pdf(uploaded_file) -> str:
    """Pulls all text out of an uploaded PDF, page by page."""
    text_parts = []
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:  # some pages (e.g. pure images) return None
                text_parts.append(page_text)
    return "\n".join(text_parts)


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 150) -> list[str]:
    """
    Splits text into overlapping chunks.
    chunk_size and overlap are measured in characters here (simple and good
    enough for a first version — word/token-based chunking is a later refinement).
    """
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap  # step forward, but re-include the overlap
    return chunks


def build_faiss_index(chunks: list[str]):
    """
    Embeds every chunk and builds a FAISS index for similarity search.
    Returns the index AND the original chunk texts, because FAISS only
    stores vectors — it has no idea what text a vector came from unless
    we keep that mapping ourselves.
    """
    embeddings = _embedding_model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")  # FAISS requires float32

    dimension = embeddings.shape[1]  # length of each embedding vector
    index = faiss.IndexFlatL2(dimension)  # L2 = straight-line distance between vectors
    index.add(embeddings)

    return index, chunks


def process_pdf(uploaded_file) -> tuple:
    """The full pipeline: PDF in, (index, chunks) out."""
    raw_text = extract_text_from_pdf(uploaded_file)

    if not raw_text.strip():
        raise ValueError(
            "No readable text found in this PDF. It may be a scanned document "
            "or image-based file — try a PDF with selectable text instead."
        )

    chunks = chunk_text(raw_text)
    index, chunks = build_faiss_index(chunks)
    return index, chunks