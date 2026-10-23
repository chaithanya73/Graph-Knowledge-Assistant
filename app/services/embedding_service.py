"""
Embedding Service — Generate vector embeddings using sentence-transformers.

Converts text into dense numerical vectors (768-dimensional by default) that capture
semantic meaning. Two sentences with similar meaning produce similar vectors,
even if they use completely different words.

INTERVIEW: "What is a vector embedding?"
- A fixed-size numerical representation of text where semantic similarity = vector proximity.
- "Python is a programming language" and "Python is a coding language" → nearly identical vectors.
- "Python is a programming language" and "The weather is sunny" → very different vectors.
- We measure similarity using cosine similarity (angle between vectors), ranging from -1 to 1.

INTERVIEW: "Why sentence-transformers instead of OpenAI embeddings?"
- Free, runs locally, no API latency, no rate limits, no data leaves your machine.
- In production, you'd benchmark both and choose based on quality vs cost tradeoffs.
- sentence-transformers uses the 'all-MiniLM-L6-v2' model: 384 dimensions, very fast.

INTERVIEW: "Why not just use TF-IDF or BM25?"
- TF-IDF/BM25 are keyword-based: "automobile" and "car" are completely different.
- Embeddings are semantic: "automobile" and "car" have ~0.85 cosine similarity.
- For knowledge retrieval, semantic understanding is critical.
"""

from typing import List
from sentence_transformers import SentenceTransformer

# Model loaded once at module level — ~80MB, takes ~2 seconds on first load.
# Subsequent calls reuse the cached model (no reload).
# INTERVIEW: "Why load at module level?"
# - Module-level = loaded once when first imported, then reused.
# - If loaded inside a function, it would reload the 80MB model on EVERY call.
_model: SentenceTransformer | None = None

# Model name — small, fast, good quality. 384-dimensional embeddings.
MODEL_NAME = "all-MiniLM-L6-v2"


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model on first use."""
    global _model
    if _model is None:
        print(f"[EMBEDDING] Loading model: {MODEL_NAME}...")
        _model = SentenceTransformer(MODEL_NAME)
        print(f"[EMBEDDING] Model loaded. Embedding dimension: {_model.get_sentence_embedding_dimension()}")
    return _model


def generate_embedding(text: str) -> List[float]:
    """
    Generate a single embedding vector for a text string.

    Returns:
        List of floats (384 dimensions for all-MiniLM-L6-v2).
    """
    model = _get_model()
    embedding = model.encode(text, normalize_embeddings=True)
    return embedding.tolist()


def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in a single batch.

    INTERVIEW: "Why batch instead of one-by-one?"
    - GPU/CPU SIMD parallelism: processing 100 texts in one batch is ~10x faster
      than processing them one at a time.
    - Reduces Python overhead (function call overhead, GIL context switching).
    - sentence-transformers internally pads and stacks inputs into a single tensor.
    """
    model = _get_model()
    embeddings = model.encode(texts, normalize_embeddings=True, batch_size=32)
    return [emb.tolist() for emb in embeddings]


def get_embedding_dimension() -> int:
    """Return the dimensionality of the embedding model (384 for MiniLM)."""
    model = _get_model()
    return model.get_sentence_embedding_dimension()
