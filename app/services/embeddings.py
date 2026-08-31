"""
Wraps a local sentence-transformers model for embeddings, and the
VARBINARY(MAX) <-> vector serialization spec §3.3 calls for ("Embeddings
stored as VARBINARY(MAX) (serialized float array)").

Runs entirely on this machine: no API key, no per-call cost, and document
text never leaves your network. The tradeoff is a one-time ~90MB model
download (from Hugging Face, on first run) and CPU time per chunk that's
slower than a hosted embeddings API -- fine at the "single-app scale"
spec §3.3 already assumes.
"""

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

_model: Optional[SentenceTransformer] = None


def _get_model() -> SentenceTransformer:
    # Loaded lazily, once, on first use -- not at import time -- so
    # importing this module (e.g. for tests) doesn't force a model
    # download/load just to read the code.
    global _model
    if _model is None:
        _model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _model


def embed_texts(texts: list[str]) -> list[np.ndarray]:
    """Embeds a batch of chunks locally, order-preserving."""
    vectors = _get_model().encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [np.asarray(vector, dtype=np.float32) for vector in vectors]


def serialize_embedding(vector: np.ndarray) -> bytes:
    return vector.astype(np.float32).tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)
