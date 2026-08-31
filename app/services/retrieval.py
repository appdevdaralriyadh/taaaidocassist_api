"""
Retrieval + query routing (spec §3.4, §8 build order item 3): given a
user's message, finds the most similar document chunks in the shared
knowledge base via cosine similarity, computed in-process with numpy --
no vector DB, per spec §3.3 ("fast enough at single-app scale (milliseconds
even at tens of thousands of chunks)").

The similarity search itself *is* the query router: the caller (chat.py)
treats "nothing cleared the threshold" as the general-knowledge signal.
There's no separate classifier call.
"""

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Document, DocumentChunk
from app.services.embeddings import deserialize_embedding, embed_texts

logger = logging.getLogger("darai.retrieval")


@dataclass
class RetrievedChunk:
    chunk_text: str
    filename: str
    score: float


def search(
    db: Session,
    query: str,
    *,
    top_k: Optional[int] = None,
    threshold: Optional[float] = None,
) -> list[RetrievedChunk]:
    """
    Embeds `query` and returns the top-K chunks whose cosine similarity to
    it meets `threshold`, best match first. Returns [] if the knowledge
    base is empty or nothing clears the bar -- that's the "fall back to
    general knowledge" signal the caller acts on.
    """
    top_k = settings.TOP_K_CHUNKS if top_k is None else top_k
    threshold = settings.SIMILARITY_THRESHOLD if threshold is None else threshold

    rows = (
        db.query(DocumentChunk.ChunkText, DocumentChunk.Embedding, Document.FileName)
        .join(Document, Document.Id == DocumentChunk.DocumentId)
        .all()
    )
    if not rows:
        return []

    query_vector = embed_texts([query])[0]
    query_dim = query_vector.shape[0]

    texts: list[str] = []
    filenames: list[str] = []
    vectors: list[np.ndarray] = []
    for chunk_text, embedding_blob, filename in rows:
        vector = deserialize_embedding(embedding_blob)
        if vector.shape[0] != query_dim:
            # Guards against a stale chunk embedded under a since-changed
            # EMBEDDING_MODEL -- skip it rather than crash the whole
            # search over one bad row.
            logger.warning(
                "Skipping chunk from '%s': embedding dim %d != query dim %d "
                "(EMBEDDING_MODEL changed since it was ingested?)",
                filename,
                vector.shape[0],
                query_dim,
            )
            continue
        texts.append(chunk_text)
        filenames.append(filename)
        vectors.append(vector)

    if not vectors:
        return []

    matrix = np.vstack(vectors)  # (N, dim)

    # Cosine similarity, vectorized over every chunk at once: normalize
    # both sides, then a single dot product.
    matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix_norms[matrix_norms == 0] = 1e-8  # guard a degenerate all-zero embedding
    normalized_matrix = matrix / matrix_norms

    query_norm = np.linalg.norm(query_vector)
    normalized_query = query_vector / (query_norm if query_norm else 1e-8)

    scores = normalized_matrix @ normalized_query  # (N,)

    order = np.argsort(-scores)  # descending

    results: list[RetrievedChunk] = []
    for idx in order[:top_k]:
        score = float(scores[idx])
        if score < threshold:
            break  # descending order -- nothing further out clears the bar either
        results.append(
            RetrievedChunk(chunk_text=texts[idx], filename=filenames[idx], score=score)
        )

    return results
