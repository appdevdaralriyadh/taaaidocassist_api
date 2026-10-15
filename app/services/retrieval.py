"""
Retrieval + query routing (spec §3.4, §8 build order item 3): given a
user's message, finds the most similar document chunks in the shared
knowledge base using SQL Server 2025's native VECTOR_DISTANCE('cosine',
...) function, computed inside the database. This retires the old
in-process numpy brute-force cosine-similarity search entirely -- no
embedding is ever fetched into Python and compared by hand anymore; the
database does the comparison and the sorting.

VECTOR_DISTANCE('cosine', ...) returns a DISTANCE in [0, 2] -- 0 =
identical, 2 = completely opposite -- the exact inverse of the old
hand-computed cosine SIMILARITY score this module used to produce
(higher = better). LOWER is better now. app/config.py's
MAX_COSINE_DISTANCE is the cutoff; the old SIMILARITY_THRESHOLD setting
is gone, not kept alongside this one under the old name.

The distance search itself *is* the query router: the caller (chat.py)
treats "nothing cleared the cutoff" as the general-knowledge signal.
There's no separate classifier call.

Optional reranker (RERANK_MODEL in app/config.py, off by default): SQL
Server returns the RERANK_CANDIDATES closest chunks within the cutoff,
Voyage AI's reranker re-orders them by how well they answer the question,
and the best TOP_K_CHUNKS (scoring at least RERANK_MIN_SCORE) are returned
in that order. If the reranker fails, the distance order is used instead.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
# embed_query raises EmbeddingServiceError when the provider (e.g. Voyage AI)
# can't be reached -- callers turn that into a clear "search unavailable" message.
from app.services.embeddings import (
    EmbeddingServiceError,
    embed_query,
    rerank,
    rerank_enabled,
    to_vector_literal,
)

logger = logging.getLogger("darai.retrieval")


@dataclass
class RetrievedChunk:
    chunk_text: str
    filename: str
    distance: float  # cosine distance, LOWER is better -- see module docstring
    # Voyage AI reranker relevance (0-1, HIGHER is better); None when the
    # reranker is off or its call failed and the distance order was used.
    rerank_score: Optional[float] = None


# Wrapped in a derived table so VECTOR_DISTANCE is computed once per row,
# then filtered/sorted/limited in the outer query, rather than repeating
# it once in a WHERE clause and again in ORDER BY.
#
# The EmbeddingModel provenance check (WHERE c.EmbeddingModel = ...)
# replaces the old Python-side skip-and-warn loop that used to run per
# mismatched chunk -- it's now enforced directly in SQL, so a model
# switch with no re-ingest yields an empty result rather than a
# per-chunk warning log. That's a deliberate simplification of this
# rewrite; if you need to know *why* a query came back empty (genuinely
# no match vs. every chunk still stamped with a stale model), check
# GET /api/chat/debug-retrieval or query DarAI_DocumentChunks.EmbeddingModel
# directly.
#
# The dimension in CAST(... AS VECTOR(n)) has to be a literal (a type's
# size can't be a bind parameter in T-SQL) -- safe here since it comes
# from our own settings, never from user input.
#
# CAST(:query_vec AS NVARCHAR(MAX)) BEFORE casting to VECTOR -- same
# reason as app/services/ingestion.py's _INSERT_CHUNK_SQL: pyodbc/ODBC
# Driver 17 sends this long a string parameter as a wire type SQL Server
# reports as `ntext`, and VECTOR's CAST rejects ntext as a source type
# directly (though it accepts nvarchar). The intermediate cast avoids
# that without touching how pyodbc binds the parameter.
_SEARCH_SQL = text(
    f"""
    DECLARE @q VECTOR({settings.EMBEDDING_DIMENSIONS}) =
        CAST(CAST(:query_vec AS NVARCHAR(MAX)) AS VECTOR({settings.EMBEDDING_DIMENSIONS}));

    SELECT ChunkText, FileName, Distance
    FROM (
        SELECT c.ChunkText, d.FileName,
               VECTOR_DISTANCE('cosine', c.Embedding, @q) AS Distance
        FROM dbo.DarAI_DocumentChunks c
        JOIN dbo.DarAI_Documents d ON d.Id = c.DocumentId
        WHERE c.EmbeddingModel = :embedding_model
    ) AS scored
    WHERE Distance <= :max_distance
    ORDER BY Distance ASC
    OFFSET 0 ROWS FETCH NEXT :top_k ROWS ONLY;
    """
)


def search(
    db: Session,
    query: str,
    *,
    top_k: Optional[int] = None,
    max_distance: Optional[float] = None,
) -> list[RetrievedChunk]:
    """
    Embeds `query` and returns the top-K chunks whose cosine distance to
    it is at or below `max_distance`, best (lowest-distance) match first.
    Returns [] if the knowledge base is empty, every chunk is stamped
    with a different EMBEDDING_MODEL than the current one, or nothing
    clears the cutoff -- any of those is the "fall back to general
    knowledge" signal the caller acts on, and this function doesn't
    distinguish between them (see the module-level comment on _SEARCH_SQL
    for how to tell them apart when debugging).

    With the reranker on (RERANK_MODEL), more candidates are fetched and
    the result is in reranker order (most relevant first) instead.
    """
    top_k = settings.TOP_K_CHUNKS if top_k is None else top_k
    max_distance = settings.MAX_COSINE_DISTANCE if max_distance is None else max_distance

    use_reranker = rerank_enabled() and top_k > 0
    fetch_k = max(settings.RERANK_CANDIDATES, top_k) if use_reranker else top_k

    query_vector = embed_query(query)

    rows = db.execute(
        _SEARCH_SQL,
        {
            "query_vec": to_vector_literal(query_vector),
            "embedding_model": settings.EMBEDDING_MODEL,
            "max_distance": max_distance,
            "top_k": fetch_k,
        },
    ).mappings().all()

    candidates = [
        RetrievedChunk(
            chunk_text=row["ChunkText"],
            filename=row["FileName"],
            distance=float(row["Distance"]),
        )
        for row in rows
    ]
    if not use_reranker or not candidates:
        return candidates[:top_k]
    return _rerank(query, candidates, top_k)


def _rerank(query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
    try:
        ranked = rerank(query, [c.chunk_text for c in candidates], top_k)
    except EmbeddingServiceError as exc:
        # Chat keeps working on the plain distance order rather than failing.
        logger.warning("Reranker unavailable, using distance order instead: %s", exc)
        return candidates[:top_k]

    min_score = settings.RERANK_MIN_SCORE
    kept = []
    for index, score in ranked:
        if score < min_score:
            continue
        chunk = candidates[index]
        kept.append(
            RetrievedChunk(
                chunk_text=chunk.chunk_text,
                filename=chunk.filename,
                distance=chunk.distance,
                rerank_score=score,
            )
        )
    logger.info(
        "Reranked %d candidates: kept %d (min score %.2f, best %s)",
        len(candidates),
        len(kept),
        min_score,
        f"{ranked[0][1]:.3f}" if ranked else "n/a",
    )
    return kept[:top_k]
