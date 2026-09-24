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
"""

import logging
from dataclasses import dataclass
from typing import Optional

from sqlalchemy import text
from sqlalchemy.orm import Session

from app.config import settings
from app.services.embeddings import embed_query, to_vector_literal

logger = logging.getLogger("darai.retrieval")


@dataclass
class RetrievedChunk:
    chunk_text: str
    filename: str
    distance: float  # cosine distance, LOWER is better -- see module docstring


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
    """
    top_k = settings.TOP_K_CHUNKS if top_k is None else top_k
    max_distance = settings.MAX_COSINE_DISTANCE if max_distance is None else max_distance

    query_vector = embed_query(query)

    rows = db.execute(
        _SEARCH_SQL,
        {
            "query_vec": to_vector_literal(query_vector),
            "embedding_model": settings.EMBEDDING_MODEL,
            "max_distance": max_distance,
            "top_k": top_k,
        },
    ).mappings().all()

    return [
        RetrievedChunk(
            chunk_text=row["ChunkText"],
            filename=row["FileName"],
            distance=float(row["Distance"]),
        )
        for row in rows
    ]
