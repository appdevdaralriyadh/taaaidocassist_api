"""
Wraps a local sentence-transformers model for embeddings.

Storage format: SQL Server 2025's native VECTOR(EMBEDDING_DIMENSIONS)
type (see sql/sqlserver2025/001_create_schema.sql), not a serialized byte
buffer -- the old VARBINARY(MAX) approach (spec §3.3's "Embeddings stored
as VARBINARY(MAX) (serialized float array)") is retired along with the
old SQL Server 2019 database. SQLAlchemy has no built-in VECTOR column
type, and its ORM/Core query builder can't express VECTOR_DISTANCE()
either, so every read or write of DarAI_DocumentChunks.Embedding goes
through raw parameterized SQL (see app/services/ingestion.py and
app/services/retrieval.py) rather than the ORM. A vector crosses that raw
-SQL boundary as a JSON-array string -- SQL Server's own documented text
representation for VECTOR, e.g. '[0.1, 0.2, ...]' -- produced by
to_vector_literal() below, bound as a plain string parameter and
explicitly CAST(... AS VECTOR(n)) in the SQL text. There's no confirmed
native pyodbc wire-level VECTOR type as of this writing (checked pyodbc's
own issue tracker and Microsoft's docs, found nothing) -- this uses the
JSON-array fallback SQL Server documents for non-vector-aware clients,
and it's worth a quick empirical smoke test against your real pyodbc
connection before trusting it under load.

Runs entirely on this machine: no API key, no per-call cost, and document
text never leaves your network. The tradeoff is a one-time ~1.3GB model
download (from Hugging Face, on first run) and CPU time per chunk that's
slower than a hosted embeddings API -- fine at the "single-app scale"
spec §3.3 already assumes.
"""

import json
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


def _validate_dimension(vector: np.ndarray) -> None:
    # Catches a mismatch as early and as loudly as possible -- at
    # embedding time, not as a silent bad INSERT/CAST failure later.
    if vector.shape[0] != settings.EMBEDDING_DIMENSIONS:
        raise ValueError(
            f"{settings.EMBEDDING_MODEL} produced a {vector.shape[0]}-dim "
            f"vector, but app/config.py's EMBEDDING_DIMENSIONS is set to "
            f"{settings.EMBEDDING_DIMENSIONS}. That constant must also "
            f"match the VECTOR(n) size in "
            f"sql/sqlserver2025/001_create_schema.sql -- update "
            f"EMBEDDING_DIMENSIONS (and the schema, if already applied) "
            f"before ingesting anything with this model."
        )


def embed_texts(texts: list[str]) -> list[np.ndarray]:
    """
    Embeds a batch of document/passage chunks locally, order-preserving.
    Deliberately does NOT apply EMBEDDING_QUERY_INSTRUCTION_PREFIX -- for
    models that use that asymmetric convention (see app/config.py's
    comment on that setting), only queries get the prefix; passages are
    always embedded plain. Called from app/services/ingestion.py for
    every chunk of a document being indexed. For a live search query, use
    embed_query() below instead.
    """
    vectors = _get_model().encode(texts, convert_to_numpy=True, show_progress_bar=False)
    result = [np.asarray(vector, dtype=np.float32) for vector in vectors]
    for vector in result:
        _validate_dimension(vector)
    return result


def embed_query(query: str) -> np.ndarray:
    """
    Embeds a single live search query -- NOT for document/passage chunks
    (use embed_texts() for those). Prepends
    settings.EMBEDDING_QUERY_INSTRUCTION_PREFIX first: a harmless no-op
    for most models (the setting defaults to "" for anything that doesn't
    use this convention), but required for BGE-*-en-v1.5-family models to
    reach the retrieval accuracy their model card documents. Called from
    app/services/retrieval.py's search().
    """
    prefixed_query = settings.EMBEDDING_QUERY_INSTRUCTION_PREFIX + query
    return embed_texts([prefixed_query])[0]


def to_vector_literal(vector: np.ndarray) -> str:
    """
    Renders a vector as the JSON-array text SQL Server's VECTOR type
    accepts (e.g. '[0.1, 0.2, ...]'), for binding as a plain string
    parameter into a CAST(:param AS VECTOR(n)) in raw SQL -- see
    app/services/ingestion.py and app/services/retrieval.py. The
    float(x) conversion on each element matters: json.dumps on a bare
    numpy float32 either raises or (depending on numpy version) emits an
    overly long repr, neither of which is the plain decimal text SQL
    Server expects.
    """
    return json.dumps([float(x) for x in vector])
