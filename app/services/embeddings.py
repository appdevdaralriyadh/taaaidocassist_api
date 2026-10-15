"""
Creates the embeddings (vectors) for document chunks and chat questions.

Two providers, chosen by EMBEDDING_PROVIDER in .env (app/config.py):

  * "local" (default): a sentence-transformers model on this machine
    (EMBEDDING_MODEL, e.g. BAAI/bge-large-en-v1.5) -- described below.
  * "voyage": the Voyage AI embeddings API (EMBEDDING_MODEL = a Voyage
    model such as voyage-3.5, VOYAGE_API_KEY). Documents are sent with
    input_type="document" and questions with input_type="query" (Voyage's
    replacement for the bge query prefix), always asking for
    EMBEDDING_DIMENSIONS-sized vectors so the VECTOR(1024) columns stay as
    they are. Document text and questions leave this server. Token usage is
    logged per call ("Voyage AI embeddings: ... tokens").

Every caller goes through embed_texts() (documents) or embed_query()
(questions); nothing else in the app knows which provider is used. Each
stored chunk is stamped with EMBEDDING_MODEL, and search only compares
chunks of the current model -- switching provider or model therefore
needs Clear Knowledge Base + re-adding the documents.

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
import logging
import threading
import time
from typing import Optional

import numpy as np

from app.config import settings

logger = logging.getLogger("darai.embeddings")

_model = None  # local sentence-transformers model, loaded on first use
_model_lock = threading.Lock()
_voyage_client = None
_voyage_lock = threading.Lock()

# Voyage AI accepts up to 1,000 texts per call; smaller batches keep each
# request well inside its per-request token limit (chunks are ~1,000
# characters).
_VOYAGE_MAX_TEXTS = 128
_VOYAGE_MAX_CHARS = 200_000


class EmbeddingServiceError(RuntimeError):
    """The embedding provider couldn't create vectors -- the message is shown to people."""


def provider() -> str:
    return (settings.EMBEDDING_PROVIDER or "local").strip().lower()


def uses_local_model() -> bool:
    return provider() != "voyage"


# ---------------------------------------------------------------------------
# Local model (sentence-transformers)
# ---------------------------------------------------------------------------


def _get_model():
    # Loaded lazily, once, on first use -- not at import time -- so
    # importing this module (e.g. for tests, or with EMBEDDING_PROVIDER=voyage)
    # doesn't load torch or download a model.
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

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


def _local_embed(texts: list[str]) -> list[np.ndarray]:
    vectors = _get_model().encode(texts, convert_to_numpy=True, show_progress_bar=False)
    return [np.asarray(vector, dtype=np.float32) for vector in vectors]


# ---------------------------------------------------------------------------
# Voyage AI
# ---------------------------------------------------------------------------


def _voyage():
    """The client for embeddings: also checks EMBEDDING_MODEL is a Voyage model."""
    model = (settings.EMBEDDING_MODEL or "").strip()
    client = _voyage_client_for("Voyage AI is selected (EMBEDDING_PROVIDER=voyage)")
    if not model or "/" in model:
        # e.g. still "BAAI/bge-large-en-v1.5" -- a Hugging Face name, not a Voyage model
        raise EmbeddingServiceError(
            f"EMBEDDING_MODEL is '{model}', which isn't a Voyage AI model -- set it in .env "
            "to the Voyage model you use (for example voyage-3.5)."
        )
    return client


def _voyage_client_for(purpose: str):
    """The shared Voyage AI client (embeddings and reranker); needs VOYAGE_API_KEY."""
    global _voyage_client
    key = (settings.VOYAGE_API_KEY or "").strip()
    if not key:
        raise EmbeddingServiceError(
            f"{purpose} but VOYAGE_API_KEY isn't set in the API's .env."
        )
    if _voyage_client is None:
        with _voyage_lock:
            if _voyage_client is None:
                try:
                    import voyageai
                except ImportError as exc:  # pragma: no cover -- depends on the install
                    raise EmbeddingServiceError(
                        "The 'voyageai' package isn't installed -- run 'pip install voyageai' on "
                        "the API server and restart it."
                    ) from exc
                _voyage_client = voyageai.Client(
                    api_key=key,
                    max_retries=settings.VOYAGE_MAX_RETRIES,
                    timeout=settings.VOYAGE_TIMEOUT_SECONDS,
                )
    return _voyage_client


def _voyage_batches(texts: list[str]):
    batch, chars = [], 0
    for text in texts:
        if batch and (len(batch) >= _VOYAGE_MAX_TEXTS or chars + len(text) > _VOYAGE_MAX_CHARS):
            yield batch
            batch, chars = [], 0
        batch.append(text)
        chars += len(text)
    if batch:
        yield batch


def _voyage_error_message(exc: Exception) -> str:
    name = exc.__class__.__name__
    detail = str(exc)[:300]
    if name == "AuthenticationError":
        return "Voyage AI rejected the API key -- check VOYAGE_API_KEY in .env."
    if name == "RateLimitError":
        return "Voyage AI's rate limit was reached -- please try again in a minute."
    if name in ("APIConnectionError", "Timeout", "ServiceUnavailableError", "ServerError", "TryAgain"):
        return f"Voyage AI couldn't be reached right now ({detail or name}) -- please try again shortly."
    if name in ("InvalidRequestError", "MalformedRequestError"):
        return f"Voyage AI didn't accept the request: {detail}"
    return f"Voyage AI couldn't create embeddings: {detail or name}"


def _voyage_embed(texts: list[str], input_type: str) -> list[np.ndarray]:
    client = _voyage()
    vectors: list[np.ndarray] = []
    tokens = 0
    for batch in _voyage_batches([t if t.strip() else " " for t in texts]):
        try:
            result = client.embed(
                batch,
                model=settings.EMBEDDING_MODEL.strip(),
                input_type=input_type,
                output_dimension=settings.EMBEDDING_DIMENSIONS,
            )
        except EmbeddingServiceError:
            raise
        except Exception as exc:  # noqa: BLE001 -- voyageai.error.* and network errors
            logger.warning("Voyage AI embedding call failed: %s: %s", exc.__class__.__name__, exc)
            raise EmbeddingServiceError(_voyage_error_message(exc)) from exc
        if len(result.embeddings) != len(batch):
            raise EmbeddingServiceError(
                f"Voyage AI returned {len(result.embeddings)} vectors for {len(batch)} texts."
            )
        vectors.extend(np.asarray(v, dtype=np.float32) for v in result.embeddings)
        tokens += int(getattr(result, "total_tokens", 0) or 0)
    logger.info(
        "Voyage AI embeddings: %d %s text(s), %d tokens (model %s)",
        len(texts), input_type, tokens, settings.EMBEDDING_MODEL,
    )
    return vectors


# ---------------------------------------------------------------------------
# What the rest of the app calls
# ---------------------------------------------------------------------------


def embed_texts(texts: list[str]) -> list[np.ndarray]:
    """
    Embeds a batch of document/passage chunks, order-preserving. Local
    model: plain text, no query prefix. Voyage AI: input_type="document".
    Called for every chunk of a document being indexed. For a live search
    question, use embed_query() instead.
    """
    if not texts:
        return []
    result = _voyage_embed(texts, "document") if provider() == "voyage" else _local_embed(texts)
    for vector in result:
        _validate_dimension(vector)
    return result


def embed_query(query: str) -> np.ndarray:
    """
    Embeds a single live search question -- NOT for document chunks (use
    embed_texts() for those). Local model: prepends
    EMBEDDING_QUERY_INSTRUCTION_PREFIX (required by BGE-*-en-v1.5 models).
    Voyage AI: input_type="query" instead (no prefix). Called from
    app/services/retrieval.py's search().
    """
    if provider() == "voyage":
        vector = _voyage_embed([query], "query")[0]
    else:
        vector = _local_embed([settings.EMBEDDING_QUERY_INSTRUCTION_PREFIX + query])[0]
    _validate_dimension(vector)
    return vector


def rerank_enabled() -> bool:
    """True when RERANK_MODEL is set in .env (the reranker is optional and off by default)."""
    return bool((settings.RERANK_MODEL or "").strip())


def rerank(query: str, documents: list[str], top_k: int) -> list[tuple[int, float]]:
    """
    Asks the Voyage AI reranker (RERANK_MODEL) how well each of `documents`
    answers `query`. Returns up to `top_k` (index into `documents`,
    relevance score 0-1) pairs, most relevant first. Works whichever
    EMBEDDING_PROVIDER is used, as long as VOYAGE_API_KEY is set. Raises
    EmbeddingServiceError on failure -- app/services/retrieval.py then falls
    back to the plain distance order. Questions and chunk text leave this
    server, as with Voyage embeddings.
    """
    if not documents or top_k <= 0:
        return []
    model = settings.RERANK_MODEL.strip()
    client = _voyage_client_for(f"The reranker is switched on (RERANK_MODEL={model})")
    started = time.perf_counter()
    try:
        result = client.rerank(
            query,
            [d if d.strip() else " " for d in documents],
            model=model,
            top_k=min(top_k, len(documents)),
            truncation=True,
        )
    except Exception as exc:  # noqa: BLE001 -- voyageai.error.* and network errors
        logger.warning("Voyage AI rerank call failed: %s: %s", exc.__class__.__name__, exc)
        raise EmbeddingServiceError(
            _voyage_error_message(exc).replace("create embeddings", "rerank")
        ) from exc
    ranked = []
    for item in result.results:
        index = int(item.index)
        if not 0 <= index < len(documents):
            raise EmbeddingServiceError(f"Voyage AI's reranker returned an unknown position ({index}).")
        ranked.append((index, float(item.relevance_score)))
    ranked.sort(key=lambda pair: pair[1], reverse=True)
    logger.info(
        "Voyage AI rerank: %d candidates, %d tokens, %.0f ms (model %s)",
        len(documents),
        int(getattr(result, "total_tokens", 0) or 0),
        (time.perf_counter() - started) * 1000,
        model,
    )
    return ranked


def warm_up() -> None:
    """Loads the local model at start-up so the first upload doesn't wait; nothing to do for Voyage AI."""
    if uses_local_model():
        embed_texts(["warm-up"])


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
