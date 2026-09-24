"""
App configuration.

DB credentials are provided as five discrete environment variables
(DB_DRIVER, DB_SERVER, DB_NAME, DB_USER, DB_PASSWORD) -- never a single
pre-built connection string, and never hardcoded here.

Entra ID / JWT / Anthropic settings work the same way: all environment
variables. The ones below default to placeholder values so the app can
still start up -- the relevant feature simply won't work until you
replace them in .env with real values (see the comments in .env).

load_dotenv() is called with no arguments, so it uses its default search
behavior starting from the current working directory. The backend is
expected to be run from the `backend/` folder (same folder as .env and
requirements.txt), e.g. `uvicorn app.main:app` invoked from within
`backend/`.
"""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()


class Settings(BaseSettings):
    # --- Database ---
    DB_DRIVER: str
    DB_SERVER: str
    DB_NAME: str
    DB_USER: str
    DB_PASSWORD: str

    # --- Entra ID (Microsoft identity platform), spec §3.1 ---
    # PLACEHOLDER values -- replace with your app registration's real
    # tenant ID and client ID before testing login. No client secret is
    # needed here: the Angular app authenticates as a public client
    # (authorization code + PKCE), and the backend only *validates* the
    # tokens Entra issues, via Entra's published JWKS -- it never talks to
    # Entra to issue tokens itself.
    ENTRA_TENANT_ID: str = "REPLACE_WITH_ENTRA_TENANT_ID"
    ENTRA_CLIENT_ID: str = "REPLACE_WITH_ENTRA_CLIENT_ID"

    # --- App's own JWT, issued after Entra validation (spec §3.1) ---
    # PLACEHOLDER -- replace with a real random secret before testing
    # login; anyone with this value can forge valid session tokens.
    # Generate one with: python -c "import secrets; print(secrets.token_hex(32))"
    JWT_SECRET_KEY: str = "REPLACE_WITH_A_LONG_RANDOM_SECRET"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRY_MINUTES: int = 480  # ~8 hours, per spec §3.1

    # --- Anthropic Claude API (chat generation) ---
    ANTHROPIC_API_KEY: str = "REPLACE_WITH_ANTHROPIC_API_KEY"
    ANTHROPIC_MODEL: str = "claude-sonnet-4-5"

    # --- Embeddings (Phase 2, spec §3.2/§3.3) ---
    # Local sentence-transformers-compatible model -- no API key needed.
    # Anthropic doesn't offer its own embeddings API (their docs point to
    # Voyage AI as the recommended hosted option); this avoids a second
    # vendor key entirely by running the model on this machine instead.
    #
    # History: all-MiniLM-L6-v2 (384-dim, ~90MB) -> BAAI/bge-m3 (1024-dim,
    # ~2.27GB, multilingual, hybrid dense/sparse/colbert) -> now
    # BAAI/bge-large-en-v1.5 (also 1024-dim, ~1.3GB, English-only, ~335M
    # params). bge-m3's whole value proposition is massive multilinguality
    # plus hybrid retrieval modes -- app/services/embeddings.py only ever
    # calls plain dense .encode(), so none of that was actually being used,
    # while paying its size/speed cost. This app's documents are English
    # (see e.g. "Dar Al Riyadh Code of Conduct English.pdf"), so a
    # dedicated English retrieval model is the better fit.
    #
    # *** Every model switch needs a full Clear Knowledge Base + re-ingest
    # of every document, no exceptions -- see docs/documentation.md §7.
    # This particular switch is a sharper trap than the MiniLM->bge-m3 one
    # was: bge-m3 and bge-large-en-v1.5 both output 1024-dim vectors, so
    # the dimension check in retrieval.py (and scripts/check_embedding_
    # dimensions.py) will NOT catch a half-finished migration here --
    # matching byte-length does not mean compatible vector space. This is
    # exactly why DocumentChunk.EmbeddingModel exists now (added by
    # sql/004_add_embedding_model_to_chunks.sql on the old SQL Server 2019
    # database): retrieval.py excludes any chunk whose stamped model
    # doesn't match this setting, independent of whether the dimension
    # happens to match too. ***
    #
    # *** Storage itself has since moved off VARBINARY(MAX) entirely: the
    # app now targets a separate, new SQL Server 2025 database (see
    # sql/sqlserver2025/001_create_schema.sql), using that engine's native
    # VECTOR(EMBEDDING_DIMENSIONS) type end to end -- see
    # app/services/embeddings.py, app/services/ingestion.py and
    # app/services/retrieval.py. The old 2019 database and its
    # VARBINARY(MAX) rows are untouched and no longer used by this app. ***
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"

    # This model's actual output dimension -- must match both what
    # app/services/embeddings.py asserts at runtime AND the VECTOR(n)
    # size in sql/sqlserver2025/001_create_schema.sql. Kept as a single
    # named constant, referenced from all three places, specifically so
    # they can never quietly drift apart the way EMBEDDING_MODEL itself
    # has drifted from SIMILARITY_THRESHOLD's tuning twice already.
    EMBEDDING_DIMENSIONS: int = 1024

    # BGE-*-en-v1.5 models were trained asymmetrically: a live search
    # query is meant to be embedded with this instruction string
    # prepended, while the passages/documents being indexed are embedded
    # plain, with no prefix at all -- see
    # app/services/embeddings.py:embed_query(). Leave this as "" for any
    # model that doesn't use this convention (all-MiniLM-L6-v2,
    # BAAI/bge-m3, all-mpnet-base-v2, etc.) -- an empty prefix is a
    # harmless no-op there. Always set this together with EMBEDDING_MODEL
    # when you change it, never independently.
    #
    # NOT independently verified live against BAAI's own model card for
    # this change -- the web-search tool needed to check it was
    # unavailable the whole session. Confirm the exact wording against
    # https://huggingface.co/BAAI/bge-large-en-v1.5's "Usage" section
    # before relying on this in a way that matters, and correct it here if
    # it differs -- a wrong prefix doesn't error, it just silently falls
    # short of the model's documented accuracy.
    EMBEDDING_QUERY_INSTRUCTION_PREFIX: str = (
        "Represent this sentence for searching relevant passages: "
    )

    # --- Upload limits (Phase 2, spec §7) ---
    MAX_UPLOAD_SIZE_MB: int = 25

    # --- Retrieval / query routing (Phase 3, spec §3.3/§3.4) ---
    # Cosine DISTANCE cutoff a chunk must clear before the assistant
    # answers from the knowledge base instead of general knowledge --
    # this cutoff *is* the query router. Computed by SQL Server 2025's
    # native VECTOR_DISTANCE('cosine', ...) (app/services/retrieval.py),
    # which returns a value in [0, 2]: 0 = identical, 2 = opposite -- the
    # exact inverse of the old hand-computed cosine SIMILARITY score
    # (higher = better) this setting used to gate on. LOWER is better
    # here -- do not read this the old way.
    #
    # This replaces the old SIMILARITY_THRESHOLD setting (removed, not
    # kept alongside this one -- two settings with inverted meaning is a
    # standing invitation to use the wrong one by accident). 0.35 is a
    # straight mechanical conversion of that setting's last value
    # (distance = 1 - similarity => 1 - 0.65 = 0.35), NOT a re-tuning --
    # and that old 0.65 was itself never actually re-validated after two
    # prior embedding-model switches (MiniLM -> bge-m3 ->
    # bge-large-en-v1.5). Treat 0.35 as an unvalidated placeholder until
    # real distances are captured via GET /api/chat/debug-retrieval's
    # `max_distance` override param (both a known-good match and a
    # known-irrelevant query) against the current model on the actual SQL
    # Server 2025 database, and re-tune from there.
    MAX_COSINE_DISTANCE: float = 0.35
    TOP_K_CHUNKS: int = 5


settings = Settings()
