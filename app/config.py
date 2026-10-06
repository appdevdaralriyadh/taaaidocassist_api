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

    # --- Document versioning (upload rework) ---
    # The upload decides automatically whether a file is a new version of
    # an existing upload (see app/services/doc_matching.py). All similarity
    # values are 0.0-1.0 "two-way" scores: the LOWER of (share of the new
    # file's 3-word sequences found in the old) and (share of the old's
    # found in the new) -- so both documents must be mostly made of the
    # other's wording. Every decision records its actual score in the upload
    # history (DarAI_IngestionJobItems.Message), so these can be tuned
    # against real documents. Exact same-filename uploads always replace.
    #
    # Name matches once version words are stripped ("Gift Policy 2026.pdf"
    # vs "Gift Policy 2026 new version.pdf"): replace at or above this.
    VERSION_MATCH_MIN_SIMILARITY: float = 0.5
    # Name does NOT match: the content alone has to carry the decision, so
    # the two directions are judged separately -- replace only when at
    # least CONTENT_MATCH_MIN_SIMILARITY of one document's wording is in
    # the other (e.g. most of the old text carried into the new one) AND at
    # least CONTENT_MATCH_MIN_COVERAGE the other way. Added/removed
    # sections only lower one direction, so genuine revisions still pass;
    # an excerpt, or a document that merely contains another, fails the
    # second check. Never applies when the two names look like siblings in
    # a series ("NDA Vendor A" / "NDA Vendor B"), which can be
    # near-identical template text while being different documents.
    CONTENT_MATCH_MIN_SIMILARITY: float = 0.8
    CONTENT_MATCH_MIN_COVERAGE: float = 0.6
    # Name does NOT match, and the wording is close but short of the two
    # values above (at least this much one way): keep both, but flag the
    # upload as a possible version of the other document for review.
    POSSIBLE_VERSION_MIN_SIMILARITY: float = 0.5
    # A content-only match also needs both documents to have at least this
    # many words -- very short files (forms, one-liners) don't carry enough
    # wording to judge, and are always kept separate.
    CONTENT_MATCH_MIN_WORDS: int = 150
    # Finding content-only candidates: a sample of the upload's chunk
    # embeddings is searched against the knowledge base, and any document
    # with a chunk this close (cosine distance) gets its wording compared.
    # Deliberately generous -- it only decides which documents to CHECK; the
    # wording comparison above makes the actual decision.
    VERSION_CANDIDATE_MAX_DISTANCE: float = 0.15
    # A file whose year (filename or title) is earlier than the matched stored
    # document's is not added, so an old edition never replaces a newer one.
    BLOCK_OLDER_EDITIONS: bool = True
    # When on, documents whose names look like a series ("NDA Vendor A" /
    # "NDA Vendor B") are never merged on content alone -- they're flagged
    # for review instead. Off by default: wording that matches 80%+ both
    # ways is the same document under another name, so the newer one
    # replaces the older (which stays restorable in History).
    PROTECT_SIBLING_NAMES: bool = False

    # --- History and retention ---
    # Days a deleted document / a replaced (previous) version stays
    # restorable before it's permanently deleted. 0 = keep forever.
    DELETED_RETENTION_DAYS: int = 30
    PREVIOUS_VERSION_RETENTION_DAYS: int = 30

    # *** Everything in this "Document versioning" block, plus the two
    # retention settings, is a DEFAULT only: it can be changed live on the
    # app's Settings page, which stores the changed value in
    # DarAI_AppSettings (see app/services/app_settings.py). A setting with
    # no row there uses the value written here. ***

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

    # --- Background uploads / syncs (app/services/jobs.py) ---
    # How many files are read + embedded at the same time (embedding runs
    # on this server's CPU, so more at once can slow chat replies).
    # Checking versions and saving always happen one file at a time.
    JOB_WORKERS: int = 2
    # Where uploaded files wait until they're processed, so uploads still
    # queued or in progress continue after an API restart. Empty = an
    # "upload_spool" folder next to the app folder.
    UPLOAD_SPOOL_DIR: str = ""


settings = Settings()
