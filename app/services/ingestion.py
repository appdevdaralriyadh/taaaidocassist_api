"""
Orchestrates ingesting one file end-to-end: hash -> parse -> chunk ->
embed -> persist (spec §3.2).

Two entry points:

  ingest_local_upload()  -- local uploads (app/api/routes/documents.py).
      Fully automatic duplicate/version handling -- no one confirms, so
      every rule errs towards keeping documents rather than deleting them:
        1. Identical content already in the knowledge base (any source)
           -> "unchanged": nothing is embedded or stored.
           1b. Identical to a document in Deleted -> "restored": it is
               brought back as it was (no re-embedding).
           1c. Identical to a Previous version of an active document ->
               "unchanged" (restorable from that document's History).
           Deleted documents are never matched by the rules below.
        2. An earlier version of this file among local uploads is found by
           any of (see app/services/doc_matching.py for the scoring):
             a. same filename (case-insensitive)
             b. same name once version words are stripped, AND wording at
                least VERSION_MATCH_MIN_SIMILARITY the same both ways
             c. different name, but wording at least
                CONTENT_MATCH_MIN_SIMILARITY the same both ways (near-
                identical) and both at least CONTENT_MATCH_MIN_WORDS long
                -- candidates for this are found via the chunk embeddings
                (VERSION_CANDIDATE_MAX_DISTANCE), then compared by wording.
           2a. If a matched document carries a LATER year (filename or
               title area) than the upload -> "older_version": the upload
               is not stored and nothing is replaced.
           2b. Otherwise -> "updated": the most recently updated match is
               replaced in place -- same Document row and Id, Version + 1.
               Its old content is archived as a 'Previous version'
               (app/services/doc_history.py -- the chat no longer sees it,
               restorable from History) and any other matches (older
               copies/versions) move to Deleted, "merged into" it, all in
               one transaction (if anything fails, nothing is changed).
               A different-name match needs CONTENT_MATCH_MIN_SIMILARITY
               one way AND CONTENT_MATCH_MIN_COVERAGE the other: the same
               document under another name. One that is only CONTAINED in
               the other (an excerpt, a handbook) is kept and flagged.
        3. Otherwise -> "new" (Version 1). When something looked like a
           version but fell short of the rules above (name matched but
           wording differs; or different name with wording at least
           POSSIBLE_VERSION_MIN_SIMILARITY the same), both are kept and
           the upload is flagged `needs_review` with a note saying why.
      Every call is recorded as an IngestionJob + IngestionJobItem (see
      sql/sqlserver2025/003_document_versioning.sql) -- including
      failures -- which is what the Document Library's history uses.

  ingest_upload()  -- the original, additive pipeline, still used by the
      OneDrive/Google Drive connectors (app/api/routes/sources.py) and
      deliberately unchanged here: duplicates are only flagged. Step 3
      moves the connectors onto the version-aware logic too.

Each chunk's Embedding is inserted via raw SQL, not the ORM -- see
app/db/models.py's docstring and app/services/embeddings.py's module
docstring for why (SQLAlchemy has no built-in VECTOR type).
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy import func, text
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Document, DocumentChunk, IngestionJob, IngestionJobItem, User
from app.services import app_settings, doc_history
from app.services.chunking import chunk_text
from app.services.doc_matching import (
    are_sibling_names,
    doc_year,
    normalize_doc_name,
    text_shingles,
    two_way_similarity,
)
from app.services.embeddings import embed_texts, to_vector_literal
from app.services.parsing import extract_text


class IngestionResult:
    def __init__(self, *, document: Document, chunk_count: int, is_duplicate: bool):
        self.document = document
        self.chunk_count = chunk_count
        self.is_duplicate = is_duplicate


# The dimension in CAST(... AS VECTOR(n)) has to be a literal in the SQL
# text -- a type's size can't be a bind parameter in T-SQL -- so it's
# interpolated here from our own settings, never from user input, which
# is what makes that safe. Built once at import time since
# EMBEDDING_DIMENSIONS is fixed for the life of the process (same as
# every other setting in this app).
#
# CAST(:embedding AS NVARCHAR(MAX)) BEFORE casting to VECTOR, not
# straight to VECTOR: pyodbc/ODBC Driver 17 sends a long string parameter
# (the JSON-array embedding literal is ~20KB+) using a "long data" wire
# type that SQL Server reports as `ntext`, and VECTOR's CAST rules
# explicitly refuse ntext as a source type ("Explicit conversion from
# data type ntext to vector is not allowed") even though they accept
# nvarchar fine. ntext -> nvarchar(max) is always allowed, so this
# intermediate cast sidesteps the restriction without needing to change
# how pyodbc binds the parameter.
_INSERT_CHUNK_SQL = text(
    f"""
    INSERT INTO dbo.DarAI_DocumentChunks
        (DocumentId, ChunkIndex, ChunkText, Embedding, EmbeddingModel)
    VALUES
        (:document_id, :chunk_index, :chunk_text,
         CAST(CAST(:embedding AS NVARCHAR(MAX)) AS VECTOR({settings.EMBEDDING_DIMENSIONS})),
         :embedding_model)
    """
)

# Finds local-upload documents with a chunk close to one sample chunk of
# the file being uploaded -- the candidate list for a content-only version
# match (rule 2c in the module docstring). Same query shape and the same
# NVARCHAR(MAX) double cast as app/services/retrieval.py's _SEARCH_SQL.
_VERSION_CANDIDATE_SQL = text(
    f"""
    DECLARE @q VECTOR({settings.EMBEDDING_DIMENSIONS}) =
        CAST(CAST(:query_vec AS NVARCHAR(MAX)) AS VECTOR({settings.EMBEDDING_DIMENSIONS}));

    SELECT DocumentId, Distance
    FROM (
        SELECT c.DocumentId,
               VECTOR_DISTANCE('cosine', c.Embedding, @q) AS Distance
        FROM dbo.DarAI_DocumentChunks c
        JOIN dbo.DarAI_Documents d ON d.Id = c.DocumentId
        WHERE c.EmbeddingModel = :embedding_model
          AND d.SourceType = 'upload'
          AND d.Status = 'Active'
    ) AS scored
    WHERE Distance <= :max_distance
    ORDER BY Distance ASC
    OFFSET 0 ROWS FETCH NEXT :top_k ROWS ONLY;
    """
)
_CANDIDATE_SAMPLE_CHUNKS = 8  # how many of the upload's chunks are searched with
_CANDIDATE_TOP_K = 10  # nearest chunks kept per searched chunk


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _extract_chunks(filename: str, content: bytes) -> list[str]:
    extracted_text = extract_text(filename, content)
    chunks = chunk_text(extracted_text)
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=(
                f"No extractable text found in '{filename}' "
                "(possibly a scanned/image-only file)."
            ),
        )
    return chunks


def _insert_chunks(db: Session, document_id: int, chunks: list[str], vectors: list) -> None:
    for index, (chunk, vector) in enumerate(zip(chunks, vectors)):
        db.execute(
            _INSERT_CHUNK_SQL,
            {
                "document_id": document_id,
                "chunk_index": index,
                "chunk_text": chunk,
                "embedding": to_vector_literal(vector),
                # Stamped at ingestion time so retrieval.py can detect an
                # embedding-model switch later, even one that doesn't
                # change the vector dimension (see app/config.py's
                # comment on EMBEDDING_MODEL).
                "embedding_model": settings.EMBEDDING_MODEL,
            },
        )


def _chunk_count(db: Session, document_id: int) -> int:
    return (
        db.query(func.count(DocumentChunk.Id))
        .filter(DocumentChunk.DocumentId == document_id)
        .scalar()
        or 0
    )


def _display_name(db: Session, user_id: Optional[int]) -> Optional[str]:
    if user_id is None:
        return None
    user = db.get(User, user_id)
    return user.DisplayName if user else None


# ---------------------------------------------------------------------------
# Local upload: duplicate- and version-aware
# ---------------------------------------------------------------------------


@dataclass
class LocalUploadResult:
    # 'new' | 'updated' | 'unchanged' | 'older_version'
    outcome: str
    # the stored document; for 'unchanged' the existing identical one, for
    # 'older_version' the existing NEWER one (the upload itself isn't stored)
    document: Document
    chunk_count: int
    version: int
    previous_version: Optional[int] = None
    # 'updated': the version that was just replaced -- its filename (may
    # differ from the new one), and who/when stored it
    replaced_filename: Optional[str] = None
    replaced_at: Optional[datetime] = None
    replaced_by: Optional[str] = None
    # 'updated': how the earlier version was recognised --
    #   'filename'          exact same name
    #   'name_and_content'  same name once version words are stripped + wording
    #   'content'           different name, near-identical wording
    # `similarity` is the two-way wording score -- the LOWER of the two
    # directions (None for 'filename'); `similarity_max` the higher one,
    # set for 'content' matches, which judge the two directions separately
    matched_by: Optional[str] = None
    similarity: Optional[float] = None
    similarity_max: Optional[float] = None
    # 'updated': other older copies/versions removed in the same step
    removed_copies: int = 0
    removed_filenames: list = None
    # 'unchanged' / 'older_version': the existing document it was matched to
    matched_filename: Optional[str] = None
    matched_source_type: Optional[str] = None
    # 'older_version': the years that decided it
    upload_year: Optional[int] = None
    existing_year: Optional[int] = None
    # 'new': True when something looked like a version but didn't meet the
    # rules, so both were kept -- `note` says why
    needs_review: bool = False
    note: Optional[str] = None
    job_id: Optional[int] = None
    # 'updated': days the replaced version stays restorable in History, and
    # days the other removed copies stay in Deleted (0 = kept until removed
    # by hand). 'restored': days a deleted document stays restorable.
    retention_days: Optional[int] = None
    removed_retention_days: Optional[int] = None
    # 'restored': identical content of a Deleted document was uploaded, so
    # that document was brought back (no re-embedding). renamed_from is set
    # when it came back under the uploaded name instead of its old one.
    renamed_from: Optional[str] = None
    never_merge_with: Optional[str] = None

    def __post_init__(self):
        if self.removed_filenames is None:
            self.removed_filenames = []


@dataclass
class _VersionMatch:
    document: Document
    matched_by: str  # 'filename' | 'name_and_content' | 'content'
    similarity: Optional[float]  # two-way score: the LOWER direction
    text: str  # the existing document's text (rebuilt from its chunks)
    similarity_max: Optional[float] = None  # the HIGHER direction


# A different-name document only gets flagged as a possible version when
# BOTH directions show real overlap -- an excerpt (one direction tiny)
# isn't a version of anything, so it isn't flagged.
_POSSIBLE_VERSION_MIN_COVERAGE = 0.3


def _pct_range(low: float, high: Optional[float] = None) -> str:
    low_pct = round(low * 100)
    if high is None or round(high * 100) == low_pct:
        return f"{low_pct}%"
    return f"{low_pct}-{round(high * 100)}%"


@dataclass
class _NearMiss:
    document: Document
    # 'name'    -- name matched, wording too different
    # 'content' -- different name, wording close but below the replace bar
    # 'sibling' -- wording close, but the names look like two documents in a
    #              series (e.g. "NDA Vendor A" / "NDA Vendor B")
    reason: str
    similarity: float  # the LOWER direction
    similarity_max: Optional[float] = None  # the HIGHER direction ('content'/'sibling')


def _document_text(db: Session, document_id: int) -> str:
    rows = (
        db.query(DocumentChunk.ChunkText)
        .filter(DocumentChunk.DocumentId == document_id)
        .order_by(DocumentChunk.ChunkIndex)
        .all()
    )
    return " ".join(row[0] for row in rows)


def _content_candidate_ids(db: Session, vectors: list, max_distance: float) -> set:
    """
    Local-upload documents that have at least one chunk close to one of a
    sample of the upload's chunks (rule 2c). Only decides which documents
    get their wording compared -- never decides a match on its own.
    """
    if not vectors:
        return set()
    if len(vectors) <= _CANDIDATE_SAMPLE_CHUNKS:
        sample = vectors
    else:
        step = len(vectors) / _CANDIDATE_SAMPLE_CHUNKS
        sample = [vectors[int(i * step)] for i in range(_CANDIDATE_SAMPLE_CHUNKS)]

    candidate_ids: set = set()
    for vector in sample:
        rows = db.execute(
            _VERSION_CANDIDATE_SQL,
            {
                "query_vec": to_vector_literal(vector),
                "embedding_model": settings.EMBEDDING_MODEL,
                "max_distance": max_distance,
                "top_k": _CANDIDATE_TOP_K,
            },
        ).all()
        candidate_ids.update(int(row[0]) for row in rows)
    return candidate_ids


def _find_version_matches(
    db: Session,
    *,
    filename: str,
    new_shingles: set,
    new_word_count: int,
    vectors: list,
    cfg: dict,
):
    """
    Finds existing LOCAL uploads that the uploaded file is a new version of
    (module docstring, rule 2). Returns (matches, near_miss):
      matches   -- list of _VersionMatch, most recently updated first
      near_miss -- the closest _NearMiss that looked like a version but
                   didn't meet the rules, or None
    `cfg` is app_settings.get_effective() -- the thresholds currently set
    on the Settings page (or their config.py defaults).
    """
    lower_name = filename.lower()
    normalized = normalize_doc_name(filename)
    # Active documents only: Deleted ones are in the Library's Deleted tab
    # and are never matched (identical content brings one back instead --
    # see ingest_local_upload step 1b).
    uploads = {
        d.Id: d
        for d in db.query(Document)
        .filter(Document.SourceType == "upload", Document.Status == doc_history.DOC_ACTIVE)
        .all()
    }

    # Every name each document has had, not just its current one: when a
    # new version arrives under a different name, the document takes that
    # new name -- but a later upload using the ORIGINAL naming (e.g. an
    # older "Gift Policy 2024.pdf" after "Gift Policy 2026.pdf" was replaced
    # by "Gift Rules for Staff.pdf") must still be recognised by name.
    # Read from the upload history; filtered in Python rather than with a
    # large IN (...) list, which SQL Server caps at ~2100 parameters.
    known_names = {doc_id: {doc.FileName} for doc_id, doc in uploads.items()}
    history = (
        db.query(IngestionJobItem.DocumentId, IngestionJobItem.FileName)
        .filter(
            IngestionJobItem.DocumentId.isnot(None),
            IngestionJobItem.Outcome.in_(("new", "updated", "possible_version", "restored")),
        )
        .all()
    )
    for doc_id, old_name in history:
        if doc_id in known_names:
            known_names[doc_id].add(old_name)

    matches: list[_VersionMatch] = []
    near_misses: list[_NearMiss] = []
    checked: set = set()

    # a + b: same filename, or same name once version words are stripped
    # (against any name the document has had)
    for doc in uploads.values():
        if doc.FileName.lower() == lower_name:
            matches.append(_VersionMatch(doc, "filename", None, _document_text(db, doc.Id)))
            checked.add(doc.Id)
            continue
        if not normalized or not any(
            normalize_doc_name(name) == normalized for name in known_names[doc.Id]
        ):
            continue
        checked.add(doc.Id)
        doc_text = _document_text(db, doc.Id)
        score, _, _ = two_way_similarity(new_shingles, text_shingles(doc_text))
        if score >= cfg["VERSION_MATCH_MIN_SIMILARITY"]:
            matches.append(_VersionMatch(doc, "name_and_content", score, doc_text))
        else:
            near_misses.append(_NearMiss(doc, "name", score))

    # c: different name -- candidates from the embeddings, decided by wording
    if new_word_count >= cfg["CONTENT_MATCH_MIN_WORDS"]:
        candidates = _content_candidate_ids(db, vectors, cfg["VERSION_CANDIDATE_MAX_DISTANCE"])
        for doc_id in candidates - checked:
            doc = uploads.get(doc_id)
            if doc is None:
                continue
            doc_text = _document_text(db, doc_id)
            if len(doc_text.split()) < cfg["CONTENT_MATCH_MIN_WORDS"]:
                continue
            low, new_in_old, old_in_new = two_way_similarity(new_shingles, text_shingles(doc_text))
            high = max(new_in_old, old_in_new)
            if high < cfg["POSSIBLE_VERSION_MIN_SIMILARITY"] or low < _POSSIBLE_VERSION_MIN_COVERAGE:
                continue
            if cfg["PROTECT_SIBLING_NAMES"] and are_sibling_names(
                normalized, normalize_doc_name(doc.FileName)
            ):
                # e.g. "NDA Vendor A" vs "NDA Vendor B": same template,
                # different documents -- never replaced automatically
                near_misses.append(_NearMiss(doc, "sibling", low, high))
            elif (
                high >= cfg["CONTENT_MATCH_MIN_SIMILARITY"]
                and low >= cfg["CONTENT_MATCH_MIN_COVERAGE"]
            ):
                matches.append(_VersionMatch(doc, "content", low, doc_text, high))
            else:
                near_misses.append(_NearMiss(doc, "content", low, high))

    matches.sort(
        key=lambda m: (m.document.UpdatedAt or m.document.UploadedAt or datetime.min, m.document.Id),
        reverse=True,
    )
    near_miss = (
        max(near_misses, key=lambda n: n.similarity_max or n.similarity) if near_misses else None
    )
    return matches, near_miss


def _near_miss_note(near_miss: "_NearMiss", cfg: dict) -> str:
    """Why a look-alike was kept as a separate document (shown as REVIEW)."""
    note = None
    pct = _pct_range(near_miss.similarity, near_miss.similarity_max)
    if near_miss.reason == "name":
        note = (
            f"The name looks like '{near_miss.document.FileName}', but only {pct} of "
            f"the wording is the same (at least "
            f"{round(cfg['VERSION_MATCH_MIN_SIMILARITY'] * 100)}% is needed to treat it "
            "as a new version), so both are kept."
        )
    elif near_miss.reason == "sibling":
        note = (
            f"{pct} of the wording is the same as '{near_miss.document.FileName}', but "
            "the names look like two different documents of the same kind (e.g. the same "
            "template for different parties), so both are kept."
        )
    elif (
        near_miss.similarity_max is not None
        and near_miss.similarity_max >= 0.9
        and near_miss.similarity < cfg["CONTENT_MATCH_MIN_COVERAGE"]
    ):
        # one document is (almost) entirely inside the other -- e.g. a
        # handbook that includes a whole policy, or a policy that
        # includes a whole annex
        note = (
            f"One of this file and '{near_miss.document.FileName}' contains most of the "
            f"other's text ({pct} overlap), e.g. a handbook that includes a policy -- "
            "that isn't a new version, so both are kept."
        )
    else:
        note = (
            f"{pct} of the wording is the same as '{near_miss.document.FileName}' "
            f"(a different name needs at least "
            f"{round(cfg['CONTENT_MATCH_MIN_SIMILARITY'] * 100)}% one way and "
            f"{round(cfg['CONTENT_MATCH_MIN_COVERAGE'] * 100)}% the other to be replaced "
            "automatically), so both are kept -- check whether one is an older version."
        )
    return note


def _start_job(db: Session, *, user: User, filename: str, content_hash: bytes, size: int):
    job = IngestionJob(
        JobType="upload",
        Status="running",
        TotalItems=1,
        StartedBy=user.Id,
        StartedAt=func.sysutcdatetime(),
    )
    db.add(job)
    db.flush()
    item = IngestionJobItem(
        JobId=job.Id,
        FileName=filename,
        Step="extracting",
        ContentHash=content_hash,
        FileSizeBytes=size,
    )
    db.add(item)
    db.flush()
    job_id, item_id = job.Id, item.Id
    # Committed on its own, before any document work, so the attempt is in
    # the history even if everything after this fails.
    db.commit()
    return job_id, item_id


def _finish_job(
    db: Session,
    *,
    job_id: int,
    item_id: int,
    outcome: str,
    document_id: Optional[int] = None,
    related_document_id: Optional[int] = None,
    previous_version: Optional[int] = None,
    new_version: Optional[int] = None,
    chunk_count: Optional[int] = None,
    message: Optional[str] = None,
) -> None:
    """Records the item's outcome and the job's totals. Does NOT commit --
    callers commit it together with the document change it describes, so
    the history and the knowledge base can never disagree."""
    short_message = message[:1000] if message else None

    item = db.get(IngestionJobItem, item_id)
    item.Step = "done"
    item.Outcome = outcome
    item.DocumentId = document_id
    item.RelatedDocumentId = related_document_id
    item.PreviousVersion = previous_version
    item.NewVersion = new_version
    item.ChunkCount = chunk_count
    item.Message = short_message
    item.UpdatedAt = func.sysutcdatetime()

    job = db.get(IngestionJob, job_id)
    job.ProcessedItems = 1
    # 'possible_version' = added as a new document, flagged for review
    # 'restored' = a Deleted document brought back by uploading it again
    job.AddedCount = 1 if outcome in ("new", "possible_version", "restored") else 0
    job.UpdatedCount = 1 if outcome == "updated" else 0
    job.UnchangedCount = 1 if outcome == "unchanged" else 0
    job.SkippedCount = 1 if outcome == "older_version" else 0
    job.FailedCount = 1 if outcome == "failed" else 0
    job.Status = "failed" if outcome == "failed" else "completed"
    job.Message = short_message
    job.FinishedAt = func.sysutcdatetime()


def _record_failure(db: Session, job_id: int, item_id: int, message: str) -> None:
    # Called after a rollback, in a fresh transaction.
    try:
        _finish_job(db, job_id=job_id, item_id=item_id, outcome="failed", message=message)
        db.commit()
    except Exception:  # noqa: BLE001 -- never let history-writing mask the real error
        db.rollback()


def ingest_local_upload(
    *,
    db: Session,
    filename: str,
    content: bytes,
    uploaded_by: User,
) -> LocalUploadResult:
    content_hash = hashlib.sha256(content).digest()
    # One snapshot of the Settings page values for this whole upload, so
    # every rule in it is judged against the same thresholds.
    cfg = app_settings.get_effective(db)
    job_id, item_id = _start_job(
        db, user=uploaded_by, filename=filename, content_hash=content_hash, size=len(content)
    )

    try:
        # 1. Identical content anywhere in the knowledge base (any source,
        #    any filename) -> skip. Nothing is embedded or stored.
        identical = (
            db.query(Document)
            .filter(Document.ContentHash == content_hash, Document.Status.in_((doc_history.DOC_ACTIVE, doc_history.DOC_EXCLUDED)))
            .order_by(Document.Id)
            .first()
        )
        in_history = (
            doc_history.find_in_history_by_hash(db, content_hash) if identical is None else None
        )

        # 1b. Identical to a document in Deleted -> bring it back as it was
        #     (its archived chunks return; nothing is re-embedded). Under the
        #     uploaded file's name -- unless an active document already uses
        #     that name, in which case the upload is handled normally below
        #     (as a new version of that document).
        if (
            in_history is not None
            and in_history[0] == "deleted"
            and doc_history.name_in_use(db, filename, except_document_id=in_history[1].Id) is None
        ):
            deleted_doc = in_history[1]
            old_name = deleted_doc.FileName
            restored = doc_history.restore_deleted(
                db, deleted_doc, by_user_id=uploaded_by.Id, new_name=filename
            )
            deleted_doc.UpdatedAt = func.sysutcdatetime()
            deleted_doc.UpdatedBy = uploaded_by.Id
            message = f"Identical to '{old_name}' in Deleted -- restored it"
            if restored["renamed_from"]:
                message += f" as '{filename}'"
            message += " instead of adding a copy."
            if restored["never_merge_with"]:
                message += (
                    f" It had been merged into '{restored['never_merge_with']}'; the two are now "
                    "kept as different documents."
                )
            _finish_job(
                db,
                job_id=job_id,
                item_id=item_id,
                outcome="restored",
                document_id=deleted_doc.Id,
                new_version=deleted_doc.Version,
                chunk_count=restored["chunk_count"],
                message=message,
            )
            db.commit()
            db.refresh(deleted_doc)
            return LocalUploadResult(
                outcome="restored",
                document=deleted_doc,
                chunk_count=restored["chunk_count"],
                version=deleted_doc.Version,
                matched_filename=old_name,
                matched_source_type=deleted_doc.SourceType,
                renamed_from=restored["renamed_from"],
                never_merge_with=restored["never_merge_with"],
                note=message,
                job_id=job_id,
            )

        # 1c. Identical to an older version of an active document -> not
        #     added (the document has moved on; that version can be brought
        #     back from its History instead).
        if in_history is not None and in_history[0] == "previous":
            current_doc, old_version = in_history[1], in_history[2]
            message = (
                f"Identical to v{old_version.VersionNumber} of '{current_doc.FileName}', which has "
                f"since been replaced by v{current_doc.Version} -- not added, nothing changed. "
                "That version can be restored from the document's History."
            )
            chunks_existing = _chunk_count(db, current_doc.Id)
            _finish_job(
                db,
                job_id=job_id,
                item_id=item_id,
                outcome="unchanged",
                related_document_id=current_doc.Id,
                chunk_count=chunks_existing,
                message=message,
            )
            db.commit()
            db.refresh(current_doc)
            return LocalUploadResult(
                outcome="unchanged",
                document=current_doc,
                chunk_count=chunks_existing,
                version=current_doc.Version,
                previous_version=old_version.VersionNumber,
                matched_filename=current_doc.FileName,
                matched_source_type=current_doc.SourceType,
                note=message,
                job_id=job_id,
            )

        if identical is not None:
            chunks_existing = _chunk_count(db, identical.Id)
            _finish_job(
                db,
                job_id=job_id,
                item_id=item_id,
                outcome="unchanged",
                related_document_id=identical.Id,
                chunk_count=chunks_existing,
                message=f"Identical to '{identical.FileName}' already in the knowledge base -- skipped.",
            )
            db.commit()
            db.refresh(identical)
            return LocalUploadResult(
                outcome="unchanged",
                document=identical,
                chunk_count=chunks_existing,
                version=identical.Version,
                matched_filename=identical.FileName,
                matched_source_type=identical.SourceType,
                job_id=job_id,
            )

        # Heavy work happens before any document is touched: if parsing or
        # embedding fails, nothing has been deleted yet.
        chunks = _extract_chunks(filename, content)
        new_text = " ".join(chunks)
        vectors = embed_texts(chunks)

        # 2. Is this a new version of an existing local upload?
        matches, near_miss = _find_version_matches(
            db,
            filename=filename,
            new_shingles=text_shingles(new_text),
            new_word_count=len(new_text.split()),
            vectors=vectors,
            cfg=cfg,
        )

        # 2a. Older-version guard: never let an older edition replace a
        #     newer one. Only applies when both sides carry a year, and
        #     when "Block older editions" is on (Settings page).
        if matches and cfg["BLOCK_OLDER_EDITIONS"]:
            upload_year = doc_year(filename, new_text)
            newest = None  # (match, year) with the latest year among matches
            for m in matches:
                year = doc_year(m.document.FileName, m.text)
                if year is not None and (newest is None or year > newest[1]):
                    newest = (m, year)
            if upload_year is not None and newest is not None and newest[1] > upload_year:
                newer_doc = newest[0].document
                message = (
                    f"Older version of '{newer_doc.FileName}' ({upload_year} vs {newest[1]}) "
                    "-- the newer one is already in the knowledge base, so this file wasn't added."
                )
                _finish_job(
                    db,
                    job_id=job_id,
                    item_id=item_id,
                    outcome="older_version",
                    related_document_id=newer_doc.Id,
                    chunk_count=_chunk_count(db, newer_doc.Id),
                    message=message,
                )
                db.commit()
                db.refresh(newer_doc)
                return LocalUploadResult(
                    outcome="older_version",
                    document=newer_doc,
                    chunk_count=_chunk_count(db, newer_doc.Id),
                    version=newer_doc.Version,
                    matched_by=newest[0].matched_by,
                    similarity=newest[0].similarity,
                    matched_filename=newer_doc.FileName,
                    matched_source_type=newer_doc.SourceType,
                    upload_year=upload_year,
                    existing_year=newest[1],
                    job_id=job_id,
                )

        if matches:
            target_match = matches[0]
            target = target_match.document
            # Other matches are older copies/versions of the same document:
            # they move to Deleted ("merged into" the target) -- except any
            # marked "never merge" with the target, which are left alone.
            never_merge = doc_history.excluded_ids(db, target.Id)
            extras = [m.document for m in matches[1:] if m.document.Id not in never_merge]
            # Versions count per document: the target goes from its own
            # current version to the next one.
            previous_version = target.Version
            new_version = previous_version + 1
            replaced_filename = target.FileName
            replaced_at = target.UpdatedAt or target.UploadedAt
            replaced_by = _display_name(db, target.UpdatedBy or target.UploadedBy)
            extra_ids = [d.Id for d in extras]
            extra_names = [d.FileName for d in extras]
            keep_previous_days = cfg["PREVIOUS_VERSION_RETENTION_DAYS"]
            keep_deleted_days = cfg["DELETED_RETENTION_DAYS"]

            # --- one transaction from here to commit ---
            # Nothing is thrown away: the target's current content becomes a
            # 'Previous version' (archived -- the chat no longer sees it,
            # restorable from History), and each extra goes to Deleted.
            doc_history.archive_current(
                db,
                target,
                new_status=doc_history.PREVIOUS,
                by_user_id=uploaded_by.Id,
                note=f"Replaced by '{filename}' (v{new_version}).",
            )
            for extra in extras:
                doc_history.soft_delete(
                    db,
                    extra,
                    by_user_id=uploaded_by.Id,
                    reason=(
                        f"Older copy of the same document -- merged into '{filename}' "
                        f"(v{new_version}) when it was uploaded."
                    ),
                    merged_into_document_id=target.Id,
                )

            target.FileName = filename
            target.ContentHash = content_hash
            target.FileSizeBytes = len(content)
            target.Version = new_version
            target.UpdatedAt = func.sysutcdatetime()
            target.UpdatedBy = uploaded_by.Id
            db.flush()

            _insert_chunks(db, target.Id, chunks, vectors)
            doc_history.record_new_latest(
                db,
                target,
                chunk_count=len(chunks),
                how_added=doc_history.HOW_ADDED[target_match.matched_by],
                by_user_id=uploaded_by.Id,
                score=target_match.similarity,
            )

            kept = (
                f"kept in History for {keep_previous_days} days"
                if keep_previous_days
                else "kept in History"
            )
            if target_match.matched_by == "filename":
                message = (
                    f"Replaced v{previous_version} with v{new_version} (same filename); "
                    f"v{previous_version} {kept}."
                )
            elif target_match.matched_by == "name_and_content":
                message = (
                    f"Replaced '{replaced_filename}' v{previous_version} with v{new_version} "
                    f"(name matched, wording {round(target_match.similarity * 100)}% the same); "
                    f"v{previous_version} {kept}."
                )
            else:
                message = (
                    f"Same content as '{replaced_filename}' "
                    f"({_pct_range(target_match.similarity, target_match.similarity_max)} match) "
                    f"under a different name -- treated as a new version: '{replaced_filename}' "
                    f"v{previous_version} was replaced and {kept}."
                )
            if extra_names:
                message += (
                    " Older copies moved to Deleted"
                    + (f" (restorable for {keep_deleted_days} days)" if keep_deleted_days else "")
                    + ": "
                    + ", ".join(f"'{name}'" for name in extra_names)
                    + "."
                )
            _finish_job(
                db,
                job_id=job_id,
                item_id=item_id,
                outcome="updated",
                document_id=target.Id,
                previous_version=previous_version,
                new_version=new_version,
                chunk_count=len(chunks),
                message=message,
            )
            db.commit()
            db.refresh(target)

            return LocalUploadResult(
                outcome="updated",
                document=target,
                chunk_count=len(chunks),
                version=new_version,
                previous_version=previous_version,
                replaced_filename=replaced_filename,
                replaced_at=replaced_at,
                replaced_by=replaced_by,
                matched_by=target_match.matched_by,
                similarity=target_match.similarity,
                similarity_max=target_match.similarity_max,
                removed_copies=len(extra_ids),
                removed_filenames=extra_names,
                retention_days=keep_previous_days,
                removed_retention_days=keep_deleted_days if extra_names else None,
                job_id=job_id,
            )

        # 3. Brand-new document. If something looked like a version but
        #    didn't meet the rules, keep both and flag it for review.
        note = _near_miss_note(near_miss, cfg) if near_miss is not None else None

        document = Document(
            SourceType="upload",
            SourcePath=None,
            FileName=filename,
            UploadedBy=uploaded_by.Id,
            ContentHash=content_hash,
            FileSizeBytes=len(content),
            Version=1,
            UpdatedBy=uploaded_by.Id,
            Status=doc_history.DOC_ACTIVE,
        )
        db.add(document)
        db.flush()  # assigns document.Id without ending the transaction

        _insert_chunks(db, document.Id, chunks, vectors)
        doc_history.record_new_latest(
            db,
            document,
            chunk_count=len(chunks),
            how_added="First upload",
            by_user_id=uploaded_by.Id,
        )

        _finish_job(
            db,
            job_id=job_id,
            item_id=item_id,
            # 'possible_version' in the history = added, but worth a look
            # (the Document Library can list these for review later)
            outcome="possible_version" if near_miss is not None else "new",
            document_id=document.Id,
            related_document_id=near_miss.document.Id if near_miss is not None else None,
            new_version=1,
            chunk_count=len(chunks),
            message="Added as a new document." + (f" {note}" if note else ""),
        )
        db.commit()
        db.refresh(document)

        return LocalUploadResult(
            outcome="new",
            document=document,
            chunk_count=len(chunks),
            version=1,
            matched_filename=near_miss.document.FileName if near_miss is not None else None,
            similarity=near_miss.similarity if near_miss is not None else None,
            similarity_max=near_miss.similarity_max if near_miss is not None else None,
            needs_review=near_miss is not None,
            note=note,
            job_id=job_id,
        )

    except HTTPException as exc:
        db.rollback()
        _record_failure(db, job_id, item_id, str(exc.detail))
        raise
    except Exception as exc:
        db.rollback()
        _record_failure(db, job_id, item_id, str(exc)[:1000] or exc.__class__.__name__)
        raise


# ---------------------------------------------------------------------------
# Single-document permanent delete, and restore from Deleted (Document Library)
# ---------------------------------------------------------------------------


@dataclass
class DeleteResult:
    document_id: int
    filename: str
    chunks_deleted: int
    deleted_at: Optional[datetime] = None
    # days it stays restorable in the Library's Deleted tab (0 = until
    # removed by hand) and the date it will be permanently deleted
    retention_days: int = 0
    restorable_until: Optional[datetime] = None
    was_in_deleted: bool = False


def _record_library_job(
    db: Session,
    *,
    job_type: str,
    outcome: str,
    document: Document,
    by_user: User,
    chunk_count: int,
    message: str,
    removed: bool,
    updated: bool = False,
    previous_version: Optional[int] = None,
) -> None:
    job = IngestionJob(
        JobType=job_type,
        Status="completed",
        TotalItems=1,
        ProcessedItems=1,
        RemovedCount=1 if removed else 0,
        AddedCount=0 if (removed or updated) else 1,
        UpdatedCount=1 if updated else 0,
        StartedBy=by_user.Id,
        StartedAt=func.sysutcdatetime(),
        FinishedAt=func.sysutcdatetime(),
        Message=message[:1000],
    )
    db.add(job)
    db.flush()
    db.add(
        IngestionJobItem(
            JobId=job.Id,
            FileName=document.FileName,
            SourcePath=document.SourcePath,
            ExternalId=document.ExternalId,
            Step="done",
            Outcome=outcome,
            DocumentId=document.Id,
            PreviousVersion=document.Version if removed else previous_version,
            NewVersion=None if removed else document.Version,
            ContentHash=document.ContentHash,
            FileSizeBytes=document.FileSizeBytes,
            ChunkCount=chunk_count,
            Message=message[:1000],
        )
    )


def delete_document(db: Session, *, document_id: int, deleted_by: User) -> DeleteResult:
    """
    Permanently deletes one document -- from the Documents tab (a person
    deleting it on purpose) or from the Deleted tab (after checking an
    older copy the automatic matching set aside). Its content and every
    version are removed for good and the chat stops using it at once; it
    cannot be restored. Only the automatic version matching moves
    documents to Deleted (doc_history.soft_delete). Recorded in the upload
    history. Any account may delete any document -- same equal-permission
    model as Data Management.
    """
    document = db.get(Document, document_id)
    if document is None or document.Status == doc_history.DOC_PERMANENTLY_DELETED:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    filename = document.FileName
    was_in_deleted = document.Status == doc_history.DOC_DELETED
    where = "the Deleted tab" if was_in_deleted else "the Document Library"
    chunks_deleted = doc_history.purge_document(
        db, document, by_user_id=deleted_by.Id, reason=f"Permanently deleted from {where}."
    )
    _record_library_job(
        db,
        job_type="delete",
        outcome="permanently_deleted",
        document=document,
        by_user=deleted_by,
        chunk_count=chunks_deleted,
        message=f"Permanently deleted '{filename}' from {where}.",
        removed=True,
    )
    db.commit()
    db.refresh(document)

    return DeleteResult(
        document_id=document_id,
        filename=filename,
        chunks_deleted=chunks_deleted,
        deleted_at=document.DeletedAt,
        retention_days=0,
        restorable_until=None,
        was_in_deleted=was_in_deleted,
    )


@dataclass
class RestoreResult:
    document: Document
    chunk_count: int
    renamed_from: Optional[str] = None
    never_merge_with: Optional[str] = None


def restore_document(
    db: Session, *, document_id: int, restored_by: User, new_name: Optional[str] = None
) -> RestoreResult:
    """
    Brings a document back from the Deleted tab, optionally under a new
    name (required when an active document already has its name -- a 409
    with a suggested name says so). Recorded in the upload history.
    """
    document = db.get(Document, document_id)
    if document is None or document.Status == doc_history.DOC_PERMANENTLY_DELETED:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")

    restored = doc_history.restore_deleted(
        db, document, by_user_id=restored_by.Id, new_name=new_name
    )
    message = f"Restored '{document.FileName}' from Deleted"
    if restored["renamed_from"]:
        message += f" (was '{restored['renamed_from']}')"
    message += "."
    if restored["never_merge_with"]:
        message += (
            f" It had been merged into '{restored['never_merge_with']}'; the two are now kept as "
            "different documents."
        )
    _record_library_job(
        db,
        job_type="restore",
        outcome="restored",
        document=document,
        by_user=restored_by,
        chunk_count=restored["chunk_count"],
        message=message,
        removed=False,
    )
    db.commit()
    db.refresh(document)
    return RestoreResult(
        document=document,
        chunk_count=restored["chunk_count"],
        renamed_from=restored["renamed_from"],
        never_merge_with=restored["never_merge_with"],
    )


# ---------------------------------------------------------------------------
# History panel actions (Document Library)
# ---------------------------------------------------------------------------


def _active_document(db: Session, document_id: int) -> Document:
    document = db.get(Document, document_id)
    if document is None or document.Status == doc_history.DOC_PERMANENTLY_DELETED:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found.")
    return document


def document_history(db: Session, *, document_id: int) -> dict:
    document = _active_document(db, document_id)
    cfg = app_settings.get_effective(db)
    history = doc_history.document_history(
        db,
        document,
        previous_days=cfg["PREVIOUS_VERSION_RETENTION_DAYS"],
        deleted_days=cfg["DELETED_RETENTION_DAYS"],
    )
    db.commit()  # keeps any repair of stale version rows (see doc_history.document_history)
    return history


def restore_version(db: Session, *, document_id: int, version_id: int, restored_by: User) -> dict:
    """Restore an earlier version as the new current version (recorded in the history)."""
    document = _active_document(db, document_id)
    result = doc_history.restore_version(db, document, version_id, by_user_id=restored_by.Id)
    message = (
        f"Restored v{result['restored_version']} of '{document.FileName}' as v{result['new_version']}; "
        f"v{result['replaced_version']} kept as a previous version."
    )
    _record_library_job(
        db,
        job_type="restore",
        outcome="restored",
        document=document,
        by_user=restored_by,
        chunk_count=result["chunk_count"],
        message=message,
        removed=False,
        updated=True,
        previous_version=result["replaced_version"],
    )
    db.commit()
    db.refresh(document)
    result["message"] = message
    return result


def restore_version_as_separate(
    db: Session, *, document_id: int, version_id: int, new_name: str, restored_by: User
) -> dict:
    """Copy an earlier version out as its own document (recorded in the history)."""
    document = _active_document(db, document_id)
    result = doc_history.restore_version_as_separate(
        db, document, version_id, new_name=new_name, by_user_id=restored_by.Id
    )
    separate = result["document"]
    message = (
        f"Restored v{result['source_version']} of '{document.FileName}' as a separate document "
        f"'{separate.FileName}' -- the two are kept as different documents."
    )
    _record_library_job(
        db,
        job_type="restore",
        outcome="restored",
        document=separate,
        by_user=restored_by,
        chunk_count=result["chunk_count"],
        message=message,
        removed=False,
    )
    db.commit()
    db.refresh(separate)
    result["message"] = message
    result["source_filename"] = document.FileName
    return result


def rename_document(db: Session, *, document_id: int, new_name: str, renamed_by: User) -> dict:
    """Rename an active document (recorded in the history)."""
    document = _active_document(db, document_id)
    result = doc_history.rename_document(db, document, new_name, by_user_id=renamed_by.Id)
    if result["changed"]:
        _record_library_job(
            db,
            job_type="rename",
            outcome="renamed",
            document=document,
            by_user=renamed_by,
            chunk_count=_chunk_count(db, document.Id),
            message=f"Renamed '{result['old_name']}' to '{document.FileName}'.",
            removed=False,
            updated=True,
            previous_version=document.Version,
        )
        db.commit()
        db.refresh(document)
    return result


def set_excluded(db: Session, *, document_id: int, excluded: bool, by_user: User) -> dict:
    """Exclude a document from the chat, or include it again (recorded in the history)."""
    document = _active_document(db, document_id)
    if excluded:
        chunks = doc_history.exclude_document(db, document, by_user_id=by_user.Id)
        outcome, message = "excluded", f"Excluded '{document.FileName}' from the chat."
    else:
        chunks = doc_history.include_document(db, document, by_user_id=by_user.Id)
        outcome, message = "included", f"Included '{document.FileName}' in the chat again."
    _record_library_job(
        db,
        job_type="exclude" if excluded else "include",
        outcome=outcome,
        document=document,
        by_user=by_user,
        chunk_count=chunks,
        message=message,
        removed=False,
        updated=True,
        previous_version=document.Version,
    )
    db.commit()
    db.refresh(document)
    return {"document": document, "chunk_count": chunks}


# ---------------------------------------------------------------------------
# OneDrive / Google Drive sync: original additive pipeline (until step 3)
# ---------------------------------------------------------------------------


def ingest_upload(
    *,
    db: Session,
    filename: str,
    content: bytes,
    uploaded_by: User,
    source_type: str = "upload",
    source_path: Optional[str] = None,
) -> IngestionResult:
    content_hash = hashlib.sha256(content).digest()

    # Flag only -- per spec §7 this is "dedupe by content hash or warn",
    # and you chose warn: an existing match doesn't stop ingestion. Dedupe
    # is by content hash only, so the same file arriving via a different
    # source_type (e.g. uploaded locally, then also pulled from Google
    # Drive) still gets flagged.
    is_duplicate = (
        db.query(Document)
        .filter(Document.ContentHash == content_hash, Document.Status == doc_history.DOC_ACTIVE)
        .first()
        is not None
    )

    chunks = _extract_chunks(filename, content)
    vectors = embed_texts(chunks)

    document = Document(
        SourceType=source_type,
        SourcePath=source_path,
        FileName=filename,
        UploadedBy=uploaded_by.Id,
        ContentHash=content_hash,
        FileSizeBytes=len(content),
        Version=1,
        Status=doc_history.DOC_ACTIVE,
    )
    db.add(document)
    db.flush()  # assigns document.Id without ending the transaction

    _insert_chunks(db, document.Id, chunks, vectors)
    doc_history.record_new_latest(
        db, document, chunk_count=len(chunks), how_added="First upload", by_user_id=uploaded_by.Id
    )

    db.commit()
    db.refresh(document)

    return IngestionResult(
        document=document, chunk_count=len(chunks), is_duplicate=is_duplicate
    )
