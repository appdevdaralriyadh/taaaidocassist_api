"""
Read-only diagnostic: reports every distinct embedding byte-length
currently stored in DarAI_DocumentChunks, and which document(s) each one
belongs to.

Why this matters right now: app/config.py's EMBEDDING_MODEL was switched
from all-MiniLM-L6-v2 (384-dim) to BAAI/bge-m3 (1024-dim). Nothing in the
codebase converts old embeddings when the model changes -- every chunk's
Embedding column just holds whatever raw float32 bytes were produced by
whatever model was active at the moment it was ingested. This script
answers, directly against the live table, whether more than one
dimension is currently present.

Run from inside taaaidocassist_api/, with the venv active (so `app`
resolves and .env loads the same way app/config.py already does):

    cd taaaidocassist_api
    venv\\Scripts\\activate
    python scripts\\check_embedding_dimensions.py

Makes no writes of any kind -- safe to run at any time.
"""

import sys
from collections import defaultdict
from pathlib import Path

# Running `python scripts\check_embedding_dimensions.py` puts this
# script's own folder (scripts/) on sys.path, not the project root
# (taaaidocassist_api/) where the `app` package actually lives -- that's
# just how Python resolves imports for a directly-invoked script file,
# regardless of which directory you `cd`'d into first. Inserting the
# parent directory explicitly makes this work no matter how/from-where
# it's run, without needing `python -m` or a scripts/__init__.py.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func

from app.db.models import Document, DocumentChunk
from app.db.session import SessionLocal

# float32 = 4 bytes per vector component -- matches
# app/services/embeddings.py's serialize_embedding()/deserialize_embedding(),
# which is the only code that ever writes or reads this column.
BYTES_PER_DIM = 4


def main() -> None:
    db = SessionLocal()
    try:
        rows = (
            db.query(
                func.datalength(DocumentChunk.Embedding).label("byte_len"),
                Document.FileName,
                Document.SourceType,
            )
            .join(Document, Document.Id == DocumentChunk.DocumentId)
            .all()
        )
    finally:
        db.close()

    if not rows:
        print("DarAI_DocumentChunks is empty -- nothing to check.")
        return

    by_dim: dict[int, dict] = defaultdict(lambda: {"chunks": 0, "files": set()})
    for byte_len, filename, source_type in rows:
        dim = byte_len // BYTES_PER_DIM
        by_dim[dim]["chunks"] += 1
        by_dim[dim]["files"].add((filename, source_type))

    print(f"{len(rows)} chunk(s) total, {len(by_dim)} distinct embedding dimension(s):\n")
    for dim, info in sorted(by_dim.items()):
        print(f"  {dim}-dim  ->  {info['chunks']} chunk(s) across {len(info['files'])} file(s)")

    if len(by_dim) == 1:
        (dim,) = by_dim.keys()
        print(f"\nAll stored vectors are {dim}-dim -- no mixed-model contamination.")
        print(
            "If this dimension doesn't match your current EMBEDDING_MODEL's real output "
            "size (bge-m3 -> 1024), every one of these chunks is still being silently "
            "skipped by retrieval.py's dimension guard -- see the note below."
        )
        return

    print("\n*** MIXED DIMENSIONS -- confirmed ***")
    print(
        "app/services/retrieval.py's search() compares each stored chunk's dimension "
        "against the live query's dimension and silently skips (logs a warning, doesn't "
        "crash) anything that doesn't match. So this isn't 'invalid math being computed' "
        "-- it's chunks being entirely excluded from every search. Whichever dimension "
        "does NOT match your current EMBEDDING_MODEL's real output size is currently "
        "invisible to the knowledge base, full stop, regardless of how relevant the text "
        "actually is. Affected files, by dimension:\n"
    )
    for dim, info in sorted(by_dim.items()):
        print(f"  --- {dim}-dim ({info['chunks']} chunks) ---")
        for filename, source_type in sorted(info["files"]):
            print(f"    {filename}  (source: {source_type})")


if __name__ == "__main__":
    main()
