"""
Splits extracted document text into overlapping chunks for embedding
(spec §3.2: "Each file is parsed, chunked, embedded"). Character-based,
not token-based -- simple, and independent of whatever embedding model is
configured.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

_CHUNK_SIZE = 1000
_CHUNK_OVERLAP = 150

_splitter = RecursiveCharacterTextSplitter(
    chunk_size=_CHUNK_SIZE,
    chunk_overlap=_CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""],
)


def chunk_text(text: str) -> list[str]:
    chunks = [chunk.strip() for chunk in _splitter.split_text(text)]
    return [chunk for chunk in chunks if chunk]
