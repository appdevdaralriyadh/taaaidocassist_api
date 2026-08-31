"""
Extracts plain text from an uploaded file's raw bytes, dispatching on file
extension. Covers every source format spec §3.2 lists for local upload:
PDF, DOCX, TXT, CSV, MD, XLSX.
"""

import io

import pandas as pd
from docx import Document as DocxDocument
from fastapi import HTTPException, status
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".csv", ".xlsx"}


def extract_text(filename: str, content: bytes) -> str:
    ext = _extension(filename)

    if ext == ".pdf":
        return _extract_pdf(content)
    if ext == ".docx":
        return _extract_docx(content)
    if ext in (".txt", ".md"):
        return _extract_plain_text(content)
    if ext == ".csv":
        return _extract_csv(content)
    if ext == ".xlsx":
        return _extract_xlsx(content)

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail=(
            f"Unsupported file type '{ext or filename}'. Supported types: "
            f"{', '.join(sorted(SUPPORTED_EXTENSIONS))}."
        ),
    )


def _extension(filename: str) -> str:
    return "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _extract_pdf(content: bytes) -> str:
    reader = PdfReader(io.BytesIO(content))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def _extract_docx(content: bytes) -> str:
    doc = DocxDocument(io.BytesIO(content))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)


def _extract_plain_text(content: bytes) -> str:
    return content.decode("utf-8", errors="replace")


def _extract_csv(content: bytes) -> str:
    df = pd.read_csv(io.BytesIO(content))
    return df.to_csv(index=False)


def _extract_xlsx(content: bytes) -> str:
    sheets = pd.read_excel(io.BytesIO(content), sheet_name=None)
    parts = [f"# Sheet: {name}\n{df.to_csv(index=False)}" for name, df in sheets.items()]
    return "\n\n".join(parts)
