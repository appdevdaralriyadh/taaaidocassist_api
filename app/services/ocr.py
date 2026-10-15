"""
Reads scanned / image-only PDF pages with Claude ("OCR").

pypdf (app/services/parsing.py) can only read text that is stored in a PDF.
A scanned document or an exported newsletter that is one picture per page
has none, so those files used to be skipped. With OCR_ENABLED=true in .env,
parsing.py sends just the pages that have (almost) no text to Claude, which
reads the page images and types out what is on them -- English only by
default (OCR_ENGLISH_ONLY; false = Arabic too), tables as Markdown tables,
a one-line [description] for photos and charts.
Pages that already have text are used as they are and never sent.

  * Pages are sent OCR_PAGES_PER_CALL at a time as a small PDF made of only
    those pages (Claude reads PDFs natively). A group that is too large or
    whose answer runs out of room is split in half and retried.
  * Files with more than OCR_MAX_PAGES pages to read are not sent (cost cap).
  * If Claude can't be reached, OcrUnavailableError is raised: the file is
    shown as Failed (not Skipped), so the next sync tries it again.
  * Uses ANTHROPIC_API_KEY (the same key as chat); the model is OCR_MODEL,
    or ANTHROPIC_MODEL when that's empty. Each call is logged with its page
    count, tokens and time ("Claude OCR: ...").

The whole page image of the scanned pages goes to Anthropic -- the same
provider chat already uses.
"""

import base64
import io
import logging
import re
import threading
import time

from fastapi import HTTPException, status
from pypdf import PdfReader, PdfWriter

from app.config import settings

logger = logging.getLogger("darai.ocr")

_client = None
_client_lock = threading.Lock()

# Anthropic accepts requests up to 32 MB; the PDF is sent base64-encoded
# (one third bigger), so the raw PDF sent per call is kept under this.
_MAX_PDF_BYTES = 22 * 1024 * 1024

_PAGE_MARKER = re.compile(r"^[ \t]*=+[ \t]*PAGE[ \t]+(\d+)[ \t]*=+[ \t]*$", re.IGNORECASE | re.MULTILINE)

_PROMPT = """The attached PDF has {n} page(s) of scanned or image-only pages from a company document. Your transcription will be indexed so people can search the document.

{scope} Do not translate, summarise, correct or add anything.
- Keep the reading order, headings and paragraphs.
- Write tables as Markdown tables.
- For a photo, chart, logo or diagram, write one short line in square brackets describing it, including any names, numbers or labels it shows, e.g. [Photo: employees at the 2024 safety award ceremony].
- Start each page with a line "=== PAGE k ===" (k = 1 to {n}, in order), also for a blank page.

Output only the transcription."""

_SCOPE_ALL = (
    "Transcribe ALL visible text on every page exactly as written, in its original language "
    "(Arabic stays Arabic, English stays English)."
)
_SCOPE_ENGLISH = (
    "Transcribe all visible ENGLISH text on every page exactly as written. Skip Arabic text "
    "entirely -- do not copy or translate it; if a page has no English text, leave it empty "
    "after its page line."
)


class OcrUnavailableError(RuntimeError):
    """Claude couldn't be used right now (network, rate limit, key...) -- the message is shown to people."""


def enabled() -> bool:
    return bool(settings.OCR_ENABLED)


def model() -> str:
    return (settings.OCR_MODEL or "").strip() or settings.ANTHROPIC_MODEL


def skip_marker() -> str:
    """
    Part of the "remember this file was unreadable" record kept per cloud
    file (app/services/cloud_sync.py unreadable_map). It changes when OCR is
    switched on or OCR_MAX_PAGES changes, so files skipped under the old
    setting are downloaded and tried again on the next sync.
    """
    if not enabled():
        return ""
    return f"ocr{settings.OCR_MAX_PAGES}{'en' if settings.OCR_ENGLISH_ONLY else ''}:"


# ---------------------------------------------------------------------------
# Called from app/services/parsing.py
# ---------------------------------------------------------------------------


def fill_image_pages(filename: str, content: bytes, pages: list[str]) -> list[str]:
    """
    `pages` is the text pypdf found on each page. Pages with fewer than
    OCR_MIN_CHARS_PER_PAGE characters are read with Claude and replaced by
    what Claude found (when that's longer). Returns the new list of page
    texts. Raises HTTPException 503 when Claude can't be reached, and 422
    ("No extractable text found ...") when a file with no text at all has
    more pages to read than OCR_MAX_PAGES.
    """
    threshold = max(0, settings.OCR_MIN_CHARS_PER_PAGE)
    to_read = [i for i, text in enumerate(pages) if len(text.strip()) < max(threshold, 1)]
    if not to_read:
        return pages

    limit = settings.OCR_MAX_PAGES
    if len(to_read) > limit:
        if not any(text.strip() for text in pages):
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"No extractable text found in '{filename}' -- it has {len(to_read)} scanned "
                    f"pages, more than the {limit} that are read with Claude (OCR_MAX_PAGES)."
                ),
            )
        logger.warning(
            "%s: %d pages without text, more than OCR_MAX_PAGES=%d -- only its typed text is used",
            filename, len(to_read), limit,
        )
        return pages

    started = time.perf_counter()
    try:
        found = read_pages(content, to_read, filename)
    except OcrUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                f"Couldn't read the scanned pages of '{filename}' with Claude right now -- {exc} "
                "A sync tries it again next time; an upload can simply be repeated."
            ),
        ) from exc

    result = list(pages)
    replaced = 0
    for index, text in found.items():
        if len(text.strip()) > len(result[index].strip()):
            result[index] = text
            replaced += 1
    logger.info(
        "%s: read %d of %d page(s) with Claude, text found on %d, %.1f s",
        filename, len(to_read), len(pages), replaced, time.perf_counter() - started,
    )
    return result


def read_pages(content: bytes, page_indexes: list[int], filename: str = "") -> dict[int, str]:
    """{page index: text Claude read on it} for the given 0-based pages of the PDF."""
    reader = PdfReader(io.BytesIO(content))
    size = max(1, settings.OCR_PAGES_PER_CALL)
    queue = [page_indexes[i:i + size] for i in range(0, len(page_indexes), size)]
    found: dict[int, str] = {}
    while queue:
        group = queue.pop(0)
        pdf = _pages_pdf(reader, group)
        if len(pdf) > _MAX_PDF_BYTES:
            if len(group) > 1:
                half = len(group) // 2
                queue[:0] = [group[:half], group[half:]]
                continue
            logger.warning(
                "%s: page %d is %.0f MB, too large to send to Claude -- left without text",
                filename, group[0] + 1, len(pdf) / 1024 / 1024,
            )
            continue
        text, cut_off = _ask_claude(pdf, group, filename)
        if cut_off and len(group) > 1:
            # the answer ran out of room: read the pages in two smaller groups
            half = len(group) // 2
            queue[:0] = [group[:half], group[half:]]
            continue
        if cut_off:
            logger.warning("%s: page %d has more text than OCR_MAX_OUTPUT_TOKENS -- kept what was read",
                           filename, group[0] + 1)
        for index, page_text in zip(group, _split_pages(text, len(group))):
            found[index] = page_text
    return found


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _get_client():
    global _client
    if _client is None:
        with _client_lock:
            if _client is None:
                from anthropic import Anthropic

                _client = Anthropic(
                    api_key=settings.ANTHROPIC_API_KEY,
                    max_retries=settings.OCR_MAX_RETRIES,
                    timeout=settings.OCR_TIMEOUT_SECONDS,
                )
    return _client


def _pages_pdf(reader: PdfReader, page_indexes: list[int]) -> bytes:
    writer = PdfWriter()
    for index in page_indexes:
        writer.add_page(reader.pages[index])
    buffer = io.BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _ask_claude(pdf: bytes, page_indexes: list[int], filename: str) -> tuple[str, bool]:
    """(text, cut off because it ran out of output room)"""
    started = time.perf_counter()
    try:
        response = _get_client().messages.create(
            model=model(),
            max_tokens=settings.OCR_MAX_OUTPUT_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": base64.standard_b64encode(pdf).decode("ascii"),
                            },
                        },
                        {"type": "text", "text": _prompt(len(page_indexes))},
                    ],
                }
            ],
        )
    except Exception as exc:  # noqa: BLE001 -- anthropic.* errors and network errors
        logger.warning("Claude OCR call failed for %s: %s: %s", filename, exc.__class__.__name__, exc)
        raise OcrUnavailableError(_error_message(exc)) from exc

    text = "".join(
        getattr(block, "text", "") for block in response.content if getattr(block, "type", "") == "text"
    )
    usage = getattr(response, "usage", None)
    logger.info(
        "Claude OCR: %s pages %s, %d input / %d output tokens, %.1f s (model %s)",
        filename,
        _page_range(page_indexes),
        int(getattr(usage, "input_tokens", 0) or 0),
        int(getattr(usage, "output_tokens", 0) or 0),
        time.perf_counter() - started,
        model(),
    )
    return text, getattr(response, "stop_reason", None) == "max_tokens"


def _prompt(count: int) -> str:
    scope = _SCOPE_ENGLISH if settings.OCR_ENGLISH_ONLY else _SCOPE_ALL
    return _PROMPT.format(n=count, scope=scope)


def _split_pages(text: str, count: int) -> list[str]:
    """Splits Claude's answer on the "=== PAGE k ===" lines into `count` page texts."""
    parts = _PAGE_MARKER.split(text)
    if len(parts) < 3:
        # no page markers: keep the whole text (on the group's first page)
        return [text.strip()] + [""] * (count - 1)
    pages = [""] * count
    leading = parts[0].strip()
    for number, body in zip(parts[1::2], parts[2::2]):
        k = int(number) - 1
        if 0 <= k < count:
            pages[k] = (pages[k] + "\n\n" + body.strip()).strip()
        elif body.strip():
            pages[-1] = (pages[-1] + "\n\n" + body.strip()).strip()
    if leading:
        pages[0] = (leading + "\n\n" + pages[0]).strip()
    return pages


def _page_range(page_indexes: list[int]) -> str:
    first, last = page_indexes[0] + 1, page_indexes[-1] + 1
    return str(first) if len(page_indexes) == 1 else f"{first}-{last}"


def _error_message(exc: Exception) -> str:
    name = exc.__class__.__name__
    detail = str(exc)[:300]
    if name == "AuthenticationError":
        return "Claude rejected the API key (ANTHROPIC_API_KEY in .env)."
    if name == "PermissionDeniedError":
        return f"the API key isn't allowed to use this model ({model()})."
    if name == "NotFoundError":
        return f"the model '{model()}' wasn't found -- check OCR_MODEL / ANTHROPIC_MODEL in .env."
    if name == "RateLimitError":
        return "Claude's rate limit was reached."
    if name in ("APIConnectionError", "APITimeoutError", "InternalServerError", "OverloadedError",
                "ServiceUnavailableError", "DeadlineExceededError"):
        return f"Claude couldn't be reached ({detail or name})."
    return f"Claude returned an error: {detail or name}"
