"""
Helpers for deciding -- automatically, with no one confirming -- whether an
uploaded file is a new version of a document already in the knowledge
base, even when its filename differs.

Used by app/services/ingestion.py's ingest_local_upload(), which applies
these rules (thresholds in app/config.py):

  * Same filename                                   -> new version
  * Same name once version words are stripped       -> new version, if the
    ("Gift Policy 2026 new version.pdf" vs            wording is at least
     "Gift Policy.pdf" -- normalize_doc_name)         VERSION_MATCH_MIN_SIMILARITY
                                                      the same BOTH ways
  * Different name                                  -> new version only if
                                                      at least CONTENT_MATCH_
                                                      MIN_SIMILARITY of the
                                                      wording matches one way
                                                      and CONTENT_MATCH_MIN_
                                                      COVERAGE the other way,
                                                      both are long enough to
                                                      judge, and the names
                                                      aren't siblings in a
                                                      series (are_sibling_names)
  * Anything less                                   -> kept as a separate
                                                      document (flagged when
                                                      it looks like a version)

"Both ways" (two_way_similarity) is what keeps a short excerpt from
replacing the full document it was taken from: the excerpt's wording is
100% inside the full document, but only a small part of the full
document's wording is inside the excerpt, so the two-way score is low.

doc_year() is the older-version guard: if a matching document's name/title
carries a later year than the upload's, the upload is treated as an older
version and not stored, rather than replacing the newer one.

Pure functions, no database or model access -- easy to test on their own.
"""

import re
from typing import Optional

# Words that describe *which version* a file is, not *what document* it is.
# Stripped from filenames before comparing them.
_VERSION_WORDS = {
    "new",
    "version",
    "ver",
    "final",
    "latest",
    "updated",
    "update",
    "revised",
    "revision",
    "rev",
    "copy",
    "draft",
    "old",
    "current",
    "edition",
    "amended",
}

_SHINGLE_SIZE = 3  # consecutive words per shingle
_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")
_TITLE_WORDS = 60  # how much of a document's opening text counts as its "title area"


def normalize_doc_name(filename: str) -> str:
    """
    Reduces a filename to the part that identifies the document:
    lowercased, extension removed, possessives/punctuation removed, and
    years, dates, version markers (v2, ver 3, v1.2), copy suffixes ("(1)")
    and version words ("new", "final", "updated", ...) dropped.

    "Dar Al Riyadh's Gift Policy 2026 new version.pdf"
        -> "dar al riyadh gift policy"

    Plain numbers that aren't years/dates are kept, so "Annex 1.pdf" and
    "Annex 2.pdf" stay different documents. Returns "" when nothing
    identifying is left (e.g. "2026.pdf") -- callers must treat "" as
    "no name match".
    """
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    s = stem.lower()
    s = re.sub(r"['’`]s\b", "", s)                          # possessive 's
    s = re.sub(r"['’`]", "", s)                             # other apostrophes
    s = re.sub(r"\b\d{1,4}[-./]\d{1,2}[-./]\d{1,4}\b", " ", s)   # dates 2026-10-04, 04.10.2026
    s = re.sub(r"\bv(?:er)?\.?\s?\d+(?:\.\d+)*\b", " ", s)       # v2, v.2, ver 3, v1.2
    s = re.sub(r"\(\d+\)", " ", s)                               # "(1)" copy suffixes
    s = s.replace("_", " ")
    s = re.sub(r"[^\w]+", " ", s)                                # remaining punctuation

    tokens = []
    for token in s.split():
        if re.fullmatch(r"(?:19|20)\d{2}", token):  # years
            continue
        if re.fullmatch(r"\d{6,8}", token):         # compact dates 20261004
            continue
        if token in _VERSION_WORDS:
            continue
        tokens.append(token)
    return " ".join(tokens)


def text_shingles(text: str) -> set:
    """
    Overlapping 3-word sequences of the text (case- and punctuation-
    insensitive). Comparing these sets, rather than chunks, means an edit
    in one place only affects the shingles around it -- it doesn't shift
    every chunk boundary after it the way chunk-by-chunk comparison would.
    """
    words = re.findall(r"\w+", text.lower())
    if len(words) < _SHINGLE_SIZE:
        return {tuple(words)} if words else set()
    return {tuple(words[i : i + _SHINGLE_SIZE]) for i in range(len(words) - _SHINGLE_SIZE + 1)}


def two_way_similarity(new: set, old: set) -> tuple[float, float, float]:
    """
    Returns (score, new_in_old, old_in_new):
      new_in_old -- share of the NEW text's 3-word sequences found in the old
      old_in_new -- share of the OLD text's 3-word sequences found in the new
      score      -- the lower of the two (0.0-1.0)

    Using the lower of the two means BOTH documents have to be mostly made
    of the other's wording. A genuine revision passes (most of the old
    wording kept, most of the new wording already there); an excerpt, or a
    document that only contains a copied section of another, does not.
    """
    if not new or not old:
        return 0.0, 0.0, 0.0
    shared = len(new & old)
    new_in_old = shared / len(new)
    old_in_new = shared / len(old)
    return min(new_in_old, old_in_new), new_in_old, old_in_new


def are_sibling_names(normalized_a: str, normalized_b: str) -> bool:
    """
    True when two (normalize_doc_name'd) names look like two different
    documents of the same kind -- mostly the same name, but EACH with its
    own distinguishing words: "nda vendor a" / "nda vendor b",
    "employment contract ahmed" / "employment contract sara",
    "dar al riyadh gift policy" / "dar al riyadh travel policy".

    Documents generated from a template (contracts, letters, policies
    from one master layout) can share 90%+ of their wording while being
    genuinely different documents, so a content-only match between
    sibling names is never replaced automatically -- it's flagged instead.

    Not siblings: unrelated names ("dar al riyadh gift policy" / "gift
    rules for staff"), or one name simply extending the other ("leave
    policy" / "annual leave policy").
    """
    a, b = set(normalized_a.split()), set(normalized_b.split())
    if not a or not b:
        return False
    shared = a & b
    if not shared or not (a - b) or not (b - a):
        return False
    return len(shared) / len(a | b) >= 0.5


def doc_year(filename: str, text: str = "") -> Optional[int]:
    """
    The latest year (19xx/20xx) in a document's filename -- or, if the
    filename has none, in the opening words of its text (its title area,
    e.g. "Gift Policy 2026", "Revised 2025"). None when neither has one.
    Used only to stop an older edition from replacing a newer one.
    """
    stem = filename.rsplit(".", 1)[0] if "." in filename else filename
    years = [int(y) for y in _YEAR.findall(stem)]
    if not years and text:
        head = " ".join(re.findall(r"\S+", text)[:_TITLE_WORDS])
        years = [int(y) for y in _YEAR.findall(head)]
    return max(years) if years else None
