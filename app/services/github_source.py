"""
GitHub connector (spec §3.2, §8 item 4): paste a GitHub URL or an
owner/repo[/path] shorthand, and every supported file at or under that
path (recursively) is pulled and ingested through the same pipeline as a
local upload (app/services/ingestion.py).

Uses PyGithub (already listed in requirements.txt) against the real
GitHub REST API -- api.github.com is reachable from wherever this backend
runs (confirmed from the sandbox this was built in, unlike
graph.microsoft.com/huggingface.co).
"""

import requests
from dataclasses import dataclass
from typing import Optional

from github import Github, GithubException, UnknownObjectException
from github.Repository import Repository

from app.config import settings
from app.services.parsing import SUPPORTED_EXTENSIONS

_MAX_BYTES = settings.MAX_UPLOAD_SIZE_MB * 1024 * 1024


class GithubPathError(ValueError):
    """The pasted path/URL couldn't be parsed."""


class GithubConnectionError(RuntimeError):
    """The parsed path couldn't be reached with the given token."""


@dataclass
class GithubPathRef:
    owner: str
    repo: str
    ref: Optional[str]  # branch/sha; None -> caller resolves the repo's default branch
    path: str  # '' -> repo root
    single_file: bool = False  # True for a /blob/ link -- sync just that one file


def parse_github_path(raw: str) -> GithubPathRef:
    """
    Accepts:
      - https://github.com/{owner}/{repo}
      - https://github.com/{owner}/{repo}/tree/{branch}/{path...}
      - https://github.com/{owner}/{repo}/blob/{branch}/{path/to/file}
      - github.com/{owner}/{repo}[...]  (scheme optional)
      - {owner}/{repo}[/{path...}]      (shorthand, uses the default branch)
    """
    text = (raw or "").strip()
    if not text:
        raise GithubPathError("Paste a GitHub repository URL or owner/repo[/path].")

    # Track whether this looks like a URL at all (scheme, "www.", or a bare
    # "github.com/..." host) *before* stripping anything -- a pasted URL on
    # some other host (e.g. gitlab.com) must be rejected clearly rather
    # than silently misread as "owner/repo" shorthand.
    lowered = text.lower()
    looks_like_url = (
        lowered.startswith("https://")
        or lowered.startswith("http://")
        or lowered.startswith("www.")
        or lowered.startswith("github.com")
    )

    for prefix in ("https://", "http://"):
        if text.lower().startswith(prefix):
            text = text[len(prefix) :]
            break
    if text.lower().startswith("www."):
        text = text[4:]

    if looks_like_url:
        if not text.lower().startswith("github.com"):
            host = text.split("/", 1)[0]
            raise GithubPathError(f"Only github.com URLs are supported (got '{host}').")
        text = text[len("github.com") :]

    text = text.strip("/")
    if not text:
        raise GithubPathError("Paste a GitHub repository URL or owner/repo[/path].")

    segments = [s for s in text.split("/") if s]
    if len(segments) < 2:
        raise GithubPathError(
            "Expected a GitHub URL or 'owner/repo', optionally with a /path."
        )

    owner, repo = segments[0], segments[1]
    if repo.endswith(".git"):
        repo = repo[: -len(".git")]
    rest = segments[2:]

    if rest and rest[0] in ("tree", "blob"):
        kind = rest[0]
        remainder = rest[1:]
        if not remainder:
            raise GithubPathError(f"Missing branch name after '/{kind}/' in the pasted URL.")
        branch, *path_parts = remainder
        return GithubPathRef(
            owner=owner,
            repo=repo,
            ref=branch,
            path="/".join(path_parts),
            single_file=(kind == "blob"),
        )

    return GithubPathRef(owner=owner, repo=repo, ref=None, path="/".join(rest))


def _client(pat: str) -> Github:
    return Github(pat)


def _resolve_repo(pat: str, ref: GithubPathRef) -> tuple[Github, Repository, str]:
    gh = _client(pat)
    try:
        repo = gh.get_repo(f"{ref.owner}/{ref.repo}")
    except UnknownObjectException:
        raise GithubConnectionError(
            f"Repository '{ref.owner}/{ref.repo}' not found, or not accessible with this token."
        )
    except GithubException as exc:
        message = (exc.data or {}).get("message", str(exc)) if hasattr(exc, "data") else str(exc)
        raise GithubConnectionError(f"GitHub API error: {message}")

    branch = ref.ref or repo.default_branch
    return gh, repo, branch


def _extension(path: str) -> str:
    basename = path.rsplit("/", 1)[-1]
    if "." not in basename:
        return ""
    return "." + basename.rsplit(".", 1)[-1].lower()


def _classify(path: str, size: int) -> Optional[str]:
    """Returns a skip reason, or None if the file should be pulled."""
    ext = _extension(path)
    if ext not in SUPPORTED_EXTENSIONS:
        return "unsupported file type"
    if size > _MAX_BYTES:
        return f"exceeds {settings.MAX_UPLOAD_SIZE_MB}MB limit"
    return None


def list_target_files(
    pat: str, ref: GithubPathRef
) -> tuple[Repository, str, list[dict], list[tuple[str, str]]]:
    """
    Returns (repo, resolved_branch, target_files, skipped) where
    target_files is [{"path", "size"}, ...] for every in-scope, supported,
    within-size-cap file, and skipped is [(path, reason), ...] for
    everything else found at/under ref.path (wrong extension, too large).
    Directories are never included in either list.
    """
    _, repo, branch = _resolve_repo(pat, ref)

    target_files: list[dict] = []
    skipped: list[tuple[str, str]] = []

    if ref.single_file:
        try:
            content_file = repo.get_contents(ref.path, ref=branch)
        except UnknownObjectException:
            raise GithubConnectionError(f"File '{ref.path}' not found on branch '{branch}'.")
        except GithubException as exc:
            message = (exc.data or {}).get("message", str(exc)) if hasattr(exc, "data") else str(exc)
            raise GithubConnectionError(f"GitHub API error: {message}")
        if isinstance(content_file, list):
            raise GithubConnectionError(f"'{ref.path}' is a directory, not a file -- paste a /blob/ link to a single file.")
        reason = _classify(content_file.path, content_file.size)
        if reason:
            skipped.append((content_file.path, reason))
        else:
            target_files.append({"path": content_file.path, "size": content_file.size})
        return repo, branch, target_files, skipped

    try:
        tree = repo.get_git_tree(branch, recursive=True)
    except GithubException as exc:
        message = (exc.data or {}).get("message", str(exc)) if hasattr(exc, "data") else str(exc)
        raise GithubConnectionError(f"GitHub API error listing repository tree: {message}")

    prefix = ref.path.strip("/")
    matched_any = False
    for element in tree.tree:
        if element.type != "blob":
            continue
        path = element.path
        if prefix and not (path == prefix or path.startswith(prefix + "/")):
            continue
        matched_any = True
        reason = _classify(path, element.size)
        if reason:
            skipped.append((path, reason))
        else:
            target_files.append({"path": path, "size": element.size})

    if prefix and not matched_any:
        raise GithubConnectionError(f"No files found under '{prefix}' on branch '{branch}'.")

    return repo, branch, target_files, skipped


def fetch_file_content(repo: Repository, path: str, ref: str) -> bytes:
    content_file = repo.get_contents(path, ref=ref)
    if isinstance(content_file, list):
        raise GithubConnectionError(f"'{path}' resolved to a directory unexpectedly.")
    try:
        content = content_file.decoded_content
        if content is not None:
            return content
    except Exception:  # noqa: BLE001 -- fall through to the raw download below
        pass

    # decoded_content can come back empty/None for files GitHub doesn't
    # inline in the Contents API response (roughly >1MB) -- fetch the raw
    # bytes directly instead.
    resp = requests.get(content_file.download_url, timeout=30)
    resp.raise_for_status()
    return resp.content
