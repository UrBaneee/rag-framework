"""Web URL connector — fetches web pages and GitHub content for ingestion.

Uses trafilatura's built-in URL fetching (already a project dependency) to
download HTML pages.  GitHub raw URLs are detected automatically and saved
with a ``.md`` extension so the Markdown parser is used instead of HTML.

Usage::

    connector = WebConnector()
    artifacts = connector.fetch(["https://example.com/page", "https://..."])
    for artifact in artifacts:
        pipeline.ingest(artifact.tmp_path, canonical_name=artifact.canonical_name)
        artifact.cleanup()
"""

from __future__ import annotations

import logging
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# GitHub raw content host — treat as Markdown
_GITHUB_RAW_HOST = "raw.githubusercontent.com"

# Common GitHub repo URL patterns that should be converted to raw URLs
_GITHUB_HOST = "github.com"


def _to_raw_github_url(url: str) -> str:
    """Convert a GitHub blob URL to a raw content URL.

    e.g. https://github.com/user/repo/blob/main/README.md
      → https://raw.githubusercontent.com/user/repo/main/README.md

    Args:
        url: GitHub URL to convert.

    Returns:
        Raw content URL, or the original URL if conversion is not applicable.
    """
    parsed = urlparse(url)
    if parsed.hostname != _GITHUB_HOST:
        return url
    parts = parsed.path.split("/")
    # /user/repo/blob/branch/path...
    if len(parts) >= 5 and parts[3] == "blob":
        raw_path = "/" + "/".join(parts[1:3] + parts[4:])
        return f"https://{_GITHUB_RAW_HOST}{raw_path}"
    return url


def _canonical_name_from_url(url: str) -> str:
    """Derive a human-readable canonical name from a URL.

    Uses the last path component, falling back to the hostname.

    Args:
        url: Source URL.

    Returns:
        Short string suitable as a document identifier.
    """
    parsed = urlparse(url)
    path = parsed.path.rstrip("/")
    if path:
        name = Path(path).name or parsed.hostname or url
    else:
        name = parsed.hostname or url
    # Ensure it has an extension (used by the MIME sniffer)
    if "." not in name:
        name = name + ".html"
    return name


def _extension_for_url(url: str) -> str:
    """Return the appropriate file extension for the fetched content.

    Markdown for raw GitHub / .md URLs, HTML otherwise.

    Args:
        url: Resolved URL.

    Returns:
        Extension string including the leading dot.
    """
    parsed = urlparse(url)
    suffix = Path(parsed.path).suffix.lower()
    if suffix in (".md", ".markdown", ".rst", ".txt"):
        return suffix
    if parsed.hostname == _GITHUB_RAW_HOST:
        return ".md"
    return ".html"


@dataclass
class WebArtifact:
    """Temporary file created by WebConnector.fetch().

    Attributes:
        tmp_path: Absolute path to the temporary file on disk.
        canonical_name: Human-readable name for the document (used as the
            ``canonical_name`` argument to ``IngestPipeline.ingest()``).
        url: Original URL that was fetched.
    """

    tmp_path: str
    canonical_name: str
    url: str
    _files_to_cleanup: list[str] = field(default_factory=list, repr=False)

    def cleanup(self) -> None:
        """Delete the temporary file from disk."""
        import os

        for path in self._files_to_cleanup:
            try:
                os.unlink(path)
            except OSError:
                pass


class WebConnector:
    """Fetches web pages and GitHub documents for ingestion.

    Uses ``trafilatura.fetch_url()`` for downloading — no additional
    HTTP library required.

    Args:
        timeout: Request timeout in seconds. Defaults to 30.
        user_agent: HTTP User-Agent string sent with requests.
    """

    _DEFAULT_USER_AGENT = (
        "Mozilla/5.0 (compatible; RAGStudioBot/1.0; +https://github.com)"
    )

    def __init__(
        self,
        timeout: int = 30,
        user_agent: str | None = None,
    ) -> None:
        try:
            import trafilatura  # noqa: F401
        except ImportError as exc:
            raise ImportError(
                "trafilatura is required for WebConnector. "
                "Install with: pip install trafilatura"
            ) from exc

        self._timeout = timeout
        self._user_agent = user_agent or self._DEFAULT_USER_AGENT

    def fetch(self, urls: list[str]) -> list[WebArtifact]:
        """Fetch a list of URLs and save each to a temporary file.

        GitHub blob URLs are automatically converted to raw content URLs.
        Failed fetches are logged and skipped (no exception raised).

        Args:
            urls: List of URLs to fetch.

        Returns:
            List of WebArtifact instances for successfully fetched URLs.
            Call ``artifact.cleanup()`` after ingestion to remove temp files.
        """
        import trafilatura

        artifacts: list[WebArtifact] = []

        for raw_url in urls:
            url = raw_url.strip()
            if not url:
                continue

            # Convert GitHub blob → raw URL
            resolved_url = _to_raw_github_url(url)
            if resolved_url != url:
                logger.info("GitHub URL converted: %s → %s", url, resolved_url)

            try:
                html_content = trafilatura.fetch_url(resolved_url)
            except Exception as exc:
                logger.error("Failed to fetch '%s': %s", resolved_url, exc)
                continue

            if not html_content:
                logger.warning("Empty response from '%s' — skipping.", resolved_url)
                continue

            ext = _extension_for_url(resolved_url)
            canonical = _canonical_name_from_url(resolved_url)

            # Ensure canonical name has the right extension
            if not canonical.endswith(ext):
                stem = Path(canonical).stem
                canonical = stem + ext

            try:
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=ext, mode="w", encoding="utf-8"
                ) as tmp:
                    tmp.write(html_content)
                    tmp_path = tmp.name
            except Exception as exc:
                logger.error("Failed to write temp file for '%s': %s", resolved_url, exc)
                continue

            artifacts.append(
                WebArtifact(
                    tmp_path=tmp_path,
                    canonical_name=canonical,
                    url=resolved_url,
                    _files_to_cleanup=[tmp_path],
                )
            )
            logger.info(
                "Fetched '%s' → %s (%d chars)",
                resolved_url,
                canonical,
                len(html_content),
            )

        return artifacts
