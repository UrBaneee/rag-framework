"""Google Docs connector — Task 15.5.

Pulls Google Docs files from one or more Google Drive folders since a
saved cursor (ISO 8601 ``modifiedTime`` timestamp), converts each
document's structural content into a plain-text ``SourceArtifact``, and
advances the cursor.

Authentication uses a Service Account JSON key file whose path is read
from an environment variable:

    RAG_GOOGLE_SERVICE_ACCOUNT_JSON — Path to the service account JSON key
    RAG_GOOGLE_DRIVE_FOLDER_IDS     — Comma-separated Drive folder IDs to
                                      sync (optional; empty = search all
                                      files accessible to the service account)

The connector calls two Google APIs via ``urllib`` (no ``google-api-python-client``
needed):

- **Google Drive API v3** (``files.list``) — enumerate Docs files since cursor
- **Google Docs API v1** (``documents.get``) — fetch full document content

Access tokens are obtained by minting a short-lived OAuth 2.0 JWT assertion
signed with the service account private key (RS256) and exchanging it at the
Google token endpoint — entirely with the standard library (``json``,
``urllib``, ``base64``, ``hmac``).  When ``cryptography`` is not available
the connector falls back to ``subprocess`` calling ``openssl`` for signing.

Document content is converted to plain text by iterating the ``body.content``
structural elements: paragraphs (with heading styles), lists, and tables.

Usage::

    import os
    os.environ["RAG_GOOGLE_SERVICE_ACCOUNT_JSON"] = "/path/to/key.json"
    os.environ["RAG_GOOGLE_DRIVE_FOLDER_IDS"] = "1BxiM..."

    connector = GoogleDocsConnector()
    artifacts = connector.list_items(since_cursor="")
    cursor = connector.next_cursor()
"""

from __future__ import annotations

import base64
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Optional

from rag.core.contracts.source_artifact import SourceArtifact
from rag.core.interfaces.source_connector import BaseSourceConnector

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DRIVE_API = "https://www.googleapis.com/drive/v3"
_DOCS_API = "https://docs.googleapis.com/v1"
_TOKEN_URI = "https://oauth2.googleapis.com/token"
_SCOPES = "https://www.googleapis.com/auth/drive.readonly https://www.googleapis.com/auth/documents.readonly"
_TOKEN_LIFETIME = 3600  # seconds
_DEFAULT_PAGE_SIZE = 20

# Heading styles → markdown-like prefixes
_HEADING_PREFIX = {
    "HEADING_1": "# ",
    "HEADING_2": "## ",
    "HEADING_3": "### ",
    "HEADING_4": "#### ",
    "TITLE": "# ",
    "SUBTITLE": "## ",
}


# ---------------------------------------------------------------------------
# JWT / OAuth helpers (stdlib only)
# ---------------------------------------------------------------------------


def _b64url(data: bytes) -> str:
    """URL-safe base64 encode without padding."""
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _sign_rs256(message: bytes, private_key_pem: str) -> bytes:
    """Sign ``message`` with RS256 using the given PEM private key.

    Tries ``cryptography`` first; falls back to ``subprocess openssl``.

    Args:
        message: Bytes to sign.
        private_key_pem: PEM-encoded RSA private key string.

    Returns:
        Raw RS256 signature bytes.

    Raises:
        RuntimeError: If neither signing method is available.
    """
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding

        private_key = serialization.load_pem_private_key(
            private_key_pem.encode(), password=None
        )
        return private_key.sign(message, padding.PKCS1v15(), hashes.SHA256())
    except ImportError:
        pass

    # Fallback: openssl via subprocess
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".pem", delete=False) as f:
        f.write(private_key_pem.encode())
        key_path = f.name
    try:
        result = subprocess.run(
            ["openssl", "dgst", "-sha256", "-sign", key_path],
            input=message,
            capture_output=True,
            check=True,
        )
        return result.stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        raise RuntimeError(
            "RS256 signing requires either the 'cryptography' package "
            "or 'openssl' on PATH.  Install with: pip install cryptography"
        ) from exc
    finally:
        os.unlink(key_path)


def _mint_jwt(service_account: dict[str, Any]) -> str:
    """Create a signed JWT assertion for a Google service account.

    Args:
        service_account: Parsed service account JSON dict.

    Returns:
        Signed JWT string.
    """
    now = int(time.time())
    header = {"alg": "RS256", "typ": "JWT"}
    payload = {
        "iss": service_account["client_email"],
        "scope": _SCOPES,
        "aud": _TOKEN_URI,
        "iat": now,
        "exp": now + _TOKEN_LIFETIME,
    }
    header_b64 = _b64url(json.dumps(header).encode())
    payload_b64 = _b64url(json.dumps(payload).encode())
    signing_input = f"{header_b64}.{payload_b64}".encode()
    signature = _sign_rs256(signing_input, service_account["private_key"])
    return f"{header_b64}.{payload_b64}.{_b64url(signature)}"


def _exchange_jwt_for_token(jwt: str) -> str:
    """Exchange a JWT assertion for a Google OAuth 2.0 access token.

    Args:
        jwt: Signed JWT assertion.

    Returns:
        Access token string.

    Raises:
        RuntimeError: On HTTP error or missing token in response.
    """
    body = urllib.parse.urlencode(
        {
            "grant_type": "urn:ietf:params:oauth:grant-type:jwt-bearer",
            "assertion": jwt,
        }
    ).encode()
    req = urllib.request.Request(
        _TOKEN_URI,
        data=body,
        method="POST",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Token exchange failed: {exc}") from exc
    token = data.get("access_token")
    if not token:
        raise RuntimeError(f"No access_token in response: {data}")
    return token


# ---------------------------------------------------------------------------
# Document content helpers
# ---------------------------------------------------------------------------


def _paragraph_to_text(paragraph: dict[str, Any]) -> Optional[str]:
    """Convert a Google Docs paragraph element to a plain-text line.

    Args:
        paragraph: A ``paragraph`` structural element from the Docs API.

    Returns:
        Plain-text string with heading prefix, or ``None`` if empty.
    """
    style = paragraph.get("paragraphStyle", {}).get("namedStyleType", "NORMAL_TEXT")
    elements = paragraph.get("elements", [])
    text = "".join(
        el.get("textRun", {}).get("content", "")
        for el in elements
        if "textRun" in el
    ).rstrip("\n")

    if not text.strip():
        return None

    prefix = _HEADING_PREFIX.get(style, "")
    return f"{prefix}{text}"


def _table_to_text(table: dict[str, Any]) -> str:
    """Convert a Google Docs table element to a plain-text block.

    Each table row is rendered as a pipe-separated line.

    Args:
        table: A ``table`` structural element from the Docs API.

    Returns:
        Multi-line plain-text string.
    """
    rows: list[str] = []
    for row in table.get("tableRows", []):
        cells: list[str] = []
        for cell in row.get("tableCells", []):
            cell_text = doc_content_to_text(cell.get("content", []))
            cells.append(cell_text.replace("\n", " ").strip())
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join(rows)


def doc_content_to_text(content: list[dict[str, Any]]) -> str:
    """Convert a Google Docs ``body.content`` list to plain text.

    Handles paragraphs and tables recursively.  Other element types
    (inline objects, section breaks, etc.) are silently skipped.

    Args:
        content: List of structural elements from the Docs API response.

    Returns:
        Multi-line plain-text string.
    """
    lines: list[str] = []
    for element in content:
        if "paragraph" in element:
            line = _paragraph_to_text(element["paragraph"])
            if line:
                lines.append(line)
        elif "table" in element:
            table_text = _table_to_text(element["table"])
            if table_text:
                lines.append(table_text)
    return "\n".join(lines)


def _stable_source_id(file_id: str) -> str:
    return f"gdocs:{file_id}"


# ---------------------------------------------------------------------------
# Connector
# ---------------------------------------------------------------------------


class GoogleDocsConnector(BaseSourceConnector):
    """Google Docs connector — pulls Docs files via Drive + Docs APIs.

    The cursor is the ``modifiedTime`` ISO timestamp of the most recently
    modified file seen.  On each ``list_items`` call a short-lived OAuth 2.0
    access token is minted from the service account credentials.

    Args:
        service_account_path: Path to the service account JSON key file.
            Falls back to ``RAG_GOOGLE_SERVICE_ACCOUNT_JSON``.
        folder_ids: List of Drive folder IDs to restrict the search.
            Falls back to ``RAG_GOOGLE_DRIVE_FOLDER_IDS`` (comma-separated).
            Empty = all files accessible to the service account.
        page_size: Number of files per Drive API list call.
        fetch_content: If True, fetch the full Docs document body.
            Set to False for metadata-only syncs.
        _access_token: Optional pre-minted token (for testing without credentials).
    """

    connector_name = "google_docs"

    def __init__(
        self,
        service_account_path: Optional[str] = None,
        folder_ids: Optional[list[str]] = None,
        page_size: int = _DEFAULT_PAGE_SIZE,
        fetch_content: bool = True,
        _access_token: Optional[str] = None,
    ) -> None:
        sa_path = service_account_path or os.environ.get(
            "RAG_GOOGLE_SERVICE_ACCOUNT_JSON", ""
        )
        self._sa_path = sa_path
        self._sa_data: Optional[dict[str, Any]] = None
        if sa_path and os.path.isfile(sa_path):
            try:
                with open(sa_path, encoding="utf-8") as f:
                    self._sa_data = json.load(f)
            except Exception as exc:
                logger.warning("Could not load service account JSON: %s", exc)

        raw_ids = os.environ.get("RAG_GOOGLE_DRIVE_FOLDER_IDS", "")
        self._folder_ids: list[str] = folder_ids or (
            [fid.strip() for fid in raw_ids.split(",") if fid.strip()] if raw_ids else []
        )
        self._page_size = page_size
        self._fetch_content = fetch_content
        self._forced_token = _access_token  # injected in tests
        self._cursor: str = ""

    # ------------------------------------------------------------------
    # BaseSourceConnector implementation
    # ------------------------------------------------------------------

    def list_items(self, since_cursor: str = "") -> list[SourceArtifact]:
        """Fetch Google Docs files modified after ``since_cursor``.

        Args:
            since_cursor: ISO 8601 ``modifiedTime`` timestamp.
                Empty = fetch all accessible Docs files.

        Returns:
            List of ``SourceArtifact`` objects, one per document.
        """
        token = self._get_access_token()
        files = self._list_drive_files(token, since_cursor)

        artifacts: list[SourceArtifact] = []
        max_ts = since_cursor

        for file_meta in files:
            try:
                artifact = self._file_to_artifact(token, file_meta)
                if artifact:
                    artifacts.append(artifact)
                    ts = file_meta.get("modifiedTime", "")
                    if ts > max_ts:
                        max_ts = ts
            except Exception as exc:
                logger.warning(
                    "Failed to convert file %s: %s", file_meta.get("id"), exc
                )

        self._cursor = max_ts
        return artifacts

    def next_cursor(self) -> str:
        """Return the ``modifiedTime`` of the most recently modified file seen."""
        return self._cursor

    def healthcheck(self) -> dict:
        """Verify credentials and Drive API access."""
        if not self._sa_path and not self._forced_token:
            return {
                "status": "error",
                "connector": self.connector_name,
                "detail": "RAG_GOOGLE_SERVICE_ACCOUNT_JSON is not configured.",
            }
        try:
            token = self._get_access_token()
            # List one file to verify Drive access
            self._drive_get(token, "/files", {"pageSize": "1", "fields": "files(id)"})
            return {"status": "ok", "connector": self.connector_name, "detail": ""}
        except Exception as exc:
            return {
                "status": "error",
                "connector": self.connector_name,
                "detail": str(exc),
            }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_access_token(self) -> str:
        """Return an access token: injected token → minted from SA credentials."""
        if self._forced_token:
            return self._forced_token
        if not self._sa_data:
            raise RuntimeError(
                "No Google service account credentials available. "
                "Set RAG_GOOGLE_SERVICE_ACCOUNT_JSON."
            )
        jwt = _mint_jwt(self._sa_data)
        return _exchange_jwt_for_token(jwt)

    def _drive_get(
        self, token: str, path: str, params: dict[str, str]
    ) -> dict[str, Any]:
        url = f"{_DRIVE_API}{path}?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token}"}
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Drive API {exc.code} for {path}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Drive API request failed: {exc}") from exc

    def _docs_get(self, token: str, document_id: str) -> dict[str, Any]:
        url = f"{_DOCS_API}/documents/{document_id}"
        req = urllib.request.Request(
            url, headers={"Authorization": f"Bearer {token}"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Docs API {exc.code} for {document_id}: {body}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(f"Docs API request failed: {exc}") from exc

    def _list_drive_files(
        self, token: str, since_cursor: str
    ) -> list[dict[str, Any]]:
        """List Google Docs files from Drive matching the cursor and folders."""
        query_parts = ["mimeType='application/vnd.google-apps.document'"]
        if since_cursor:
            query_parts.append(f"modifiedTime > '{since_cursor}'")
        if self._folder_ids:
            folder_clauses = " or ".join(
                f"'{fid}' in parents" for fid in self._folder_ids
            )
            query_parts.append(f"({folder_clauses})")

        params = {
            "q": " and ".join(query_parts),
            "fields": "files(id,name,modifiedTime,createdTime,webViewLink,owners)",
            "pageSize": str(self._page_size),
            "orderBy": "modifiedTime asc",
        }
        data = self._drive_get(token, "/files", params)
        return data.get("files", [])

    def _file_to_artifact(
        self, token: str, file_meta: dict[str, Any]
    ) -> Optional[SourceArtifact]:
        """Convert a Drive file metadata dict + Docs content to an artifact."""
        file_id = file_meta.get("id", "")
        if not file_id:
            return None

        name = file_meta.get("name", "(Untitled)")
        modified = file_meta.get("modifiedTime", "")
        created = file_meta.get("createdTime", "")
        web_link = file_meta.get("webViewLink", "")
        owners = [
            o.get("emailAddress", "") for o in file_meta.get("owners", [])
        ]

        content_text = name
        if self._fetch_content:
            try:
                doc = self._docs_get(token, file_id)
                body_content = doc.get("body", {}).get("content", [])
                body_text = doc_content_to_text(body_content)
                if body_text:
                    content_text = f"{name}\n\n{body_text}"
            except Exception as exc:
                logger.warning("Failed to fetch doc content for %s: %s", file_id, exc)

        return SourceArtifact(
            source_type="google_docs",
            source_id=_stable_source_id(file_id),
            external_url=web_link,
            content_text=content_text,
            mime_type="text/plain",
            metadata={
                "file_id": file_id,
                "name": name,
                "modified_time": modified,
                "created_time": created,
                "owners": owners,
                "web_view_link": web_link,
            },
            cursor_after=modified,
        )
