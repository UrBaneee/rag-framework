"""Unit tests for the sniffer components."""

import pytest

from rag.infra.loading.local_file_loader import LocalFileLoader, RawArtifact
from rag.infra.sniffing.composite_sniffer import CompositeSniffer
from rag.infra.sniffing.sniffer_magic import MagicSniffer
from rag.infra.sniffing.sniffer_mime import MimeSniffer


def make_artifact(raw_bytes: bytes, extension: str = ".bin") -> RawArtifact:
    """Helper to construct a minimal RawArtifact for testing."""
    return RawArtifact(
        source_path=f"/tmp/test{extension}",
        raw_bytes=raw_bytes,
        metadata={"extension": extension, "file_name": f"test{extension}"},
    )


# ── MagicSniffer ──────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestMagicSniffer:
    def setup_method(self):
        self.sniffer = MagicSniffer()

    def test_detects_pdf(self):
        artifact = make_artifact(b"%PDF-1.4 ...", ".pdf")
        result = self.sniffer.sniff(artifact)
        assert result.detected_type == "pdf"
        assert result.mime_type == "application/pdf"
        assert result.confidence == 1.0

    def test_detects_html_doctype(self):
        artifact = make_artifact(b"<!doctype html><html><body></body></html>", ".html")
        result = self.sniffer.sniff(artifact)
        assert result.detected_type == "html"

    def test_detects_html_tag(self):
        artifact = make_artifact(b"<html><head></head></html>", ".html")
        result = self.sniffer.sniff(artifact)
        assert result.detected_type == "html"

    def test_detects_png(self):
        artifact = make_artifact(b"\x89PNG\r\n\x1a\nDATA", ".png")
        result = self.sniffer.sniff(artifact)
        assert result.detected_type == "unsupported"
        assert result.mime_type == "image/png"

    def test_detects_zip_container(self):
        artifact = make_artifact(b"PK\x03\x04" + b"\x00" * 100, ".docx")
        result = self.sniffer.sniff(artifact)
        assert result.detected_type == "zip_container"

    def test_returns_none_for_unknown(self):
        artifact = make_artifact(b"random binary content xyz", ".bin")
        result = self.sniffer.sniff(artifact)
        assert result.detected_type is None
        assert result.confidence == 0.0


# ── MimeSniffer ───────────────────────────────────────────────────────────────

@pytest.mark.unit
class TestMimeSniffer:
    def setup_method(self):
        self.sniffer = MimeSniffer()

    def test_detects_txt(self):
        artifact = make_artifact(b"hello", ".txt")
        result = self.sniffer.sniff(artifact)
        assert result.detected_type == "txt"
        assert result.mime_type == "text/plain"

    def test_detects_markdown(self):
        artifact = make_artifact(b"# Title", ".md")
        result = self.sniffer.sniff(artifact)
        assert result.detected_type == "markdown"

    def test_detects_markdown_long_ext(self):
        artifact = make_artifact(b"# Title", ".markdown")
        result = self.sniffer.sniff(artifact)
        assert result.detected_type == "markdown"

    def test_detects_html(self):
        artifact = make_artifact(b"<html/>", ".html")
        result = self.sniffer.sniff(artifact)
        assert result.detected_type == "html"

    def test_detects_docx(self):
        artifact = make_artifact(b"PK...", ".docx")
        result = self.sniffer.sniff(artifact)
        assert result.detected_type == "docx"

    def test_unknown_extension_returns_none(self):
        artifact = make_artifact(b"data", ".xyz")
        result = self.sniffer.sniff(artifact)
        assert result.detected_type is None


# ── CompositeSniffer ──────────────────────────────────────────────────────────

@pytest.mark.unit
class TestCompositeSniffer:
    def setup_method(self):
        self.sniffer = CompositeSniffer()

    def test_detects_pdf(self):
        artifact = make_artifact(b"%PDF-1.7 content", ".pdf")
        result = self.sniffer.sniff(artifact)
        assert result.detected_type == "pdf"
        assert result.strategy == "magic"

    def test_detects_html_by_magic(self):
        artifact = make_artifact(b"<!DOCTYPE html><html></html>", ".html")
        result = self.sniffer.sniff(artifact)
        assert result.detected_type == "html"
        assert result.strategy == "magic"

    def test_detects_txt_by_mime(self):
        artifact = make_artifact(b"Plain text content.", ".txt")
        result = self.sniffer.sniff(artifact)
        assert result.detected_type == "txt"
        assert result.strategy == "mime"

    def test_detects_markdown_by_mime(self):
        artifact = make_artifact(b"# Heading\n\nBody text.", ".md")
        result = self.sniffer.sniff(artifact)
        assert result.detected_type == "markdown"
        assert result.strategy == "mime"

    def test_docx_specialised_from_zip(self, tmp_path):
        artifact = make_artifact(b"PK\x03\x04" + b"\x00" * 50, ".docx")
        result = self.sniffer.sniff(artifact)
        assert result.detected_type == "docx"
        assert result.strategy == "zip_specialised"

    def test_unsupported_explicit_for_png(self):
        artifact = make_artifact(b"\x89PNG\r\n\x1a\nDATA", ".png")
        result = self.sniffer.sniff(artifact)
        assert result.detected_type == "unsupported"

    def test_unknown_file_returns_unknown(self):
        artifact = make_artifact(b"totally random xyz 1234", ".xyz")
        result = self.sniffer.sniff(artifact)
        assert result.detected_type == "unknown"
        assert result.strategy == "unknown"

    def test_result_detected_type_is_never_none(self):
        artifact = make_artifact(b"\x00\x00\x00\x00", ".bin")
        result = self.sniffer.sniff(artifact)
        assert result.detected_type is not None


# ── Integration: load real file then sniff ────────────────────────────────────

@pytest.mark.integration
def test_load_and_sniff_txt_file(tmp_path):
    txt_file = tmp_path / "doc.txt"
    txt_file.write_text("This is plain text.", encoding="utf-8")
    artifact = LocalFileLoader().load(txt_file)
    result = CompositeSniffer().sniff(artifact)
    assert result.detected_type == "txt"
