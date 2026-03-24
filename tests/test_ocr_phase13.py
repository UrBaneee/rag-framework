"""Tests for Phase 13 — OCR Support (Tasks 13.1–13.4).

Uses mocking throughout since PaddleOCR and real scanned PDFs are not
available in this environment.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from rag.core.contracts.ir_block import BoundingBox, BlockType, IRBlock
from rag.core.interfaces.ocr_provider import BaseOCRProvider
from rag.core.interfaces.page_renderer import BasePageRenderer


# ---------------------------------------------------------------------------
# Task 13.1 — OCR provider interface
# ---------------------------------------------------------------------------


class TestBaseOCRProvider:
    def test_is_abstract(self):
        import inspect
        assert inspect.isabstract(BaseOCRProvider)

    def test_ocr_method_is_abstract(self):
        assert "ocr" in BaseOCRProvider.__abstractmethods__

    def test_concrete_subclass_must_implement_ocr(self):
        class GoodOCR(BaseOCRProvider):
            def ocr(self, image):
                return []

        provider = GoodOCR()
        assert provider.ocr(None) == []

    def test_incomplete_subclass_raises(self):
        class BadOCR(BaseOCRProvider):
            pass

        with pytest.raises(TypeError):
            BadOCR()


class TestPaddleOCRProviderImportError:
    """PaddleOCRProvider raises ImportError when paddleocr is not installed."""

    def test_import_error_when_paddleocr_missing(self):
        import sys
        # Temporarily hide paddleocr from sys.modules
        with patch.dict(sys.modules, {"paddleocr": None}):
            # Re-import with paddleocr unavailable
            import importlib
            import rag.infra.ocr.paddleocr_provider as mod
            orig = mod._PADDLE_AVAILABLE
            mod._PADDLE_AVAILABLE = False
            try:
                from rag.infra.ocr.paddleocr_provider import PaddleOCRProvider
                with pytest.raises(ImportError) as exc_info:
                    PaddleOCRProvider()
                assert "paddleocr" in str(exc_info.value).lower() or "paddle" in str(exc_info.value).lower()
            finally:
                mod._PADDLE_AVAILABLE = orig

    def test_import_error_message_contains_install_hint(self):
        import rag.infra.ocr.paddleocr_provider as mod
        orig = mod._PADDLE_AVAILABLE
        mod._PADDLE_AVAILABLE = False
        try:
            from rag.infra.ocr.paddleocr_provider import PaddleOCRProvider
            with pytest.raises(ImportError) as exc_info:
                PaddleOCRProvider()
            assert "pip install" in str(exc_info.value)
        finally:
            mod._PADDLE_AVAILABLE = orig


class TestPaddleOCRProviderMocked:
    """PaddleOCRProvider with mocked engine."""

    def _make_provider(self, ocr_result=None):
        """Build a PaddleOCRProvider with mocked internals."""
        import rag.infra.ocr.paddleocr_provider as mod
        orig = mod._PADDLE_AVAILABLE
        orig_cls = mod._PaddleOCR
        mod._PADDLE_AVAILABLE = True

        mock_engine = MagicMock()
        if ocr_result is None:
            # Default: one line with text and confidence
            mock_engine.ocr.return_value = [
                [
                    [[[10, 20], [100, 20], [100, 40], [10, 40]], ("Hello World", 0.95)],
                ]
            ]
        else:
            mock_engine.ocr.return_value = ocr_result

        mock_cls = MagicMock(return_value=mock_engine)
        mod._PaddleOCR = mock_cls

        try:
            from rag.infra.ocr.paddleocr_provider import PaddleOCRProvider
            provider = PaddleOCRProvider.__new__(PaddleOCRProvider)
            provider._ocr_engine = mock_engine
            provider._lang = "en"
            return provider, mock_engine
        finally:
            mod._PADDLE_AVAILABLE = orig
            mod._PaddleOCR = orig_cls

    def test_ocr_returns_ir_blocks(self):
        provider, _ = self._make_provider()
        mock_image = MagicMock()
        with patch("numpy.array", return_value=MagicMock()):
            blocks = provider.ocr(mock_image)
        assert len(blocks) == 1
        assert isinstance(blocks[0], IRBlock)

    def test_ocr_text_extracted(self):
        provider, _ = self._make_provider()
        mock_image = MagicMock()
        with patch("numpy.array", return_value=MagicMock()):
            blocks = provider.ocr(mock_image)
        assert blocks[0].text == "Hello World"

    def test_ocr_confidence_set(self):
        provider, _ = self._make_provider()
        mock_image = MagicMock()
        with patch("numpy.array", return_value=MagicMock()):
            blocks = provider.ocr(mock_image)
        assert abs(blocks[0].confidence - 0.95) < 1e-6

    def test_ocr_bbox_set(self):
        provider, _ = self._make_provider()
        mock_image = MagicMock()
        with patch("numpy.array", return_value=MagicMock()):
            blocks = provider.ocr(mock_image)
        assert blocks[0].bbox is not None
        assert blocks[0].bbox.x0 == 10.0
        assert blocks[0].bbox.y0 == 20.0

    def test_ocr_empty_result(self):
        provider, _ = self._make_provider(ocr_result=[[]])
        mock_image = MagicMock()
        with patch("numpy.array", return_value=MagicMock()):
            blocks = provider.ocr(mock_image)
        assert blocks == []

    def test_ocr_none_result(self):
        provider, _ = self._make_provider(ocr_result=[None])
        mock_image = MagicMock()
        with patch("numpy.array", return_value=MagicMock()):
            blocks = provider.ocr(mock_image)
        assert blocks == []


# ---------------------------------------------------------------------------
# Task 13.2 — Page renderer interface
# ---------------------------------------------------------------------------


class TestBasePageRenderer:
    def test_is_abstract(self):
        import inspect
        assert inspect.isabstract(BasePageRenderer)

    def test_required_abstract_methods(self):
        assert "render" in BasePageRenderer.__abstractmethods__
        assert "render_range" in BasePageRenderer.__abstractmethods__
        assert "page_count" in BasePageRenderer.__abstractmethods__


class TestPyMuPDFPageRendererImportError:
    def test_import_error_when_fitz_missing(self):
        import sys
        with patch.dict(sys.modules, {"fitz": None}):
            # Force re-evaluation of the import
            from rag.infra.ocr.renderer_pymupdf import PyMuPDFPageRenderer
            with pytest.raises(ImportError) as exc_info:
                PyMuPDFPageRenderer()
            assert "pymupdf" in str(exc_info.value).lower() or "fitz" in str(exc_info.value).lower()


class TestPyMuPDFPageRendererMocked:
    """PyMuPDFPageRenderer with mocked fitz."""

    def _make_renderer(self):
        mock_pix = MagicMock()
        mock_pix.width = 800
        mock_pix.height = 1000
        mock_pix.samples = b"\xff" * (800 * 1000 * 3)

        mock_page = MagicMock()
        mock_page.get_pixmap.return_value = mock_pix
        mock_page.get_transformation_matrix.return_value = MagicMock()

        mock_doc = MagicMock()
        mock_doc.__len__.return_value = 3
        mock_doc.__getitem__.return_value = mock_page

        mock_fitz = MagicMock()
        mock_fitz.open.return_value = mock_doc

        return mock_fitz, mock_doc, mock_page, mock_pix

    def test_page_count(self, tmp_path):
        mock_fitz, mock_doc, _, _ = self._make_renderer()
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            from rag.infra.ocr.renderer_pymupdf import PyMuPDFPageRenderer
            renderer = PyMuPDFPageRenderer.__new__(PyMuPDFPageRenderer)
            renderer._dpi = 150
            renderer._colorspace = "RGB"
            mock_fitz.open.return_value = mock_doc
            count = renderer.page_count(str(pdf))
        assert count == 3

    def test_render_returns_pil_image(self, tmp_path):
        mock_fitz, mock_doc, mock_page, mock_pix = self._make_renderer()
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        mock_image = MagicMock()
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            with patch("PIL.Image.frombytes", return_value=mock_image):
                from rag.infra.ocr.renderer_pymupdf import PyMuPDFPageRenderer
                renderer = PyMuPDFPageRenderer.__new__(PyMuPDFPageRenderer)
                renderer._dpi = 150
                renderer._colorspace = "RGB"
                result = renderer.render(str(pdf), page_num=1)
        assert result is mock_image

    def test_render_range_returns_list(self, tmp_path):
        mock_fitz, mock_doc, mock_page, mock_pix = self._make_renderer()
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4")

        mock_image = MagicMock()
        with patch.dict("sys.modules", {"fitz": mock_fitz}):
            with patch("PIL.Image.frombytes", return_value=mock_image):
                from rag.infra.ocr.renderer_pymupdf import PyMuPDFPageRenderer
                renderer = PyMuPDFPageRenderer.__new__(PyMuPDFPageRenderer)
                renderer._dpi = 150
                renderer._colorspace = "RGB"
                results = renderer.render_range(str(pdf), start=1, end=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Task 13.3 — PdfOCRParser
# ---------------------------------------------------------------------------


class TestPdfOCRParser:
    def _make_parser(self, pages=2, text_per_page="Sample OCR text here."):
        from rag.infra.parsing.pdf_ocr_parser import PdfOCRParser

        mock_renderer = MagicMock(spec=BasePageRenderer)
        mock_renderer.page_count.return_value = pages
        mock_renderer.render.return_value = MagicMock()  # fake image

        mock_ocr = MagicMock(spec=BaseOCRProvider)
        mock_ocr.ocr.return_value = [
            IRBlock(
                block_type=BlockType.PARAGRAPH,
                text=text_per_page,
                bbox=BoundingBox(x0=0, y0=0, x1=100, y1=20),
                confidence=0.92,
            )
        ]

        parser = PdfOCRParser(renderer=mock_renderer, ocr_provider=mock_ocr)
        return parser, mock_renderer, mock_ocr

    def test_returns_document(self, tmp_path):
        from rag.core.contracts.document import Document
        parser, _, _ = self._make_parser()
        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(b"%PDF")
        doc = parser.parse(str(pdf))
        assert isinstance(doc, Document)

    def test_parse_report_parser_used_pdf_ocr(self, tmp_path):
        parser, _, _ = self._make_parser()
        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(b"%PDF")
        doc = parser.parse(str(pdf))
        assert doc.parse_report is not None
        assert doc.parse_report.parser_used == "pdf_ocr"

    def test_blocks_include_page_number(self, tmp_path):
        parser, _, _ = self._make_parser(pages=2)
        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(b"%PDF")
        doc = parser.parse(str(pdf))
        pages = {b.page for b in doc.blocks}
        assert 1 in pages
        assert 2 in pages

    def test_blocks_include_bounding_box(self, tmp_path):
        parser, _, _ = self._make_parser()
        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(b"%PDF")
        doc = parser.parse(str(pdf))
        assert doc.blocks[0].bbox is not None

    def test_mean_confidence_in_metadata(self, tmp_path):
        parser, _, _ = self._make_parser()
        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(b"%PDF")
        doc = parser.parse(str(pdf))
        assert "mean_ocr_confidence" in doc.metadata
        assert abs(doc.metadata["mean_ocr_confidence"] - 0.92) < 0.01

    def test_supports_pdf_mime(self):
        parser, _, _ = self._make_parser()
        assert parser.supports("application/pdf")
        assert not parser.supports("text/html")

    def test_max_pages_limits_processing(self, tmp_path):
        from rag.infra.parsing.pdf_ocr_parser import PdfOCRParser
        mock_renderer = MagicMock(spec=BasePageRenderer)
        mock_renderer.page_count.return_value = 10
        mock_renderer.render.return_value = MagicMock()
        mock_ocr = MagicMock(spec=BaseOCRProvider)
        mock_ocr.ocr.return_value = [IRBlock(text="text", block_type=BlockType.PARAGRAPH)]
        parser = PdfOCRParser(mock_renderer, mock_ocr, max_pages=3)
        pdf = tmp_path / "scan.pdf"
        pdf.write_bytes(b"%PDF")
        doc = parser.parse(str(pdf))
        assert mock_renderer.render.call_count == 3


# ---------------------------------------------------------------------------
# Task 13.4 — OCR router integration
# ---------------------------------------------------------------------------


class TestOCRRouterIntegration:
    def test_ocr_disabled_skips_pdf_ocr_candidate(self, tmp_path):
        """When ocr.enabled=false, pdf_ocr is skipped even if registered."""
        from rag.infra.parsing.pdf_ocr_parser import PdfOCRParser
        from rag.pipelines.parsing.orchestrator import ParserOrchestrator

        # Create a config with OCR disabled
        cfg = tmp_path / "parser_candidates.yaml"
        cfg.write_text("""
ocr:
  enabled: false
  min_chars_threshold: 100
routes:
  pdf:
    candidates: [pymupdf, pdf_ocr]
""")
        mock_pymupdf = MagicMock()
        from rag.core.contracts.document import Document
        from rag.core.contracts.parse_report import ParseReport
        mock_pymupdf.parse.return_value = Document(
            doc_id="d1", source_path="/tmp/f.pdf", mime_type="application/pdf",
            blocks=[IRBlock(text="Lots of text " * 20, block_type=BlockType.PARAGRAPH)],
            parse_report=ParseReport(char_count=300, block_count=1,
                                     non_printable_ratio=0.0, repetition_score=0.0,
                                     parser_used="pymupdf"),
        )
        mock_pdf_ocr = MagicMock()

        registry = {"pymupdf": mock_pymupdf, "pdf_ocr": mock_pdf_ocr}
        orch = ParserOrchestrator(registry, router_config_path=cfg)
        assert orch._ocr_enabled is False

    def test_low_char_count_triggers_ocr_fallback(self, tmp_path):
        """Primary parser with few chars triggers OCR fallback."""
        from rag.pipelines.parsing.orchestrator import ParserOrchestrator
        from rag.pipelines.parsing.plans import ParsePlan
        from rag.core.contracts.document import Document
        from rag.core.contracts.parse_report import ParseReport
        from rag.infra.loading.local_file_loader import RawArtifact

        cfg = tmp_path / "parser_candidates.yaml"
        cfg.write_text("""
ocr:
  enabled: true
  min_chars_threshold: 100
routes:
  pdf:
    candidates: [pymupdf, pdf_ocr]
""")
        # pymupdf returns only 10 chars (scanned PDF)
        mock_pymupdf = MagicMock()
        mock_pymupdf.parse.return_value = Document(
            doc_id="d1", source_path="/tmp/f.pdf", mime_type="application/pdf",
            blocks=[IRBlock(text="Few chars", block_type=BlockType.PARAGRAPH)],
            parse_report=ParseReport(char_count=9, block_count=1,
                                     non_printable_ratio=0.0, repetition_score=0.0,
                                     parser_used="pymupdf"),
        )
        ocr_doc = Document(
            doc_id="d2", source_path="/tmp/f.pdf", mime_type="application/pdf",
            blocks=[IRBlock(text="OCR extracted text " * 10, block_type=BlockType.PARAGRAPH)],
            parse_report=ParseReport(char_count=200, block_count=1,
                                     non_printable_ratio=0.0, repetition_score=0.0,
                                     parser_used="pdf_ocr"),
        )
        mock_pdf_ocr = MagicMock()
        mock_pdf_ocr.parse.return_value = ocr_doc

        registry = {"pymupdf": mock_pymupdf, "pdf_ocr": mock_pdf_ocr}
        orch = ParserOrchestrator(registry, router_config_path=cfg)

        artifact = RawArtifact(
            source_path="/tmp/f.pdf",
            raw_bytes=b"%PDF",
            metadata={},
        )
        plan = ParsePlan(
            detected_type="pdf",
            mime_type="application/pdf",
            candidates=["pymupdf", "pdf_ocr"],
            unsupported=False,
        )
        doc = orch.parse(artifact, plan)
        assert doc.parse_report.parser_used == "pdf_ocr"
        assert doc.parse_report.fallback_triggered is True

    def test_sufficient_chars_does_not_trigger_ocr(self, tmp_path):
        """Text PDF with enough chars should NOT trigger OCR."""
        from rag.pipelines.parsing.orchestrator import ParserOrchestrator
        from rag.pipelines.parsing.plans import ParsePlan
        from rag.core.contracts.document import Document
        from rag.core.contracts.parse_report import ParseReport
        from rag.infra.loading.local_file_loader import RawArtifact

        cfg = tmp_path / "parser_candidates.yaml"
        cfg.write_text("""
ocr:
  enabled: true
  min_chars_threshold: 100
routes:
  pdf:
    candidates: [pymupdf, pdf_ocr]
""")
        text_doc = Document(
            doc_id="d1", source_path="/tmp/f.pdf", mime_type="application/pdf",
            blocks=[IRBlock(text="Lots of text here! " * 20, block_type=BlockType.PARAGRAPH)],
            parse_report=ParseReport(char_count=380, block_count=1,
                                     non_printable_ratio=0.0, repetition_score=0.0,
                                     parser_used="pymupdf"),
        )
        mock_pymupdf = MagicMock()
        mock_pymupdf.parse.return_value = text_doc
        mock_pdf_ocr = MagicMock()

        registry = {"pymupdf": mock_pymupdf, "pdf_ocr": mock_pdf_ocr}
        orch = ParserOrchestrator(registry, router_config_path=cfg)

        artifact = RawArtifact(source_path="/tmp/f.pdf", raw_bytes=b"%PDF", metadata={})
        plan = ParsePlan(detected_type="pdf", mime_type="application/pdf",
                         candidates=["pymupdf", "pdf_ocr"], unsupported=False)
        doc = orch.parse(artifact, plan)
        assert doc.parse_report.parser_used == "pymupdf"
        mock_pdf_ocr.parse.assert_not_called()
