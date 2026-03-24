"""Abstract base class for page renderer plugins — Task 13.2."""

from __future__ import annotations

from abc import ABC, abstractmethod


class BasePageRenderer(ABC):
    """Interface that all page renderer plugins must implement.

    A page renderer converts a single PDF page into a PIL ``Image`` at a
    configurable DPI, suitable for OCR processing.
    """

    @abstractmethod
    def render(self, pdf_path: str, page_num: int) -> object:
        """Render a single PDF page to a PIL Image.

        Args:
            pdf_path: Absolute path to the PDF file.
            page_num: 1-based page number to render.

        Returns:
            A PIL ``Image`` object in RGB mode.

        Raises:
            FileNotFoundError: If ``pdf_path`` does not exist.
            IndexError: If ``page_num`` is out of range for the PDF.
            ImportError: If the underlying rendering library is not installed.
        """

    @abstractmethod
    def render_range(self, pdf_path: str, start: int, end: int) -> list[object]:
        """Render a range of pages to PIL Images.

        Args:
            pdf_path: Absolute path to the PDF file.
            start: 1-based first page number (inclusive).
            end: 1-based last page number (inclusive).

        Returns:
            List of PIL ``Image`` objects, one per page.
        """

    @abstractmethod
    def page_count(self, pdf_path: str) -> int:
        """Return the number of pages in a PDF.

        Args:
            pdf_path: Absolute path to the PDF file.

        Returns:
            Total page count.
        """
