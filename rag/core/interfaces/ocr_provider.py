"""Abstract base class for OCR provider plugins — Task 13.1."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rag.core.contracts.ir_block import IRBlock


class BaseOCRProvider(ABC):
    """Interface that all OCR provider plugins must implement.

    An OCR provider takes a page image (as raw bytes or a PIL Image) and
    returns a list of ``IRBlock`` objects containing the extracted text,
    bounding boxes, and confidence scores.

    Implementations should raise ``ImportError`` with a clear installation
    message if the underlying OCR library is not installed.
    """

    @abstractmethod
    def ocr(self, image: object) -> list["IRBlock"]:
        """Run OCR on a single page image.

        Args:
            image: A PIL ``Image`` object representing one page of a document.

        Returns:
            Ordered list of ``IRBlock`` objects, each containing extracted
            text, bounding box coordinates, and a confidence score in [0, 1].

        Raises:
            ImportError: If the underlying OCR library is not installed.
            RuntimeError: If OCR processing fails.
        """
