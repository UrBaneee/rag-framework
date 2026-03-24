"""PaddleOCR provider implementation — Task 13.1.

Wraps the ``paddleocr`` library with a graceful ImportError when not
installed, and converts raw PaddleOCR output into ``IRBlock`` objects.

Install PaddleOCR::

    pip install paddlepaddle paddleocr

Usage::

    from rag.infra.ocr.paddleocr_provider import PaddleOCRProvider
    provider = PaddleOCRProvider(lang="en")
    blocks = provider.ocr(pil_image)
"""

from __future__ import annotations

import logging
from typing import Any

from rag.core.contracts.ir_block import BoundingBox, BlockType, IRBlock
from rag.core.interfaces.ocr_provider import BaseOCRProvider

logger = logging.getLogger(__name__)

_PADDLE_AVAILABLE = False
_PaddleOCR: Any = None

try:
    from paddleocr import PaddleOCR as _PaddleOCR  # type: ignore[no-redef]
    _PADDLE_AVAILABLE = True
except ImportError:
    pass


class PaddleOCRProvider(BaseOCRProvider):
    """OCR provider backed by PaddleOCR.

    Args:
        lang: Language code for PaddleOCR (e.g. ``"en"``, ``"ch"``).
            Defaults to ``"en"``.
        use_angle_cls: Enable text angle classification.  Defaults to True.
        use_gpu: Use GPU acceleration if available.  Defaults to False.

    Raises:
        ImportError: At instantiation time if ``paddleocr`` is not installed.

    Example::

        provider = PaddleOCRProvider(lang="en")
        blocks = provider.ocr(pil_image)
    """

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = True,
        use_gpu: bool = False,
    ) -> None:
        if not _PADDLE_AVAILABLE:
            raise ImportError(
                "PaddleOCR is not installed. Install it with:\n"
                "  pip install paddlepaddle paddleocr\n"
                "For GPU support, install paddlepaddle-gpu instead of paddlepaddle."
            )
        self._ocr_engine = _PaddleOCR(
            use_angle_cls=use_angle_cls,
            lang=lang,
            use_gpu=use_gpu,
            show_log=False,
        )
        self._lang = lang

    def ocr(self, image: object) -> list[IRBlock]:
        """Run OCR on a PIL Image and return IRBlocks.

        Args:
            image: PIL ``Image`` in RGB mode.

        Returns:
            List of ``IRBlock`` objects with text, bbox, and confidence.

        Raises:
            RuntimeError: If PaddleOCR processing fails.
        """
        import numpy as np  # type: ignore[import]

        try:
            # PaddleOCR accepts numpy arrays
            img_array = np.array(image)
            result = self._ocr_engine.ocr(img_array, cls=True)
        except Exception as exc:
            raise RuntimeError(f"PaddleOCR processing failed: {exc}") from exc

        blocks: list[IRBlock] = []
        if not result or not result[0]:
            return blocks

        for line in result[0]:
            # PaddleOCR format: [[[x0,y0],[x1,y1],[x2,y2],[x3,y3]], (text, confidence)]
            coords, (text, confidence) = line
            if not text.strip():
                continue

            # Convert quad bbox to axis-aligned bbox
            xs = [pt[0] for pt in coords]
            ys = [pt[1] for pt in coords]
            bbox = BoundingBox(
                x0=float(min(xs)),
                y0=float(min(ys)),
                x1=float(max(xs)),
                y1=float(max(ys)),
            )

            blocks.append(
                IRBlock(
                    block_type=BlockType.PARAGRAPH,
                    text=text.strip(),
                    bbox=bbox,
                    confidence=float(confidence),
                )
            )

        logger.debug(
            "PaddleOCR extracted %d blocks (lang=%s)", len(blocks), self._lang
        )
        return blocks
