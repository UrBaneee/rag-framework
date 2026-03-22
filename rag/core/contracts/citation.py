"""Citation contracts — Span and Citation for grounded answer generation."""

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SpanType(str, Enum):
    """Semantic type of a text span within an answer."""

    ANSWER = "answer"           # Regular answer text
    CITATION_MARKER = "citation_marker"  # Inline reference, e.g. "[1]"
    ABSTAIN = "abstain"         # Abstention notice when evidence is insufficient
    PREAMBLE = "preamble"       # Introductory text before the answer body


class Span(BaseModel):
    """A labelled region of text within an Answer.

    Answers are segmented into Spans so that downstream consumers (UI,
    evaluation) can distinguish answer body text from citation markers
    and abstention notices without parsing the raw string.

    Attributes:
        text: The span text content.
        span_type: Semantic role of this span.
        start: Character offset in the full answer string (inclusive).
        end: Character offset in the full answer string (exclusive).
    """

    text: str
    span_type: SpanType = SpanType.ANSWER
    start: int = Field(default=0, ge=0)
    end: int = Field(default=0, ge=0)


class Citation(BaseModel):
    """A source reference linking an inline marker to a retrieved chunk.

    The citation format follows:
        Inline:  "...container orchestration.[1]"
        Sources: "[1] architecture.pdf — page 12"

    Attributes:
        ref_number: 1-based reference index used in inline markers.
        chunk_id: Identifier of the supporting Chunk.
        doc_id: Parent document identifier.
        source_label: Human-readable source description, e.g.
            "architecture.pdf — page 12".
        page: Page number within the source document, if applicable.
        display_text: Short excerpt from the chunk for context display.
    """

    ref_number: int = Field(ge=1)
    chunk_id: str
    doc_id: str
    source_label: str
    page: Optional[int] = None
    display_text: str = ""
