"""Answer contract — the final output of the RAG query pipeline."""

from pydantic import BaseModel, Field

from rag.core.contracts.citation import Citation, Span


class Answer(BaseModel):
    """The grounded answer produced by the generation stage.

    Answers are strictly evidence-based: when retrieved context is
    insufficient, the system sets ``abstained=True`` and ``text`` contains
    an abstention message rather than a hallucinated answer.

    Attributes:
        text: Full answer string including inline citation markers, e.g.
            "Cloud deployment requires orchestration.[1]"
        citations: Ordered list of Citation objects (1-based ref_number).
        spans: Optional segmentation of ``text`` into typed Spans for
            structured rendering.
        abstained: True when the system declined to answer due to
            insufficient evidence.
        query: The original user query that produced this answer.
    """

    text: str
    citations: list[Citation] = Field(default_factory=list)
    spans: list[Span] = Field(default_factory=list)
    abstained: bool = False
    query: str = ""
