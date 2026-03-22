"""ParsePlan — the routing decision produced by the parser router."""

from dataclasses import dataclass, field


@dataclass
class ParsePlan:
    """Routing plan for a single document.

    Produced by ``ParserOrchestrator.route()`` after the sniffer has
    identified the document type. Consumed by ``ParserOrchestrator.parse()``
    to try candidate parsers in order.

    Attributes:
        detected_type: Stable type string from CompositeSniffer, e.g.
            "pdf", "html", "txt", "markdown", "unsupported", "unknown".
        mime_type: MIME type string from the sniffer.
        candidates: Ordered list of parser names to try. Empty list means
            the format is explicitly unsupported.
        unsupported: True when detected_type is "unsupported" or "unknown"
            and no candidates are available.
    """

    detected_type: str
    mime_type: str
    candidates: list[str] = field(default_factory=list)
    unsupported: bool = False
