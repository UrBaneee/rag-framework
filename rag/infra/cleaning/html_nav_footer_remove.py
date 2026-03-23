"""HTML nav/footer cleaner — removes navigation and footer boilerplate blocks."""

from rag.core.contracts.ir_block import IRBlock
from rag.core.interfaces.cleaner import BaseCleaner

# Keywords that strongly indicate a navigation or footer block.
# Matched case-insensitively against the stripped block text.
_NAV_FOOTER_KEYWORDS: tuple[str, ...] = (
    "skip to content",
    "skip to main",
    "back to top",
    "privacy policy",
    "terms of service",
    "terms and conditions",
    "cookie policy",
    "all rights reserved",
    "copyright ©",
    "© ",
    "site map",
    "sitemap",
    "contact us",
    "subscribe to our newsletter",
    "follow us on",
    "share this page",
    "breadcrumb",
)

# If a block contains only short pipe/dash/bullet-separated tokens it is
# likely a navigation bar (e.g. "Home | About | Contact | Login").
_NAV_SEPARATOR_CHARS = frozenset("|•·–—/")
_MAX_NAV_TOKENS = 10
_MAX_NAV_TOKEN_LENGTH = 30


def _is_nav_link_list(text: str) -> bool:
    """Return True when text looks like a navigation link list.

    Heuristic: text is split by separator characters; if all tokens are
    short words and there are between 2 and _MAX_NAV_TOKENS of them,
    the block is likely a nav bar.

    Args:
        text: Stripped block text.

    Returns:
        True if the text looks like a pipe/bullet-separated nav list.
    """
    for sep in _NAV_SEPARATOR_CHARS:
        if sep in text:
            tokens = [t.strip() for t in text.split(sep) if t.strip()]
            if 2 <= len(tokens) <= _MAX_NAV_TOKENS and all(
                len(t) <= _MAX_NAV_TOKEN_LENGTH for t in tokens
            ):
                return True
    return False


def _is_nav_or_footer(text: str) -> bool:
    """Return True if block text matches known nav/footer patterns.

    Args:
        text: Stripped block text to evaluate.

    Returns:
        True if the block should be treated as navigation/footer boilerplate.
    """
    lower = text.lower()
    if any(kw in lower for kw in _NAV_FOOTER_KEYWORDS):
        return True
    if _is_nav_link_list(text):
        return True
    return False


class HtmlNavFooterRemover(BaseCleaner):
    """Cleaner that removes HTML navigation and footer boilerplate blocks.

    Applies two heuristics to each block:
    1. Keyword matching against a list of common nav/footer phrases.
    2. Structural detection of pipe/bullet-separated link lists.

    Blocks that match either heuristic are discarded; all others are kept
    unchanged.

    Usage::

        cleaner = HtmlNavFooterRemover()
        cleaned_blocks = cleaner.clean(blocks)
    """

    def clean(self, blocks: list[IRBlock]) -> list[IRBlock]:
        """Remove navigation and footer boilerplate blocks.

        Args:
            blocks: Input IRBlocks from the previous pipeline stage.

        Returns:
            Filtered list with nav/footer blocks removed.
        """
        return [b for b in blocks if not _is_nav_or_footer(b.text.strip())]
