"""Tests for the UnicodeFixer cleaner."""

import pytest

from rag.core.contracts.ir_block import BlockType, IRBlock
from rag.infra.cleaning.unicode_fix import UnicodeFixer, _normalize_text


def _block(text: str) -> IRBlock:
    return IRBlock(block_type=BlockType.PARAGRAPH, text=text)


@pytest.fixture()
def cleaner() -> UnicodeFixer:
    return UnicodeFixer()


# ---------------------------------------------------------------------------
# _normalize_text helper
# ---------------------------------------------------------------------------


def test_normalize_smart_quotes_single():
    assert _normalize_text("\u2018hello\u2019") == "'hello'"


def test_normalize_smart_quotes_double():
    assert _normalize_text("\u201chello\u201d") == '"hello"'


def test_normalize_en_dash():
    assert _normalize_text("page 1\u20132") == "page 1-2"


def test_normalize_em_dash():
    assert _normalize_text("word\u2014word") == "word--word"


def test_normalize_ellipsis():
    assert _normalize_text("wait\u2026") == "wait..."


def test_normalize_non_breaking_space():
    assert _normalize_text("hello\u00a0world") == "hello world"


def test_normalize_soft_hyphen_removed():
    assert _normalize_text("hyp\u00adhen") == "hyphen"


def test_normalize_ligature_fi():
    assert _normalize_text("\ufb01le") == "file"


def test_normalize_ligature_fl():
    assert _normalize_text("\ufb02oor") == "floor"


def test_normalize_ligature_ff():
    assert _normalize_text("\ufb00ect") == "ffect"


def test_normalize_plain_text_unchanged():
    text = "Hello, world! This is plain ASCII."
    assert _normalize_text(text) == text


def test_normalize_nfc_composition():
    # e + combining acute accent → é (NFC)
    composed = _normalize_text("e\u0301")
    assert composed == "\u00e9"


# ---------------------------------------------------------------------------
# UnicodeFixer.clean()
# ---------------------------------------------------------------------------


def test_clean_returns_same_count(cleaner):
    blocks = [_block("Hello \u2018world\u2019"), _block("Normal text")]
    result = cleaner.clean(blocks)
    assert len(result) == 2


def test_clean_normalises_smart_quotes(cleaner):
    blocks = [_block("\u201cQuoted text\u201d")]
    result = cleaner.clean(blocks)
    assert result[0].text == '"Quoted text"'


def test_clean_leaves_plain_block_unchanged(cleaner):
    block = _block("Plain text, no changes needed.")
    result = cleaner.clean([block])
    assert result[0] is block  # same object — not copied


def test_clean_preserves_block_type(cleaner):
    from rag.core.contracts.ir_block import BlockType
    block = IRBlock(block_type=BlockType.HEADING, text="\u2018Title\u2019")
    result = cleaner.clean([block])
    assert result[0].block_type == BlockType.HEADING


def test_clean_empty_list(cleaner):
    assert cleaner.clean([]) == []


def test_clean_multiple_substitutions_in_one_block(cleaner):
    text = "\u201cHello\u201d \u2014 it\u2019s a \u2026 test"
    result = cleaner.clean([_block(text)])
    assert result[0].text == '"Hello" -- it\'s a ... test'


def test_clean_preserves_section_path(cleaner):
    block = IRBlock(
        block_type=BlockType.PARAGRAPH,
        text="\u2018hi\u2019",
        section_path=["Chapter 1"],
    )
    result = cleaner.clean([block])
    assert result[0].section_path == ["Chapter 1"]
