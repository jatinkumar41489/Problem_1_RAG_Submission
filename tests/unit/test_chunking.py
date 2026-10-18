import pytest

from app.exceptions import ConfigError
from app.ingestion.chunking import Chunker


def words(n):
    return " ".join(f"w{i}" for i in range(n))


@pytest.mark.parametrize("size,overlap", [(0, 0), (-5, 0), (10, -1), (10, 10), (10, 11)])
def test_invalid_params_rejected(size, overlap):
    with pytest.raises(ConfigError):
        Chunker(size, overlap)


def test_short_document_single_chunk():
    cs = Chunker(100, 10).chunk_text(words(7))
    assert len(cs) == 1 and cs[0].token_count == 7


def test_empty_and_whitespace_yield_nothing():
    assert Chunker(10, 2).chunk_text("") == []
    assert Chunker(10, 2).chunk_text("   \n\t  ") == []


def test_window_sizes_and_overlap():
    cs = Chunker(10, 3).chunk_text(words(25))
    # stride 7: windows start at 0,7,14,21; the last has 3 overlap words + 1 new word
    assert [c.token_count for c in cs] == [10, 10, 10, 4]
    assert cs[0].text.split()[-3:] == cs[1].text.split()[:3]      # 3-word overlap
    assert cs[-1].text.split()[-1] == "w24"                        # covers the final word


def test_every_word_covered_and_terminates():
    text = words(1234)
    cs = Chunker(50, 49).chunk_text(text)  # stride 1: worst case, must still terminate
    covered = set()
    for c in cs:
        covered.update(c.text.split())
    assert covered == set(text.split())


def test_no_fragment_contained_in_predecessor():
    cs = Chunker(10, 3).chunk_text(words(24))
    for a, b in zip(cs, cs[1:]):
        assert b.text.split()[-1] not in a.text.split()


def test_exact_multiple_boundary():
    cs = Chunker(10, 0).chunk_text(words(20))
    assert len(cs) == 2 and all(c.token_count == 10 for c in cs)


def test_sections_follow_heading_offsets():
    text = "alpha beta gamma delta epsilon zeta"
    cs = Chunker(3, 0).chunk_text(text, headings=[(0, "A"), (text.index("delta"), "A > B")])
    assert [c.section for c in cs] == ["A", "A > B"]


def test_page_and_start_index():
    cs = Chunker(3, 1).chunk_text(words(8), page=4, start_index=10)
    assert all(c.page == 4 for c in cs) and cs[0].index == 10
