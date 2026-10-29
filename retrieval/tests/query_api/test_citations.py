"""Parsing and validation of ``[n]`` citations."""

import pytest

from scene_rag.retrieval.query_api.citations import parse_citations, validate


class TestGivenAnswers:
    """Texts with and without markers."""

    @pytest.mark.parametrize(
        "text, expected",
        [
            ("no citation", ()),
            ("see [1].", (1,)),
            ("[1][3] and [1]", (1, 3)),
            ("[1, 2] then [3,1]", (1, 2, 3)),
            ("[0] and [12]", (0, 12)),
            ("[a] [1a] [ 1 ] [-1]", ()),
        ],
    )
    def test_when_parsed_then_numbers_in_order_of_first_mention(
        self, text: str, expected: tuple[int, ...]
    ) -> None:
        assert parse_citations(text) == expected

    @pytest.mark.parametrize(
        "citations, count, valid, invalid",
        [
            ((), 3, (), ()),
            ((1, 3), 3, (1, 3), ()),
            ((0, 2, 4), 3, (2,), (0, 4)),
            ((1,), 0, (), (1,)),
        ],
    )
    def test_when_validated_then_split_by_source_count(
        self,
        citations: tuple[int, ...],
        count: int,
        valid: tuple[int, ...],
        invalid: tuple[int, ...],
    ) -> None:
        assert validate(citations, count) == (valid, invalid)
