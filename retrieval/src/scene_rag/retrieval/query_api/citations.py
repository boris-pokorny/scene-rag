"""Citations in a generated answer: ``[n]`` markers naming the numbered sources.

Part of the ``query_config`` id: what counts as a citation decides what evaluation measures.
"""

import re
from typing import Final

_MARKER: Final = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def parse_citations(text: str) -> tuple[int, ...]:
    """Return the source numbers the text cites, in order of first mention, each once.

    A marker is ``[n]`` or ``[n, m]``; anything else in brackets is not a citation.
    """
    seen: dict[int, None] = {}
    for marker in _MARKER.finditer(text):
        for number in marker.group(1).split(","):
            seen.setdefault(int(number), None)
    return tuple(seen)


def validate(
    citations: tuple[int, ...], source_count: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split citations into those naming a source (``1..source_count``) and the rest."""
    valid = tuple(n for n in citations if 1 <= n <= source_count)
    invalid = tuple(n for n in citations if not 1 <= n <= source_count)
    return valid, invalid
