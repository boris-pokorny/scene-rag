"""Relevance by span overlap: a source is relevant when it overlaps a labelled span.

Matching is by asset version and time, never by chunk, so the same labels judge any
chunking. Spans that only touch at an endpoint do not overlap.
"""

from collections.abc import Sequence

from scene_rag.evaluation.models import Source, Span


def overlaps(source: Source, span: Span) -> bool:
    """Return whether ``source`` shares asset version and some time with ``span``."""
    return (
        source.asset_version == span.asset_version
        and source.start_seconds < span.end_seconds
        and span.start_seconds < source.end_seconds
    )


def covered(sources: Sequence[Source], spans: Sequence[Span]) -> set[int]:
    """Return the indices of the spans at least one source overlaps."""
    return {i for i, span in enumerate(spans) if any(overlaps(s, span) for s in sources)}


def gains(sources: Sequence[Source], spans: Sequence[Span]) -> list[int]:
    """Return a binary gain per source, in rank order.

    A source gains 1 when it overlaps a span no earlier source overlapped, so several sources
    on one span are credited once and nDCG stays at most 1.
    """
    credited: set[int] = set()
    result: list[int] = []
    for source in sources:
        new = {i for i, span in enumerate(spans) if i not in credited and overlaps(source, span)}
        credited |= new
        result.append(1 if new else 0)
    return result
