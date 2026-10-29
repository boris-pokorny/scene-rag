"""The prompt the generator answers under: numbered sources, then the question.

Part of the ``query_config`` id: the templates come from `PromptConfig`, the rendering of a
source line is decided here.
"""

from collections.abc import Sequence

from scene_rag.retrieval.query_api import models
from scene_rag.retrieval.query_api.config import PromptConfig


def _filename(source: models.Source) -> str:
    return source.sources[0].filename if source.sources else source.asset_version


class PromptBuilder:
    """Renders a chat prompt from the configured templates."""

    def __init__(self, config: PromptConfig) -> None:
        """Bind to the templates."""
        self._config = config

    def render_sources(self, sources: Sequence[models.Source]) -> str:
        """Render one line per source, numbered from one, in the order given."""
        return "\n".join(
            self._config.source_template.format(
                number=number,
                filename=_filename(source),
                start=source.start_seconds,
                end=source.end_seconds,
                text=source.text,
            )
            for number, source in enumerate(sources, start=1)
        )

    def build(self, question: str, sources: Sequence[models.Source]) -> tuple[models.Message, ...]:
        """Return the system and user messages for the question over the sources."""
        user = self._config.user_template.format(
            sources=self.render_sources(sources), question=question
        )
        return (
            models.Message("system", self._config.system_template),
            models.Message("user", user),
        )
