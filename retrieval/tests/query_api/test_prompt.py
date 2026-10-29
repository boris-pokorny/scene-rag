"""Rendering of the generator's prompt."""

from scene_rag.retrieval.query_api import models
from scene_rag.retrieval.query_api.config import PromptConfig
from scene_rag.retrieval.query_api.prompt import PromptBuilder
from tests.support.query_fakes import make_entry_view, make_source

CONFIG = PromptConfig(
    system_template="sys",
    user_template="S:\n{sources}\nQ: {question}",
    source_template="[{number}] {filename} {start:.1f}-{end:.1f}s: {text}",
)


class TestGivenSources:
    """Two sources and a question."""

    def test_when_built_then_numbered_lines_then_question(self) -> None:
        sources = [make_source(make_entry_view(0), 0.9), make_source(make_entry_view(1), 0.8)]

        messages = PromptBuilder(CONFIG).build("when? {not a field}", sources)

        assert messages == (
            models.Message("system", "sys"),
            models.Message(
                "user",
                "S:\n[1] clip.mp4 0.0-4.0s: chunk 0 text\n[2] clip.mp4 4.0-8.0s: chunk 1 text\n"
                "Q: when? {not a field}",
            ),
        )

    def test_when_source_has_no_source_reference_then_asset_version_named(self) -> None:
        source = models.Source(
            chunk_version="sha256:00",
            asset_version="sha256:aa",
            scene_set_version="sha256:ss",
            scene_indices=(0,),
            start_seconds=0.0,
            end_seconds=1.0,
            score=0.5,
            text="t",
            sources=(),
            entitlements=frozenset({"public"}),
        )

        assert PromptBuilder(CONFIG).render_sources([source]) == "[1] sha256:aa 0.0-1.0s: t"

    def test_when_defaults_used_then_system_asks_for_citations(self) -> None:
        messages = PromptBuilder(PromptConfig()).build("q", [])

        assert "[2]" in messages[0].content
        assert messages[1].content.endswith("Question: q")
