"""Validation and redaction of the settings."""

import pytest

from scene_rag.retrieval.chunking.settings import ChunkingSettings, redact_uri


def make_settings(**overrides: object) -> ChunkingSettings:
    values: dict[str, object] = {
        "scene_api_url": "http://app:s3cret@scene-api:8080",
        "chunk_store_uri": "mongodb://app:s3cret@chunks:27017",
        "chunk_store_database": "chunks",
    }
    values.update(overrides)
    return ChunkingSettings(**values)  # type: ignore[arg-type]


class TestGivenSettings:
    """A complete, valid set of values."""

    def test_when_constructed_then_defaults_applied(self) -> None:
        settings = make_settings()

        assert settings.scene_api_timeout_seconds == 30.0
        assert settings.poll_interval_seconds is None
        assert settings.workers == 1

    def test_when_redacted_then_credentials_hidden(self) -> None:
        settings = make_settings()

        assert settings.redacted_chunk_store_uri == "mongodb://***@chunks:27017"
        assert settings.redacted_scene_api_url == "http://***@scene-api:8080"
        assert redact_uri("http://h/x") == "http://h/x"

    @pytest.mark.parametrize(
        "field, value",
        [
            ("scene_api_url", ""),
            ("scene_api_url", "scene-api:8080"),
            ("chunk_store_uri", ""),
            ("chunk_store_database", ""),
            ("scene_api_timeout_seconds", 0.0),
            ("scene_api_timeout_seconds", float("inf")),
            ("poll_interval_seconds", 0.0),
            ("poll_interval_seconds", float("nan")),
            ("workers", 0),
        ],
    )
    def test_when_value_invalid_then_value_error(self, field: str, value: object) -> None:
        with pytest.raises(ValueError, match=field.split("_")[0]):
            make_settings(**{field: value})
