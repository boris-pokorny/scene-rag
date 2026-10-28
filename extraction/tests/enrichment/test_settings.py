"""Validation and redaction of the settings."""

from pathlib import Path

import pytest

from scene_rag.extraction.enrichment.settings import EnrichmentSettings, redact_uri


def make_settings(**overrides: object) -> EnrichmentSettings:
    values: dict[str, object] = {
        "registry_uri": "mongodb://app:s3cret@registry:27017",
        "registry_database": "registry",
        "scene_store_uri": "mongodb://scenes:27017",
        "scene_store_database": "scenes",
        "enrichment_store_uri": "mongodb://enrichments:27017",
        "enrichment_store_database": "enrichments",
    }
    values.update(overrides)
    return EnrichmentSettings(**values)  # type: ignore[arg-type]


class TestGivenSettings:
    """A complete, valid set of values."""

    def test_when_constructed_then_defaults_applied(self) -> None:
        settings = make_settings()

        assert settings.ollama_url == "http://localhost:11434"
        assert settings.ollama_timeout_seconds == 600.0
        assert settings.whisper_cache_dir is None
        assert settings.whisper_cpu_threads == 0
        assert settings.poll_interval_seconds is None
        assert settings.workers == 1

    def test_when_redacted_then_credentials_hidden(self) -> None:
        settings = make_settings()

        assert settings.redacted_registry_uri == "mongodb://***@registry:27017"
        assert settings.redacted_scene_store_uri == "mongodb://scenes:27017"
        assert settings.redacted_enrichment_store_uri == "mongodb://enrichments:27017"
        assert redact_uri("http://u:p@h/x") == "http://***@h/x"

    def test_when_cache_dir_given_then_kept(self) -> None:
        assert make_settings(whisper_cache_dir=Path("/models")).whisper_cache_dir == Path("/models")

    @pytest.mark.parametrize(
        "field, value",
        [
            ("registry_uri", ""),
            ("registry_database", ""),
            ("scene_store_uri", ""),
            ("scene_store_database", ""),
            ("enrichment_store_uri", ""),
            ("enrichment_store_database", ""),
            ("ollama_url", ""),
            ("ollama_url", "localhost:11434"),
            ("ollama_timeout_seconds", 0.0),
            ("ollama_timeout_seconds", float("inf")),
            ("whisper_cpu_threads", -1),
            ("poll_interval_seconds", 0.0),
            ("poll_interval_seconds", float("nan")),
            ("workers", 0),
        ],
    )
    def test_when_value_invalid_then_value_error(self, field: str, value: object) -> None:
        with pytest.raises(ValueError, match=field.split("_")[0]):
            make_settings(**{field: value})
