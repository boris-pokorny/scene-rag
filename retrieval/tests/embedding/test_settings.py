"""Validation and redaction of the settings; the names shared with the chunking step."""

import pytest

from scene_rag.retrieval.chunking import settings as chunking_settings
from scene_rag.retrieval.embedding import settings
from scene_rag.retrieval.embedding.settings import EmbeddingSettings, OllamaSettings, redact_uri


def make_settings(**overrides: object) -> EmbeddingSettings:
    values: dict[str, object] = {
        "chunk_store_uri": "mongodb://app:s3cret@chunks:27017",
        "chunk_store_database": "chunks",
        "search_index_url": "http://app:s3cret@qdrant:6333",
    }
    values.update(overrides)
    return EmbeddingSettings(**values)  # type: ignore[arg-type]


class TestGivenSettings:
    """A complete, valid set of values."""

    def test_when_constructed_then_defaults_applied(self) -> None:
        loaded = make_settings()

        assert loaded.search_index_timeout_seconds == 30.0
        assert loaded.ollama == OllamaSettings("http://localhost:11434", 120.0)
        assert loaded.batch_size == 32
        assert loaded.poll_interval_seconds is None
        assert loaded.workers == 1

    def test_when_redacted_then_credentials_hidden(self) -> None:
        loaded = make_settings(ollama=OllamaSettings("http://u:p@ollama:11434"))

        assert loaded.redacted_chunk_store_uri == "mongodb://***@chunks:27017"
        assert loaded.redacted_search_index_url == "http://***@qdrant:6333"
        assert loaded.ollama.redacted_url == "http://***@ollama:11434"
        assert redact_uri("http://h/x") == "http://h/x"

    @pytest.mark.parametrize(
        "field, value",
        [
            ("chunk_store_uri", ""),
            ("chunk_store_database", ""),
            ("search_index_url", ""),
            ("search_index_url", "qdrant:6333"),
            ("search_index_timeout_seconds", 0.0),
            ("search_index_timeout_seconds", float("inf")),
            ("batch_size", 0),
            ("poll_interval_seconds", 0.0),
            ("poll_interval_seconds", float("nan")),
            ("workers", 0),
        ],
    )
    def test_when_value_invalid_then_value_error(self, field: str, value: object) -> None:
        with pytest.raises(ValueError, match=field.split("_")[0]):
            make_settings(**{field: value})

    @pytest.mark.parametrize("field, value", [("url", "ollama:11434"), ("timeout_seconds", 0.0)])
    def test_when_ollama_value_invalid_then_value_error(self, field: str, value: object) -> None:
        with pytest.raises(ValueError, match="ollama"):
            OllamaSettings(**{field: value})  # type: ignore[arg-type]


class TestGivenSharedNames:
    """One database holds the chunks and the index ledger: the variables are the same."""

    def test_when_compared_with_chunking_then_store_and_poll_names_equal(self) -> None:
        assert settings.CHUNK_STORE_URI_ENV == chunking_settings.CHUNK_STORE_URI_ENV
        assert settings.CHUNK_STORE_DATABASE_ENV == chunking_settings.CHUNK_STORE_DATABASE_ENV
        assert settings.POLL_INTERVAL_SECONDS_ENV == chunking_settings.POLL_INTERVAL_SECONDS_ENV
        assert settings.WORKERS_ENV.endswith("EMBEDDING_WORKERS")
