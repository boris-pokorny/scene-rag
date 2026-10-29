"""Validation and redaction of the settings; the names shared with the steps."""

import pytest

from scene_rag.retrieval.chunking import settings as chunking_settings
from scene_rag.retrieval.embedding import settings as embedding_settings
from scene_rag.retrieval.query_api import settings
from scene_rag.retrieval.query_api.settings import OllamaSettings, QuerySettings, redact_uri


def make_settings(**overrides: object) -> QuerySettings:
    values: dict[str, object] = {
        "chunk_store_uri": "mongodb://app:s3cret@chunks:27017",
        "chunk_store_database": "chunks",
        "search_index_url": "http://app:s3cret@qdrant:6333",
        "scene_api_url": "http://app:s3cret@scene-api:8080",
    }
    values.update(overrides)
    return QuerySettings(**values)  # type: ignore[arg-type]


class TestGivenSettings:
    """A complete, valid set of values."""

    def test_when_constructed_then_defaults_applied(self) -> None:
        loaded = make_settings()

        assert loaded.search_index_timeout_seconds == 30.0
        assert loaded.scene_api_timeout_seconds == 30.0
        assert loaded.ollama == OllamaSettings("http://localhost:11434", 120.0, 300.0)
        assert (loaded.host, loaded.port) == ("127.0.0.1", 8081)

    def test_when_redacted_then_credentials_hidden(self) -> None:
        loaded = make_settings(ollama=OllamaSettings("http://u:p@ollama:11434"))

        assert loaded.redacted_chunk_store_uri == "mongodb://***@chunks:27017"
        assert loaded.redacted_search_index_url == "http://***@qdrant:6333"
        assert loaded.redacted_scene_api_url == "http://***@scene-api:8080"
        assert loaded.ollama.redacted_url == "http://***@ollama:11434"
        assert redact_uri("http://h/x") == "http://h/x"

    @pytest.mark.parametrize(
        "field, value",
        [
            ("chunk_store_uri", ""),
            ("chunk_store_database", ""),
            ("search_index_url", "qdrant:6333"),
            ("scene_api_url", ""),
            ("search_index_timeout_seconds", 0.0),
            ("scene_api_timeout_seconds", float("inf")),
            ("host", ""),
            ("port", -1),
            ("port", 65536),
        ],
    )
    def test_when_value_invalid_then_value_error(self, field: str, value: object) -> None:
        with pytest.raises(ValueError, match=field.split("_")[0]):
            make_settings(**{field: value})

    @pytest.mark.parametrize(
        "field, value",
        [("url", "ollama:11434"), ("timeout_seconds", 0.0), ("generator_timeout_seconds", -1.0)],
    )
    def test_when_ollama_value_invalid_then_value_error(self, field: str, value: object) -> None:
        with pytest.raises(ValueError, match="ollama"):
            OllamaSettings(**{field: value})  # type: ignore[arg-type]


class TestGivenSharedNames:
    """The role reads what the steps wrote: the variables are the steps' own."""

    def test_when_compared_with_steps_then_names_equal(self) -> None:
        assert settings.CHUNK_STORE_URI_ENV == embedding_settings.CHUNK_STORE_URI_ENV
        assert settings.CHUNK_STORE_DATABASE_ENV == embedding_settings.CHUNK_STORE_DATABASE_ENV
        assert settings.SEARCH_INDEX_URL_ENV == embedding_settings.SEARCH_INDEX_URL_ENV
        assert settings.OLLAMA_URL_ENV == embedding_settings.OLLAMA_URL_ENV
        assert settings.OLLAMA_TIMEOUT_SECONDS_ENV == embedding_settings.OLLAMA_TIMEOUT_SECONDS_ENV
        assert settings.SCENE_API_URL_ENV == chunking_settings.SCENE_API_URL_ENV
        assert settings.HOST_ENV.endswith("QUERY_API_HOST")
