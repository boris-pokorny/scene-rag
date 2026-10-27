import pytest

from scene_rag.extraction.ingestion.settings import redact_uri


class TestGivenRegistryUri:
    """Connection strings as they may appear in the environment."""

    @pytest.mark.parametrize(
        ("uri", "expected"),
        [
            ("mongodb://localhost:27017", "mongodb://localhost:27017"),
            ("mongodb://app:s3cret@mongo:27017/db", "mongodb://***@mongo:27017/db"),
            (
                "mongodb+srv://app:p%40ss@cluster.example.net/",
                "mongodb+srv://***@cluster.example.net/",
            ),
            ("mongodb://app@mongo:27017", "mongodb://***@mongo:27017"),
        ],
    )
    def test_when_redacted_then_userinfo_hidden(self, uri: str, expected: str) -> None:
        assert redact_uri(uri) == expected
