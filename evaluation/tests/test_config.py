"""Hyperparameters and settings of the run role validate."""

from pathlib import Path
from typing import Any

import pytest

from scene_rag.evaluation.config import RunConfig
from scene_rag.evaluation.settings import RunSettings, redact_uri


class TestGivenRunConfig:
    """Cut-offs, count and use cases."""

    @pytest.mark.parametrize(
        ("overrides", "message"),
        [
            ({"ks": ()}, "ks"),
            ({"ks": (0, 5)}, "ks"),
            ({"top_k": 4, "ks": (5,)}, "top_k"),
            ({"use_cases": ()}, "use_cases"),
            ({"use_cases": ("rerank",)}, "use_cases"),
            ({"use_cases": ("search", "search")}, "repeat"),
        ],
    )
    def test_when_invalid_then_rejected(self, overrides: dict[str, Any], message: str) -> None:
        with pytest.raises(ValueError, match=message):
            RunConfig(**overrides)


class TestGivenRunSettings:
    """URL and timeout."""

    def test_when_not_http_then_rejected(self) -> None:
        with pytest.raises(ValueError, match="http"):
            RunSettings("ftp://q", Path("x"))

    @pytest.mark.parametrize("timeout", [0.0, float("inf")])
    def test_when_timeout_not_positive_finite_then_rejected(self, timeout: float) -> None:
        with pytest.raises(ValueError, match="timeout"):
            RunSettings("http://q", Path("x"), timeout)

    def test_when_url_has_credentials_then_redacted(self) -> None:
        settings = RunSettings("http://user:secret@q:8081", Path("x"))

        assert settings.redacted_query_api_url == "http://***@q:8081"
        assert redact_uri("http://q") == "http://q"
