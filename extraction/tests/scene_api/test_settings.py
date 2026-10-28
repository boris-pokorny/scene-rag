"""Validation of the scene-api settings."""

import dataclasses
from typing import Any, cast

import pytest

from scene_rag.extraction.scene_api.settings import SceneApiSettings, redact_uri

VALID = SceneApiSettings(
    registry_uri="mongodb://user:secret@localhost:27017",
    registry_database="scene_rag",
    scene_store_uri="mongodb://localhost:27017",
    scene_store_database="scene_rag",
    enrichment_store_uri="mongodb://localhost:27017",
    enrichment_store_database="scene_rag",
)


class TestGivenSettings:
    """Defaults, validation and redaction."""

    def test_when_defaults_then_local_host_and_page_of_a_hundred(self) -> None:
        assert (VALID.host, VALID.port, VALID.page_size) == ("127.0.0.1", 8080, 100)

    @pytest.mark.parametrize(
        "field",
        [
            "registry_uri",
            "registry_database",
            "scene_store_uri",
            "scene_store_database",
            "enrichment_store_uri",
            "enrichment_store_database",
            "host",
        ],
    )
    def test_when_text_empty_then_rejected(self, field: str) -> None:
        with pytest.raises(ValueError, match=field):
            dataclasses.replace(VALID, **cast(dict[str, Any], {field: ""}))

    @pytest.mark.parametrize("port", [-1, 65536])
    def test_when_port_out_of_range_then_rejected(self, port: int) -> None:
        with pytest.raises(ValueError, match="port"):
            dataclasses.replace(VALID, port=port)

    def test_when_port_zero_then_accepted(self) -> None:
        assert dataclasses.replace(VALID, port=0).port == 0

    def test_when_page_size_below_one_then_rejected(self) -> None:
        with pytest.raises(ValueError, match="page_size"):
            dataclasses.replace(VALID, page_size=0)

    def test_when_uri_redacted_then_credentials_hidden(self) -> None:
        assert VALID.redacted_registry_uri == "mongodb://***@localhost:27017"
        assert VALID.redacted_scene_store_uri == "mongodb://localhost:27017"
        assert VALID.redacted_enrichment_store_uri == "mongodb://localhost:27017"
        assert redact_uri("mongodb://a:b@h/db?x=1") == "mongodb://***@h/db?x=1"
