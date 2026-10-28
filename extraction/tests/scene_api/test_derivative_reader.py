"""Reading derivatives from the filesystem, verified against their digest."""

import hashlib
from pathlib import Path

import pytest

from scene_rag.extraction.scene_api import models
from scene_rag.extraction.scene_api.derivative_reader import FileDerivativeReader
from scene_rag.extraction.scene_api.protocols import DerivativeUnavailableError


def filed(directory: Path, data: bytes, algorithm: str = "sha256") -> models.DerivativeRefView:
    digest = hashlib.new(algorithm, data).hexdigest()
    path = directory / f"{digest}.jpg"
    path.write_bytes(data)
    return models.DerivativeRefView(
        digest=f"{algorithm}:{digest}",
        uri=path.as_uri(),
        media_type="image/jpeg",
        size_bytes=len(data),
    )


class TestGivenFiledDerivative:
    """A file whose name is its content hash."""

    def test_when_read_then_bytes_returned(self, tmp_path: Path) -> None:
        reference = filed(tmp_path, b"jpeg bytes")

        assert FileDerivativeReader().read(reference) == b"jpeg bytes"

    def test_when_other_algorithm_then_verified_with_it(self, tmp_path: Path) -> None:
        reference = filed(tmp_path, b"jpeg bytes", "sha1")

        assert FileDerivativeReader().read(reference) == b"jpeg bytes"

    def test_when_uri_has_escaped_characters_then_resolved(self, tmp_path: Path) -> None:
        directory = tmp_path / "with space"
        directory.mkdir()
        reference = filed(directory, b"x")

        assert "%20" in reference.uri
        assert FileDerivativeReader().read(reference) == b"x"

    def test_when_bytes_tampered_then_unavailable(self, tmp_path: Path) -> None:
        reference = filed(tmp_path, b"jpeg bytes")
        Path(tmp_path / reference.uri.rsplit("/", 1)[1]).write_bytes(b"other bytes")

        with pytest.raises(DerivativeUnavailableError, match="does not match"):
            FileDerivativeReader().read(reference)

    def test_when_file_missing_then_unavailable(self, tmp_path: Path) -> None:
        reference = filed(tmp_path, b"jpeg bytes")
        Path(tmp_path / reference.uri.rsplit("/", 1)[1]).unlink()

        with pytest.raises(DerivativeUnavailableError, match=reference.uri):
            FileDerivativeReader().read(reference)

    @pytest.mark.parametrize(
        "uri, digest",
        [
            ("s3://bucket/key.jpg", "sha256:" + "ab" * 32),
            ("file:///x.jpg", "nohash:" + "ab" * 32),
            ("file:///x.jpg", "sha256"),
        ],
    )
    def test_when_reference_unusable_then_unavailable(self, uri: str, digest: str) -> None:
        reference = models.DerivativeRefView(
            digest=digest, uri=uri, media_type="image/jpeg", size_bytes=1
        )

        with pytest.raises(DerivativeUnavailableError):
            FileDerivativeReader().read(reference)
