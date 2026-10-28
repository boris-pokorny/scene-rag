"""Store contracts and the query interface of the scene-api role.

`AssetVersionReader`, `SceneSetReader` and `EnrichmentReader` are the role's read-only views of
the stores the steps write; `DerivativeReader` yields the bytes a scene set references.
`SceneCatalog` is what the HTTP adapter calls: the role's use cases, free of transport.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Protocol

from scene_rag.extraction.scene_api import models


class NotFoundError(LookupError):
    """A requested resource does not exist.

    Attributes:
        kind: What was looked for: ``scene-set``, ``scene``, ``asset-version``, ``derivative``.
        key: The identifier that was looked up.
    """

    def __init__(self, kind: str, key: str) -> None:
        """Name the resource kind and the key that missed."""
        super().__init__(f"no {kind} {key}")
        self.kind = kind
        self.key = key


class InvalidQueryError(ValueError):
    """A request parameter cannot be used as asked."""


class DerivativeUnavailableError(RuntimeError):
    """A derivative cannot be read as segmentation filed it: missing, or its bytes changed."""


class AssetVersionReader(Protocol):
    """Queries over the Asset Registry, as far as the role needs them."""

    def get(self, version: str) -> models.AssetVersionView | None:
        """Return the version with the given content hash, live or retired."""
        ...

    def get_many(self, versions: Iterable[str]) -> Mapping[str, models.AssetVersionView]:
        """Return the versions found among those asked for, keyed by content hash."""
        ...


class SceneSetReader(Protocol):
    """Queries over the Scene Store, as far as the role needs them."""

    def get(self, version: str) -> models.SceneSetView | None:
        """Return the scene set with the given data version."""
        ...

    def list_summaries(
        self,
        *,
        after: str | None,
        limit: int,
        asset_version: str | None,
        segmentation_fingerprint: str | None,
    ) -> Sequence[models.SceneSetSummaryView]:
        """Return up to ``limit`` summaries in version order, those after the cursor.

        Args:
            after: Version the page starts after; from the beginning when ``None``.
            limit: Most summaries to return.
            asset_version: Only sets cut from this asset version, when given.
            segmentation_fingerprint: Only sets cut under this fingerprint, when given.
        """
        ...

    def find_derivative(self, digest: str) -> models.DerivativeRefView | None:
        """Return the reference some scene set holds to the derivative with this digest."""
        ...


class EnrichmentReader(Protocol):
    """Queries over the Enrichment Store, as far as the role needs them."""

    def list_coverage(self, scene_set_versions: Iterable[str]) -> Sequence[models.Coverage]:
        """Return, per scene set, enricher and fingerprint, how many scenes have a record.

        In no particular order.
        """
        ...

    def list_for_scenes(
        self,
        scene_set_version: str,
        scene_indices: Iterable[int],
        pins: Sequence[models.EnrichmentPin] = (),
    ) -> Sequence[models.EnrichmentView]:
        """Return the records of the given scenes, in (index, enricher, fingerprint) order.

        Args:
            scene_set_version: The scene set the scenes belong to.
            scene_indices: Which scenes.
            pins: Only records written under one of these (enricher, fingerprint) pairs; every
                record when empty.
        """
        ...


class DerivativeReader(Protocol):
    """Reads the bytes a derivative reference points at."""

    def read(self, reference: models.DerivativeRefView) -> bytes:
        """Return the bytes of the derivative, verified against its digest.

        Raises:
            DerivativeUnavailableError: When the file cannot be read or its content hash is
                not the one the reference names.
        """
        ...


class SceneCatalog(Protocol):
    """The use cases of the role, as the delivery adapters call them."""

    def list_scene_sets(self, query: models.SceneSetQuery) -> models.Page[models.SceneSetSummary]:
        """Return a page of scene sets of live asset versions, complete under the pins.

        Raises:
            InvalidQueryError: When the limit is below one.
        """
        ...

    def get_scene_set(self, version: str) -> models.SceneSetSummary:
        """Return the summary of one scene set, live or not.

        Raises:
            NotFoundError: When no scene set has the version.
        """
        ...

    def list_scene_records(
        self, version: str, query: models.SceneQuery
    ) -> models.Page[models.SceneRecord]:
        """Return a page of the scene records of one scene set.

        Raises:
            NotFoundError: When no scene set has the version.
            InvalidQueryError: When the limit is below one.
        """
        ...

    def get_scene_record(
        self, version: str, index: int, pins: Sequence[models.EnrichmentPin]
    ) -> models.SceneRecord:
        """Return one scene record.

        Raises:
            NotFoundError: When no scene set has the version, or it has no such scene.
        """
        ...

    def get_asset_version(self, version: str) -> models.AssetVersionView:
        """Return one asset version, live or retired.

        Raises:
            NotFoundError: When no asset version has the content hash.
        """
        ...

    def find_derivative(self, digest: str) -> models.DerivativeRefView:
        """Return the reference some scene set holds to a derivative, without its bytes.

        Raises:
            NotFoundError: When no scene set references the digest.
        """
        ...

    def get_derivative(self, digest: str) -> models.DerivativeContent:
        """Return the verified bytes of a derivative some scene set references.

        Raises:
            NotFoundError: When no scene set references the digest.
            DerivativeUnavailableError: When the bytes cannot be read as filed.
        """
        ...
