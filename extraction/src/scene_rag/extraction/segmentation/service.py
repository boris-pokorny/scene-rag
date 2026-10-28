"""The unit of segmentation work: cut one asset version into one scene set.

`scheduler` decides which versions a pass processes and how many at a time; this module decides
what a scene set contains, and is fingerprinted for it.
"""

import dataclasses
import logging
import tempfile
from pathlib import Path
from typing import Final, Literal

from scene_rag.extraction.segmentation import models
from scene_rag.extraction.segmentation.config import SegmentationConfig
from scene_rag.extraction.segmentation.fingerprint import compute_fingerprint
from scene_rag.extraction.segmentation.protocols import (
    AssetVersionReader,
    AudioClipper,
    Clock,
    DerivativeStore,
    KeyframeRenderer,
    MediaLocator,
    SceneStoreWriter,
    SegmentExtractor,
    VersionRetiredError,
)

logger = logging.getLogger(__name__)

KEYFRAME_MEDIA_TYPE: Final = "image/jpeg"
AUDIO_MEDIA_TYPE: Final = "audio/wav"


class SegmentationService:
    """`VersionSegmenter` over the reference backends: the step's output, version by version.

    Each version is cut, rendered and clipped in a scratch workspace; the derivatives are filed
    first and the scene set last, so a stored set never points at a file that is missing.
    Derivatives are content-addressed, so a failed or repeated run leaves nothing to undo.

    A version whose bytes cannot be read as ingestion recorded them is left to the caller to
    defer: ingestion reconciles the source, after which the version is either retired or still
    pending. A version retired while being processed is dropped before the scene set is
    written. Versions are independent, so a `Dispatcher` may run several at once.
    """

    def __init__(
        self,
        *,
        assets: AssetVersionReader,
        scene_writer: SceneStoreWriter,
        derivatives: DerivativeStore,
        locator: MediaLocator,
        extractor: SegmentExtractor,
        keyframes: KeyframeRenderer,
        audio: AudioClipper,
        clock: Clock,
        config: SegmentationConfig,
        workspace_root: Path | None = None,
    ) -> None:
        """Compose the pass from its collaborators; `bootstrap` is the only caller.

        Args:
            assets: Read view of the Asset Registry, for the final liveness check.
            scene_writer: Command side of the Scene Store.
            derivatives: Where keyframes and clips are filed.
            locator: Finds a version's bytes on the local filesystem.
            extractor: Cuts a video into segments.
            keyframes: Renders keyframes per segment.
            audio: Slices audio clips per segment.
            clock: Source of ``created_at``.
            config: Hyperparameters in force.
            workspace_root: Directory the per-version scratch workspaces are created under;
                the system temporary directory when ``None``. Put it on the derivative
                store's filesystem so filing a derivative is a rename, not a copy.
        """
        self._assets = assets
        self._scene_writer = scene_writer
        self._derivatives = derivatives
        self._locator = locator
        self._extractor = extractor
        self._keyframes = keyframes
        self._audio = audio
        self._clock = clock
        self._config = config
        self._workspace_root = workspace_root
        self._fingerprint = compute_fingerprint(config)

    @property
    def fingerprint(self) -> str:
        """Fingerprint stamped on the lineage of every scene set this service writes."""
        return self._fingerprint

    def segment(self, record: models.AssetVersionView) -> Literal["segmented", "duplicate"]:
        """Cut, render, clip and store one version; see `VersionSegmenter.segment`."""
        path = self._locator.locate(record)
        logger.info("segmenting %s from %s", record.version, path.name)
        with tempfile.TemporaryDirectory(
            prefix="scene-rag-segmentation-", dir=self._workspace_root
        ) as workspace_name:
            workspace = Path(workspace_name)
            analysis = self._extractor.extract(path)
            rendered = self._keyframes.render(path, analysis.segments, workspace / "keyframes")
            clips = self._audio.clip(path, analysis.segments, workspace / "audio")
            scenes = tuple(
                models.Scene(
                    **dataclasses.asdict(segment),
                    keyframes=tuple(
                        models.Keyframe(
                            derivative=self._derivatives.put(frame.path, KEYFRAME_MEDIA_TYPE),
                            frame=frame.frame,
                            seconds=frame.seconds,
                        )
                        for frame in rendered.get(segment.index, ())
                    ),
                    audio_clip=None
                    if segment.index not in clips
                    else self._derivatives.put(clips[segment.index], AUDIO_MEDIA_TYPE),
                )
                for segment in analysis.segments
            )

        current = self._assets.get(record.version)
        if current is None or not current.is_live:
            raise VersionRetiredError(record.version)
        version = models.scene_set_version(
            record.version, self._fingerprint, self._config.hash_algorithm
        )
        scene_set = models.SceneSet(
            version=version,
            asset_version=record.version,
            media=analysis.media,
            scenes=scenes,
            created_at=self._clock.now(),
            lineage=models.Lineage(
                fingerprint=self._fingerprint,
                inputs=(models.ASSET_VERSION_INPUT_PREFIX + record.version,),
                version=version,
            ),
        )
        if not self._scene_writer.put(scene_set):
            logger.info("scene set %s already written for %s", version, record.version)
            return "duplicate"
        logger.info("wrote scene set %s: %d scene(s) for %s", version, len(scenes), record.version)
        return "segmented"
