"""The segmentation pass: cut every pending asset version into a scene set."""

import dataclasses
import logging
import tempfile
from pathlib import Path
from typing import Final

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
    MediaUnavailableError,
    SceneStoreReader,
    SceneStoreWriter,
    SegmentExtractor,
)

logger = logging.getLogger(__name__)

KEYFRAME_MEDIA_TYPE: Final = "image/jpeg"
AUDIO_MEDIA_TYPE: Final = "audio/wav"


class _Dropped(Exception):
    """The version was retired while it was being segmented."""


class SegmentationService:
    """Segment every live asset version that has no scene set under the current fingerprint.

    Pending work is derived, never flagged: a pass lists the live versions and the asset
    versions already segmented under this fingerprint, and processes the difference. Each
    version is cut, rendered and clipped in a scratch workspace; the derivatives are filed
    first and the scene set last, so a stored set never points at a file that is missing.
    Derivatives are content-addressed, so a failed or repeated run leaves nothing to undo.

    A version whose bytes cannot be read as ingestion recorded them is deferred: ingestion
    reconciles the source, after which the version is either retired or still pending. A
    version retired while being processed is dropped before the scene set is written.
    """

    def __init__(
        self,
        *,
        assets: AssetVersionReader,
        scene_reader: SceneStoreReader,
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
            assets: Read view of the Asset Registry.
            scene_reader: Query side of the Scene Store.
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
        self._scene_reader = scene_reader
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

    def run_once(self) -> models.SegmentationReport:
        """Run one pass over the pending versions.

        Returns:
            Counters of what the pass did. A failing version is counted and skipped; a failing
            listing propagates, since nothing can be decided without it.
        """
        done = self._scene_reader.list_asset_versions(self._fingerprint)
        pending = [record for record in self._assets.list_live() if record.version not in done]
        logger.info(
            "%d asset version(s) pending under fingerprint %s", len(pending), self._fingerprint
        )
        report = models.SegmentationReport()
        for record in pending:
            try:
                outcome = self._segment(record)
            except MediaUnavailableError as exc:
                logger.warning("deferred %s: %s", record.version, exc)
                outcome = "deferred"
            except _Dropped:
                logger.info("dropped %s: retired while segmenting", record.version)
                outcome = "dropped"
            except Exception:
                logger.exception("failed to segment %s", record.version)
                outcome = "failed"
            report = dataclasses.replace(report, **{outcome: getattr(report, outcome) + 1})
        logger.info("segmentation pass: %s", report)
        return report

    def _segment(self, record: models.AssetVersionView) -> str:
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
            raise _Dropped
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
