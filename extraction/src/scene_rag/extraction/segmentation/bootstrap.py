"""Composition root of the segmentation step.

Binds settings from the environment and hyperparameters from `SegmentationConfig`, then wires
the registry view, the stores, the locator and the media backends into `SegmentationService`.
Nothing else reads the environment.
"""

import contextlib
import logging
import os
import shutil
import time
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

from pymongo import MongoClient
from pymongo.errors import PyMongoError

from scene_rag.extraction.segmentation import settings as settings_module
from scene_rag.extraction.segmentation.asset_registry import Document, MongoAssetVersionReader
from scene_rag.extraction.segmentation.config import SegmentationConfig
from scene_rag.extraction.segmentation.derivative_store import FilesystemDerivativeStore
from scene_rag.extraction.segmentation.ffmpeg_backend import FfmpegAudioClipper, SubprocessRunner
from scene_rag.extraction.segmentation.file_locator import FileUriLocator
from scene_rag.extraction.segmentation.models import truncate_to_millisecond
from scene_rag.extraction.segmentation.scene_store import MongoSceneStore
from scene_rag.extraction.segmentation.service import SegmentationService
from scene_rag.extraction.segmentation.settings import SegmentationSettings

# Silence ffmpeg decoder warnings (e.g. "mmco: unref short failure" on imperfect h264
# streams): a tool knob, not a setting. It must be set before OpenCV is loaded by the backend.
os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "-8")

from scene_rag.extraction.segmentation.scenedetect_backend import (  # noqa: E402
    ContentSegmentExtractor,
    SceneDetectKeyframeRenderer,
)

logger = logging.getLogger(__name__)

# Operational guardrails, not tunables: the connectivity check's patience, and the longest
# pause between retries once passes keep failing.
_SERVER_SELECTION_TIMEOUT_MS: Final = 2000
_MAX_BACKOFF_SECONDS: Final = 300.0
# Subdirectory of the derivative root holding the scratch workspaces of versions in progress.
WORKSPACE_DIRECTORY: Final = ".workspace"

type ClientFactory = Callable[..., MongoClient[Document]]


class SystemClock:
    """`Clock` over the system time, truncated to the millisecond the stores keep."""

    def now(self) -> datetime:
        """Return the current UTC time at millisecond precision."""
        return truncate_to_millisecond(datetime.now(UTC))


def _require(environ: Mapping[str, str], name: str, hint: str) -> str:
    value = environ.get(name)
    if not value:
        raise RuntimeError(f"{name} is not set. {hint}")
    return value


def load_settings(environ: Mapping[str, str] | None = None) -> SegmentationSettings:
    """Bind `SegmentationSettings` from the environment.

    The poll interval is optional: a single pass never sleeps. `run_worker` demands it before
    looping. The ffmpeg binary defaults to the one on the path.

    Args:
        environ: Variables to read; the process environment by default.

    Returns:
        Validated settings.

    Raises:
        RuntimeError: When a variable is missing or the settings do not validate.
    """
    env = os.environ if environ is None else environ
    poll_interval = env.get(settings_module.POLL_INTERVAL_SECONDS_ENV)
    mongo_hint = "Start MongoDB and point this at it, e.g. mongodb://localhost:27017."
    try:
        return SegmentationSettings(
            registry_uri=_require(env, settings_module.REGISTRY_URI_ENV, mongo_hint),
            registry_database=_require(
                env, settings_module.REGISTRY_DATABASE_ENV, "Name the registry database."
            ),
            scene_store_uri=_require(env, settings_module.SCENE_STORE_URI_ENV, mongo_hint),
            scene_store_database=_require(
                env, settings_module.SCENE_STORE_DATABASE_ENV, "Name the scene store database."
            ),
            derivative_root=Path(
                _require(
                    env,
                    settings_module.DERIVATIVE_ROOT_ENV,
                    "Point it at the directory to file keyframes and audio clips under.",
                )
            ).resolve(),
            ffmpeg_binary=env.get(settings_module.FFMPEG_BINARY_ENV) or "ffmpeg",
            poll_interval_seconds=None if not poll_interval else float(poll_interval),
        )
    except ValueError as exc:
        raise RuntimeError(f"invalid settings: {exc}") from exc


def connect_mongo(
    uri: str, *, redacted: str, client_factory: ClientFactory = MongoClient
) -> MongoClient[Document]:
    """Connect to a MongoDB and verify it answers.

    Args:
        uri: The connection string.
        redacted: The connection string as it may appear in logs and errors.
        client_factory: Builds the client from the connection string and keyword options;
            `MongoClient` itself outside tests.

    Returns:
        A connected client; the caller owns it and closes it, e.g. with ``with``.

    Raises:
        RuntimeError: When the connection string is malformed or the server cannot be
            reached.
    """
    try:
        client = client_factory(
            uri, tz_aware=True, serverSelectionTimeoutMS=_SERVER_SELECTION_TIMEOUT_MS
        )
        client.admin.command("ping")
    except (PyMongoError, ValueError) as exc:  # pymongo raises ValueError for a bad port
        raise RuntimeError(
            f"Could not reach MongoDB at {redacted!r}: {exc}. Check that it is running and the "
            "connection string is correct."
        ) from exc
    logger.info("connected to MongoDB at %s", redacted)
    return client


def require_ffmpeg(binary: str) -> None:
    """Check that the ffmpeg executable can be found, before any version is attempted.

    Args:
        binary: Name looked up on the path, or a path to the executable.

    Raises:
        RuntimeError: When nothing executable is found; every version would fail otherwise,
            pass after pass.
    """
    if shutil.which(binary) is None:
        raise RuntimeError(
            f"ffmpeg not found as {binary!r}. Install it on the path or point "
            f"{settings_module.FFMPEG_BINARY_ENV} at the executable."
        )


def create_service(
    settings: SegmentationSettings,
    registry_client: MongoClient[Document],
    scene_client: MongoClient[Document],
    config: SegmentationConfig | None = None,
) -> SegmentationService:
    """Assemble the step from its reference backends.

    Mongo registry view and Scene Store, filesystem Derivative Store, file locator,
    PySceneDetect cuts and keyframes, ffmpeg clips, system clock. Scratch workspaces go
    under ``<derivative root>/.workspace``, the store's own filesystem, so filing a rendered
    file is a rename; the directory is created here.
    """
    config = SegmentationConfig() if config is None else config
    scene_store = MongoSceneStore(scene_client, settings.scene_store_database)
    workspace_root = settings.derivative_root / WORKSPACE_DIRECTORY
    workspace_root.mkdir(parents=True, exist_ok=True)
    return SegmentationService(
        assets=MongoAssetVersionReader(registry_client, settings.registry_database),
        scene_reader=scene_store,
        scene_writer=scene_store,
        derivatives=FilesystemDerivativeStore(settings.derivative_root, config.hash_algorithm),
        locator=FileUriLocator(),
        extractor=ContentSegmentExtractor(config),
        keyframes=SceneDetectKeyframeRenderer(config),
        audio=FfmpegAudioClipper(config, SubprocessRunner(), settings.ffmpeg_binary),
        clock=SystemClock(),
        config=config,
        workspace_root=workspace_root,
    )


def _backoff_seconds(interval: float, failures: int) -> float:
    """Pause after ``failures`` consecutive failed passes: doubling from the interval, capped."""
    return min(interval * 2.0**failures, max(interval, _MAX_BACKOFF_SECONDS))


def run_worker(*, once: bool, sleep: Callable[[float], None] = time.sleep) -> None:
    """Entry point of the ``segmentation`` step in the worker role.

    A pass that raises (a store outage, say) ends the process in ``once`` mode. In the loop it
    is logged and the next pass runs after a pause that doubles with every consecutive
    failure, from the poll interval up to a few minutes.

    Args:
        once: Run a single pass instead of looping until interrupted.
        sleep: Pause function between passes; injectable for tests.

    Raises:
        RuntimeError: When settings are missing, the loop has no poll interval, ffmpeg cannot
            be found or a store is unreachable.
    """
    settings = load_settings()
    interval = settings.poll_interval_seconds
    if not once and interval is None:
        raise RuntimeError(
            f"{settings_module.POLL_INTERVAL_SECONDS_ENV} is not set. "
            "Give the pause between passes in seconds, or run with --once."
        )
    require_ffmpeg(settings.ffmpeg_binary)
    with contextlib.ExitStack() as stack:
        registry_client = stack.enter_context(
            connect_mongo(settings.registry_uri, redacted=settings.redacted_registry_uri)
        )
        scene_client = stack.enter_context(
            connect_mongo(settings.scene_store_uri, redacted=settings.redacted_scene_store_uri)
        )
        service = create_service(settings, registry_client, scene_client)
        logger.info("segmentation fingerprint %s", service.fingerprint)
        if once:
            service.run_once()
            return
        assert interval is not None  # checked above, before connecting
        failures = 0
        while True:
            try:
                service.run_once()
            except Exception:
                pause = _backoff_seconds(interval, failures)
                failures += 1
                logger.exception("segmentation pass failed; retrying in %.1f s", pause)
            else:
                pause = interval
                failures = 0
            sleep(pause)
