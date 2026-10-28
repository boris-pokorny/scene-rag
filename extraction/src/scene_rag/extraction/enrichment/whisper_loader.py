"""Resolves and loads the faster-whisper model: the only module importing the library.

Deployment code, deliberately outside the transcript fingerprint: where a snapshot is cached,
how many threads it runs on and how a startup failure is worded change nothing about what the
model hears. The snapshot digest computed here is what the fingerprint pins.
"""

import hashlib
import logging
from pathlib import Path
from typing import Final

from scene_rag.extraction.enrichment import models
from scene_rag.extraction.enrichment.config import TranscriptionConfig
from scene_rag.extraction.enrichment.whisper_backend import FasterWhisperTranscriber, WhisperLike

logger = logging.getLogger(__name__)

# The files of a CTranslate2 Whisper snapshot that define its output: the weights, the model
# and feature-extractor configuration, and the tokenizer (one of several layouts).
_DIGESTED_FILES: Final = (
    "model.bin",
    "config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "vocabulary.json",
    "vocabulary.txt",
)
_READ_CHUNK_BYTES: Final = 1 << 20


def snapshot_digest(directory: Path, algorithm: str = "sha256") -> str:
    """Hash the output-defining files of a model snapshot; absent ones are skipped.

    Raises:
        FileNotFoundError: When the directory holds no such file.
    """
    digest = hashlib.new(algorithm)
    found = False
    for name in _DIGESTED_FILES:
        path = directory / name
        if not path.is_file():
            continue
        found = True
        digest.update(name.encode())
        with path.open("rb") as stream:
            while chunk := stream.read(_READ_CHUNK_BYTES):
                digest.update(chunk)
    if not found:
        raise FileNotFoundError(f"no model files among {_DIGESTED_FILES} in {directory}")
    return f"{algorithm}:{digest.hexdigest()}"


def load_transcriber(
    config: TranscriptionConfig, *, cache_dir: Path | None, cpu_threads: int
) -> FasterWhisperTranscriber:
    """Download (if needed) and load the configured model, computing its digest.

    Args:
        config: Hyperparameters; the model name and precision.
        cache_dir: Where snapshots are kept; the library's cache when ``None``.
        cpu_threads: Threads the model computes on; zero leaves it to the library.

    Raises:
        RuntimeError: When the library is not installed or the model cannot be fetched or
            loaded; a startup failure, never a per-scene one.
    """
    try:
        from faster_whisper import WhisperModel, download_model
    except ImportError as exc:
        raise RuntimeError(
            f"faster-whisper cannot be imported: {exc}. Install the enrichment extra with "
            "`pdm install -G enrichment`."
        ) from exc
    try:
        snapshot = Path(
            download_model(config.model, cache_dir=None if cache_dir is None else str(cache_dir))
        )
        digest = snapshot_digest(snapshot)
        model: WhisperLike = WhisperModel(
            str(snapshot), device="cpu", compute_type=config.compute_type, cpu_threads=cpu_threads
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load speech model {config.model!r}: {exc}. Check the model name and "
            "that the model cache is reachable."
        ) from exc
    logger.info("loaded speech model %s (%s)", config.model, digest)
    return FasterWhisperTranscriber(model, models.ModelRef(config.model, digest), config)
