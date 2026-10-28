"""Fingerprint of the segmentation step: hyperparameters plus the code that defines its output."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Final

from scene_rag.extraction.segmentation.config import SegmentationConfig

_PACKAGE_DIR = Path(__file__).parent

# The modules that decide what the step writes. Deployment code (`bootstrap`, `settings`,
# the store backends), orchestration (`scheduler`, `dispatch`) and the interfaces
# (`protocols`) are left out: a changed log message, connection timeout or worker count must
# not imply reprocessing.
_FINGERPRINTED_MODULES: Final = (
    "config.py",
    "ffmpeg_backend.py",
    "fingerprint.py",
    "models.py",
    "scenedetect_backend.py",
    "service.py",
)


def compute_fingerprint(config: SegmentationConfig) -> str:
    """Hash the hyperparameters and the output-defining modules of this subpackage.

    Only this step's code is hashed, not the whole image, so changes to other steps do not
    rerun segmentation. The step uses no model; the versions of the tools it drives (the
    detector library, OpenCV, ffmpeg) are pinned by the lockfile and the image, not hashed
    here. The sources are read at runtime, which assumes the installed files match the
    repository they were built from.

    Args:
        config: The hyperparameters in force.

    Returns:
        The first twelve hex digits of the digest.

    Raises:
        FileNotFoundError: When a fingerprinted module was renamed without updating the list.
    """
    digest = hashlib.sha256()
    digest.update(json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode())
    for name in _FINGERPRINTED_MODULES:
        digest.update(name.encode())
        digest.update((_PACKAGE_DIR / name).read_bytes())
    return digest.hexdigest()[:12]
