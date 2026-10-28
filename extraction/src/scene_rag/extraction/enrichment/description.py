"""The vision description enricher: what a scene shows, from its keyframes."""

import logging
from collections.abc import Sequence
from typing import Final

from scene_rag.extraction.enrichment import models
from scene_rag.extraction.enrichment.config import DescriptionConfig
from scene_rag.extraction.enrichment.fingerprint import compute_fingerprint
from scene_rag.extraction.enrichment.protocols import DerivativeReader, VisionModel

logger = logging.getLogger(__name__)

NAME: Final = "description"
CLASSIFICATION: Final = "internal"
# The modules that decide what this enricher writes; see `fingerprint.compute_fingerprint`.
FINGERPRINTED_MODULES: Final = (
    "models.py",
    "fingerprint.py",
    "description.py",
    "ollama_backend.py",
)


def pick_evenly[T](items: Sequence[T], count: int) -> list[T]:
    """Return up to ``count`` items spaced evenly over ``items``.

    Every item when there are no more than ``count``; the middle one when ``count`` is one;
    otherwise the first, the last and the ones in between at equal steps.

    Args:
        items: Candidates in order.
        count: Most items to return; at least one.
    """
    if len(items) <= count:
        return list(items)
    if count == 1:
        return [items[len(items) // 2]]
    step = (len(items) - 1) / (count - 1)
    return [items[round(position * step)] for position in range(count)]


class DescriptionEnricher:
    """`Enricher` describing each scene in prose from an even pick of its keyframes.

    One segment spans the whole scene; a scene without keyframes, or one the model has nothing
    to say about, gets none.
    """

    def __init__(
        self, *, vision: VisionModel, derivatives: DerivativeReader, config: DescriptionConfig
    ) -> None:
        """Compose the enricher; `bootstrap` is the only caller.

        Args:
            vision: The vision-language model.
            derivatives: Reader of the keyframe bytes.
            config: Hyperparameters in force.
        """
        self._vision = vision
        self._derivatives = derivatives
        self._config = config
        self._fingerprint = compute_fingerprint(config, FINGERPRINTED_MODULES, vision.model.digest)

    @property
    def name(self) -> str:
        """Return ``"description"``."""
        return NAME

    @property
    def classification(self) -> str:
        """Return the data classification of the descriptions."""
        return CLASSIFICATION

    @property
    def model(self) -> models.ModelRef:
        """Return the vision model and its digest."""
        return self._vision.model

    @property
    def fingerprint(self) -> str:
        """Return the enricher's fingerprint."""
        return self._fingerprint

    def enrich(
        self, scene_set: models.SceneSetView, scene: models.SceneView
    ) -> models.EnricherOutput:
        """Describe one scene; see `Enricher.enrich`."""
        picked = pick_evenly(scene.keyframes, self._config.max_keyframes)
        if not picked:
            logger.info("scene %d of %s has no keyframes", scene.index, scene_set.version)
            return models.EnricherOutput()
        images = [self._derivatives.read(keyframe.derivative) for keyframe in picked]
        text = self._vision.describe(images, self._config.prompt).strip()
        inputs = tuple(keyframe.derivative.digest for keyframe in picked)
        if not text:
            logger.info("no description for scene %d of %s", scene.index, scene_set.version)
            return models.EnricherOutput(inputs=inputs)
        segment = models.SemanticSegment(
            start_seconds=scene.start_seconds, end_seconds=scene.end_seconds, text=text
        )
        return models.EnricherOutput(segments=(segment,), inputs=inputs)
