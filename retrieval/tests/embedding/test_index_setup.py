"""Preparing the index of a fingerprint: collection and record, created or verified."""

import dataclasses

import pytest

from scene_rag.retrieval.embedding import models
from scene_rag.retrieval.embedding.config import EmbeddingConfig, IndexConfig
from scene_rag.retrieval.embedding.index_setup import prepare_index
from scene_rag.retrieval.embedding.protocols import IndexMismatchError
from tests.support.embedding_fakes import (
    STEP_FP,
    FakeClock,
    FakeEmbeddingModel,
    FakeIndexStore,
    FakeSearchIndex,
    make_index,
)


def prepare(
    search_index: FakeSearchIndex,
    store: FakeIndexStore,
    *,
    model: FakeEmbeddingModel | None = None,
    config: EmbeddingConfig | None = None,
) -> models.Index:
    return prepare_index(
        model=model or FakeEmbeddingModel(),
        index_writer=search_index,
        store_reader=store,
        store_writer=store,
        config=config or EmbeddingConfig(),
        clock=FakeClock(),
    )


class TestGivenNothingPrepared:
    """An empty Search Index and ledger."""

    def test_when_prepared_then_collection_created_and_record_written(self) -> None:
        search_index, store = FakeSearchIndex(), FakeIndexStore()

        index = prepare(search_index, store)

        assert index == make_index()
        assert search_index.describe("chunks_" + STEP_FP) == models.CollectionInfo(8, "cosine", 0)
        assert store.get_index(STEP_FP) == index

    def test_when_config_differs_then_record_reflects_it(self) -> None:
        config = EmbeddingConfig(index=IndexConfig(distance="dot"))

        index = prepare(FakeSearchIndex(), FakeIndexStore(), config=config)

        assert index.distance == "dot"
        assert index.query_prefix == config.model.query_prefix


class TestGivenPreparedIndex:
    """A collection and record already there."""

    @pytest.fixture
    def prepared(self) -> tuple[FakeSearchIndex, FakeIndexStore]:
        search_index, store = FakeSearchIndex(), FakeIndexStore()
        prepare(search_index, store)
        return search_index, store

    def test_when_prepared_again_then_stored_record_returned(
        self, prepared: tuple[FakeSearchIndex, FakeIndexStore]
    ) -> None:
        search_index, store = prepared

        index = prepare(search_index, store)

        assert index == make_index()
        assert len(store.indexes()) == 1

    def test_when_collection_differs_then_mismatch(
        self, prepared: tuple[FakeSearchIndex, FakeIndexStore]
    ) -> None:
        search_index, store = prepared

        with pytest.raises(IndexMismatchError):
            prepare(search_index, store, model=FakeEmbeddingModel(dimensions=4))

    def test_when_record_differs_then_mismatch(
        self, prepared: tuple[FakeSearchIndex, FakeIndexStore]
    ) -> None:
        search_index, store = prepared
        stored = store.get_index(STEP_FP)
        assert stored is not None
        store._indexes[STEP_FP] = dataclasses.replace(stored, num_ctx=1)

        with pytest.raises(IndexMismatchError, match="other values"):
            prepare(search_index, store)
