
import pytest

from src.constraints.metadata_store import (
    ItemKey,
    RegulatoryMetadataStore,
)
from src.constraints.regulatory_metadata import RegulatoryMetadata


def test_same_uid_different_datasets():
    store = RegulatoryMetadataStore()

    store.register(
        ItemKey("rs", "ITEM-001"),
        RegulatoryMetadata(item_uid="ITEM-001", source="RS"),
    )
    store.register(
        ItemKey("csd", "ITEM-001"),
        RegulatoryMetadata(item_uid="ITEM-001", source="CSD"),
    )

    assert len(store) == 2
    assert store.get(ItemKey("rs", "ITEM-001")).source == "RS"
    assert store.get(ItemKey("csd", "ITEM-001")).source == "CSD"


def test_duplicate_registration_rejected():
    store = RegulatoryMetadataStore()
    key = ItemKey("rs", "ITEM-001")
    metadata = RegulatoryMetadata(item_uid="ITEM-001")

    store.register(key, metadata)

    with pytest.raises(ValueError, match="이미 등록"):
        store.register(key, metadata)


def test_missing_metadata_returns_none():
    store = RegulatoryMetadataStore()

    assert store.get(ItemKey("rs", "UNKNOWN")) is None


def test_mismatched_uid_rejected():
    store = RegulatoryMetadataStore()

    with pytest.raises(ValueError, match="item_uid"):
        store.register(
            ItemKey("rs", "ITEM-001"),
            RegulatoryMetadata(item_uid="ITEM-002"),
        )


def test_empty_key_rejected():
    with pytest.raises(ValueError):
        ItemKey("", "ITEM-001")

    with pytest.raises(ValueError):
        ItemKey("rs", "")
