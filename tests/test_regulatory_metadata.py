
import pytest

from src.constraints.regulatory_metadata import (
    RegulatoryMetadata,
    attach_regulatory_metadata,
)


def sample_item():
    return {
        "item_uid": "TEST-001",
        "category": "화학품",
        "hazard_type": "flammable",
        "sealed": True,
    }


def test_original_item_unchanged():
    item = sample_item()
    original = dict(item)

    metadata = RegulatoryMetadata(item_uid="TEST-001")
    result = attach_regulatory_metadata(item, metadata)

    assert item == original
    assert "regulatory_metadata" not in item
    assert "regulatory_metadata" in result


def test_item_uid_mismatch():
    item = sample_item()
    metadata = RegulatoryMetadata(item_uid="OTHER-001")

    with pytest.raises(ValueError, match="item_uid"):
        attach_regulatory_metadata(item, metadata)


def test_missing_metadata():
    item = sample_item()
    result = attach_regulatory_metadata(item)

    assert result["regulatory_metadata"] is None
    assert item.get("regulatory_metadata") is None


def test_unverified_metadata_preserved():
    item = sample_item()
    metadata = RegulatoryMetadata(
        item_uid="TEST-001",
        un_number=None,
        imdg_class=None,
        source="RESEARCH_DRAFT",
        source_reference="TEST_ONLY",
        verified=False,
    )

    result = attach_regulatory_metadata(item, metadata)
    info = result["regulatory_metadata"]

    assert info["un_number"] is None
    assert info["imdg_class"] is None
    assert info["source"] == "RESEARCH_DRAFT"
    assert info["source_reference"] == "TEST_ONLY"
    assert info["verified"] is False


def test_verified_metadata_preserved():
    item = sample_item()
    metadata = RegulatoryMetadata(
        item_uid="TEST-001",
        un_number="TEST-UN",
        source="TEST_ONLY",
        source_reference="TEST-SECTION",
        verified=True,
    )

    result = attach_regulatory_metadata(item, metadata)
    info = result["regulatory_metadata"]

    assert info["un_number"] == "TEST-UN"
    assert info["verified"] is True
    assert info["source_reference"] == "TEST-SECTION"
