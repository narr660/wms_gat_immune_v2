
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from src.constraints.regulatory_metadata import RegulatoryMetadata


@dataclass(frozen=True)
class ItemKey:
    dataset_id: str
    item_uid: str

    def __post_init__(self):
        if not self.dataset_id or not self.item_uid:
            raise ValueError("dataset_id와 item_uid는 필수입니다.")


class RegulatoryMetadataStore:
    def __init__(self):
        self._records: Dict[
            Tuple[str, str], RegulatoryMetadata
        ] = {}

    def register(
        self,
        key: ItemKey,
        metadata: RegulatoryMetadata,
    ) -> None:
        if key.item_uid != str(metadata.item_uid):
            raise ValueError("item_uid 불일치")

        record_key = (key.dataset_id, key.item_uid)

        if record_key in self._records:
            raise ValueError("이미 등록된 상품 메타데이터입니다.")

        self._records[record_key] = metadata

    def get(
        self,
        key: ItemKey,
    ) -> Optional[RegulatoryMetadata]:
        return self._records.get(
            (key.dataset_id, key.item_uid)
        )

    def __len__(self):
        return len(self._records)
