
from dataclasses import dataclass, asdict
from typing import Optional, Dict


@dataclass
class RegulatoryMetadata:
    item_uid: str

    # IMDG 관련 정보
    un_number: Optional[str] = None
    imdg_class: Optional[str] = None
    subsidiary_hazard: Optional[str] = None
    packing_group: Optional[str] = None
    segregation_code: Optional[str] = None

    # 국내 위험물 관련 정보
    kr_dg_class: Optional[str] = None
    quantity: Optional[float] = None
    designated_quantity: Optional[float] = None
    regulatory_applicability: Optional[bool] = None

    # 메타데이터의 출처와 검증 상태
    source: Optional[str] = None
    source_reference: Optional[str] = None
    verified: bool = False


def attach_regulatory_metadata(
    item: Dict,
    metadata: Optional[RegulatoryMetadata] = None,
) -> Dict:
    """
    원본 상품을 수정하지 않고 규정 메타데이터 결합.

    메타데이터가 없는 경우에도 임의의 법적 분류를 생성하지 않는다.
    """
    result = dict(item)

    if metadata is None:
        result["regulatory_metadata"] = None
        return result

    if str(item["item_uid"]) != str(metadata.item_uid):
        raise ValueError("item_uid 불일치")

    result["regulatory_metadata"] = asdict(metadata)
    return result
