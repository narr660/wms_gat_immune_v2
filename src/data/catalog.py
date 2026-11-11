"""SKU 카탈로그: 같은 상품(분류 + 치수)은 언제 도착해도 같은 속성을 가진다.

원본 벤치마크에는 SKU 식별자가 없으므로 (category, size)를 SKU로 본다.
속성은 SKU마다 한 번만 생성하므로, 같은 SKU가 여러 시퀀스에 반복해서
나타난다. 그래야 "이 상품 쌍은 전에 이랬다"는 경험이 의미를 가진다.
"""
from typing import Dict, List, Tuple

from src.data.augment import augment_item


def sku_key(item: Dict) -> Tuple:
    return (str(item["category"]), tuple(float(v) for v in item["size"]))


class SkuCatalog:
    def __init__(self, sequences: List[List[Dict]], seed: int = 42):
        keys = sorted({sku_key(it) for seq in sequences for it in seq})
        self.ids: Dict[Tuple, int] = {key: i for i, key in enumerate(keys)}
        self.attributes: Dict[int, Dict] = {}
        for key, sku_id in self.ids.items():
            base = {"category": key[0], "size": list(key[1]),
                    "hazard_level": self._level(sequences, key)}
            attrs = augment_item(base, dataset_name="sku", seq_idx=0,
                                 item_idx=sku_id, seed=seed)
            attrs["sku_id"] = sku_id
            self.attributes[sku_id] = attrs

    @staticmethod
    def _level(sequences, key) -> float:
        for seq in sequences:
            for it in seq:
                if sku_key(it) == key:
                    return float(it.get("hazard_level", 0.0))
        return 0.0

    def __len__(self):
        return len(self.ids)

    def apply(self, sequences: List[List[Dict]]) -> List[List[Dict]]:
        """각 상품을 SKU 속성으로 바꾼다. 도착 순서와 고유 번호는 유지한다."""
        result = []
        for s, seq in enumerate(sequences):
            items = []
            for i, it in enumerate(seq):
                item = dict(self.attributes[self.ids[sku_key(it)]])
                item["arrival_index"] = int(it.get("arrival_index", i))
                item["item_uid"] = str(it.get("item_uid", f"{s}:{i}"))
                items.append(item)
            result.append(items)
        return result
