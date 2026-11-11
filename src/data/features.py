"""상품 특징 인코딩 (26차원)과 aiNet 기억 키.

관측 가능한 속성(치수·무게·분류·포장·표시)과 도착 시점에는 알 수 없는
속성(누출 위험, 냄새 프로필, 지지 한계)을 구분한다. 부분 관측 설정에서는
숨은 속성 자리를 0으로 둔다. 숨은 속성은 적재 후 관측된 노출 경험으로만
드러나며, 그 경험을 aiNet이 기억한다.
"""
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

CATEGORIES = ("위험물", "식품", "화학품", "일반")
HAZARD_TYPES = ("none", "flammable", "toxic", "corrosive", "oxidizing", "other")
ODOR_PROFILES = ("release", "absorb", "neutral")

CONTINUOUS = (
    "size_x", "size_y", "size_z", "volume",
    "weight", "max_supported_weight",
    "log_weight", "log_max_supported_weight",
    "hazard_level", "leak_risk",
)
BINARY = ("sealed", "fragile", "is_electronic")

FEATURE_NAMES: List[str] = (
    list(CONTINUOUS) + list(BINARY)
    + [f"category={c}" for c in CATEGORIES]
    + [f"hazard_type={h}" for h in HAZARD_TYPES]
    + [f"odor_profile={o}" for o in ODOR_PROFILES]
)
FEATURE_DIM = len(FEATURE_NAMES)
N_CONTINUOUS = len(CONTINUOUS)
assert FEATURE_DIM == 26

# 도착 시점에 상품 마스터 정보로는 알 수 없는 속성
HIDDEN_FEATURES = ("max_supported_weight", "log_max_supported_weight", "leak_risk",
                   "odor_profile=release", "odor_profile=absorb", "odor_profile=neutral")


def observable_mask(observability: str = "partial") -> np.ndarray:
    """1 = 모델에 보이는 특징. full 은 모든 속성을 본다 (상한 비교용)."""
    if observability == "full":
        return np.ones(FEATURE_DIM, dtype=np.float32)
    if observability != "partial":
        raise ValueError(f"observability 는 partial 또는 full: {observability}")
    return np.asarray([0.0 if n in HIDDEN_FEATURES else 1.0 for n in FEATURE_NAMES],
                      dtype=np.float32)


def _one_hot(value, choices: Sequence[str], field: str) -> List[float]:
    if value not in choices:
        raise ValueError(f"{field} 값을 알 수 없습니다: {value!r}")
    return [1.0 if value == c else 0.0 for c in choices]


def encode_item(item: Dict) -> np.ndarray:
    """상품 dict -> 정규화 전 26차원 벡터."""
    sx, sy, sz = (float(v) for v in item["size"])
    weight = float(item["weight_sim"])
    support = float(item["max_supported_weight_sim"])
    values = [
        sx, sy, sz, float(item.get("volume", sx * sy * sz)),
        weight, support, np.log1p(weight), np.log1p(support),
        float(item.get("hazard_level", 0.0)), float(item["leak_risk"]),
        float(bool(item["sealed"])), float(bool(item["fragile"])),
        float(bool(item["is_electronic"])),
    ]
    values += _one_hot(item["category"], CATEGORIES, "category")
    values += _one_hot(item["hazard_type"], HAZARD_TYPES, "hazard_type")
    values += _one_hot(item["odor_profile"], ODOR_PROFILES, "odor_profile")
    return np.asarray(values, dtype=np.float32)


@dataclass
class FeatureNormalizer:
    """연속 특징만 z-score 정규화한다. 통계는 학습셋에서만 계산한다."""
    mean: np.ndarray
    std: np.ndarray

    @classmethod
    def fit(cls, matrices: Sequence[np.ndarray]) -> "FeatureNormalizer":
        stacked = np.concatenate([m[:, :N_CONTINUOUS] for m in matrices], axis=0)
        std = stacked.std(axis=0)
        return cls(mean=stacked.mean(axis=0), std=np.where(std < 1e-8, 1.0, std))

    def transform(self, matrix: np.ndarray) -> np.ndarray:
        out = matrix.astype(np.float32, copy=True)
        out[:, :N_CONTINUOUS] = (out[:, :N_CONTINUOUS] - self.mean) / self.std
        return out

    def to_dict(self):
        return {"mean": self.mean.tolist(), "std": self.std.tolist()}

    @classmethod
    def from_dict(cls, data):
        return cls(mean=np.asarray(data["mean"], dtype=np.float32),
                   std=np.asarray(data["std"], dtype=np.float32))


# ---------------------------------------------------------------- 기억 키
def memory_key(item: Dict) -> tuple:
    """aiNet 기억 키 = SKU. 같은 SKU 쌍의 경험이 한곳에 쌓인다."""
    if "sku_id" in item:
        return (int(item["sku_id"]),)
    return (str(item["category"]), tuple(float(v) for v in item["size"]))
