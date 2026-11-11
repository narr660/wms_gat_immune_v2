"""상품 특징 인코딩 (26차원)과 aiNet 기억용 상품 시그니처.

레이블 규칙이 사용하는 속성을 모두 포함한다. 도착 순서처럼 위험과
무관한 값은 넣지 않는다.
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


# ---------------------------------------------------------------- 시그니처
# aiNet 기억은 개별 상품이 아니라 "어떤 종류의 상품인가"로 저장한다.
# 그래야 처음 보는 상품에도 같은 종류의 경험을 적용할 수 있다.
# 관계마다 그 위험과 관련된 속성만 키로 쓴다 (관계별 기억).
_LEAK_BINS = np.array([0.15, 0.35, 0.65, 0.8])
_WEIGHT_BINS = np.array([1.0, 3.0, 8.0, 20.0, 50.0])


def item_signatures(item: Dict) -> Dict[str, tuple]:
    return {
        "odor": (item["odor_profile"], bool(item["sealed"])),
        "contamination": (
            item["hazard_type"], float(item.get("hazard_level", 0.0)) >= 0.7,
            int(np.digitize(float(item["leak_risk"]), _LEAK_BINS)),
            bool(item["sealed"]), item["category"] == "식품", bool(item["is_electronic"])),
        "physical": (
            bool(item["fragile"]),
            int(np.digitize(float(item["weight_sim"]), _WEIGHT_BINS)),
            int(np.digitize(float(item["max_supported_weight_sim"]), _WEIGHT_BINS))),
    }
