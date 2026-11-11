"""방향성 상품 쌍 (i -> j)의 관계별 위험 등급을 한 번에 계산한다.

src.data.oracle.relation_risks 와 같은 규칙의 벡터화 구현이며,
tests/test_labels.py 가 두 구현의 일치를 검증한다.
"""
from typing import Dict, List

import numpy as np


def pair_labels(items: List[Dict], src: np.ndarray, dst: np.ndarray) -> Dict[str, np.ndarray]:
    def col(key, cast=float):
        return np.asarray([cast(it[key]) for it in items])

    release = np.asarray([it["odor_profile"] == "release" for it in items])
    absorb = np.asarray([it["odor_profile"] == "absorb" for it in items])
    sealed = col("sealed", bool)
    fragile = col("fragile", bool)
    electronic = col("is_electronic", bool)
    food = np.asarray([it["category"] == "식품" for it in items])
    hazard_none = np.asarray([it["hazard_type"] == "none" for it in items])
    severe = np.asarray([it["hazard_type"] in ("toxic", "corrosive") for it in items])
    level = np.asarray([float(it.get("hazard_level", 0.0)) for it in items])
    leak = col("leak_risk")
    weight = col("weight_sim")
    support = np.maximum(col("max_supported_weight_sim"), 1e-6)

    # 냄새: 방출 상품 -> 흡수 상품
    odor = np.where(release[src] & absorb[dst],
                    3 - sealed[src].astype(int) - sealed[dst].astype(int), 0)

    # 오염: 누출원 노출도 x 수신 상품 민감도
    exposure = leak[src] * np.where(sealed[src], 0.4, 1.0)
    sensitivity = 1.0 + 0.5 * food[dst] + 0.3 * electronic[dst] + 0.2 * (~sealed[dst])
    score = exposure * sensitivity * np.where(hazard_none[src], 0.5, 1.0)
    contamination = np.select(
        [severe[src] & (level[src] >= 0.7) & (score >= 0.8),
         score >= 0.65, score >= 0.35, score >= 0.15],
        [4, 3, 2, 1], default=0)

    # 물리: 올라가는 상품 무게 / 받치는 상품의 지지 한계
    ratio = weight[src] / support[dst]
    physical = np.where(
        fragile[dst],
        np.select([ratio >= 1.5, ratio >= 1.0, ratio >= 0.5], [4, 3, 2], default=1),
        np.select([ratio >= 2.0, ratio >= 1.0, ratio >= 0.5], [3, 2, 1], default=0))

    return {"odor": odor.astype(np.int64),
            "contamination": contamination.astype(np.int64),
            "physical": physical.astype(np.int64)}
