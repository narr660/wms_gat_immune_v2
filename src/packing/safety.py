"""PART 3 안전 필터 — 3단 판정.

 1. 금지 규칙 (하드 제약): 예측과 무관하게 차단한다. 도착 시점에 관측 가능한 속성만 쓴다.
 2. 고위험 마스킹: 어느 관계든 P(최고 등급) >= theta_r 인 쌍이 생기는 박스는 후보에서 뺀다.
 3. 위험 비용: 보정 반발강도(rho)를 박스 후보의 비용으로 더한다 (정책이 낮은 쪽을 선호).

판정은 양방향 쌍 (신규 -> 기존, 기존 -> 신규) 모두 본다.
"""
from typing import Dict, List

import numpy as np

from src.schema import RELATIONS


def forbidden_pair(a: Dict, b: Dict) -> bool:
    """관측 가능한 속성만으로 판정하는 WMS 금지 규칙 (1단계 합성 규정).

    독성·부식성 고위험(hazard_level >= 0.7) 상품은 식품과 같은 박스에 담지 않는다.
    실제 운영에서는 이 자리에 검증된 규정 카탈로그(src/constraints)가 들어간다.
    """
    def severe(item):
        return item["hazard_type"] in ("toxic", "corrosive") and float(item.get("hazard_level", 0.0)) >= 0.7

    def food(item):
        return item["category"] == "식품"

    return (severe(a) and food(b)) or (severe(b) and food(a))


class PairwiseRisk:
    """한 시퀀스의 쌍별 예측값 테이블. [N, N] 배열로 보관한다 (대각선 0)."""

    def __init__(self, n: int, rho: Dict[str, np.ndarray], p_high: Dict[str, np.ndarray],
                 edge_index: np.ndarray):
        self.rho = {r: np.zeros((n, n), dtype=np.float32) for r in RELATIONS}
        self.p_high = {r: np.zeros((n, n), dtype=np.float32) for r in RELATIONS}
        src, dst = edge_index
        for r in RELATIONS:
            self.rho[r][src, dst] = rho[r]
            self.p_high[r][src, dst] = p_high[r]

    def pair_masked(self, i: int, j: int, thresholds: Dict[str, float]) -> bool:
        return any(max(self.p_high[r][i, j], self.p_high[r][j, i]) >= thresholds[r]
                   for r in RELATIONS)

    def pair_cost(self, i: int, j: int) -> float:
        return max(max(self.rho[r][i, j], self.rho[r][j, i]) for r in RELATIONS)


class SafetyFilter:
    def __init__(self, items: List[Dict], risk: PairwiseRisk = None,
                 thresholds: Dict[str, float] = None,
                 use_forbidden=True, use_mask=True):
        self.items = items
        self.risk = risk
        self.thresholds = thresholds
        self.use_forbidden = use_forbidden
        self.use_mask = use_mask

    def box_allowed(self, new_index: int, box_item_indices: List[int]) -> bool:
        new_item = self.items[new_index]
        for j in box_item_indices:
            if self.use_forbidden and forbidden_pair(new_item, self.items[j]):
                return False
            if (self.use_mask and self.risk is not None
                    and self.risk.pair_masked(new_index, j, self.thresholds)):
                return False
        return True

    def box_cost(self, new_index: int, box_item_indices: List[int]) -> float:
        """박스 안 기존 상품들과의 최대 쌍 위험 (rho)."""
        if self.risk is None or not box_item_indices:
            return 0.0
        return max(self.risk.pair_cost(new_index, j) for j in box_item_indices)
