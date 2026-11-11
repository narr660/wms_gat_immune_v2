"""PART 3 안전 필터 — 3단 판정, 관계의 물리적 성격에 맞는 범위로.

 1. 금지 규칙 (하드 제약): 예측과 무관하게 차단한다. 도착 시점에 관측 가능한 속성만 쓴다.
 2. 고위험 마스킹 (P(최고 등급) >= theta_r):
    - 냄새·오염: 공기·접촉으로 박스 안 전체에 퍼지므로 **박스 단위**로 막는다 (양방향).
    - 물리(지지 한계): 위에 얹을 때만 위험하므로 **배치 위치 단위**로 막는다 —
      새 상품(위)의 바닥이 충돌 상품(아래)을 덮는 자리만 차단하고 옆자리는 허용한다.
      방향은 위 -> 아래 (새 상품이 아래 상품의 지지 한계를 누르는 방향)만 본다.
 3. 위험 비용: 보정 반발강도(rho). 냄새·오염은 박스 쌍 최대, 물리는 그 자리에서
    실제로 아래에 깔리는 쌍의 최대.
"""
from typing import Dict, List

import numpy as np

from src.packing.environment import BOX
from src.schema import RELATIONS

BOX_LEVEL = ("odor", "contamination")        # 박스 단위 관계
STACK_LEVEL = "physical"                      # 쌓기(위->아래) 단위 관계


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

    def box_pair_masked(self, i: int, j: int, thresholds: Dict[str, float]) -> bool:
        """박스 단위 관계(냄새·오염)만, 양방향."""
        return any(max(self.p_high[r][i, j], self.p_high[r][j, i]) >= thresholds[r]
                   for r in BOX_LEVEL)

    def box_pair_cost(self, i: int, j: int) -> float:
        return max(max(self.rho[r][i, j], self.rho[r][j, i]) for r in BOX_LEVEL)

    def stack_masked(self, top: int, bottom: int, thresholds: Dict[str, float]) -> bool:
        """위(top)가 아래(bottom)를 누르는 방향의 물리 고위험."""
        return self.p_high[STACK_LEVEL][top, bottom] >= thresholds[STACK_LEVEL]

    def stack_cost(self, top: int, bottom: int) -> float:
        return float(self.rho[STACK_LEVEL][top, bottom])


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
        """박스 단위 판정: 금지 규칙 + 냄새·오염 마스킹."""
        new_item = self.items[new_index]
        for j in box_item_indices:
            if self.use_forbidden and forbidden_pair(new_item, self.items[j]):
                return False
            if (self.use_mask and self.risk is not None
                    and self.risk.box_pair_masked(new_index, j, self.thresholds)):
                return False
        return True

    def blocked_columns(self, new_index: int, box) -> np.ndarray:
        """물리 마스킹: 새 상품을 그 위에 얹으면 안 되는 바닥 칸들 [BOX, BOX] bool.

        충돌(위->아래 물리 고위험) 상품이 깔려 있는 칸을 True 로 표시한다.
        """
        blocked = np.zeros((BOX, BOX), dtype=bool)
        if not (self.use_mask and self.risk is not None):
            return blocked
        for p in box.placements:
            if self.risk.stack_masked(new_index, p.item_index, self.thresholds):
                w, d, _ = p.dims
                blocked[p.x:p.x + w, p.y:p.y + d] = True
        return blocked

    def under_items(self, box, spot, dims) -> List[int]:
        """이 자리에 놓으면 아래에 깔리는(바닥이 겹치는) 기존 상품들."""
        x, y, _ = spot
        w, d, _ = dims
        under = []
        for p in box.placements:
            pw, pd, _ = p.dims
            if x < p.x + pw and p.x < x + w and y < p.y + pd and p.y < y + d:
                under.append(p.item_index)
        return under

    def spot_cost(self, new_index: int, box, spot, dims) -> float:
        """냄새·오염은 박스 쌍 최대, 물리는 이 자리에서 아래에 깔리는 쌍의 최대."""
        if self.risk is None:
            return 0.0
        cost = 0.0
        for j in box.item_indices():
            cost = max(cost, self.risk.box_pair_cost(new_index, j))
        for j in self.under_items(box, spot, dims):
            cost = max(cost, self.risk.stack_cost(new_index, j))
        return cost
