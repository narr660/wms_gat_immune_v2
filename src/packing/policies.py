"""탐욕 정책: 허용된 (박스, 자리) 후보에 비용을 매겨 가장 싼 것을 고른다.

비용 = risk_weight * 박스 쌍 위험 + space_weight * (z / 박스높이) + newbox_weight * 새 박스 여부
가중치는 복구된 실험 설정(위험 1.0 · 공간 0.15 · 새 박스 0.25)을 기본으로 쓴다.
"""
from dataclasses import dataclass

from src.packing.environment import BOX, PackingEnv
from src.packing.safety import SafetyFilter


@dataclass
class PolicyConfig:
    risk_weight: float = 1.0
    space_weight: float = 0.15
    newbox_weight: float = 0.25


def choose(env: PackingEnv, safety: SafetyFilter, item_index: int,
           cfg: PolicyConfig):
    """(박스 번호, 자리)를 돌려준다. 모든 기존 박스가 막히면 새 박스를 연다."""
    best, best_cost = None, float("inf")
    for box_id, spot in env.feasible_actions(item_index):
        is_new = box_id == len(env.boxes)
        box_items = [] if is_new else env.boxes[box_id].item_indices()
        if not is_new and not safety.box_allowed(item_index, box_items):
            continue
        cost = (cfg.risk_weight * safety.box_cost(item_index, box_items)
                + cfg.space_weight * spot[2] / BOX
                + cfg.newbox_weight * float(is_new))
        if cost < best_cost:
            best, best_cost = (box_id, spot), cost
    return best
