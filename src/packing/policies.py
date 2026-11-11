"""탐욕 정책: 허용된 (박스, 자리) 후보에 비용을 매겨 가장 싼 것을 고른다.

비용 = risk_weight * 박스 쌍 위험 + space_weight * (z / 박스높이) + newbox_weight * 새 박스 여부
가중치 기본값은 스윕으로 확정한 균형점: 위험 1.0 · 공간 0.15 · 새 박스 1.0 (물리 재현율 99%와 함께,
시드 42 테스트 315개에서 박스 9.82 · 활용률 44.6% · 고위험 동적재 14건 — 프런티어의 무릎).
"""
from dataclasses import dataclass

from src.packing.environment import BOX, PackingEnv
from src.packing.safety import SafetyFilter


@dataclass
class PolicyConfig:
    risk_weight: float = 1.0
    space_weight: float = 0.15
    newbox_weight: float = 1.0


def choose(env: PackingEnv, safety: SafetyFilter, item_index: int,
           cfg: PolicyConfig):
    """(박스 번호, 자리, 회전된 치수)를 돌려준다. 모든 기존 박스가 막히면 새 박스를 연다.

    물리 '쌓기 금지' 칸(blocked_columns)을 자리 탐색에 먼저 반영하므로, 충돌 상품
    옆자리는 후보로 남고 그 위만 제외된다.
    """
    blocked_by_box = [safety.blocked_columns(item_index, box) for box in env.boxes]
    best, best_cost = None, float("inf")
    for box_id, spot, dims in env.feasible_actions(item_index, blocked_by_box):
        is_new = box_id == len(env.boxes)
        if not is_new and not safety.box_allowed(item_index, env.boxes[box_id].item_indices()):
            continue
        risk_cost = 0.0 if is_new else safety.spot_cost(item_index, env.boxes[box_id], spot, dims)
        cost = (cfg.risk_weight * risk_cost
                + cfg.space_weight * spot[2] / BOX
                + cfg.newbox_weight * float(is_new))
        if cost < best_cost:
            best, best_cost = (box_id, spot, dims), cost
    return best
