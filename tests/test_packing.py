"""적재 환경·안전 필터·정책의 핵심 동작 확인."""
import numpy as np

from src.packing.environment import BOX, Box, PackingEnv, item_dims
from src.packing.policies import PolicyConfig, choose
from src.packing.safety import PairwiseRisk, SafetyFilter, forbidden_pair
from src.schema import RELATIONS


def _item(size=(3, 3, 3), category="가전", hazard="none", level=0.0):
    return {"size": size, "category": category, "hazard_type": hazard, "hazard_level": level}


def test_heightmap_stacks_and_respects_ceiling():
    box = Box()
    box.place(0, (10, 10, 6), (0, 0, 0))
    spots = box.candidates((10, 10, 5))
    assert spots == []                                   # 6 + 5 > 10
    spot = box.best_spot((4, 4, 4))
    assert spot == (0, 0, 6)                             # 위에 쌓인다
    box.place(1, (4, 4, 4), spot)
    assert int(box.heightmap[:4, :4].max()) == 10


def test_oversized_item_goes_to_dedicated_box():
    env = PackingEnv([_item(size=(12, 3, 3))])
    assert env.feasible_actions(0) == []
    env.step(0, 0, (0, 0, 0))
    assert env.oversized == [0] and env.boxes == []


def test_forbidden_pair_blocks_box():
    items = [_item(category="식품"), _item(hazard="toxic", level=0.9)]
    env = PackingEnv(items)
    env.step(0, 0, env.feasible_actions(0)[0][1])        # 새 박스에 식품
    safety = SafetyFilter(items, use_forbidden=True, use_mask=False)
    action = choose(env, safety, 1, PolicyConfig())
    assert action[0] == 1                                # 기존 박스가 막혀 새 박스를 연다
    assert forbidden_pair(items[0], items[1])


def _risk(n, high_pairs=(), rho_pairs=()):
    src = np.repeat(np.arange(n), n - 1)
    dst = np.concatenate([np.delete(np.arange(n), i) for i in range(n)])
    rho = {r: np.zeros(len(src), dtype=np.float32) for r in RELATIONS}
    p = {r: np.zeros(len(src), dtype=np.float32) for r in RELATIONS}
    for (i, j, v) in high_pairs:
        p["physical"][(src == i) & (dst == j)] = v
    for (i, j, v) in rho_pairs:
        rho["physical"][(src == i) & (dst == j)] = v
    return PairwiseRisk(n, rho, p, np.stack([src, dst]))


def test_threshold_mask_blocks_risky_box():
    items = [_item(), _item()]
    risk = _risk(2, high_pairs=[(0, 1, 0.95)])
    env = PackingEnv(items)
    env.step(0, 0, (0, 0, 0))
    safety = SafetyFilter(items, risk, {r: 0.9 for r in RELATIONS},
                          use_forbidden=False, use_mask=True)
    action = choose(env, safety, 1, PolicyConfig())
    assert action[0] == 1                                # 마스킹 → 새 박스


def test_risk_cost_prefers_safer_existing_box():
    items = [_item(), _item(), _item()]
    risk = _risk(3, rho_pairs=[(2, 0, 0.9), (0, 2, 0.9), (2, 1, 0.05)])
    env = PackingEnv(items)
    env.step(0, 0, (0, 0, 0))                            # 박스0: 위험 높은 상대
    env.step(1, 1, (0, 0, 0))                            # 박스1: 위험 낮은 상대
    safety = SafetyFilter(items, risk, {r: 1.1 for r in RELATIONS},
                          use_forbidden=False, use_mask=False)
    action = choose(env, safety, 2, PolicyConfig(risk_weight=1.0, newbox_weight=0.25))
    assert action[0] == 1                                # rho 0.05 쪽 박스 선택

    space_only = choose(env, safety, 2, PolicyConfig(risk_weight=0.0, newbox_weight=0.25))
    assert space_only[0] in (0, 1)                       # 위험 무시 시 기존 박스 아무 곳
