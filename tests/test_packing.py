"""적재 환경·안전 필터·정책의 핵심 동작 확인."""
import numpy as np

from src.packing.environment import BOX, Box, PackingEnv, item_dims, orientations
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
    env.step(0, 0, (0, 0, 0), item_dims(env.items[0]))
    assert env.oversized == [0] and env.boxes == []


def test_orientations_unique_and_flat_first():
    assert len(orientations((2, 3, 4), True)) == 6
    assert len(orientations((2, 2, 4), True)) == 3
    assert orientations((5, 5, 5), True) == [(5, 5, 5)]
    assert orientations((2, 3, 4), False) == [(2, 3, 4)]
    first = orientations((2, 3, 4), True)[0]
    assert first[2] == 2 and first[0] * first[1] == 12      # 가장 납작하고 바닥 넓은 회전 먼저


def test_rotation_enables_placement_in_narrow_strip():
    """바닥에 깊이 2짜리 좁은 띠만 남은 박스: 회전 없이는 못 넣고, 회전하면 들어간다."""
    items = [_item(size=(10, 8, 10)), _item(size=(3, 3, 2))]
    for allow, expect_new_box in [(False, True), (True, False)]:
        env = PackingEnv(items, allow_rotation=allow)
        env.step(0, 0, (0, 0, 0), (10, 8, 10))               # y 0~7 가득, y 8~9 띠만 남음
        actions = env.feasible_actions(1)
        existing = [a for a in actions if a[0] == 0]
        if allow:
            assert existing and existing[0][2][1] <= 2       # 깊이 2 이하 회전으로 들어감
        else:
            assert not existing                              # (3,3,2)는 깊이 3이라 못 들어감


def test_vectorized_best_spot_matches_bruteforce():
    rng = __import__("numpy").random.default_rng(0)
    for _ in range(20):
        box = Box()
        box.heightmap[:] = rng.integers(0, 11, size=(BOX, BOX))
        dims = tuple(int(v) for v in rng.integers(1, 6, size=3))
        w, d, h = dims
        brute = []
        for x in range(BOX - w + 1):
            for y in range(BOX - d + 1):
                z = int(box.heightmap[x:x + w, y:y + d].max())
                if z + h <= BOX:
                    brute.append((x, y, z))
        expected = min(brute, key=lambda s: (s[2], s[0], s[1])) if brute else None
        assert box.best_spot(dims) == expected


def test_forbidden_pair_blocks_box():
    items = [_item(category="식품"), _item(hazard="toxic", level=0.9)]
    env = PackingEnv(items)
    _, spot, dims = env.feasible_actions(0)[0]
    env.step(0, 0, spot, dims)                            # 새 박스에 식품
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


def test_physical_mask_blocks_stacking_but_allows_side_by_side():
    """물리 고위험: 충돌 상품 '위'만 막고 옆자리는 허용한다."""
    items = [_item(size=(4, 10, 2)), _item(size=(3, 3, 3))]
    risk = _risk(2, high_pairs=[(1, 0, 0.95)])           # 1(위) -> 0(아래) 물리 고위험
    env = PackingEnv(items)
    env.step(0, 0, (0, 0, 0), (4, 10, 2))                # x 0~3 바닥 점유
    safety = SafetyFilter(items, risk, {r: 0.9 for r in RELATIONS},
                          use_forbidden=False, use_mask=True)
    action = choose(env, safety, 1, PolicyConfig())
    box_id, (x, y, z), dims = action
    assert box_id == 0 and x >= 4 and z == 0             # 같은 박스, 옆 바닥 (위가 아님)

    # 바닥이 꽉 차서 위에밖에 못 놓으면 → 새 박스
    items2 = [_item(size=(10, 10, 2)), _item(size=(3, 3, 3))]
    risk2 = _risk(2, high_pairs=[(1, 0, 0.95)])
    env2 = PackingEnv(items2)
    env2.step(0, 0, (0, 0, 0), (10, 10, 2))
    safety2 = SafetyFilter(items2, risk2, {r: 0.9 for r in RELATIONS},
                           use_forbidden=False, use_mask=True)
    action2 = choose(env2, safety2, 1, PolicyConfig())
    assert action2[0] == 1


def test_odor_mask_still_blocks_whole_box():
    items = [_item(size=(4, 10, 2)), _item(size=(3, 3, 3))]
    src = np.array([0, 1]); dst = np.array([1, 0])
    rho = {r: np.zeros(2, dtype=np.float32) for r in RELATIONS}
    p = {r: np.zeros(2, dtype=np.float32) for r in RELATIONS}
    p["odor"][:] = 0.95                                   # 냄새 고위험 (양방향)
    risk = PairwiseRisk(2, rho, p, np.stack([src, dst]))
    env = PackingEnv(items)
    env.step(0, 0, (0, 0, 0), (4, 10, 2))
    safety = SafetyFilter(items, risk, {r: 0.9 for r in RELATIONS},
                          use_forbidden=False, use_mask=True)
    action = choose(env, safety, 1, PolicyConfig())
    assert action[0] == 1                                 # 옆자리가 비어 있어도 박스 전체 금지


def test_risk_cost_prefers_safer_existing_box():
    items = [_item(), _item(), _item()]
    src = np.array([2, 0, 2, 1, 0, 1]); dst = np.array([0, 2, 1, 2, 1, 0])
    rho = {r: np.zeros(6, dtype=np.float32) for r in RELATIONS}
    p = {r: np.zeros(6, dtype=np.float32) for r in RELATIONS}
    rho["odor"][0] = rho["odor"][1] = 0.9                 # 2<->0 냄새 위험 (박스 비용)
    rho["odor"][2] = 0.05                                 # 2->1 낮음
    risk = PairwiseRisk(3, rho, p, np.stack([src, dst]))
    env = PackingEnv(items)
    env.step(0, 0, (0, 0, 0), item_dims(items[0]))       # 박스0: 위험 높은 상대
    env.step(1, 1, (0, 0, 0), item_dims(items[1]))       # 박스1: 위험 낮은 상대
    safety = SafetyFilter(items, risk, {r: 1.1 for r in RELATIONS},
                          use_forbidden=False, use_mask=False)
    action = choose(env, safety, 2, PolicyConfig(risk_weight=1.0, newbox_weight=0.25))
    assert action[0] == 1                                # rho 0.05 쪽 박스 선택

    space_only = choose(env, safety, 2, PolicyConfig(risk_weight=0.0, newbox_weight=0.25))
    assert space_only[0] in (0, 1)                       # 위험 무시 시 기존 박스 아무 곳


def test_theta_for_recall_picks_largest_threshold_meeting_target():
    from scripts.select_threshold import theta_for_recall
    probs = np.array([0.1, 0.2, 0.3, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 0.6])
    is_high = np.array([False, False, False] + [True] * 9 + [False])
    theta = theta_for_recall(probs, is_high, 1.0)
    assert theta == 0.91                                   # 전부 잡으려면 최솟값
    theta = theta_for_recall(probs, is_high, 0.9)          # 9개 중 8개(88.9%)로는 부족 -> 9개 전부
    assert (probs[is_high] >= theta).mean() >= 0.9
    assert theta_for_recall(probs, np.zeros_like(is_high), 0.99) is None
