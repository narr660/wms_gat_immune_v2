"""PPO 모듈 테스트 — 후보 일치, 보상=-비용, 마스킹 존중, 학습 전진."""
import numpy as np
import torch

from src.packing.environment import PackingEnv
from src.packing.policies import PolicyConfig, choose
from src.packing.ppo import (PolicyNet, action_cost, candidate_actions,
                             featurize, gae, ppo_update, rollout)
from src.packing.safety import SafetyFilter
from src.schema import RELATIONS

from src.packing.safety import PairwiseRisk

def _item(size=(3, 3, 3), category="가전", hazard="none", level=0.0):
    return {"size": size, "category": category, "hazard_type": hazard, "hazard_level": level}

def _risk(n, high_pairs=(), rho_pairs=()):
    import numpy as _np
    src = _np.repeat(_np.arange(n), n - 1)
    dst = _np.concatenate([_np.delete(_np.arange(n), i) for i in range(n)])
    rho = {r: _np.zeros(len(src), dtype=_np.float32) for r in RELATIONS}
    p = {r: _np.zeros(len(src), dtype=_np.float32) for r in RELATIONS}
    for (i, j, v) in high_pairs:
        p["physical"][(src == i) & (dst == j)] = v
    for (i, j, v) in rho_pairs:
        rho["physical"][(src == i) & (dst == j)] = v
    return PairwiseRisk(n, rho, p, _np.stack([src, dst]))


THR = {r: 0.5 for r in RELATIONS}


def setup(n=12, seed=0, high_frac=0.2):
    """합성 상품 n개 + 무작위 물리 High 쌍 비율 high_frac."""
    rng = np.random.default_rng(seed)
    items = [_item(size=tuple(int(v) for v in rng.integers(2, 6, 3))) for _ in range(n)]
    pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    chosen = rng.choice(len(pairs), int(high_frac * len(pairs)), replace=False)
    high = []
    for k in chosen:
        i, j = pairs[k]
        high.append((i, j, 0.9)); high.append((j, i, 0.9))
    risk = _risk(n, high_pairs=high, rho_pairs=[(i, j, 0.8) for i, j, _ in high])
    return items, SafetyFilter(items, risk, THR)


def test_candidates_match_greedy_choice():
    """탐욕이 고른 행동은 항상 PPO 후보 집합 안에 있다 (같은 마스킹·금지)."""
    items, safety = setup()
    env = PackingEnv(items)
    for i in range(len(items)):
        actions = candidate_actions(env, safety, i)
        a = choose(env, safety, i, PolicyConfig())
        if a is None:
            assert not actions
            env.step(i, 0, (0, 0, 0))
        else:
            assert any(a[0] == b and tuple(a[1]) == tuple(s) for b, s, _ in actions)
            env.step(i, *a)


def test_reward_equals_negative_greedy_cost():
    items, safety = setup()
    env = PackingEnv(items)
    cfg = PolicyConfig()
    actions = candidate_actions(env, safety, 0)
    _, cands, _ = featurize(env, safety, 0, actions)
    new_idx = [k for k, (b, _, _) in enumerate(actions) if b == len(env.boxes)]
    assert new_idx
    assert abs(action_cost(cfg, cands[new_idx[0]]) - cfg.newbox_weight) < 1e-6


def test_rollout_places_everything_and_respects_blocks():
    items, safety = setup(n=16, high_frac=0.4)
    env = PackingEnv(items)
    net = PolicyNet()
    torch.manual_seed(0)
    rollout(env, safety, net, PolicyConfig(), torch.device("cpu"), greedy=True)
    assert sum(len(b.placements) for b in env.boxes) + len(env.oversized) == len(items)
    # 물리 High 쌍이 겹쳐 쌓이지 않았는지 (쌓기 단위 마스킹 존중)
    from scripts.run_packing import _stacked_pairs
    for box in env.boxes:
        for top, bottom in _stacked_pairs(box):
            assert not safety.risk.stack_masked(top, bottom, THR), (top, bottom)


def test_ppo_update_improves_reward():
    """작은 문제에서 PPO 업데이트가 그리디 평균 보상을 떨어뜨리지 않고 올린다."""
    torch.manual_seed(0)
    net = PolicyNet(hidden=32)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    cfg = PolicyConfig()
    device = torch.device("cpu")

    def make_batch(k=6, base=100):
        eps = []
        for e in range(k):
            items, safety = setup(n=10, seed=base + e, high_frac=0.3)
            env = PackingEnv(items)
            steps = rollout(env, safety, net, cfg, device, greedy=False)
            adv, ret = gae(steps)
            eps.append({"steps": steps, "adv": adv, "ret": ret})
        return eps

    def mean_reward():
        vals = []
        for e in range(8):
            items, safety = setup(n=10, seed=500 + e, high_frac=0.3)
            env = PackingEnv(items)
            steps = rollout(env, safety, net, cfg, device, greedy=True)
            vals.append(sum(s.reward for s in steps))
        return float(np.mean(vals))

    before = mean_reward()
    for it in range(12):
        ppo_update(net, opt, make_batch(base=100 + 10 * it), device, epochs=4)
    after = mean_reward()
    assert after >= before - 0.3, f"보상 퇴보: {before:.2f} -> {after:.2f}"
