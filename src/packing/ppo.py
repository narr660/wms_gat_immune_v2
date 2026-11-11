"""PART 3 2단계 — PPO 적재 정책.

탐욕과 같은 후보·같은 안전 필터·같은 비용계 위에서, 보상 r = -C(a)의 장기 합을
최대화한다. 후보가 가변 개수이므로 후보별 특징을 점수화해 마스크드 softmax로 고른다.

- 후보 특징 (8): [새 박스 여부, z/10, 박스 채움률, 박스 상품 수/100,
                  위험 비용(spot_cost), 박스 내 최대 위험, 놓인 뒤 채움률, 회전 높이/10]
- 전역 특징 (4): [진행률, 열린 박스 수/20, 상품 부피/125, 남은 상품 수/100]
- 정책: (전역 ‖ 후보) -> MLP -> 점수 -> softmax. 가치: 전역 -> MLP -> V.
"""
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
import torch.nn as nn

from src.packing.environment import BOX, PackingEnv, item_dims  # noqa: F401
from src.packing.policies import PolicyConfig
from src.packing.safety import SafetyFilter

CAND_DIM, GLOB_DIM = 8, 4


def candidate_actions(env: PackingEnv, safety: SafetyFilter, i: int):
    """탐욕(choose)과 완전히 같은 후보 집합: 마스킹·금지 반영."""
    blocked = [safety.blocked_columns(i, box) for box in env.boxes]
    actions = []
    for box_id, spot, dims in env.feasible_actions(i, blocked):
        is_new = box_id == len(env.boxes)
        if not is_new and not safety.box_allowed(i, env.boxes[box_id].item_indices()):
            continue
        actions.append((box_id, spot, dims))
    return actions


def featurize(env: PackingEnv, safety: SafetyFilter, i: int, actions):
    n_items = len(env.items)
    w, d, h = item_dims(env.items[i])
    glob = np.array([i / n_items, len(env.boxes) / 20.0,
                     (w * d * h) / 125.0, (n_items - i) / n_items], dtype=np.float32)
    box_stats = {}                     # 박스당 1회: (채움률, 상품수, 박스 내 최대 위험)
    for box_id, _, _ in actions:
        if box_id == len(env.boxes) or box_id in box_stats:
            continue
        box = env.boxes[box_id]
        idx = box.item_indices()
        box_risk = max((safety.risk.box_pair_cost(j, k)
                        for j in idx for k in idx if j < k), default=0.0) \
            if safety.risk is not None else 0.0
        box_stats[box_id] = (box.used_volume() / float(BOX ** 3),
                             len(box.placements) / 100.0, box_risk)
    feats, costs = [], []
    for box_id, spot, dims in actions:
        is_new = box_id == len(env.boxes)
        vol = dims[0] * dims[1] * dims[2]
        fill, count, box_risk = (0.0, 0.0, 0.0) if is_new else box_stats[box_id]
        risk_cost = 0.0 if is_new else safety.spot_cost(i, env.boxes[box_id], spot, dims)
        feats.append([float(is_new), spot[2] / BOX, fill, count, risk_cost, box_risk,
                      fill + vol / float(BOX ** 3), dims[2] / BOX])
        costs.append(risk_cost)
    return glob, np.array(feats, dtype=np.float32), np.array(costs, dtype=np.float32)


def action_cost(cfg: PolicyConfig, feat_row) -> float:
    """탐욕의 비용함수와 동일 (보상 = -비용)."""
    is_new, z_norm, _, _, risk_cost = feat_row[0], feat_row[1], feat_row[2], feat_row[3], feat_row[4]
    return (cfg.risk_weight * risk_cost + cfg.space_weight * z_norm
            + cfg.newbox_weight * is_new)


class PolicyNet(nn.Module):
    def __init__(self, hidden: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(GLOB_DIM + CAND_DIM, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        self.critic = nn.Sequential(
            nn.Linear(GLOB_DIM, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def scores(self, glob: torch.Tensor, cands: torch.Tensor) -> torch.Tensor:
        g = glob.unsqueeze(0).expand(cands.shape[0], -1)
        return self.actor(torch.cat([g, cands], dim=-1)).squeeze(-1)

    def value(self, glob: torch.Tensor) -> torch.Tensor:
        return self.critic(glob).squeeze(-1)


@dataclass
class Step:
    glob: np.ndarray
    cands: np.ndarray
    action: int
    logp: float
    value: float
    reward: float


def rollout(env: PackingEnv, safety: SafetyFilter, net: PolicyNet, cfg: PolicyConfig,
            device, greedy: bool = False) -> List[Step]:
    steps = []
    for i in range(len(env.items)):
        actions = candidate_actions(env, safety, i)
        if not actions:                                   # 박스보다 큰 상품
            env.step(i, 0, (0, 0, 0))
            continue
        glob, cands, _ = featurize(env, safety, i, actions)
        g = torch.from_numpy(glob).to(device)
        c = torch.from_numpy(cands).to(device)
        with torch.no_grad():
            logits = net.scores(g, c)
            dist = torch.distributions.Categorical(logits=logits)
            a = int(torch.argmax(logits).item()) if greedy else int(dist.sample().item())
            logp = float(dist.log_prob(torch.tensor(a, device=device)).item())
            value = float(net.value(g).item())
        reward = -action_cost(cfg, cands[a])
        steps.append(Step(glob, cands, a, logp, value, reward))
        env.step(i, *actions[a])
    return steps


def gae(steps: List[Step], gamma: float = 0.99, lam: float = 0.95):
    adv, ret, last = [], [], 0.0
    next_value = 0.0
    for s in reversed(steps):
        delta = s.reward + gamma * next_value - s.value
        last = delta + gamma * lam * last
        adv.append(last)
        ret.append(last + s.value)
        next_value = s.value
    return list(reversed(adv)), list(reversed(ret))


def pad_batch(steps: List[Step], device):
    """가변 후보 수를 패딩·마스크로 배치화."""
    K = max(len(s.cands) for s in steps)
    N = len(steps)
    glob = torch.from_numpy(np.stack([s.glob for s in steps])).to(device)
    cands = torch.zeros(N, K, CAND_DIM, device=device)
    mask = torch.zeros(N, K, dtype=torch.bool, device=device)
    for n, s in enumerate(steps):
        k = len(s.cands)
        cands[n, :k] = torch.from_numpy(s.cands)
        mask[n, :k] = True
    acts = torch.tensor([s.action for s in steps], device=device)
    old_logp = torch.tensor([s.logp for s in steps], device=device)
    return glob, cands, mask, acts, old_logp


def masked_dist(net: PolicyNet, glob, cands, mask):
    N, K, _ = cands.shape
    g = glob.unsqueeze(1).expand(N, K, GLOB_DIM)
    logits = net.actor(torch.cat([g, cands], dim=-1)).squeeze(-1)
    logits = logits.masked_fill(~mask, -1e9)
    return torch.distributions.Categorical(logits=logits)


def ppo_update(net: PolicyNet, opt, episodes, device, clip=0.2, epochs=4,
               vf_coef=0.5, ent_coef=0.01, max_grad=0.5):
    steps = [s for ep in episodes for s in ep["steps"]]
    advs = np.array([a for ep in episodes for a in ep["adv"]], dtype=np.float32)
    rets = np.array([r for ep in episodes for r in ep["ret"]], dtype=np.float32)
    advs = (advs - advs.mean()) / (advs.std() + 1e-8)
    glob, cands, mask, acts, old_logp = pad_batch(steps, device)
    adv_t = torch.from_numpy(advs).to(device)
    ret_t = torch.from_numpy(rets).to(device)
    stats = {}
    for _ in range(epochs):
        dist = masked_dist(net, glob, cands, mask)
        logp = dist.log_prob(acts)
        ratio = torch.exp(logp - old_logp)
        l_pol = -torch.min(ratio * adv_t,
                           torch.clamp(ratio, 1 - clip, 1 + clip) * adv_t).mean()
        v = net.value(glob)
        l_val = ((v - ret_t) ** 2).mean()
        ent = dist.entropy().mean()
        loss = l_pol + vf_coef * l_val - ent_coef * ent
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_grad)
        opt.step()
        stats = {"policy_loss": float(l_pol.item()), "value_loss": float(l_val.item()),
                 "entropy": float(ent.item())}
    return stats
