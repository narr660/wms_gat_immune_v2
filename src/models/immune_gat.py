"""PART 2 - 면역 피드백 GAT = 관계 분리형 GAT + DCA + aiNet 보정.

DCA  : rho_dca = sigmoid(logit(rho) + beta_r * (D - S))
aiNet: rho_m4  = clip(rho_dca + gamma * (M * max(s,0) + C * min(s,0)), 0, 1)

beta를 0으로 고정하고 기억을 주지 않으면 관계 분리형 GAT와 같은 계산이 된다.
"""
from typing import Dict, Optional

import torch
from torch import nn

from src.models.outputs import EdgeMemory, ModelOutput
from src.models.relation_gat import RelationGAT
from src.schema import RELATIONS


class DCASignal(nn.Module):
    """상품 쌍의 위험(Danger)·안전(Safe) 신호. DCA에서 착안한 구조."""

    def __init__(self, pair_dim, hidden_dim):
        super().__init__()
        self.shared = nn.Sequential(nn.Linear(pair_dim, hidden_dim), nn.ReLU())
        self.danger = nn.Linear(hidden_dim, 1)
        self.safe = nn.Linear(hidden_dim, 1)

    def forward(self, features):
        hidden = self.shared(features)
        return torch.sigmoid(self.danger(hidden)), torch.sigmoid(self.safe(hidden))


class ImmuneGAT(RelationGAT):
    def __init__(self, input_dim=26, hidden_dim=32, heads=4, lambda_repulsion=1.0,
                 initial_beta=0.5, memory_gain=0.2, train_beta=True):
        super().__init__(input_dim, hidden_dim, heads, lambda_repulsion)
        pair_dim = 2 * (hidden_dim + input_dim)
        self.dca = nn.ModuleDict({r: DCASignal(pair_dim, hidden_dim) for r in RELATIONS})
        self.beta = nn.ParameterDict({
            r: nn.Parameter(torch.tensor(float(initial_beta)), requires_grad=train_beta)
            for r in RELATIONS})
        self.memory_gain = float(memory_gain)

    def freeze_beta_at_zero(self):
        """통제 조건: DCA 보정을 끈다 (나머지 구조·초기값은 동일)."""
        for p in self.beta.values():
            p.requires_grad_(False)
            p.data.zero_()

    def adjust(self, relation, features, base_repulsion,
               memory: Optional[Dict[str, EdgeMemory]], out: ModelOutput):
        danger, safe = self.dca[relation](features)
        out.danger[relation], out.safe[relation] = danger, safe

        # DCA: 현재 위험 신호로 보정
        safe_rho = base_repulsion.clamp(1e-6, 1 - 1e-6)
        rho = torch.sigmoid(torch.logit(safe_rho) + self.beta[relation] * (danger - safe))
        out.dca_repulsion[relation] = rho

        # aiNet: 과거 경험으로 양방향 보정
        if memory is not None and relation in memory:
            m = memory[relation]
            direction = m.direction.clamp(-1.0, 1.0)
            delta = self.memory_gain * (
                m.strength.clamp(0, 1) * direction.clamp(min=0)
                + m.confidence.clamp(0, 1) * direction.clamp(max=0))
            adjusted = (rho + delta).clamp(0.0, 1.0)
            out.memory_delta[relation] = adjusted - rho
            rho = adjusted
        return rho
