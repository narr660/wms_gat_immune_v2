"""PART 2 - 면역 피드백 GAT = 관계 분리형 GAT + DCA + aiNet 보정.

DCA  : rho_dca = sigmoid(logit(rho) + beta_r * (D - S))
aiNet: rho_m4  = clip(rho_dca + gamma_up * M * max(s,0) + gamma_down * C * min(s,0), 0, 1)

강화(+ 방향)는 강도 M(클론 친화도 x 신뢰도)으로, 완화(- 방향)는 신뢰도 C만으로 세기를
건다 — C는 "몇 번 봤는가"만 볼 뿐 그 기억이 실제 경험과 일치하는지(친화도)는 보지
않으므로, 완화가 강화보다 쉽게 걸린다. 그래서 강화 세기(gamma_up)와 완화 세기
(gamma_down)를 관계별로 따로 둔다: 양끝 등급은 강화로 좋아지고 중간 등급은 과도한
완화로 나빠지는 경우, gamma_down만 낮춰서 중간 등급을 보호할 수 있다.

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
                 initial_beta=0.5, memory_gain=0.2, memory_gain_relax=None, train_beta=True):
        super().__init__(input_dim, hidden_dim, heads, lambda_repulsion)
        pair_dim = 2 * (hidden_dim + input_dim)
        self.dca = nn.ModuleDict({r: DCASignal(pair_dim, hidden_dim) for r in RELATIONS})
        self.beta = nn.ParameterDict({
            r: nn.Parameter(torch.tensor(float(initial_beta)), requires_grad=train_beta)
            for r in RELATIONS})

        # 강화(+) 세기. 스칼라를 주면 모든 관계에 같은 값을 쓴다.
        if isinstance(memory_gain, dict):
            self.memory_gain_up = {r: float(memory_gain.get(r, 0.0)) for r in RELATIONS}
        else:
            self.memory_gain_up = {r: float(memory_gain) for r in RELATIONS}

        # 완화(-) 세기. 따로 주지 않은 관계는 강화와 같은 값을 쓴다(기존과 동일한 대칭 동작).
        memory_gain_relax = memory_gain_relax or {}
        self.memory_gain_down = {
            r: float(memory_gain_relax[r]) if r in memory_gain_relax else self.memory_gain_up[r]
            for r in RELATIONS}

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

        # aiNet: 과거 경험으로 양방향 보정 (강화·완화 세기를 따로 적용)
        gain_up, gain_down = self.memory_gain_up[relation], self.memory_gain_down[relation]
        if memory is not None and relation in memory and (gain_up > 0 or gain_down > 0):
            m = memory[relation]
            direction = m.direction.clamp(-1.0, 1.0)
            delta = (
                gain_up * m.strength.clamp(0, 1) * direction.clamp(min=0)
                + gain_down * m.confidence.clamp(0, 1) * direction.clamp(max=0))
            adjusted = (rho + delta).clamp(0.0, 1.0)
            if m.target is not None:
                # 기억된 관측값을 지나치지 않는다: 등급 3(0.75)인 쌍을 위로 끌어도
                # 0.75에서 멈춰, 등급 4 영역으로 넘어가 오분류되는 것을 막는다.
                target = m.target.clamp(0.0, 1.0)
                adjusted = torch.where(delta > 0, torch.minimum(adjusted, torch.maximum(rho, target)),
                                       adjusted)
                adjusted = torch.where(delta < 0, torch.maximum(adjusted, torch.minimum(rho, target)),
                                       adjusted)
            out.memory_delta[relation] = adjusted - rho
            rho = adjusted
        return rho
