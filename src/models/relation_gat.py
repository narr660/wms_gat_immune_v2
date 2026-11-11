"""PART 1 - 관계 분리형 GAT.

관계(냄새·오염·물리)마다 독립된 인코더를 두고, 방향성 반발강도 rho로
2차 메시지 전달을 증폭한다:  m_ij = alpha_ij (1 + lambda * rho_ij) W h_i

등급 헤드는 간선 입력 [h_i, h_j, x_i, x_j]에 그 간선의 반발강도 rho_ij를 더해
받는다. 면역이 보정한 반발강도가 해당 상품 쌍의 등급 판단에 직접 반영된다.
"""
import torch
from torch import nn

from src.models.layers import GATLayer, mlp, pair_features
from src.models.outputs import ModelOutput
from src.schema import NUM_CLASSES, RELATIONS


class RelationChannel(nn.Module):
    def __init__(self, input_dim, hidden_dim, heads, lambda_repulsion):
        super().__init__()
        self.lambda_repulsion = float(lambda_repulsion)
        pair_dim = 2 * (hidden_dim + input_dim)
        self.gat1 = GATLayer(input_dim, hidden_dim, heads)
        self.repulsion_head = mlp(pair_dim, hidden_dim, 1)
        self.gat2 = GATLayer(hidden_dim, hidden_dim, heads)

    def encode(self, x, edge_index):
        """1차 GAT -> (h1, 간선 입력, 기본 반발강도 rho)."""
        h1 = torch.relu(self.gat1(x, edge_index))
        features = pair_features(h1, x, edge_index)
        return h1, features, torch.sigmoid(self.repulsion_head(features))

    def propagate(self, h1, edge_index, repulsion):
        """반발강도로 증폭한 2차 GAT."""
        gate = 1.0 + self.lambda_repulsion * repulsion
        return torch.relu(self.gat2(h1, edge_index, gate=gate))


class RelationGAT(nn.Module):
    def __init__(self, input_dim=26, hidden_dim=32, heads=4, lambda_repulsion=1.0):
        super().__init__()
        self.channels = nn.ModuleDict({
            r: RelationChannel(input_dim, hidden_dim, heads, lambda_repulsion)
            for r in RELATIONS})
        pair_dim = 2 * (hidden_dim + input_dim)
        self.heads = nn.ModuleDict(
            {r: mlp(pair_dim + 1, hidden_dim, NUM_CLASSES[r]) for r in RELATIONS})

    def adjust(self, relation, features, base_repulsion, memory, out: ModelOutput):
        """면역 보정 지점. 관계 분리형 GAT는 보정하지 않는다."""
        return base_repulsion

    def forward(self, x, edge_index, memory=None) -> ModelOutput:
        out = ModelOutput(logits={}, embeddings={})
        for r in RELATIONS:
            channel = self.channels[r]
            h1, features, base = channel.encode(x, edge_index)
            rho = self.adjust(r, features, base, memory, out)
            h2 = channel.propagate(h1, edge_index, rho)
            out.base_repulsion[r] = base
            out.repulsion[r] = rho
            out.embeddings[r] = h2
            out.logits[r] = self.heads[r](
                torch.cat([pair_features(h2, x, edge_index), rho], dim=-1))
        return out
