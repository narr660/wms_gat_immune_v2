"""그래프 어텐션 층. 외부 그래프 라이브러리 없이 PyTorch만 사용한다."""
from typing import Optional

import torch
from torch import nn


def edge_softmax(scores: torch.Tensor, target: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """같은 목적지 노드로 들어오는 간선끼리 softmax. scores: [E, H]."""
    index = target.unsqueeze(-1).expand_as(scores)
    peak = scores.new_full((num_nodes, scores.shape[1]), float("-inf"))
    peak = peak.scatter_reduce(0, index, scores, reduce="amax", include_self=True)
    weights = (scores - peak[target]).exp()
    total = scores.new_zeros((num_nodes, scores.shape[1])).index_add_(0, target, weights)
    return weights / total[target].clamp_min(1e-12)


class GATLayer(nn.Module):
    """다중 헤드 그래프 어텐션.

    gate가 주어지면 메시지를 간선별로 증폭한다:
        m_ij = alpha_ij * gate_ij * W h_i        (gate = 1 + lambda * rho)
    """

    def __init__(self, in_dim: int, out_dim: int, heads: int = 4):
        super().__init__()
        if out_dim % heads:
            raise ValueError("out_dim은 heads의 배수여야 합니다.")
        self.heads, self.head_dim = heads, out_dim // heads
        self.projection = nn.Linear(in_dim, out_dim, bias=False)
        self.att_source = nn.Parameter(torch.empty(heads, self.head_dim))
        self.att_target = nn.Parameter(torch.empty(heads, self.head_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.xavier_uniform_(self.projection.weight)
        nn.init.xavier_uniform_(self.att_source)
        nn.init.xavier_uniform_(self.att_target)
        self.last_attention: Optional[torch.Tensor] = None

    def forward(self, h, edge_index, gate: Optional[torch.Tensor] = None):
        source, target = edge_index
        n = h.shape[0]
        projected = self.projection(h).view(n, self.heads, self.head_dim)
        src_h, dst_h = projected[source], projected[target]
        scores = nn.functional.leaky_relu(
            (src_h * self.att_source).sum(-1) + (dst_h * self.att_target).sum(-1), 0.2)
        attention = edge_softmax(scores, target, n)
        self.last_attention = attention.detach()
        messages = attention.unsqueeze(-1) * src_h
        if gate is not None:
            messages = messages * gate.unsqueeze(-1)
        out = projected.new_zeros((n, self.heads, self.head_dim)).index_add_(0, target, messages)
        return out.reshape(n, -1) + self.bias


def pair_features(h, x, edge_index):
    """간선 (i -> j)의 입력: [h_i, h_j, x_i, x_j]."""
    source, target = edge_index
    return torch.cat([h[source], h[target], x[source], x[target]], dim=-1)


def mlp(in_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, out_dim))
