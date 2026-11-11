"""기본 GAT: 하나의 공통 인코더 + 관계별 분류 헤드."""
import torch
from torch import nn

from src.models.layers import GATLayer, mlp, pair_features
from src.models.outputs import ModelOutput
from src.schema import NUM_CLASSES, RELATIONS


class BasicGAT(nn.Module):
    def __init__(self, input_dim=26, hidden_dim=32, heads=4):
        super().__init__()
        self.gat1 = GATLayer(input_dim, hidden_dim, heads)
        self.gat2 = GATLayer(hidden_dim, hidden_dim, heads)
        pair_dim = 2 * (hidden_dim + input_dim)
        self.heads = nn.ModuleDict(
            {r: mlp(pair_dim, hidden_dim, NUM_CLASSES[r]) for r in RELATIONS})

    def forward(self, x, edge_index, memory=None) -> ModelOutput:
        h = torch.relu(self.gat1(x, edge_index))
        h = torch.relu(self.gat2(h, edge_index))
        features = pair_features(h, x, edge_index)
        return ModelOutput(
            logits={r: self.heads[r](features) for r in RELATIONS},
            embeddings={r: h for r in RELATIONS},
        )
