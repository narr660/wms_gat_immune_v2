import numpy as np
import torch

from src.config import AiNetConfig
from src.immune.ainet import AiNetMemory
from src.schema import RELATIONS

NODES = [("a",), ("b",), ("c",)]
SIGS = {r: NODES for r in RELATIONS}
EDGES = torch.tensor([[0, 1, 0], [1, 0, 2]])      # a->b, b->a, a->c


def _values(odor):
    zeros = np.zeros(3)
    return {"odor": np.asarray(odor, dtype=float), "contamination": zeros, "physical": zeros}


def test_unseen_pairs_have_no_memory():
    memory, coverage = AiNetMemory().lookup(SIGS, EDGES)
    for r in RELATIONS:
        assert coverage[r] == 0.0
        assert memory[r].strength.abs().sum() == 0 and memory[r].direction.abs().sum() == 0


def test_risky_experience_strengthens_only_that_direction():
    ainet = AiNetMemory(AiNetConfig())
    # a->b 는 예측(0.5)보다 위험(1.0), b->a 는 예측(0.5)보다 안전(0.0)
    for _ in range(3):
        ainet.observe(SIGS, EDGES, expected=_values([0.5, 0.5, 0.5]),
                      exposure=_values([1.0, 0.0, 0.5]))
    memory, coverage = ainet.lookup(SIGS, EDGES)
    odor = memory["odor"]
    assert coverage["odor"] == 1.0
    assert odor.direction[0] > 0 and odor.strength[0] > 0      # 강화
    assert odor.direction[1] < 0 and odor.strength[1] == 0     # 완화 (기억은 늘지 않음)
    assert odor.direction[2] == 0                              # 예측과 같으면 그대로
    assert torch.isclose(odor.confidence[0], torch.tensor(3 / (3 + 3.0)))


def test_memory_is_keyed_by_signature_not_item():
    ainet = AiNetMemory()
    ainet.observe(SIGS, EDGES, _values([0.2, 0.2, 0.2]), _values([1.0, 1.0, 1.0]))
    # 다른 그래프, 다른 노드 순서지만 같은 종류의 상품 쌍
    other = {r: [("b",), ("a",)] for r in RELATIONS}
    memory, coverage = ainet.lookup(other, torch.tensor([[1], [0]]))
    assert coverage["odor"] == 1.0 and memory["odor"].strength[0] > 0


def test_clone_count_is_bounded_and_values_valid():
    ainet = AiNetMemory(AiNetConfig(clone_count=5))
    for _ in range(20):
        ainet.observe(SIGS, EDGES, _values([0.6, 0.3, 0.9]), _values([0.9, 0.1, 0.9]))
    clones = ainet.stores["odor"].clones[: len(ainet.stores["odor"].rows)]
    assert clones.shape[1] == 5
    valid = clones[~np.isnan(clones)]
    assert valid.min() >= 0 and valid.max() <= 1
