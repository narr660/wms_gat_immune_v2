import numpy as np
import torch

from src.config import AiNetConfig
from src.immune.ainet import AiNetMemory
from src.schema import RELATIONS

SIGS = [("a",), ("b",), ("c",)]
EDGES = torch.tensor([[0, 1, 0], [1, 0, 2]])      # a->b, b->a, a->c


def _values(odor, other=0.0):
    rest = np.full(3, other)
    return {"odor": np.asarray(odor, dtype=float), "contamination": rest, "physical": rest}


def test_unseen_pairs_have_no_memory():
    memory, coverage = AiNetMemory().lookup(SIGS, EDGES)
    assert coverage == 0.0
    for r in RELATIONS:
        assert memory[r].strength.abs().sum() == 0 and memory[r].direction.abs().sum() == 0


def test_risky_experience_strengthens_only_that_direction():
    ainet = AiNetMemory(AiNetConfig())
    # a->b 는 예측(0.5)보다 위험(1.0), b->a 는 예측(0.5)보다 안전(0.0)
    for _ in range(3):
        ainet.observe(SIGS, EDGES, expected=_values([0.5, 0.5, 0.5]),
                      exposure=_values([1.0, 0.0, 0.5]))
    memory, coverage = ainet.lookup(SIGS, EDGES)
    odor = memory["odor"]
    assert coverage == 1.0
    assert odor.direction[0] > 0 and odor.strength[0] > 0      # 강화
    assert odor.direction[1] < 0                               # 완화
    assert odor.direction[2] == 0                              # 예측과 같으면 그대로
    assert (odor.strength <= odor.confidence + 1e-6).all()     # 강도 = 친화도 x 신뢰도
    assert torch.isclose(odor.confidence[0], torch.tensor(3 / (3 + 3.0)))


def test_memory_is_keyed_by_sku_pair_not_graph_position():
    ainet = AiNetMemory()
    ainet.observe(SIGS, EDGES, _values([0.2, 0.2, 0.2]), _values([1.0, 1.0, 1.0]))
    # 다른 그래프, 다른 노드 순서지만 같은 종류의 상품 쌍
    memory, coverage = ainet.lookup([("b",), ("a",)], torch.tensor([[1], [0]]))
    assert coverage == 1.0 and memory["odor"].strength[0] > 0


def test_shared_memory_mixes_relations():
    """관계를 구분하지 않으면 냄새 경험이 오염·물리 보정으로 번진다."""
    ainet = AiNetMemory()
    for _ in range(3):
        ainet.observe(SIGS, EDGES, expected=_values([0.1, 0.1, 0.1], other=0.1),
                      exposure=_values([1.0, 1.0, 1.0], other=0.1))
    separate, _ = ainet.lookup(SIGS, EDGES)
    shared, _ = ainet.lookup(SIGS, EDGES, shared=True)
    assert separate["odor"].direction[0] > 0 and separate["physical"].direction[0] == 0
    assert shared["physical"].direction[0] > 0


def test_clone_count_is_bounded_and_values_valid():
    ainet = AiNetMemory(AiNetConfig(clone_count=5))
    for _ in range(20):
        ainet.observe(SIGS, EDGES, _values([0.6, 0.3, 0.9]), _values([0.9, 0.1, 0.9]))
    clones = ainet.stores["odor"].clones[: len(ainet.stores["odor"].rows)]
    assert clones.shape[1] == 5
    valid = clones[~np.isnan(clones)]
    assert valid.min() >= 0 and valid.max() <= 1
