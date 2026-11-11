import torch

from src.config import ModelConfig
from src.data.graphs import build_datasets
from src.models.factory import build_model
from src.models.layers import edge_softmax
from src.models.outputs import EdgeMemory
from src.schema import NUM_CLASSES, RELATIONS


def _graph(sequences):
    datasets, _ = build_datasets(sequences, {"train": [0, 1], "val": [2], "test": [3]})
    return datasets["train"][0]


def test_edge_softmax_sums_to_one_per_target():
    target = torch.tensor([0, 0, 1, 1, 1])
    weights = edge_softmax(torch.randn(5, 4), target, num_nodes=2)
    sums = torch.zeros(2, 4).index_add_(0, target, weights)
    assert torch.allclose(sums, torch.ones(2, 4), atol=1e-5)


def test_all_models_output_expected_shapes(sequences):
    g = _graph(sequences)
    for name in ("basic", "relation", "immune", "immune_beta0"):
        out = build_model(name, ModelConfig())(g.x, g.edge_index)
        for r in RELATIONS:
            assert out.logits[r].shape == (g.edge_index.shape[1], NUM_CLASSES[r])
            assert torch.isfinite(out.logits[r]).all()


def test_beta_zero_equals_relation_gat_computation(sequences):
    """DCA를 끄고 기억을 주지 않으면 관계 분리형 GAT와 같은 반발강도를 쓴다."""
    g = _graph(sequences)
    torch.manual_seed(0)
    model = build_model("immune_beta0", ModelConfig()).eval()
    out = model(g.x, g.edge_index)
    for r in RELATIONS:
        assert torch.allclose(out.repulsion[r], out.base_repulsion[r], atol=1e-5)


def test_same_seed_gives_identical_initial_weights_for_control():
    torch.manual_seed(7)
    a = build_model("immune", ModelConfig())
    torch.manual_seed(7)
    b = build_model("immune_beta0", ModelConfig())
    for (name, pa), (_, pb) in zip(a.named_parameters(), b.named_parameters()):
        if not name.startswith("beta"):
            assert torch.equal(pa, pb), name


def test_memory_only_changes_its_own_relation(sequences):
    """냄새 기억은 냄새 예측만 바꾸고, 오염·물리는 그대로 둔다."""
    g = _graph(sequences)
    model = build_model("immune", ModelConfig()).eval()
    e = g.edge_index.shape[1]
    memory = {"odor": EdgeMemory(torch.ones(e, 1), torch.ones(e, 1), torch.ones(e, 1))}
    base, changed = model(g.x, g.edge_index), model(g.x, g.edge_index, memory=memory)
    assert (changed.logits["odor"] - base.logits["odor"]).abs().max() > 0
    for r in ("contamination", "physical"):
        assert torch.equal(changed.logits[r], base.logits[r])


def test_memory_strengthens_and_relaxes(sequences):
    g = _graph(sequences)
    model = build_model("immune", ModelConfig()).eval()
    e = g.edge_index.shape[1]
    ones, zeros = torch.ones(e, 1), torch.zeros(e, 1)
    up = model(g.x, g.edge_index, memory={"odor": EdgeMemory(ones, ones, zeros)})
    down = model(g.x, g.edge_index, memory={"odor": EdgeMemory(zeros, -ones, ones)})
    assert (up.memory_delta["odor"] >= 0).all() and up.memory_delta["odor"].max() > 0
    assert (down.memory_delta["odor"] <= 0).all() and down.memory_delta["odor"].min() < 0
    assert up.memory_delta["odor"].max() <= model.memory_gain + 1e-6
