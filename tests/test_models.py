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
    cfg = ModelConfig()
    cfg.memory_gain = {"odor": 0.5, "contamination": 0.5, "physical": 0.5}
    model = build_model("immune", cfg).eval()
    e = g.edge_index.shape[1]
    memory = {"odor": EdgeMemory(torch.ones(e, 1), torch.ones(e, 1), torch.ones(e, 1))}
    base, changed = model(g.x, g.edge_index), model(g.x, g.edge_index, memory=memory)
    assert (changed.logits["odor"] - base.logits["odor"]).abs().max() > 0
    for r in ("contamination", "physical"):
        assert torch.equal(changed.logits[r], base.logits[r])


def test_memory_strengthens_and_relaxes(sequences):
    g = _graph(sequences)
    cfg = ModelConfig()
    cfg.memory_gain = {"odor": 0.5, "contamination": 0.5, "physical": 0.5}
    model = build_model("immune", cfg).eval()
    e = g.edge_index.shape[1]
    ones, zeros = torch.ones(e, 1), torch.zeros(e, 1)
    up = model(g.x, g.edge_index, memory={"odor": EdgeMemory(ones, ones, zeros)})
    down = model(g.x, g.edge_index, memory={"odor": EdgeMemory(zeros, -ones, ones)})
    assert (up.memory_delta["odor"] >= 0).all() and up.memory_delta["odor"].max() > 0
    assert (down.memory_delta["odor"] <= 0).all() and down.memory_delta["odor"].min() < 0
    assert up.memory_delta["odor"].max() <= model.memory_gain_up["odor"] + 1e-6


def test_zero_gain_relation_ignores_memory(sequences):
    """관계별 보정 세기를 0으로 두면 그 관계는 기억이 있어도 반발강도가 그대로다."""
    g = _graph(sequences)
    cfg = ModelConfig()
    cfg.memory_gain = {"odor": 0.0, "contamination": 0.5, "physical": 0.5}
    cfg.memory_gain_relax = {}
    model = build_model("immune", cfg).eval()
    e = g.edge_index.shape[1]
    ones = torch.ones(e, 1)
    base = model(g.x, g.edge_index)
    with_memory = model(g.x, g.edge_index, memory={"odor": EdgeMemory(ones, ones, torch.zeros(e, 1))})
    assert torch.equal(base.repulsion["odor"], with_memory.repulsion["odor"])
    assert "odor" not in with_memory.memory_delta


def test_relax_gain_defaults_to_reinforce_gain_when_unset(sequences):
    """memory_gain_relax 에 없는 관계는 memory_gain 과 같은 값을 써서 기존과 같은 대칭 동작이 된다."""
    g = _graph(sequences)
    cfg = ModelConfig()
    cfg.memory_gain = {"odor": 0.5, "contamination": 0.0, "physical": 0.0}
    cfg.memory_gain_relax = {}
    model = build_model("immune", cfg).eval()
    assert model.memory_gain_down["odor"] == model.memory_gain_up["odor"] == 0.5


def test_reinforce_and_relax_gain_apply_independently(sequences):
    """강화(+ 방향)와 완화(- 방향) 세기를 다르게 주면, 그 방향의 보정 크기만 달라진다."""
    g = _graph(sequences)
    cfg = ModelConfig()
    cfg.memory_gain = {"odor": 0.6, "contamination": 0.0, "physical": 0.0}      # 강화
    cfg.memory_gain_relax = {"odor": 0.1}                                       # 완화만 약하게
    model = build_model("immune", cfg).eval()
    e = g.edge_index.shape[1]
    ones, zeros = torch.ones(e, 1), torch.zeros(e, 1)

    up = model(g.x, g.edge_index, memory={"odor": EdgeMemory(ones, ones, zeros)})      # + 방향
    down = model(g.x, g.edge_index, memory={"odor": EdgeMemory(zeros, -ones, ones)})   # - 방향

    # 두 보정 모두 올바른 부호이고, 강화(0.6)가 완화(0.1)보다 또렷하게 크다
    # (rho가 이미 1에 가까운 소수 간선에서는 클리핑으로 둘 다 눌릴 수 있어 median으로 비교한다)
    assert up.memory_delta["odor"].min() > 0 and down.memory_delta["odor"].max() < 0
    assert up.memory_delta["odor"].median() > -down.memory_delta["odor"].median() * 3


def test_prequential_lookup_happens_before_observation(sequences):
    """그래프의 정답은 그 그래프의 기억 조회에 들어가지 않는다."""
    from src.immune.ainet import AiNetMemory, prequential_pass
    datasets, _ = build_datasets(sequences, {"train": [0, 0], "val": [2], "test": [3]})
    model = build_model("immune", ModelConfig()).eval()
    lookups, coverage = prequential_pass(model, datasets["train"], AiNetMemory(), "cpu")
    assert coverage[0] == 0.0 and coverage[1] == 1.0
    assert all(m.direction.abs().sum() == 0 for m in lookups[0].values())


def test_memory_pull_stops_at_remembered_target(sequences):
    """보정은 기억된 관측값(target)을 향해 끌어당기되 지나치지 않는다."""
    g = _graph(sequences)
    cfg = ModelConfig()
    cfg.memory_gain = {"odor": 1.0, "contamination": 0.0, "physical": 0.0}   # 일부러 큰 세기
    cfg.memory_gain_relax = {"odor": 1.0}
    model = build_model("immune", cfg).eval()
    e = g.edge_index.shape[1]
    ones = torch.ones(e, 1)

    base = model(g.x, g.edge_index).repulsion["odor"]
    target = torch.full((e, 1), 0.75)                        # 등급 3/4 에 해당하는 관측값

    up = model(g.x, g.edge_index,
               memory={"odor": EdgeMemory(ones, ones, torch.zeros(e, 1), target)})
    down = model(g.x, g.edge_index,
                 memory={"odor": EdgeMemory(torch.zeros(e, 1), -ones, ones, target)})

    # 위로 끌 때: target보다 낮던 rho는 target을 넘지 못하고, 이미 높던 rho는 그대로
    below = base < 0.75
    assert (up.repulsion["odor"][below] <= 0.75 + 1e-6).all()
    assert torch.allclose(up.repulsion["odor"][~below], base[~below])
    # 아래로 끌 때: target보다 높던 rho는 target 아래로 내려가지 않는다
    above = base > 0.75
    assert (down.repulsion["odor"][above] >= 0.75 - 1e-6).all()
    assert torch.allclose(down.repulsion["odor"][~above], base[~above])
