from src.config import ModelConfig
from src.data.features import FEATURE_DIM
from src.models.basic_gat import BasicGAT
from src.models.immune_gat import ImmuneGAT
from src.models.relation_gat import RelationGAT

MODEL_NAMES = ("basic", "relation", "immune", "immune_beta0")


def build_model(name: str, cfg: ModelConfig):
    """basic: 기본 GAT / relation: 관계 분리형 / immune: 면역 피드백 /
    immune_beta0: 면역 구조에서 DCA 보정만 끈 통제 모델."""
    common = dict(input_dim=FEATURE_DIM, hidden_dim=cfg.hidden_dim, heads=cfg.heads)
    if name == "basic":
        return BasicGAT(**common)
    if name == "relation":
        return RelationGAT(lambda_repulsion=cfg.lambda_repulsion, **common)
    if name in ("immune", "immune_beta0"):
        model = ImmuneGAT(lambda_repulsion=cfg.lambda_repulsion,
                          initial_beta=cfg.initial_beta,
                          memory_gain=cfg.memory_gain,
                          memory_gain_relax=cfg.memory_gain_relax, **common)
        if name == "immune_beta0":
            model.freeze_beta_at_zero()
        return model
    raise ValueError(f"알 수 없는 모델: {name} (가능: {MODEL_NAMES})")
