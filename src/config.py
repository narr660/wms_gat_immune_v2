"""실험 설정: YAML 한 파일이 실험 하나를 정의한다."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class DataConfig:
    dataset: str = "data/processed/rs_v3.pkl"
    split: str = "data/splits/rs_seed42.json"
    max_nodes: int = 100
    sku_catalog: bool = True        # 같은 (분류, 치수) = 같은 SKU = 같은 속성
    observability: str = "partial"  # partial: 숨은 속성 비관측 / full: 모두 관측
    limit_train: Optional[int] = None
    limit_val: Optional[int] = None
    limit_test: Optional[int] = None


@dataclass
class ModelConfig:
    hidden_dim: int = 32
    heads: int = 4
    lambda_repulsion: float = 1.0   # 메시지 증폭 m = a(1 + lambda*rho)Wh
    initial_beta: float = 0.5       # DCA 반응 계수 초기값 (관계별 학습)
    memory_gain: float = 1.0        # aiNet 보정 계수 gamma


@dataclass
class TrainConfig:
    max_epochs: int = 12
    patience: int = 3
    lr: float = 1e-3
    weight_decay: float = 0.0
    lambda_rep_loss: float = 0.5    # 반발 보조 손실 가중치
    class_weight_cap: float = 8.0   # w_c = min(cap, sqrt(N0/Nc))
    memory_epochs: int = 4          # 기억을 주입한 상태로 추가 학습하는 epoch 수
    memory_dropout: float = 0.3     # 그 중 기억을 가리는 간선 비율 (처음 보는 쌍 대비)


@dataclass
class AiNetConfig:
    clone_count: int = 5
    learning_rate: float = 0.25
    mutation_rate: float = 0.04
    suppression_distance: float = 0.04
    tau: float = 1.0                # 방향 s = clip((O - R)/tau, -1, 1)
    kappa: float = 3.0              # 신뢰도 C = n / (n + kappa)


@dataclass
class EvalConfig:
    bootstrap: int = 2000


@dataclass
class ExperimentConfig:
    experiment: str = "main"
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    ainet: AiNetConfig = field(default_factory=AiNetConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self):
        return asdict(self)


_SECTIONS = {"data": DataConfig, "model": ModelConfig, "train": TrainConfig,
             "ainet": AiNetConfig, "eval": EvalConfig}


def load_config(path) -> ExperimentConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    kwargs = {"experiment": raw.get("experiment", "main")}
    for name, cls in _SECTIONS.items():
        section = raw.get(name) or {}
        unknown = set(section) - set(cls.__dataclass_fields__)
        if unknown:
            raise KeyError(f"{name} 설정에 알 수 없는 키: {sorted(unknown)}")
        kwargs[name] = cls(**section)
    return ExperimentConfig(**kwargs)
