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
    limit_train: Optional[int] = None
    limit_val: Optional[int] = None
    limit_test: Optional[int] = None


@dataclass
class ModelConfig:
    hidden_dim: int = 32
    heads: int = 4
    lambda_repulsion: float = 1.0   # 메시지 증폭 m = a(1 + lambda*rho)Wh
    initial_beta: float = 0.5       # DCA 반응 계수 초기값 (관계별 학습)
    memory_gain: float = 0.2        # aiNet 보정 상한 gamma


@dataclass
class TrainConfig:
    max_epochs: int = 12
    patience: int = 3
    lr: float = 1e-3
    weight_decay: float = 0.0
    lambda_rep_loss: float = 0.5    # 반발 보조 손실 가중치
    class_weight_cap: float = 8.0   # w_c = min(cap, sqrt(N0/Nc))


@dataclass
class AiNetConfig:
    clone_count: int = 5
    learning_rate: float = 0.25
    mutation_rate: float = 0.04
    suppression_distance: float = 0.04
    tau: float = 0.25               # 방향 s = clip((O - R)/tau, -1, 1)
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
