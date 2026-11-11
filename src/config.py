"""실험 설정: YAML 한 파일이 실험 하나를 정의한다."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional

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
    memory_gain: Dict[str, float] = field(default_factory=lambda: {
        "odor": 0.0, "contamination": 0.0, "physical": 0.4})
    # aiNet 강화(+ 방향) 계수 gamma_up, 관계별. DCA가 이미 거의 완벽한 관계(냄새·오염)는
    # 반발강도가 분류 정확도와 별개로 움직여도 되므로, 기억이 그걸 "틀렸다"고
    # 보고 끌어당기면 오히려 해친다. 물리처럼 DCA가 실제로 부족한 관계에서만
    # 보정을 켠다. 0으로 두면 그 관계는 aiNet 기억을 받지 않는다 (DCA 그대로).
    memory_gain_relax: Dict[str, float] = field(default_factory=lambda: {"physical": 0.15})
    # aiNet 완화(- 방향) 계수 gamma_down, 관계별. 완화는 신뢰도(C, 관측 횟수)만으로
    # 세기가 걸리고 강화처럼 클론 친화도의 검증을 받지 않는다 — 그래서 과도하게
    # 걸리면 중간 등급(실제로는 등급 2·3인데 예측이 아래로 끌려 더 낮은 등급으로
    # 보이는 경우)을 해칠 수 있다. 여기 없는 관계는 memory_gain과 같은 값을 쓴다
    # (기존과 같은 대칭 동작).


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
    memory_lr_scale: float = 0.1    # 추가 학습 학습률 = lr * 이 값 (급격한 변화로 원래 가중치가 무너지는 것을 막음)


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
