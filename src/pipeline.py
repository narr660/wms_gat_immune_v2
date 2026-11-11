"""실험 공통 절차: 데이터 준비, 실행 폴더, 결과 저장."""
import json
from pathlib import Path

import numpy as np
import torch

from src.config import ExperimentConfig
from src.data.graphs import build_datasets, load_sequences, load_split
from src.training.losses import class_weights


def device_auto() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def prepare_data(cfg: ExperimentConfig):
    sequences = load_sequences(cfg.data.dataset)
    split = load_split(cfg.data.split)
    limits = {"train": cfg.data.limit_train, "val": cfg.data.limit_val,
              "test": cfg.data.limit_test}
    datasets, normalizer = build_datasets(sequences, split, cfg.data.max_nodes, limits)
    weights = class_weights(datasets["train"], cfg.train.class_weight_cap)
    return datasets, normalizer, weights


def run_dir(cfg: ExperimentConfig, model_name: str, seed: int, root="outputs") -> Path:
    return Path(root) / cfg.experiment / f"{model_name}_seed{seed}"


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def save_confusion(path: Path, cms):
    np.savez_compressed(path, **cms)


def load_confusion(path: Path):
    with np.load(path) as data:
        return {k: data[k] for k in data.files}
