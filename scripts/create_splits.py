"""시퀀스 단위 70/15/15 분할을 만든다 (재현 가능).

    python -m scripts.create_splits --dataset data/processed/rs_v3.pkl --seed 42
"""
import argparse
import json
from pathlib import Path

import numpy as np

from src.data.graphs import load_sequences


def make_split(n: int, seed: int = 42):
    ids = np.random.default_rng(seed).permutation(n).tolist()
    n_train, n_val = int(n * 0.70), int(n * 0.15)
    return {"train": ids[:n_train], "val": ids[n_train:n_train + n_val],
            "test": ids[n_train + n_val:]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="data/processed/rs_v3.pkl")
    parser.add_argument("--name", default="rs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split = make_split(len(load_sequences(args.dataset)), args.seed)
    out = Path("data/splits") / f"{args.name}_seed{args.seed}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"dataset": args.name, "seed": args.seed,
                               "n_sequences": sum(map(len, split.values())),
                               "indices": split}, indent=2), encoding="utf-8")
    print(args.name, {k: len(v) for k, v in split.items()}, "->", out)
