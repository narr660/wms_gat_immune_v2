"""같은 체크포인트에서 aiNet 기억을 끄고 켠 차이를 측정한다 (KPI 2).

    python -m scripts.eval_ainet --config configs/main.yaml --seed 42

기억은 학습 그래프로만 만들고, 테스트에서는 조회만 한다.
"""
import argparse

import numpy as np
import torch

from src.config import load_config
from src.evaluation.bootstrap import paired_bootstrap
from src.evaluation.metrics import collect_confusion, summarize
from src.immune.ainet import build_memory_from_training
from src.models.factory import build_model
from src.schema import RELATIONS
from src.pipeline import device_auto, prepare_data, run_dir, save_confusion, save_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = device_auto()
    out = run_dir(cfg, "immune", args.seed)
    assert (out / "best.pt").exists(), f"먼저 immune 모델을 학습하세요: {out}"

    datasets, _, _ = prepare_data(cfg)
    model = build_model("immune", cfg.model).to(device)
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))

    memory = build_memory_from_training(model, datasets["train"], cfg.ainet, device, args.seed)
    memory.save(out / "ainet_memory.pkl")
    print("기억 크기(상품 쌍 종류):", memory.size())

    coverage = {r: [] for r in RELATIONS}

    def memory_fn(graph):
        edge_memory, covered = memory.lookup(graph.signatures, graph.edge_index, device)
        for r in RELATIONS:
            coverage[r].append(covered[r])
        return edge_memory

    cms_off = collect_confusion(model, datasets["test"], device)
    cms_on = collect_confusion(model, datasets["test"], device, memory_fn=memory_fn)
    save_confusion(out / "cms_test_ainet_on.npz", cms_on)

    result = {
        "seed": args.seed,
        "memory_size": memory.size(),
        "memory_coverage_by_relation": {r: float(np.mean(v)) for r, v in coverage.items()},
        "memory_coverage": float(np.mean([np.mean(v) for v in coverage.values()])),  # 기억 적용률
        "off": summarize(cms_off),
        "on": summarize(cms_on),
        "on_minus_off": paired_bootstrap(cms_on, cms_off, cfg.eval.bootstrap, args.seed),
    }
    save_json(out / "ainet_metrics.json", result)

    off, on = result["off"]["overall"], result["on"]["overall"]
    print("기억 적용률:", {r: f"{v:.1%}" for r, v in result["memory_coverage_by_relation"].items()})
    for key in ("macro_f1", "high_recall", "high_missed", "low_overestimated"):
        print(f"{key:18s} OFF {off[key]:.4f} -> ON {on[key]:.4f}")


if __name__ == "__main__":
    main()
