"""고위험 마스킹 임계값 theta_r 을 검증셋에서 고른다 (KPI 4 준비).

기준: 관계별로 "검증셋에서 실제 최고 등급 쌍의 재현율 >= --recall 을 만족하는
가장 큰 theta". 테스트셋 정답은 쓰지 않는다.

    python -m scripts.select_threshold --config configs/main.yaml --seed 42
"""
import argparse

import numpy as np
import torch

from src.config import load_config
from src.models.factory import build_model
from src.pipeline import device_auto, prepare_data, run_dir, save_json
from src.schema import RELATIONS, max_grade


def theta_for_recall(probs: np.ndarray, is_high: np.ndarray, recall: float):
    """실제 최고 등급 쌍의 재현율 >= recall 을 만족하는 가장 큰 theta. 고위험이 없으면 None."""
    high_p = np.sort(probs[is_high])
    if len(high_p) == 0:
        return None
    idx = int(np.floor((1 - recall) * len(high_p)))
    return float(high_p[idx])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recall", type=float, default=0.99)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = device_auto()
    out = run_dir(cfg, "immune", args.seed)
    assert (out / "best.pt").exists(), f"먼저 immune 모델을 학습하세요: {out}"
    datasets, _, _ = prepare_data(cfg)
    model = build_model("immune", cfg.model).to(device)
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    model.eval()

    probs = {r: [] for r in RELATIONS}
    is_high = {r: [] for r in RELATIONS}
    with torch.no_grad():
        for g in datasets["val"]:
            o = model(g.x.to(device), g.edge_index.to(device))
            for r in RELATIONS:
                probs[r].append(torch.softmax(o.logits[r], -1)[:, -1].cpu().numpy())
                is_high[r].append((g.y[r] == max_grade(r)).numpy())

    thresholds = {}
    for r in RELATIONS:
        p = np.concatenate(probs[r])
        high = np.concatenate(is_high[r])
        theta = theta_for_recall(p, high, args.recall)
        if theta is None:
            thresholds[r] = 0.5
            print(f"{r}: 검증셋에 최고 등급 쌍이 없어 0.5 사용")
            continue
        recall = float((p[high] >= theta).mean())
        masked_rate = float((p >= theta).mean())
        thresholds[r] = theta
        print(f"{r:14s} theta={theta:.4f} | 검증 고위험 재현율 {recall:.4f} | "
              f"전체 쌍 중 마스킹 비율 {masked_rate:.2%}")

    save_json(out / "thresholds.json", {"seed": args.seed, "target_recall": args.recall,
                                        "thresholds": thresholds})
    print(f"저장: {out}/thresholds.json")


if __name__ == "__main__":
    main()
