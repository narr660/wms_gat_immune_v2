"""aiNet 기억을 붙인 면역 피드백 GAT를 만들고 평가한다 (KPI 2).

    python -m scripts.train_memory --config configs/main.yaml --seed 42
    python -m scripts.train_memory --config configs/main.yaml --seed 42 --shared

먼저 `scripts.train --model immune` 으로 학습한 모델이 있어야 한다.

절차
 1. 학습 시퀀스를 순서대로 지나가며 기억을 쌓는다 (조회 -> 관측 -> 갱신).
 2. 기억을 주입한 상태로 2차 GAT와 분류 헤드를 추가 학습한다.
    반발강도를 만드는 부분은 고정하므로 기억과 예측이 어긋나지 않는다.
 3. 같은 체크포인트로 테스트셋을 세 가지 조건에서 평가한다.
      off    : 기억 없음
      frozen : 학습셋 기억만 사용 (테스트에서는 조회만)
      online : 테스트 시퀀스도 예측한 뒤 관측해 기억을 갱신 (적재 중 경험 축적)

--shared 는 관계를 구분하지 않는 기억을 쓰는 비교 모델이다.
"""
import argparse
import copy

import numpy as np
import torch

from src.config import load_config
from src.evaluation.bootstrap import paired_bootstrap
from src.evaluation.metrics import collect_confusion, summarize
from src.immune.ainet import AiNetMemory, prequential_pass
from src.models.factory import build_model
from src.pipeline import device_auto, prepare_data, run_dir, save_confusion, save_json
from src.training.trainer import finetune_with_memory


def freeze_repulsion_path(model):
    """반발강도(rho_dca)를 만드는 부분을 고정한다: 1차 GAT, 반발 헤드, DCA, beta."""
    for name, p in model.named_parameters():
        p.requires_grad_(".gat2." in name or name.startswith("heads."))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shared", action="store_true",
                        help="관계를 구분하지 않는 기억 (비교 조건)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = device_auto()
    source = run_dir(cfg, "immune", args.seed)
    name = "immune_memory_shared" if args.shared else "immune_memory"
    out = run_dir(cfg, name, args.seed)
    assert (source / "best.pt").exists(), f"먼저 immune 모델을 학습하세요: {source}"
    print(f"[{cfg.experiment}] {name} seed={args.seed} device={device} -> {out}")

    datasets, _, weights = prepare_data(cfg)
    model = build_model("immune", cfg.model).to(device)
    model.load_state_dict(torch.load(source / "best.pt", map_location=device))
    freeze_repulsion_path(model)

    # 1) 학습 시퀀스로 기억 축적
    memory = AiNetMemory(cfg.ainet, args.seed)
    train_mem, train_cov = prequential_pass(model, datasets["train"], memory, device,
                                            update=True, shared=args.shared)
    val_mem, _ = prequential_pass(model, datasets["val"], memory, device,
                                  update=False, shared=args.shared)
    print(f"기억한 SKU 쌍: {memory.size()} | 학습 중 평균 기억 적용률 {np.mean(train_cov):.1%}")
    active = {r: g for r, g in cfg.model.memory_gain.items() if g > 0}
    print(f"관계별 보정 세기: {cfg.model.memory_gain} | 기억이 적용되는 관계: {list(active) or '없음'}")
    if any(g >= 0.5 for g in cfg.model.memory_gain.values()):
        print("주의: 보정 세기가 큰 관계가 있습니다. 과적합으로 off 조건이 나빠지면 "
              "finetune 결과의 regressed 를 확인하세요.")

    # 2) 기억을 주입한 추가 학습
    info = finetune_with_memory(model, datasets, {"train": train_mem, "val": val_mem},
                                weights, cfg, args.seed, out, device)
    memory.save(out / "ainet_memory.pkl")
    if info["regressed"]:
        print("기억 주입 추가 학습이 원래 체크포인트를 개선하지 못해 원래 가중치를 그대로 사용합니다.")

    # 3) 같은 체크포인트로 OFF / frozen / online 평가
    test = datasets["test"]
    frozen_mem, frozen_cov = prequential_pass(model, test, memory, device,
                                              update=False, shared=args.shared)
    online_mem, online_cov = prequential_pass(model, test, copy.deepcopy(memory), device,
                                              update=True, shared=args.shared)
    conditions = {"off": (None, [0.0]), "frozen": (frozen_mem, frozen_cov),
                  "online": (online_mem, online_cov)}
    cms, result = {}, {"model": name, "seed": args.seed, "memory_pairs": memory.size(),
                       "finetune": info, "conditions": {}}
    for label, (memories, coverage) in conditions.items():
        cms[label], rho_error = collect_confusion(model, test, device, memories)
        save_confusion(out / f"cms_test_{label}.npz", cms[label])
        result["conditions"][label] = {**summarize(cms[label]), "rho_error": rho_error,
                                       "memory_coverage": float(np.mean(coverage))}
    result["frozen_minus_off"] = paired_bootstrap(cms["frozen"], cms["off"],
                                                  cfg.eval.bootstrap, args.seed)
    result["online_minus_off"] = paired_bootstrap(cms["online"], cms["off"],
                                                  cfg.eval.bootstrap, args.seed)
    save_json(out / "ainet_metrics.json", result)

    # 모델 비교 표에는 online 조건을 이 모델의 테스트 성능으로 올린다
    save_confusion(out / "cms_test.npz", cms["online"])
    save_json(out / "metrics.json", {"model": name, "seed": args.seed,
                                     "test": result["conditions"]["online"]})
    save_json(out / "config.json", cfg.to_dict())

    print(f"{'조건':8s} {'macro-F1':>9s} {'high-recall':>12s} {'고위험 미탐지':>10s} "
          f"{'rho 오차':>8s} {'기억 적용률':>8s}")
    for label, c in result["conditions"].items():
        o = c["overall"]
        print(f"{label:8s} {o['macro_f1']:9.4f} {o['high_recall']:12.4f} "
              f"{o['high_missed']:10d} {c['rho_error']['mean']:8.4f} {c['memory_coverage']:8.1%}")


if __name__ == "__main__":
    main()
