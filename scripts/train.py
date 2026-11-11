"""모델 하나를 학습하고 검증·테스트 지표를 저장한다.

    python -m scripts.train --config configs/main.yaml --model relation --seed 42

모델: basic(기본 GAT) / relation(관계 분리형) / immune(면역 피드백) /
      immune_beta0(DCA 보정을 끈 통제 모델)
"""
import argparse

from src.config import load_config
from src.evaluation.metrics import collect_confusion, summarize
from src.models.factory import MODEL_NAMES, build_model
from src.pipeline import device_auto, prepare_data, run_dir, save_confusion, save_json
from src.training.trainer import set_seed, train_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model", required=True, choices=MODEL_NAMES)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = device_auto()
    out = run_dir(cfg, args.model, args.seed)
    print(f"[{cfg.experiment}] {args.model} seed={args.seed} device={device} -> {out}")

    datasets, normalizer, weights = prepare_data(cfg)
    print("그래프:", {k: len(v) for k, v in datasets.items()},
          "| 노드", datasets["train"][0].x.shape[0],
          "| 간선", datasets["train"][0].edge_index.shape[1])

    set_seed(args.seed)                      # 같은 시드 -> 같은 초기 가중치
    model = build_model(args.model, cfg.model)
    info = train_model(model, datasets, weights, cfg, args.seed, out, device)

    metrics = {"model": args.model, "seed": args.seed, "best_epoch": info["best_epoch"],
               "parameters": info["parameters"], "seconds": info["seconds"]}
    for split in ("val", "test"):
        cms, rho_error = collect_confusion(model, datasets[split], device)
        save_confusion(out / f"cms_{split}.npz", cms)
        metrics[split] = summarize(cms)
        metrics[split]["rho_error"] = rho_error
    if hasattr(model, "beta"):
        metrics["beta"] = {r: float(p.detach()) for r, p in model.beta.items()}

    save_json(out / "metrics.json", metrics)
    save_json(out / "config.json", cfg.to_dict())
    save_json(out / "normalizer.json", normalizer.to_dict())
    save_json(out / "class_weights.json", {r: w.tolist() for r, w in weights.items()})

    t = metrics["test"]["overall"]
    print(f"TEST | macro-F1 {t['macro_f1']:.4f} | accuracy {t['accuracy']:.4f} | "
          f"high-recall {t['high_recall']:.4f} | 고위험 미탐지 {t['high_missed']}/{t['high_total']}")


if __name__ == "__main__":
    main()
