"""학습 루프: 그래프 단위 갱신, 검증 손실 기준 조기 종료."""
import json
import random
import time
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import torch

from src.config import ExperimentConfig
from src.evaluation.metrics import collect_confusion, summarize
from src.immune.ainet import to_device
from src.models.outputs import EdgeMemory
from src.schema import RELATIONS
from src.training.losses import total_loss


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _to_device(graph, device):
    return (graph.x.to(device), graph.edge_index.to(device),
            {r: graph.y[r].to(device) for r in RELATIONS})


@torch.no_grad()
def mean_loss(model, graphs, weights, lambda_rep, device, memories=None) -> float:
    model.eval()
    losses = []
    for i, g in enumerate(graphs):
        x, edge_index, y = _to_device(g, device)
        memory = to_device(memories[i], device) if memories is not None else None
        losses.append(float(total_loss(model(x, edge_index, memory=memory), y,
                                       weights, lambda_rep)))
    return float(np.mean(losses))


def _drop(memory, rate: float, generator):
    """일부 간선의 기억을 가린다. 처음 보는 상품 쌍에서도 예측이 무너지지 않게 한다."""
    if rate <= 0:
        return memory
    dropped = {}
    for r, m in memory.items():
        keep = (torch.rand(m.direction.shape, generator=generator) >= rate).float()
        dropped[r] = EdgeMemory(m.strength * keep, m.direction * keep, m.confidence * keep,
                                m.target)        # 가린 간선은 delta=0이라 target은 영향 없음
    return dropped


def finetune_with_memory(model, datasets, memories, weights, cfg: ExperimentConfig,
                         seed: int, out_dir: Path, device, log=print) -> Dict:
    """기억을 주입한 상태로 추가 학습한다. 모델이 보정된 반발강도를 쓰는 법을 배운다.

    memories: {"train": [...], "val": [...]} 그래프별 기억 (prequential 조회 결과)

    안전장치
      - 학습률을 본학습보다 낮춰(lr * memory_lr_scale), 이미 잘 맞춰진 2차 GAT·분류
        헤드가 몇 epoch 만에 크게 흔들리지 않게 한다.
      - 매 epoch마다 기억이 없는 조건(off)의 검증 손실도 함께 재서, 기억이 섞인
        손실만 보고 "기억에만 맞는 지점"을 고르지 않게 한다.
      - 추가 학습이 원래 체크포인트보다 나아지지 않으면(off 손실 기준) 원래
        가중치를 그대로 쓴다. 즉 추가 학습이 성능을 깎는 일은 없다.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    model.to(device)
    weights = {r: w.to(device) for r, w in weights.items()}
    lr = cfg.train.lr * cfg.train.memory_lr_scale
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=lr, weight_decay=cfg.train.weight_decay)
    generator = torch.Generator().manual_seed(seed)
    train_graphs, val_graphs = datasets["train"], datasets["val"]

    baseline_off = mean_loss(model, val_graphs, weights, cfg.train.lambda_rep_loss, device)
    best_off, history = baseline_off, []
    torch.save(model.state_dict(), out_dir / "best.pt")
    log(f"memory epoch 00 (기준, 추가 학습 전) | val(off) {baseline_off:.4f}")

    for epoch in range(1, cfg.train.memory_epochs + 1):
        model.train()
        order = np.random.default_rng(seed + 1000 + epoch).permutation(len(train_graphs))
        losses = []
        for idx in order:
            x, edge_index, y = _to_device(train_graphs[int(idx)], device)
            memory = to_device(_drop(memories["train"][int(idx)],
                                     cfg.train.memory_dropout, generator), device)
            optimizer.zero_grad(set_to_none=True)
            loss = total_loss(model(x, edge_index, memory=memory), y, weights,
                              cfg.train.lambda_rep_loss)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        val_on = mean_loss(model, val_graphs, weights, cfg.train.lambda_rep_loss,
                           device, memories["val"])
        val_off = mean_loss(model, val_graphs, weights, cfg.train.lambda_rep_loss, device)
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_loss_on": val_on, "val_loss_off": val_off})
        log(f"memory epoch {epoch:02d} | train {history[-1]['train_loss']:.4f} | "
            f"val(on) {val_on:.4f} | val(off) {val_off:.4f}")
        if val_off < best_off:          # off 기준으로만 체크포인트를 고른다
            best_off = val_off
            torch.save(model.state_dict(), out_dir / "best.pt")

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    regressed = best_off >= baseline_off
    if regressed:
        log(f"추가 학습이 기존 체크포인트를 개선하지 못함 "
            f"(off {baseline_off:.4f} -> {best_off:.4f}) — 원래 가중치를 그대로 사용")
    return {"history": history, "baseline_off_loss": baseline_off,
            "best_off_loss": best_off, "regressed": regressed}


def train_model(model, datasets: Dict[str, Sequence], weights, cfg: ExperimentConfig,
                seed: int, out_dir: Path, device, log=print) -> Dict:
    """최적 체크포인트를 out_dir/best.pt 로 저장하고 학습 이력을 돌려준다."""
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(seed)
    model.to(device)
    weights = {r: w.to(device) for r, w in weights.items()}
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)

    train_graphs, val_graphs = datasets["train"], datasets["val"]
    best_loss, best_epoch, bad_epochs, history = float("inf"), 0, 0, []
    started = time.time()

    for epoch in range(1, cfg.train.max_epochs + 1):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(len(train_graphs))
        losses = []
        for idx in order:
            x, edge_index, y = _to_device(train_graphs[int(idx)], device)
            optimizer.zero_grad(set_to_none=True)
            loss = total_loss(model(x, edge_index), y, weights, cfg.train.lambda_rep_loss)
            if not torch.isfinite(loss):
                raise RuntimeError(f"epoch {epoch}: 손실이 유한하지 않습니다.")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))

        val_loss = mean_loss(model, val_graphs, weights, cfg.train.lambda_rep_loss, device)
        val = summarize(collect_confusion(model, val_graphs, device)[0])["overall"]
        history.append({"epoch": epoch, "train_loss": float(np.mean(losses)),
                        "val_loss": val_loss, "val_macro_f1": val["macro_f1"],
                        "val_high_recall": val["high_recall"]})
        log(f"epoch {epoch:02d} | train {history[-1]['train_loss']:.4f} | "
            f"val {val_loss:.4f} | macro-F1 {val['macro_f1']:.4f} | "
            f"high-recall {val['high_recall']:.4f}")

        if val_loss < best_loss:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= cfg.train.patience:
                log(f"조기 종료: epoch {epoch}")
                break

    model.load_state_dict(torch.load(out_dir / "best.pt", map_location=device))
    info = {"best_epoch": best_epoch, "best_val_loss": best_loss,
            "seconds": time.time() - started, "history": history,
            "parameters": sum(p.numel() for p in model.parameters())}
    (out_dir / "history.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    return info
