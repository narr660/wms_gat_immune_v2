"""안전-효율 스윕: (물리 마스킹 재현율 목표 x 새 박스 비용) 격자로 프런티어를 그린다.

기준(결정됨): 고위험 동적재 <= --max-colocated (기본 0) 를 제약으로 놓고,
그 안에서 박스 수가 가장 적은 조합을 균형점으로 고른다. 탈락한 조합도 표에 남겨
"동적재를 조금 허용하면 박스가 얼마나 주는지"(프런티어 기울기)를 함께 보고한다.

빠르게 돌리기 위해 모델 추론(쌍별 rho, P(High))은 시퀀스당 한 번만 하고 모든 조합이
재사용한다. 그래서 스윕에서는 aiNet 기억을 frozen(학습셋 기억 고정)으로 쓴다 —
고리 ②(시퀀스 단위 갱신)는 조합마다 경로가 달라져 재사용이 깨지기 때문. 균형점
확정 후 본 실행(run_packing)은 고리 ②를 켠 값으로 보고한다.

    python -m scripts.sweep_packing --config configs/main.yaml --seed 42
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from scripts.run_packing import evaluate_boxes, pack_sequence, sequence_risk
from scripts.select_threshold import theta_for_recall
from src.config import load_config
from src.data.catalog import SkuCatalog
from src.data.graphs import _ordered_items, load_sequences, load_split
from src.immune.ainet import AiNetMemory
from src.models.factory import build_model
from src.packing.policies import PolicyConfig
from src.packing.safety import SafetyFilter
from src.pipeline import device_auto, prepare_data, run_dir, save_json
from src.schema import RELATIONS, max_grade

RECALL_GRID = [0.99, 0.97, 0.95, 0.90]      # 물리 마스킹 재현율 목표 (냄새·오염은 0.99 고정)
NEWBOX_GRID = [0.25, 0.5, 1.0, 2.0]          # 새 박스 비용


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-colocated", type=int, default=0,
                        help="허용하는 고위험 동적재 합 (기본 0 = 제약)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = device_auto()
    model_dir = run_dir(cfg, "immune", args.seed)
    model = build_model("immune", cfg.model).to(device)
    model.load_state_dict(torch.load(model_dir / "best.pt", map_location=device))
    model.eval()
    memory_path = Path("outputs") / cfg.experiment / f"immune_memory_seed{args.seed}" / "ainet_memory.pkl"
    memory = AiNetMemory.load(memory_path) if memory_path.exists() else None
    print(f"aiNet 기억(frozen): {'불러옴' if memory else '없음'}")

    datasets, _, _ = prepare_data(cfg)
    graphs_by_id = {g.sequence_id: g for g in datasets["test"]}
    raw = load_sequences(cfg.data.dataset)
    if cfg.data.sku_catalog:
        raw = SkuCatalog(raw).apply(raw)
    test_ids = load_split(cfg.data.split)["test"][: args.limit or None]

    # 1) 검증셋 확률 한 번 수집 -> 재현율 목표별 물리 임계값
    probs = {r: [] for r in RELATIONS}
    is_high = {r: [] for r in RELATIONS}
    with torch.no_grad():
        for g in datasets["val"]:
            o = model(g.x.to(device), g.edge_index.to(device))
            for r in RELATIONS:
                probs[r].append(torch.softmax(o.logits[r], -1)[:, -1].cpu().numpy())
                is_high[r].append((g.y[r] == max_grade(r)).numpy())
    probs = {r: np.concatenate(v) for r, v in probs.items()}
    is_high = {r: np.concatenate(v) for r, v in is_high.items()}
    base_theta = {r: theta_for_recall(probs[r], is_high[r], 0.99) or 0.5 for r in RELATIONS}
    phys_theta = {rec: theta_for_recall(probs["physical"], is_high["physical"], rec) or 0.5
                  for rec in RECALL_GRID}
    for rec, th in phys_theta.items():
        rate = float((probs["physical"] >= th).mean())
        print(f"물리 재현율 {rec:.0%} -> theta {th:.4f} | 마스킹 비율 {rate:.2%}")

    # 2) 테스트 시퀀스별 위험 테이블(frozen 기억)을 한 번만 계산
    per_seq = []
    with torch.no_grad():
        for sid in test_ids:
            graph = graphs_by_id[sid]
            items = _ordered_items(raw[sid], cfg.data.max_nodes)
            risk, _ = sequence_risk(model, graph, device, memory)
            per_seq.append((sid, items, risk))
    print(f"위험 테이블 준비: {len(per_seq)}개 시퀀스")

    # 3) 격자 평가
    rows = []
    for recall, newbox in itertools.product(RECALL_GRID, NEWBOX_GRID):
        thresholds = dict(base_theta)
        thresholds["physical"] = phys_theta[recall]
        policy = PolicyConfig(risk_weight=1.0, newbox_weight=newbox)
        agg = dict(exposure=0.0, high=0, forbidden=0, boxes=[], util=[])
        for sid, items, risk in per_seq:
            safety = SafetyFilter(items, risk, thresholds)
            env = pack_sequence(items, safety, policy)
            exposure, high_pairs, forbidden = evaluate_boxes(env, items)
            agg["exposure"] += exposure
            agg["high"] += high_pairs
            agg["forbidden"] += forbidden
            agg["boxes"].append(len(env.boxes))
            agg["util"].append(env.utilization())
        rows.append(dict(recall=recall, theta=round(phys_theta[recall], 4), newbox=newbox,
                         high_colocated=agg["high"], forbidden=agg["forbidden"],
                         exposure=round(agg["exposure"], 1),
                         boxes_mean=round(float(np.mean(agg["boxes"])), 2),
                         utilization=round(float(np.mean(agg["util"])), 4)))
        r = rows[-1]
        print(f"재현율 {recall:.0%} x 새박스 {newbox:<4} | 동적재 {r['high_colocated']:>5} | "
              f"박스 {r['boxes_mean']:>6.2f} | 활용률 {r['utilization']:.1%}")

    # 4) 균형점: 제약 만족 중 박스 최소
    ok = [r for r in rows if r["high_colocated"] <= args.max_colocated and r["forbidden"] == 0]
    chosen = min(ok, key=lambda r: r["boxes_mean"]) if ok else None

    out_dir = Path("reports") / cfg.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [f"# 안전-효율 스윕 — 시드 {args.seed}, 테스트 {len(per_seq)}개 시퀀스, "
             f"aiNet 기억 frozen, 회전 켜짐", "",
             f"제약: 고위험 동적재 <= {args.max_colocated} · 금지 위반 = 0, 목표: 박스 수 최소", "",
             "| 물리 재현율 목표 | theta | 새 박스 비용 | 고위험 동적재 | 금지 위반 | 박스 수 | 활용률 | 선택 |",
             "|---|---|---|---|---|---|---|---|"]
    for r in sorted(rows, key=lambda r: (r["high_colocated"] > args.max_colocated, r["boxes_mean"])):
        mark = "★ 균형점" if chosen is r else ("가능" if r in ok else "")
        lines.append(f"| {r['recall']:.0%} | {r['theta']} | {r['newbox']} | {r['high_colocated']:,} | "
                     f"{r['forbidden']:,} | {r['boxes_mean']:.2f} | {r['utilization']:.1%} | {mark} |")
    if chosen:
        lines += ["", f"균형점: 물리 재현율 {chosen['recall']:.0%} (theta {chosen['theta']}) x "
                  f"새 박스 비용 {chosen['newbox']} — 박스 {chosen['boxes_mean']:.2f}개, "
                  f"활용률 {chosen['utilization']:.1%}, 동적재 {chosen['high_colocated']}건"]
    else:
        lines += ["", "제약을 만족하는 조합이 없음 — --max-colocated 를 올려 다시 보거나 격자를 넓힌다."]
    (out_dir / f"sweep_seed{args.seed}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    save_json(out_dir / f"sweep_seed{args.seed}.json",
              {"seed": args.seed, "constraint_max_colocated": args.max_colocated,
               "rows": rows, "chosen": chosen})
    print("\n".join(lines))
    print(f"저장: {out_dir}/sweep_seed{args.seed}.md")


if __name__ == "__main__":
    main()
