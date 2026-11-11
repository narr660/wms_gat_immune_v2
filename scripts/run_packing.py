"""PART 3 (1단계): 테스트 시퀀스를 네 조건으로 적재하고 안전·효율 지표를 비교한다.

    python -m scripts.run_packing --config configs/main.yaml --seed 42
    python -m scripts.run_packing --config configs/main.yaml --seed 42 --limit 50   # 빠른 확인

조건
  space        안전 장치 없음. 공간만 보는 탐욕 (기준선)
  filter       + 금지 규칙 차단 + P(최고 등급) 마스킹
  risk         + 보정 반발강도를 박스 비용으로 (위험 낮은 박스 선호)
  risk_memory  + aiNet 기억: 학습셋 기억으로 시작하고, 시퀀스 하나를 다 적재할 때마다
                 같은 박스에 담겼던 쌍의 관측 노출로 기억을 갱신한다 (고리 ②,
                 시퀀스 단위). 이후 시퀀스의 rho 가 달라진다.

지표 (시퀀스 평균 또는 합)
  risk_exposure   같은 박스 쌍의 실제 등급/최고 등급 합 (관계 합산) — 위험 노출
  high_colocated  실제 최고 등급 쌍이 같은 박스에 담긴 수 — 고위험 동적재
  forbidden       금지 규칙 위반 쌍 수
  boxes, utilization, oversized
"""
import argparse
import copy
from pathlib import Path

import numpy as np
import torch

from src.config import load_config
from src.data.labels import pair_labels
from src.immune.ainet import AiNetMemory, to_device
from src.models.factory import build_model
from src.packing.environment import PackingEnv
from src.packing.policies import PolicyConfig, choose
from src.packing.safety import PairwiseRisk, SafetyFilter, forbidden_pair
from src.pipeline import device_auto, prepare_data, run_dir, save_json
from src.schema import RELATIONS, max_grade

CONDITIONS = {
    "space":       dict(forbidden=False, mask=False, risk_cost=False, memory=False),
    "filter":      dict(forbidden=True,  mask=True,  risk_cost=False, memory=False),
    "risk":        dict(forbidden=True,  mask=True,  risk_cost=True,  memory=False),
    "risk_memory": dict(forbidden=True,  mask=True,  risk_cost=True,  memory=True),
}


@torch.no_grad()
def sequence_risk(model, graph, device, memory: AiNetMemory = None):
    mem = None
    if memory is not None:
        mem, _ = memory.lookup(graph.keys, graph.edge_index)
        mem = to_device(mem, device)
    out = model(graph.x.to(device), graph.edge_index.to(device), memory=mem)
    rho, p_high = {}, {}
    for r in RELATIONS:
        rho[r] = out.repulsion[r].squeeze(-1).cpu().numpy()
        p_high[r] = torch.softmax(out.logits[r], -1)[:, -1].cpu().numpy()
    n = graph.x.shape[0]
    return PairwiseRisk(n, rho, p_high, graph.edge_index.numpy()), out


def pack_sequence(items, safety, policy_cfg, allow_rotation=True):
    env = PackingEnv(items, allow_rotation=allow_rotation)
    for i in range(len(items)):
        action = choose(env, safety, i, policy_cfg)
        if action is None:                 # 박스보다 큰 상품
            env.step(i, 0, (0, 0, 0))
            continue
        env.step(i, *action)
    return env


def _stacked_pairs(box):
    """바닥이 겹치고 위에 있는 (위, 아래) 상품 쌍 — 물리 위험이 실제로 걸리는 쌍."""
    pairs = []
    ps = box.placements
    for i, top in enumerate(ps):
        for bottom in ps[:i] + ps[i + 1:]:
            if top.z < bottom.z + bottom.dims[2]:
                continue                                     # 위에 있지 않음
            if (top.x < bottom.x + bottom.dims[0] and bottom.x < top.x + top.dims[0]
                    and top.y < bottom.y + bottom.dims[1] and bottom.y < top.y + top.dims[1]):
                pairs.append((top.item_index, bottom.item_index))
    return pairs


def evaluate_boxes(env: PackingEnv, items):
    """위험 노출: 냄새·오염은 같은 박스 쌍(양방향), 물리는 겹쳐 쌓인 쌍(위->아래)만."""
    exposure, high_pairs, forbidden = 0.0, 0, 0
    for box in env.boxes:
        idx = box.item_indices()
        if len(idx) < 2:
            continue
        src, dst = np.meshgrid(idx, idx, indexing="ij")
        keep = src != dst
        src, dst = src[keep], dst[keep]
        labels = pair_labels(items, src, dst)
        for r in ("odor", "contamination"):
            exposure += float(labels[r].sum()) / max_grade(r)
            high_pairs += int((labels[r] == max_grade(r)).sum())
        stacked = _stacked_pairs(box)
        if stacked:
            tops = np.array([t for t, _ in stacked])
            bottoms = np.array([b for _, b in stacked])
            phys = pair_labels(items, tops, bottoms)["physical"]
            exposure += float(phys.sum()) / max_grade("physical")
            high_pairs += int((phys == max_grade("physical")).sum())
        for a, b in zip(src, dst):
            if a < b and forbidden_pair(items[a], items[b]):
                forbidden += 1
    return exposure, high_pairs, forbidden


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None, help="테스트 시퀀스 수 제한")
    parser.add_argument("--no-rotation", action="store_true", help="축 정렬 회전 끄기 (비교용)")
    parser.add_argument("--conditions", default=None,
                        help="쉼표로 조건 선택 (예: space,risk). 기본: 전부")
    args = parser.parse_args()

    cfg = load_config(args.config)
    device = device_auto()
    model_dir = run_dir(cfg, "immune", args.seed)
    assert (model_dir / "best.pt").exists(), f"먼저 immune 모델을 학습하세요: {model_dir}"
    assert (model_dir / "thresholds.json").exists(), \
        "먼저 python -m scripts.select_threshold 를 실행하세요."
    import json
    thresholds = json.loads((model_dir / "thresholds.json").read_text())["thresholds"]

    datasets, _, _ = prepare_data(cfg)
    graphs_by_id = {g.sequence_id: g for g in datasets["test"]}
    model = build_model("immune", cfg.model).to(device)
    model.load_state_dict(torch.load(model_dir / "best.pt", map_location=device))
    model.eval()

    base_memory = None
    memory_path = Path("outputs") / cfg.experiment / f"immune_memory_seed{args.seed}" / "ainet_memory.pkl"
    if memory_path.exists():
        base_memory = AiNetMemory.load(memory_path)
        print(f"aiNet 기억 불러옴: {memory_path} (SKU 쌍 {base_memory.size()})")

    # 원본 상품 dict 가 필요하므로 시퀀스를 다시 만든다 (prepare_data와 같은 경로)
    from src.data.catalog import SkuCatalog
    from src.data.graphs import load_sequences, load_split, _ordered_items
    raw = load_sequences(cfg.data.dataset)
    if cfg.data.sku_catalog:
        raw = SkuCatalog(raw).apply(raw)
    split = load_split(cfg.data.split)
    test_ids = split["test"][: args.limit or None]

    policy = PolicyConfig()
    selected = {k: v for k, v in CONDITIONS.items()
                if args.conditions is None or k in args.conditions.split(",")}
    assert selected, f"--conditions 가 비었습니다: {args.conditions}"
    rotation = not args.no_rotation
    print(f"회전: {'켜짐 (축 정렬)' if rotation else '꺼짐'} | 조건: {list(selected)}")
    results = {name: [] for name in selected}
    for name, flags in selected.items():
        memory = copy.deepcopy(base_memory) if flags["memory"] and base_memory else None
        if flags["memory"] and base_memory is None:
            print("경고: ainet_memory.pkl 이 없어 risk_memory 조건은 기억 없이 시작합니다.")
            memory = AiNetMemory(cfg.ainet, args.seed)
        for sid in test_ids:
            graph = graphs_by_id[sid]
            items = _ordered_items(raw[sid], cfg.data.max_nodes)
            risk, out = sequence_risk(model, graph, device,
                                      memory if flags["memory"] else None)
            safety = SafetyFilter(items, risk if flags["mask"] or flags["risk_cost"] else None,
                                  thresholds, use_forbidden=flags["forbidden"],
                                  use_mask=flags["mask"])
            local_policy = policy if flags["risk_cost"] else PolicyConfig(risk_weight=0.0)
            env = pack_sequence(items, safety, local_policy, allow_rotation=rotation)
            exposure, high_pairs, forbidden = evaluate_boxes(env, items)
            results[name].append(dict(
                sequence=sid, exposure=exposure, high_colocated=high_pairs,
                forbidden=forbidden, boxes=len(env.boxes),
                utilization=env.utilization(), oversized=len(env.oversized)))
            if flags["memory"]:        # 고리 ②: 같은 박스 쌍의 관측으로 기억 갱신
                expected = {r: risk.rho[r][graph.edge_index[0], graph.edge_index[1]]
                            for r in RELATIONS}
                exposure_obs = {r: (graph.y[r].numpy() / max_grade(r)) for r in RELATIONS}
                co = np.zeros(graph.edge_index.shape[1], dtype=bool)
                in_box = {i: b for b, box in enumerate(env.boxes) for i in box.item_indices()}
                src, dst = graph.edge_index.numpy()
                for e in range(len(src)):
                    co[e] = in_box.get(src[e], -1) == in_box.get(dst[e], -2)
                if co.any():
                    sel = graph.edge_index[:, torch.from_numpy(co)]
                    memory.observe(graph.keys, sel,
                                   {r: expected[r][co] for r in RELATIONS},
                                   {r: exposure_obs[r][co] for r in RELATIONS})

    out_dir = Path("reports") / cfg.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    base_name = "space" if "space" in results else next(iter(results))
    base_exposure = np.mean([r["exposure"] for r in results[base_name]]) or 1.0
    lines = [f"# PART 3 (1단계) 적재 결과 — 시드 {args.seed}, 테스트 {len(test_ids)}개 시퀀스, "
             f"회전 {'켜짐' if rotation else '꺼짐'}", "",
             f"| 조건 | 위험 노출 지수 ({base_name}=100) | 고위험 동적재 (합) | 금지 위반 (합) | 박스 수 (평균) | 활용률 (평균) |",
             "|---|---|---|---|---|---|"]
    for name, rows in results.items():
        summary[name] = {
            "exposure_mean": float(np.mean([r["exposure"] for r in rows])),
            "exposure_index": float(np.mean([r["exposure"] for r in rows]) / base_exposure * 100),
            "high_colocated": int(sum(r["high_colocated"] for r in rows)),
            "forbidden": int(sum(r["forbidden"] for r in rows)),
            "boxes_mean": float(np.mean([r["boxes"] for r in rows])),
            "utilization_mean": float(np.mean([r["utilization"] for r in rows])),
            "oversized": int(sum(r["oversized"] for r in rows)),
        }
        s = summary[name]
        lines.append(f"| {name} | {s['exposure_index']:.1f} | {s['high_colocated']:,} | "
                     f"{s['forbidden']:,} | {s['boxes_mean']:.2f} | {s['utilization_mean']:.1%} |")
    suffix = "" if rotation else "_norot"
    save_json(out_dir / f"packing_seed{args.seed}{suffix}.json",
              {"seed": args.seed, "n_sequences": len(test_ids),
               "thresholds": thresholds, "summary": summary, "per_sequence": results})
    (out_dir / f"packing_seed{args.seed}{suffix}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"저장: {out_dir}/packing_seed{args.seed}{suffix}.md")


if __name__ == "__main__":
    main()
