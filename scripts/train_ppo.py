"""PART 3 2단계 — PPO 학습.

보상 = 탐욕 비용함수의 부호 반전 (같은 균형점 가중치) → "같은 목적에서 장기 최적화가
탐욕 대비 무엇을 더 얻는가"를 측정한다. 위험 테이블(rho, P_high)은 학습 시퀀스에
대해 frozen 기억으로 한 번만 계산해 캐시한다 (스윕과 같은 근거: 재사용 가능성).
평가(검증 그리디 롤아웃)는 oracle 지표(동적재·박스)로 — 학습 보상에는 oracle 미사용.

    python -m scripts.train_ppo --config configs/main.yaml --seed 42
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from scripts.run_packing import evaluate_boxes, sequence_risk
from src.config import load_config
from src.data.catalog import SkuCatalog
from src.data.graphs import _ordered_items, load_sequences, load_split
from src.immune.ainet import AiNetMemory
from src.models.factory import build_model
from src.packing.environment import PackingEnv
from src.packing.policies import PolicyConfig, choose
from src.packing.ppo import PolicyNet, gae, ppo_update, rollout
from src.packing.safety import SafetyFilter
from src.pipeline import device_auto, run_dir, save_json
from src.schema import RELATIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-seqs", type=int, default=600, help="학습에 쓸 시퀀스 수")
    parser.add_argument("--val-seqs", type=int, default=60, help="주기 평가용 검증 시퀀스 수")
    parser.add_argument("--episodes", type=int, default=3000)
    parser.add_argument("--batch-episodes", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--eval-every", type=int, default=20, help="N 업데이트마다 검증")
    args = parser.parse_args()

    cfg = load_config(args.config)
    torch.manual_seed(args.seed)
    device = device_auto()
    rng = np.random.default_rng(args.seed)

    model_dir = run_dir(cfg, "immune", args.seed)
    model = build_model("immune", cfg.model).to(device)
    model.load_state_dict(torch.load(model_dir / "best.pt", map_location=device))
    model.eval()
    mem_path = Path("outputs") / cfg.experiment / f"immune_memory_seed{args.seed}" / "ainet_memory.pkl"
    memory = AiNetMemory.load(mem_path) if mem_path.exists() else None
    thr_path = model_dir / "thresholds.json"
    thresholds = json.loads(thr_path.read_text())["thresholds"] if thr_path.exists() else {r: 0.99 for r in RELATIONS}
    print(f"aiNet 기억(frozen): {'불러옴' if memory else '없음'} | 임계값: {thresholds}")

    from src.pipeline import prepare_data
    datasets, _, _ = prepare_data(cfg)
    graphs = {g.sequence_id: g for ds in datasets.values() for g in ds}
    raw = load_sequences(cfg.data.dataset)
    if cfg.data.sku_catalog:
        raw = SkuCatalog(raw).apply(raw)
    split = load_split(cfg.data.split)
    train_ids = list(split["train"])[: args.train_seqs]
    val_ids = list(split["val"])[: args.val_seqs]

    def make_tables(ids, label):
        out = {}
        t0 = time.time()
        with torch.no_grad():
            for k, sid in enumerate(ids):
                items = _ordered_items(raw[sid], cfg.data.max_nodes)
                risk, _ = sequence_risk(model, graphs[sid], device, memory)
                out[sid] = (items, risk)
                if (k + 1) % 200 == 0:
                    print(f"  {label} 위험 테이블 {k+1}/{len(ids)} ({time.time()-t0:.0f}s)")
        print(f"{label} 위험 테이블 {len(ids)}개 준비 ({time.time()-t0:.0f}s)")
        return out

    train_tab = make_tables(train_ids, "학습")
    val_tab = make_tables(val_ids, "검증")

    policy_cfg = PolicyConfig()                      # 균형점 가중치 (risk 1.0 · space 0.15 · newbox 1.0)
    net = PolicyNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr)

    def greedy_metrics(tables, use_net):
        """그리디 롤아웃의 oracle 지표 (평가 전용 — 학습 보상 아님)."""
        high, boxes, utils, rewards = 0, [], [], []
        for sid, (items, risk) in tables.items():
            safety = SafetyFilter(items, risk, thresholds)
            env = PackingEnv(items)
            if use_net:
                steps = rollout(env, safety, net, policy_cfg, device, greedy=True)
                rewards.append(sum(s.reward for s in steps))
            else:
                for i in range(len(items)):
                    a = choose(env, safety, i, policy_cfg)
                    env.step(i, *a) if a else env.step(i, 0, (0, 0, 0))
            _, h, _ = evaluate_boxes(env, items)
            high += h; boxes.append(len(env.boxes)); utils.append(env.utilization())
        return dict(high=high, boxes=float(np.mean(boxes)), util=float(np.mean(utils)),
                    reward=float(np.mean(rewards)) if rewards else None)

    base = greedy_metrics(val_tab, use_net=False)
    print(f"탐욕 기준선 (검증 {len(val_ids)}개): 동적재 {base['high']} | 박스 {base['boxes']:.2f} | 활용률 {base['util']:.1%}")

    out_dir = Path("outputs") / cfg.experiment / f"ppo_seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    best_key, history = None, []
    n_updates = args.episodes // args.batch_episodes
    t0 = time.time()
    for upd in range(1, n_updates + 1):
        episodes = []
        for _ in range(args.batch_episodes):
            sid = train_ids[int(rng.integers(len(train_ids)))]
            items, risk = train_tab[sid]
            env = PackingEnv(items)
            safety = SafetyFilter(items, risk, thresholds)
            steps = rollout(env, safety, net, policy_cfg, device, greedy=False)
            adv, ret = gae(steps)
            episodes.append({"steps": steps, "adv": adv, "ret": ret})
        stats = ppo_update(net, opt, episodes, device)
        mean_r = float(np.mean([sum(s.reward for s in ep["steps"]) for ep in episodes]))
        if upd % args.eval_every == 0 or upd == n_updates:
            m = greedy_metrics(val_tab, use_net=True)
            # 선택 기준 = 비교 기준과 동일: 탐욕 이하 박스 중 동적재 최소, 동률이면 박스
            key = (m["high"], m["boxes"]) if m["boxes"] <= base["boxes"] + 0.3 else (float("inf"), m["boxes"])
            improved = best_key is None or key < best_key
            if improved:
                best_key = key
                torch.save(net.state_dict(), out_dir / "best.pt")
            history.append({"update": upd, "train_reward": mean_r, **m, **stats, "best": improved})
            print(f"[{upd:>4}/{n_updates}] 보상 {mean_r:7.2f} | 검증: 동적재 {m['high']:>4} 박스 {m['boxes']:6.2f} "
                  f"활용률 {m['util']:.1%} {'★best' if improved else ''} ({time.time()-t0:.0f}s)")
    save_json(out_dir / "history.json", {"baseline": base, "history": history,
                                          "train_seqs": len(train_ids), "episodes": args.episodes})
    print(f"저장: {out_dir}/best.pt, history.json")


if __name__ == "__main__":
    main()
