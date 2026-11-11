"""관계별 등급 하나하나의 F1·미탐지를 모델별 · 시드 평균으로 비교한다.

학습을 다시 돌릴 필요가 없다 — 이미 저장된 outputs/<실험>/<모델>_seed<시드>/cms_test.npz
(시퀀스별 혼동행렬)에서 바로 계산한다. GPU도 필요 없다.

    python -m scripts.relation_breakdown --config configs/main.yaml

출력: reports/<실험>/relation_breakdown.md, relation_breakdown.csv
"""
import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.config import load_config
from src.evaluation.metrics import relation_metrics
from src.pipeline import load_confusion
from src.schema import NUM_CLASSES, RELATIONS

LABELS = {
    "basic": "기본 GAT",
    "relation": "관계 분리형 GAT",
    "immune": "면역 피드백 GAT (DCA)",
    "immune_memory_shared": "+ 관계 구분 없는 기억",
    "immune_memory": "+ aiNet 기억",
}


def find_runs(exp_dir: Path):
    """model -> {seed: cms_test.npz 경로}."""
    runs = defaultdict(dict)
    for path in sorted(exp_dir.glob("*_seed*/cms_test.npz")):
        name = path.parent.name           # 예: immune_memory_seed43
        model, _, seed = name.rpartition("_seed")
        runs[model][int(seed)] = path
    return runs


def per_grade_across_seeds(runs, model: str, relation: str):
    """같은 모델, 여러 시드의 등급별 F1·표본수·미탐지를 평균/합산한다."""
    by_grade = defaultdict(lambda: {"f1": [], "total": 0, "missed": 0})
    seeds = sorted(runs.get(model, {}))
    for seed in seeds:
        cms = load_confusion(runs[model][seed])
        metrics = relation_metrics(cms[relation].sum(0))
        for row in metrics["per_grade"]:
            g = by_grade[row["grade"]]
            g["f1"].append(row["f1"])
            g["total"] += row["total"]
            g["missed"] += row["missed"]
    return seeds, {
        grade: {"f1_mean": float(np.mean(v["f1"])), "f1_values": v["f1"],
                "total": v["total"], "missed": v["missed"]}
        for grade, v in sorted(by_grade.items())
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--relation", choices=RELATIONS, default=None,
                        help="하나만 볼 관계 (기본: 셋 다)")
    args = parser.parse_args()
    cfg = load_config(args.config)
    exp_dir = Path("outputs") / cfg.experiment
    out_dir = Path("reports") / cfg.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = find_runs(exp_dir)
    assert runs, f"결과가 없습니다: {exp_dir} (cms_test.npz 가 저장된 실험 폴더가 필요합니다)"

    relations = [args.relation] if args.relation else list(RELATIONS)
    lines = [f"# 관계별 등급 상세 — {cfg.experiment}", "",
             "mid_f1(중간 등급 평균)로는 보이지 않는, 등급 하나하나의 F1과 미탐지 수. "
             "같은 모델의 여러 시드는 F1 평균 · 표본수·미탐지는 합산.", ""]
    csv_rows = [["relation", "model", "grade", "n_seeds", "f1_mean", "f1_values",
                "total", "missed"]]

    for relation in relations:
        lines += [f"## {relation}", "",
                  "| 모델 | " + " | ".join(f"등급{g}" for g in range(NUM_CLASSES[relation])) + " |",
                  "|---|" + "---|" * NUM_CLASSES[relation]]
        for model, label in LABELS.items():
            if model not in runs:
                continue
            seeds, by_grade = per_grade_across_seeds(runs, model, relation)
            cells = []
            for g in range(NUM_CLASSES[relation]):
                if g in by_grade:
                    v = by_grade[g]
                    cells.append(f"F1 {v['f1_mean']*100:.2f} · 미탐지 {v['missed']}/{v['total']}")
                    csv_rows.append([relation, model, g, len(seeds), v["f1_mean"],
                                     ";".join(f"{x:.4f}" for x in v["f1_values"]),
                                     v["total"], v["missed"]])
                else:
                    cells.append("—")
            lines.append(f"| {label} (시드 {','.join(map(str, seeds))}) | " + " | ".join(cells) + " |")
        lines.append("")

    (out_dir / "relation_breakdown.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out_dir / "relation_breakdown.csv", "w", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerows(csv_rows)

    print("\n".join(lines))
    print(f"\n저장: {out_dir}/relation_breakdown.md, relation_breakdown.csv")


if __name__ == "__main__":
    main()
