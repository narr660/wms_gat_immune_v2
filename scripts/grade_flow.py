"""한 관계의 혼동행렬 전체를 모델별로 보여준다: 각 등급의 오류가 어디로 갔는가.

재학습·GPU 필요 없음 — 저장된 cms_test.npz 에서 바로 계산한다.

    python -m scripts.grade_flow --config configs/main.yaml --relation physical
"""
import argparse
from pathlib import Path

import numpy as np

from scripts.relation_breakdown import LABELS, find_runs
from src.config import load_config
from src.pipeline import load_confusion
from src.schema import NUM_CLASSES, RELATIONS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--relation", choices=RELATIONS, default="physical")
    args = parser.parse_args()
    cfg = load_config(args.config)
    runs = find_runs(Path("outputs") / cfg.experiment)
    assert runs, "저장된 결과가 없습니다."
    k = NUM_CLASSES[args.relation]

    lines = [f"# 등급 이동 — {args.relation} (시드 합산)", "",
             "행 = 실제 등급, 열 = 예측 등급. 각 칸은 그 행에서의 비율(%)이고 괄호는 건수. "
             "대각선이 정답, 대각선 오른쪽은 '더 위험하게'(보수적), 왼쪽은 '덜 위험하게'(위험한 오류) 본 것.", ""]
    for model, label in LABELS.items():
        if model not in runs:
            continue
        total = None
        for seed, path in sorted(runs[model].items()):
            cm = load_confusion(path)[args.relation].sum(0)
            total = cm if total is None else total + cm
        row_sum = total.sum(1, keepdims=True).clip(min=1)
        pct = total / row_sum * 100
        lines += [f"## {label} (시드 {','.join(map(str, sorted(runs[model])))})", "",
                  "| 실제\\예측 | " + " | ".join(f"{g}" for g in range(k)) + " | 보수적 오류 | 위험한 오류 |",
                  "|---|" + "---|" * (k + 2)]
        for g in range(k):
            cells = [f"{pct[g, j]:.2f}% ({total[g, j]:,})" for j in range(k)]
            conservative = total[g, g + 1:].sum()        # 실제보다 높게 (과잉 경계)
            risky = total[g, :g].sum()                   # 실제보다 낮게 (위험)
            lines.append(f"| **{g}** | " + " | ".join(cells)
                         + f" | {conservative:,} | {risky:,} |")
        lines.append("")

    out = Path("reports") / cfg.experiment / f"grade_flow_{args.relation}.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"저장: {out}")


if __name__ == "__main__":
    main()
