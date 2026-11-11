"""저장된 실험 결과를 모아 KPI 표를 만든다.

    python -m scripts.make_kpi_report --config configs/main.yaml

출력: reports/<실험>/kpi_summary.md, kpi1_models.csv, kpi2_immune.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.config import load_config
from src.evaluation.bootstrap import paired_bootstrap
from src.pipeline import load_confusion

LABELS = {"basic": "기본 GAT", "relation": "관계 분리형 GAT",
          "immune_beta0": "면역 구조 (DCA 끔)", "immune": "면역 피드백 GAT"}
KEYS = ("macro_f1", "accuracy", "high_recall", "high_precision", "low_f1",
        "high_missed", "low_overestimated")


def load_runs(exp_dir: Path):
    runs = {}
    for path in sorted(exp_dir.glob("*_seed*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        runs.setdefault(m["model"], {})[m["seed"]] = (path.parent, m)
    return runs


def fmt(values, pct=True):
    values = np.asarray(values, dtype=float)
    scale = 100 if pct else 1
    if len(values) == 1:
        return f"{values[0] * scale:.2f}"
    return f"{values.mean() * scale:.2f} ± {values.std(ddof=1) * scale:.2f}"


def compare(runs, a, b, n_boot):
    """같은 시드끼리 A - B paired bootstrap."""
    rows = []
    for seed in sorted(set(runs.get(a, {})) & set(runs.get(b, {}))):
        cms_a = load_confusion(runs[a][seed][0] / "cms_test.npz")
        cms_b = load_confusion(runs[b][seed][0] / "cms_test.npz")
        rows.append((seed, paired_bootstrap(cms_a, cms_b, n_boot, seed)))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    exp_dir = Path("outputs") / cfg.experiment
    out_dir = Path("reports") / cfg.experiment
    out_dir.mkdir(parents=True, exist_ok=True)
    runs = load_runs(exp_dir)
    assert runs, f"결과가 없습니다: {exp_dir}"

    lines = [f"# KPI 요약 — {cfg.experiment}", "",
             "테스트셋 기준. 여러 시드는 평균 ± 표준편차, 차이는 시퀀스 단위 paired bootstrap 95% CI.", ""]

    # ---------------------------------------------------------------- 모델별 표
    lines += ["## 모델별 성능", "",
              "| 모델 | 시드 수 | Macro-F1 (%) | Accuracy (%) | 고위험 Recall (%) | 고위험 미탐지 | 저위험 과대평가 |",
              "|---|---|---|---|---|---|---|"]
    with open(out_dir / "kpi1_models.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "seed", *KEYS])
        for model in LABELS:
            if model not in runs:
                continue
            tests = [m["test"]["overall"] for _, m in runs[model].values()]
            for seed, (_, m) in runs[model].items():
                writer.writerow([model, seed, *[m["test"]["overall"][k] for k in KEYS]])
            lines.append(
                f"| {LABELS[model]} | {len(tests)} | {fmt([t['macro_f1'] for t in tests])} | "
                f"{fmt([t['accuracy'] for t in tests])} | {fmt([t['high_recall'] for t in tests])} | "
                f"{fmt([t['high_missed'] for t in tests], pct=False)} | "
                f"{fmt([t['low_overestimated'] for t in tests], pct=False)} |")

    # ---------------------------------------------------------------- 비교
    comparisons = [
        ("KPI ① 관계 분리 효과", "relation", "basic"),
        ("KPI ② DCA 효과 (같은 구조·초기값, β만 다름)", "immune", "immune_beta0"),
        ("참고: 면역 피드백 GAT vs 관계 분리형 GAT", "immune", "relation"),
    ]
    with open(out_dir / "kpi2_immune.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["comparison", "seed", "metric", "a", "b", "diff", "ci_low", "ci_high"])
        for title, a, b in comparisons:
            rows = compare(runs, a, b, cfg.eval.bootstrap)
            if not rows:
                continue
            lines += ["", f"## {title}", "", f"{LABELS[a]} − {LABELS[b]}", "",
                      "| 시드 | 지표 | 차이 | 95% CI | 0 제외 |", "|---|---|---|---|---|"]
            for seed, result in rows:
                for metric in ("macro_f1", "high_recall", "high_precision", "low_f1",
                               "high_missed", "low_overestimated"):
                    r = result[metric]
                    count = metric in ("high_missed", "low_overestimated")
                    s = 1 if count else 100
                    unit = "건" if count else "%p"
                    lines.append(
                        f"| {seed} | {metric} | {r['diff'] * s:+.3f}{unit} | "
                        f"[{r['ci_low'] * s:+.3f}, {r['ci_high'] * s:+.3f}] | "
                        f"{'예' if r['ci_excludes_zero'] else '아니오'} |")
                    writer.writerow([f"{a}-{b}", seed, metric, r["a"], r["b"],
                                     r["diff"], r["ci_low"], r["ci_high"]])

    # ---------------------------------------------------------------- aiNet
    ainet = sorted(exp_dir.glob("immune_seed*/ainet_metrics.json"))
    if ainet:
        lines += ["", "## KPI ② aiNet 기억 OFF/ON (같은 체크포인트)", "",
                  "| 시드 | 기억 적용률 | 지표 | OFF | ON | 차이 | 95% CI | 0 제외 |",
                  "|---|---|---|---|---|---|---|---|"]
        for path in ainet:
            a = json.loads(path.read_text(encoding="utf-8"))
            for metric in ("macro_f1", "high_recall", "high_missed", "low_overestimated"):
                r = a["on_minus_off"][metric]
                count = metric in ("high_missed", "low_overestimated")
                s = 1 if count else 100
                lines.append(
                    f"| {a['seed']} | {a['memory_coverage']:.1%} | {metric} | "
                    f"{r['b'] * s:.3f} | {r['a'] * s:.3f} | {r['diff'] * s:+.3f} | "
                    f"[{r['ci_low'] * s:+.3f}, {r['ci_high'] * s:+.3f}] | "
                    f"{'예' if r['ci_excludes_zero'] else '아니오'} |")

    (out_dir / "kpi_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n저장: {out_dir}")


if __name__ == "__main__":
    main()
