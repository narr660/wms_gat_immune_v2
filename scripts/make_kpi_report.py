"""저장된 실험 결과를 모아 KPI 표를 만든다.

    python -m scripts.make_kpi_report --config configs/main.yaml

출력: reports/<실험>/kpi_summary.md, kpi_models.csv, kpi_comparisons.csv
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.config import load_config
from src.evaluation.bootstrap import paired_bootstrap
from src.pipeline import load_confusion
from src.schema import RELATIONS

LABELS = {
    "basic": "기본 GAT",
    "relation": "관계 분리형 GAT",
    "immune": "면역 피드백 GAT (DCA)",
    "immune_memory_shared": "면역 피드백 GAT + 관계 구분 없는 기억",
    "immune_memory": "면역 피드백 GAT + aiNet 기억",
}
COMPARISONS = [
    ("KPI ① 관계 분리 효과", "relation", "basic"),
    ("KPI ② DCA 효과 (같은 초기값에서 DCA 보정만 추가)", "immune", "relation"),
    ("KPI ② aiNet 기억 효과", "immune_memory", "immune"),
    ("관계 분리가 면역의 전제인가 (관계별 기억 − 관계 구분 없는 기억)",
     "immune_memory", "immune_memory_shared"),
]
SHOWN = ("macro_f1", "high_recall", "high_precision", "low_f1",
         "high_missed", "low_overestimated")
COUNTS = ("high_missed", "low_overestimated")


def load_runs(exp_dir: Path):
    runs = {}
    for path in sorted(exp_dir.glob("*_seed*/metrics.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        runs.setdefault(m["model"], {})[m["seed"]] = (path.parent, m)
    return runs


def fmt(values, pct=True):
    values = np.asarray(values, dtype=float) * (100 if pct else 1)
    return f"{values[0]:.2f}" if len(values) == 1 else f"{values.mean():.2f} ± {values.std(ddof=1):.2f}"


def diff_row(metric, r):
    scale, unit = (1, "건") if metric in COUNTS else (100, "%p")
    return (f"{r['diff'] * scale:+.3f}{unit}",
            f"[{r['ci_low'] * scale:+.3f}, {r['ci_high'] * scale:+.3f}]",
            "예" if r["ci_excludes_zero"] else "아니오")


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
             f"테스트셋 기준 · 관측 설정: {cfg.data.observability} · "
             "여러 시드는 평균 ± 표준편차 · 차이는 시퀀스 단위 paired bootstrap 95% CI", ""]

    # ---------------------------------------------------------------- 모델별
    lines += ["## 모델별 성능", "",
              "| 모델 | 시드 수 | Macro-F1 (%) | 고위험 Recall (%) | 고위험 미탐지 | 저위험 과대평가 | 반발강도 오차 |",
              "|---|---|---|---|---|---|---|"]
    with open(out_dir / "kpi_models.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["model", "seed", "relation", "macro_f1", "accuracy", "high_recall",
                         "high_precision", "high_missed", "high_total", "low_overestimated"])
        for model, label in LABELS.items():
            if model not in runs:
                continue
            tests = [m["test"] for _, m in runs[model].values()]
            overall = [t["overall"] for t in tests]
            rho = [t.get("rho_error", {}).get("mean") for t in tests]
            lines.append(
                f"| {label} | {len(tests)} | {fmt([o['macro_f1'] for o in overall])} | "
                f"{fmt([o['high_recall'] for o in overall])} | "
                f"{fmt([o['high_missed'] for o in overall], pct=False)} | "
                f"{fmt([o['low_overestimated'] for o in overall], pct=False)} | "
                f"{fmt(rho, pct=False) if all(v is not None for v in rho) else '—'} |")
            for seed, (_, m) in runs[model].items():
                for name, v in [("overall", m["test"]["overall"]),
                                *m["test"]["relations"].items()]:
                    writer.writerow([model, seed, name, v["macro_f1"], v["accuracy"],
                                     v["high_recall"], v["high_precision"], v["high_missed"],
                                     v["high_total"], v["low_overestimated"]])

    # ---------------------------------------------------------------- 관계별
    lines += ["", "## 관계별 고위험 탐지 (첫 시드)", "",
              "| 모델 | " + " | ".join(f"{r} Macro-F1 / 미탐지" for r in RELATIONS) + " |",
              "|---|" + "---|" * len(RELATIONS)]
    for model, label in LABELS.items():
        if model in runs:
            rel = runs[model][min(runs[model])][1]["test"]["relations"]
            lines.append(f"| {label} | " + " | ".join(
                f"{rel[r]['macro_f1'] * 100:.2f} / {rel[r]['high_missed']}/{rel[r]['high_total']}"
                for r in RELATIONS) + " |")

    # ---------------------------------------------------------------- 비교
    with open(out_dir / "kpi_comparisons.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["comparison", "seed", "metric", "a", "b", "diff", "ci_low", "ci_high"])
        for title, a, b in COMPARISONS:
            seeds = sorted(set(runs.get(a, {})) & set(runs.get(b, {})))
            if not seeds:
                continue
            lines += ["", f"## {title}", "", f"{LABELS[a]} − {LABELS[b]}", "",
                      "| 시드 | 지표 | 차이 | 95% CI | 0 제외 |", "|---|---|---|---|---|"]
            for seed in seeds:
                result = paired_bootstrap(load_confusion(runs[a][seed][0] / "cms_test.npz"),
                                          load_confusion(runs[b][seed][0] / "cms_test.npz"),
                                          cfg.eval.bootstrap, seed)
                for metric in SHOWN:
                    r = result[metric]
                    lines.append(f"| {seed} | {metric} | " + " | ".join(diff_row(metric, r)) + " |")
                    writer.writerow([f"{a}-{b}", seed, metric, r["a"], r["b"],
                                     r["diff"], r["ci_low"], r["ci_high"]])

    # ---------------------------------------------------------------- aiNet 조건
    for path in sorted(exp_dir.glob("immune_memory_seed*/ainet_metrics.json")):
        a = json.loads(path.read_text(encoding="utf-8"))
        lines += ["", f"## aiNet 기억 OFF / frozen / online — 같은 체크포인트 (시드 {a['seed']})", "",
                  "off: 기억 없음 · frozen: 학습셋 기억만 · online: 테스트 중에도 예측 후 관측해 갱신", "",
                  "| 조건 | Macro-F1 (%) | 고위험 Recall (%) | 고위험 미탐지 | 저위험 과대평가 | 반발강도 오차 | 기억 적용률 |",
                  "|---|---|---|---|---|---|---|"]
        for label, c in a["conditions"].items():
            o = c["overall"]
            lines.append(f"| {label} | {o['macro_f1'] * 100:.2f} | {o['high_recall'] * 100:.2f} | "
                         f"{o['high_missed']} | {o['low_overestimated']} | "
                         f"{c['rho_error']['mean']:.4f} | {c['memory_coverage']:.1%} |")
        lines += ["", "| 비교 | 지표 | 차이 | 95% CI | 0 제외 |", "|---|---|---|---|---|"]
        for key, title in (("frozen_minus_off", "frozen − off"), ("online_minus_off", "online − off")):
            for metric in ("macro_f1", "high_recall", "high_missed", "low_overestimated"):
                lines.append(f"| {title} | {metric} | " + " | ".join(diff_row(metric, a[key][metric])) + " |")

    (out_dir / "kpi_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\n저장: {out_dir}")


if __name__ == "__main__":
    main()
