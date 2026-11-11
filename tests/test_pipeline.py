"""학습 -> 평가 -> aiNet OFF/ON -> KPI 리포트 전체 흐름."""
import json
import subprocess
import sys

import numpy as np

from src.evaluation.bootstrap import paired_bootstrap
from src.evaluation.metrics import relation_metrics


def test_relation_metrics_counts_misses_and_ignores_absent_classes():
    cm = np.array([[90, 10, 0, 0],
                   [0, 0, 0, 0],       # 존재하지 않는 등급
                   [0, 2, 8, 0],
                   [0, 0, 3, 7]])
    m = relation_metrics(cm)
    assert m["high_missed"] == 3 and m["high_total"] == 10
    assert m["low_overestimated"] == 10
    assert abs(m["high_recall"] - 0.7) < 1e-9
    f1_low, f1_mid, f1_high = 2 * 90 / (2 * 90 + 10), 2 * 8 / (2 * 8 + 2 + 3), 2 * 7 / (2 * 7 + 3)
    assert abs(m["macro_f1"] - np.mean([f1_low, f1_mid, f1_high])) < 1e-9


def test_bootstrap_of_identical_models_is_zero():
    rng = np.random.default_rng(0)
    cms = {r: rng.integers(0, 20, size=(12, k, k))
           for r, k in (("odor", 4), ("contamination", 5), ("physical", 5))}
    result = paired_bootstrap(cms, cms, n_boot=30)
    assert all(v["diff"] == 0 and not v["ci_excludes_zero"] for v in result.values())


def _run(cwd, *args):
    done = subprocess.run([sys.executable, "-m", *args], cwd=cwd, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    return done.stdout


def test_end_to_end(tiny_project, request):
    import os
    tmp, config = tiny_project
    os.environ["PYTHONPATH"] = str(request.config.rootpath)
    args = ["--config", str(config), "--seed", "1"]
    for model in ("basic", "relation", "immune"):
        _run(tmp, "scripts.train", *args, "--model", model)
        metrics = json.loads((tmp / "outputs/unit" / f"{model}_seed1/metrics.json").read_text())
        assert 0 <= metrics["test"]["overall"]["macro_f1"] <= 1
    _run(tmp, "scripts.train_memory", *args)
    _run(tmp, "scripts.train_memory", *args, "--shared")
    ainet = json.loads((tmp / "outputs/unit/immune_memory_seed1/ainet_metrics.json").read_text())
    assert set(ainet["conditions"]) == {"off", "frozen", "online"}
    assert ainet["conditions"]["off"]["memory_coverage"] == 0
    report = _run(tmp, "scripts.make_kpi_report", "--config", str(config))
    for title in ("관계 분리 효과", "DCA 효과", "aiNet 기억 효과", "관계 분리가 면역의 전제"):
        assert title in report
