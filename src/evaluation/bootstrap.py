"""시퀀스 단위 paired bootstrap: 같은 시퀀스를 두 모델에 함께 재표본추출한다."""
from typing import Dict, Sequence

import numpy as np

from src.evaluation.metrics import summarize

DEFAULT_METRICS = ("macro_f1", "accuracy", "low_f1", "mid_f1", "high_f1",
                   "high_recall", "high_precision", "high_missed", "low_overestimated")


def paired_bootstrap(cms_a: Dict[str, np.ndarray], cms_b: Dict[str, np.ndarray],
                     n_boot: int = 2000, seed: int = 42,
                     metrics: Sequence[str] = DEFAULT_METRICS) -> Dict[str, Dict]:
    """A - B 차이와 95% 신뢰구간."""
    n = next(iter(cms_a.values())).shape[0]
    assert n == next(iter(cms_b.values())).shape[0], "시퀀스 수가 다릅니다."
    rng = np.random.default_rng(seed)
    full_a, full_b = summarize(cms_a)["overall"], summarize(cms_b)["overall"]
    samples = {m: np.empty(n_boot) for m in metrics}
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a, b = summarize(cms_a, idx)["overall"], summarize(cms_b, idx)["overall"]
        for m in metrics:
            samples[m][i] = a[m] - b[m]
    result = {}
    for m in metrics:
        low, high = np.quantile(samples[m], [0.025, 0.975])
        result[m] = {"a": full_a[m], "b": full_b[m], "diff": full_a[m] - full_b[m],
                     "ci_low": float(low), "ci_high": float(high),
                     "ci_excludes_zero": bool(low > 0 or high < 0)}
    return result
