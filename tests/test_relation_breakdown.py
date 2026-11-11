"""relation_breakdown 이 저장된 혼동행렬만으로 등급별 F1을 맞게 모으는지 확인한다."""
import numpy as np

from scripts.relation_breakdown import find_runs, per_grade_across_seeds
from src.pipeline import save_confusion
from src.schema import NUM_CLASSES, RELATIONS


def _make_cms(seed_value: int):
    """시드별로 물리 관계의 등급 분포를 살짝 다르게 준다."""
    cms = {}
    for r in RELATIONS:
        k = NUM_CLASSES[r]
        cm = np.eye(k, dtype=np.int64)[None] * 10        # 완벽한 혼동행렬 하나
        if r == "physical":
            cm[0, -1, -1] -= seed_value                  # 시드마다 고위험 미탐지가 다르게
            cm[0, -1, 0] += seed_value
        cms[r] = cm
    return cms


def test_aggregates_f1_mean_and_sums_counts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for seed in (1, 2):
        out = tmp_path / "outputs" / "unit" / f"immune_seed{seed}"
        out.mkdir(parents=True)
        save_confusion(out / "cms_test.npz", _make_cms(seed))

    runs = find_runs(tmp_path / "outputs" / "unit")
    assert set(runs) == {"immune"}
    assert set(runs["immune"]) == {1, 2}

    seeds, by_grade = per_grade_across_seeds(runs, "immune", "physical")
    assert seeds == [1, 2]
    k = NUM_CLASSES["physical"]
    assert set(by_grade) == set(range(k))        # 모든 등급이 두 시드 다 존재
    high = by_grade[k - 1]
    assert len(high["f1_values"]) == 2            # 시드별 F1을 따로 들고 있다가
    assert high["total"] == 20                    # 표본수는 두 시드를 합산
    assert high["missed"] == 1 + 2                # 미탐지도 합산 (시드1: 1건, 시드2: 2건)
