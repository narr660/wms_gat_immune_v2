# Colab 실행 셀 (KPI 1·2)

```python
# 1) 저장소와 데이터
REPO = "https://github.com/narr660/<새 저장소 이름>.git"
!git clone {REPO} wms
%cd wms
!pip -q install pyyaml pytest
!python -m scripts.check_data
!python -m pytest -q

# 2) 동작 확인 (수 분)
for m in ["basic", "relation", "immune", "immune_beta0"]:
    !python -m scripts.train --config configs/smoke.yaml --model {m} --seed 42
!python -m scripts.eval_ainet --config configs/smoke.yaml --seed 42
!python -m scripts.make_kpi_report --config configs/smoke.yaml

# 3) 본 실험 (GPU 권장, 시드 3개)
for seed in [42, 43, 44]:
    for m in ["basic", "relation", "immune", "immune_beta0"]:
        !python -m scripts.train --config configs/main.yaml --model {m} --seed {seed}
    !python -m scripts.eval_ainet --config configs/main.yaml --seed {seed}
!python -m scripts.make_kpi_report --config configs/main.yaml
```

결과는 `outputs/main/`과 `reports/main/kpi_summary.md`에 저장된다.
런타임이 끊겨도 이미 끝난 모델 폴더는 남으므로, 남은 조합만 다시 실행하면 된다.
