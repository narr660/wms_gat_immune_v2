# GAT-인공면역시스템 기반의 WMS 혼적을 고려한 온라인 3D 패킹 최적화

연구용 합성 혼적 규칙을 사용한다. **실제 법규·안전 준수를 보증하지 않는다.**

## 구조

| PART | 내용 | 코드 | 상태 |
|---|---|---|---|
| 1 | 관계 분리형 GAT (냄새·오염·물리) | `src/models/relation_gat.py` | 구현 |
| 2 | DCA–aiNet 면역 피드백 | `src/models/immune_gat.py`, `src/immune/ainet.py` | 구현 |
| 3 | 멀티박스 PPO 3D 적재 | `src/packing/`, `src/rl/` | 예정 |

```
configs/        실험 설정 (YAML 하나 = 실험 하나)
src/schema.py   관계와 등급 수 (냄새 4, 오염 5, 물리 5)
src/data/       속성 생성·등급 규칙, 26차원 특징, 그래프 생성
src/constraints/ 금지 규칙 판정 (PART 3의 안전 필터에 사용)
src/models/     기본 GAT · 관계 분리형 GAT · 면역 피드백 GAT
src/immune/     aiNet 기억 (관계별 상품 시그니처 키)
src/training/   손실(가중 CE + 반발 보조), 학습 루프
src/evaluation/ 지표, 시퀀스 단위 paired bootstrap
scripts/        check_data · train · eval_ainet · make_kpi_report
tests/          단위 테스트와 전체 흐름 테스트
data/           rs_v3.pkl 과 분할 파일 (data/README.md 참고)
```

## 실행

```bash
pip install torch numpy pyyaml pytest
python -m scripts.check_data                # 데이터 파일 확인
python -m pytest -q                         # 전체 테스트

# KPI 1: 관계 분리 효과
python -m scripts.train --config configs/main.yaml --model basic    --seed 42
python -m scripts.train --config configs/main.yaml --model relation --seed 42

# KPI 2: 면역 효과 (같은 구조·같은 초기값에서 DCA만 끈 통제 모델과 비교)
python -m scripts.train --config configs/main.yaml --model immune       --seed 42
python -m scripts.train --config configs/main.yaml --model immune_beta0 --seed 42
python -m scripts.eval_ainet --config configs/main.yaml --seed 42   # aiNet OFF/ON

# KPI 표 생성 -> reports/main/kpi_summary.md
python -m scripts.make_kpi_report --config configs/main.yaml
```

시드를 바꿔 반복하면 리포트가 평균 ± 표준편차로 합쳐진다.
`configs/smoke.yaml`은 동작 확인용이며 그 결과는 논문에 쓰지 않는다.

## 결과 파일

`outputs/<실험>/<모델>_seed<시드>/`

| 파일 | 내용 |
|---|---|
| `metrics.json` | 검증·테스트 지표 (관계별, 관계 평균) |
| `cms_test.npz` | 시퀀스별 혼동행렬 — 모든 지표와 bootstrap의 원본 |
| `history.json` | epoch별 손실과 검증 지표 |
| `config.json`, `normalizer.json`, `class_weights.json` | 재현에 필요한 설정 |
| `ainet_metrics.json` | aiNet OFF/ON 비교와 기억 적용률 |

## 설계 원칙

- 정규화 통계, 클래스 가중치, aiNet 기억은 **학습셋에서만** 계산한다.
- 테스트셋은 모델을 확정한 뒤 평가에만 쓴다.
- 통제 비교(`immune` vs `immune_beta0`)는 같은 시드에서 같은 초기 가중치로 시작한다.
- 레이블은 속성 기반 합성 규칙(`src/data/oracle.py`)으로 만든다.
