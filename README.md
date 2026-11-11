# GAT-인공면역시스템 기반의 WMS 혼적을 고려한 온라인 3D 패킹 최적화

연구용 합성 혼적 규칙을 사용한다. **실제 법규·안전 준수를 보증하지 않는다.**

## 구조

| PART | 내용 | 코드 | 상태 |
|---|---|---|---|
| 1 | 관계 분리형 GAT (냄새·오염·물리) | `src/models/relation_gat.py` | 구현 |
| 2 | DCA–aiNet 면역 피드백 | `src/models/immune_gat.py`, `src/immune/ainet.py` | 구현 |
| 3 | 멀티박스 PPO 3D 적재 | `src/packing/`, `src/rl/` | 예정 |

## 문제 설정

- **SKU**: 같은 (분류, 치수)의 상품은 같은 SKU이고 속성이 같다. 같은 SKU가 여러 시퀀스에 반복해서 도착한다.
- **부분 관측**: 도착 시점에는 치수·무게·분류·포장·표시만 안다. 누출 위험, 냄새 프로필, 지지 한계는 모른다.
- **경험**: 숨은 속성은 함께 적재한 뒤 관측된 노출로만 드러난다. aiNet이 SKU 쌍 × 관계별로 이 경험을 기억해 반발강도를 보정한다.
- 위험 등급은 모든 속성으로 정해지므로, 속성만 보는 GAT는 한계가 있고 경험이 쌓일수록 정확해진다.

## 폴더

```
configs/        실험 설정 (YAML 하나 = 실험 하나)
data/           rs_v3.pkl 과 분할 파일 (data/README.md 참고)
src/schema.py   관계와 등급 수 (냄새 4, 오염 5, 물리 5)
src/data/       속성 생성·등급 규칙, SKU 카탈로그, 26차원 특징, 그래프 생성
src/models/     기본 GAT · 관계 분리형 GAT · 면역 피드백 GAT
src/immune/     aiNet 기억 (SKU 쌍 × 관계), 조회 -> 관측 -> 갱신
src/training/   손실(가중 CE + 반발 보조), 학습 루프, 기억 주입 학습
src/evaluation/ 지표, 시퀀스 단위 paired bootstrap
src/constraints/ 금지 규칙 판정 (PART 3의 안전 필터에 사용)
scripts/        check_data · train · train_memory · make_kpi_report
tests/          단위 테스트와 전체 흐름 테스트
```

## 실행

```bash
pip install torch numpy pyyaml pytest
python -m scripts.check_data                # 데이터 파일 확인
python -m pytest -q                         # 전체 테스트

# KPI 1: 관계 분리 효과
python -m scripts.train --config configs/main.yaml --model basic    --seed 42
python -m scripts.train --config configs/main.yaml --model relation --seed 42

# KPI 2: 면역 효과
python -m scripts.train --config configs/main.yaml --model immune --seed 42      # DCA
python -m scripts.train_memory --config configs/main.yaml --seed 42             # aiNet 기억
python -m scripts.train_memory --config configs/main.yaml --seed 42 --shared    # 관계 구분 없는 기억 (비교)

# KPI 표 생성 -> reports/main/kpi_summary.md
python -m scripts.make_kpi_report --config configs/main.yaml
```

시드를 바꿔 반복하면 리포트가 평균 ± 표준편차로 합쳐진다.
`configs/smoke.yaml`은 동작 확인용이며 그 결과는 논문에 쓰지 않는다.

## 비교하는 모델

| 이름 | 내용 | 확인하는 것 |
|---|---|---|
| `basic` | 공통 인코더 | 기준선 |
| `relation` | 관계별 인코더 + 방향성 반발강도 | 관계 분리 효과 |
| `immune` | + DCA 보정 (같은 시드에서 `relation`과 초기 가중치 동일) | DCA 효과 |
| `immune_memory` | + aiNet 기억 (관계별) | 경험 기억 효과 |
| `immune_memory_shared` | + 관계를 구분하지 않는 기억 | 관계 분리가 면역의 전제인지 |

aiNet은 같은 체크포인트에서 세 조건으로 평가한다: `off`(기억 없음), `frozen`(학습셋 기억만), `online`(테스트 시퀀스도 예측한 뒤 관측해 갱신).

## 결과 파일

`outputs/<실험>/<모델>_seed<시드>/`

| 파일 | 내용 |
|---|---|
| `metrics.json` | 테스트 지표 (관계별, 관계 평균, 반발강도 오차) |
| `cms_test.npz` | 시퀀스별 혼동행렬 — 모든 지표와 bootstrap의 원본 |
| `history.json` | epoch별 손실과 검증 지표 |
| `ainet_metrics.json` | off / frozen / online 비교와 기억 적용률 |
| `config.json`, `normalizer.json`, `class_weights.json` | 재현에 필요한 설정 |

## 설계 원칙

- 정규화 통계와 클래스 가중치는 **학습셋에서만** 계산한다.
- 기억은 항상 "조회 → 예측 → 관측 → 갱신" 순서다. 어떤 시퀀스의 정답도 그 시퀀스의 예측에 쓰이지 않는다.
- `frozen` 조건은 테스트 정답을 전혀 쓰지 않는다. `online` 조건은 예측을 마친 시퀀스의 관측만 이후 예측에 쓴다.
- 테스트셋은 모델을 확정한 뒤 평가에만 쓴다.
