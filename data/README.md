# 데이터

| 파일 | 내용 | 크기 |
|---|---|---|
| `processed/rs_v3.pkl` | 합성 속성을 붙인 시퀀스 2,100개 (시퀀스당 상품 100개) | 26.5 MB |
| `splits/rs_seed42.json` | 시퀀스 단위 분할: 학습 1,470 / 검증 315 / 테스트 315 | — |

- `rs_v3.pkl`은 일반 파일로 커밋한다 (Git LFS 사용 안 함).
- 실험에서는 `rs_v3.pkl`의 분류와 치수만 쓰고, 속성은 SKU 단위로 다시 만든다 (`src/data/catalog.py`).
- 속성 생성 규칙은 `src/data/augment.py`, 등급 규칙은 `src/data/oracle.py`.
- `python -m scripts.check_data` 로 파일이 원본과 같은지(sha256) 확인한다.
- 속성과 등급은 연구용 합성 규칙이며 실제 위험물 규정을 대신하지 않는다.
