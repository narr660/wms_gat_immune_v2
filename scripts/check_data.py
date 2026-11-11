"""데이터 파일이 연구에 쓰는 원본과 같은지 확인한다.

    python -m scripts.check_data
"""
import hashlib
import json
import sys
from pathlib import Path

from src.data.catalog import SkuCatalog
from src.data.graphs import load_sequences, load_split

EXPECTED = {
    "data/processed/rs_v3.pkl": {
        "sha256": "c6ac4213d15d6cabafaee8858c042b042103d6e37a025a0111f11884654960f0",
        "size": 26528838, "sequences": 2100, "items_per_sequence": 100},
}
REQUIRED_KEYS = ("size", "category", "weight_sim", "max_supported_weight_sim", "sealed",
                 "leak_risk", "fragile", "is_electronic", "odor_profile", "hazard_type",
                 "arrival_index", "item_uid")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    ok = True
    for name, expected in EXPECTED.items():
        path = Path(name)
        if not path.exists():
            print(f"[없음] {name} — 기존 저장소에서 복사하세요.")
            ok = False
            continue
        if path.stat().st_size < 1000:
            print(f"[오류] {name} 은 Git LFS 포인터 파일입니다. 실제 파일을 넣으세요.")
            ok = False
            continue
        same = sha256(path) == expected["sha256"]
        sequences = load_sequences(path)
        missing = [k for k in REQUIRED_KEYS if k not in sequences[0][0]]
        print(f"[{'일치' if same else '다름'}] {name} | 시퀀스 {len(sequences)} | "
              f"첫 시퀀스 상품 {len(sequences[0])} | 누락 속성 {missing or '없음'}")
        ok &= same and not missing and len(sequences) == expected["sequences"]
        catalog = SkuCatalog(sequences)
        items = sum(len(seq) for seq in sequences)
        print(f"[SKU] 종류 {len(catalog)} | 상품 {items} | SKU당 평균 {items / len(catalog):.0f}회 등장")
    split_path = Path("data/splits/rs_seed42.json")
    if split_path.exists():
        split = load_split(split_path)
        sizes = {k: len(v) for k, v in split.items()}
        print(f"[분할] {sizes}")
        ok &= sizes == {"train": 1470, "val": 315, "test": 315}
    else:
        print("[없음] data/splits/rs_seed42.json")
        ok = False
    print("데이터 확인:", "통과" if ok else "실패")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
