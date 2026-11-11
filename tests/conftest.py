"""테스트용 합성 시퀀스. 실제 데이터 파일 없이 전체 흐름을 검증한다."""
import json
import pickle
import random

import pytest

from src.data.augment import augment_dataset
from src.data.catalog import SkuCatalog

_LEVEL = {"위험물": 0.9, "화학품": 0.7, "식품": 0.1, "일반": 0.0}


def make_sequences(n_sequences=24, n_items=10, seed=0):
    rng = random.Random(seed)
    raw = []
    for _ in range(n_sequences):
        sequence = []
        for _ in range(n_items):
            category = rng.choice(list(_LEVEL))
            sequence.append({
                "category": category,
                "size": [rng.randint(1, 5), rng.randint(1, 5), rng.randint(1, 5)],
                "hazard_level": _LEVEL[category],
            })
        raw.append(sequence)
    return augment_dataset(raw, dataset_name="test", seed=42)


def make_sku_sequences(**kwargs):
    raw = make_sequences(**kwargs)
    return SkuCatalog(raw).apply(raw)


@pytest.fixture(scope="session")
def raw_sequences():
    """SKU 카탈로그 적용 전: 상품마다 속성이 따로 생성된 원본 형태."""
    return make_sequences()


@pytest.fixture(scope="session")
def sequences(raw_sequences):
    """SKU 카탈로그 적용 후: 같은 (분류, 치수)는 같은 속성."""
    return SkuCatalog(raw_sequences).apply(raw_sequences)


@pytest.fixture()
def tiny_project(tmp_path, raw_sequences):
    """데이터·분할·설정 파일을 갖춘 임시 실험 폴더."""
    data_path = tmp_path / "data.pkl"
    with open(data_path, "wb") as f:
        pickle.dump(raw_sequences, f)
    split_path = tmp_path / "split.json"
    ids = list(range(len(raw_sequences)))
    split_path.write_text(json.dumps(
        {"indices": {"train": ids[:14], "val": ids[14:19], "test": ids[19:]}}))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"experiment: unit\n"
        f"data:\n  dataset: {data_path}\n  split: {split_path}\n  max_nodes: 10\n"
        f"train:\n  max_epochs: 2\n  patience: 2\n  memory_epochs: 1\n"
        f"eval:\n  bootstrap: 50\n")
    return tmp_path, config_path
