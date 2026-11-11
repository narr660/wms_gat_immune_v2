"""테스트용 합성 시퀀스. 실제 데이터 파일 없이 전체 흐름을 검증한다."""
import json
import pickle
import random

import pytest

from src.data.augment import augment_dataset

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


@pytest.fixture(scope="session")
def sequences():
    return make_sequences()


@pytest.fixture()
def tiny_project(tmp_path, sequences):
    """데이터·분할·설정 파일을 갖춘 임시 실험 폴더."""
    data_path = tmp_path / "data.pkl"
    with open(data_path, "wb") as f:
        pickle.dump(sequences, f)
    split_path = tmp_path / "split.json"
    ids = list(range(len(sequences)))
    split_path.write_text(json.dumps(
        {"indices": {"train": ids[:14], "val": ids[14:19], "test": ids[19:]}}))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        f"experiment: unit\n"
        f"data:\n  dataset: {data_path}\n  split: {split_path}\n  max_nodes: 10\n"
        f"train:\n  max_epochs: 2\n  patience: 2\n"
        f"eval:\n  bootstrap: 50\n")
    return tmp_path, config_path
