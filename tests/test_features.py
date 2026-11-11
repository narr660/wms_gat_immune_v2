import numpy as np

from src.data.features import (FEATURE_DIM, FEATURE_NAMES, FeatureNormalizer,
                               N_CONTINUOUS, encode_item, item_signatures)
from src.data.graphs import build_datasets


def test_feature_vector_has_26_named_dimensions(sequences):
    vector = encode_item(sequences[0][0])
    assert vector.shape == (FEATURE_DIM,) == (26,)
    assert len(FEATURE_NAMES) == 26
    assert "arrival_index" not in FEATURE_NAMES          # 위험과 무관한 값은 넣지 않는다
    assert any(n.startswith("odor_profile=") for n in FEATURE_NAMES)
    assert any(n.startswith("hazard_type=") for n in FEATURE_NAMES)


def test_normalizer_uses_training_statistics_only(sequences):
    split = {"train": [0, 1, 2], "val": [3], "test": [4]}
    datasets, normalizer = build_datasets(sequences, split)
    train_x = np.concatenate([g.x.numpy() for g in datasets["train"]])
    assert np.allclose(train_x[:, :N_CONTINUOUS].mean(0), 0, atol=1e-4)
    restored = FeatureNormalizer.from_dict(normalizer.to_dict())
    assert np.allclose(restored.mean, normalizer.mean)


def test_signature_ignores_item_identity(sequences):
    a = dict(sequences[0][0])
    b = dict(a, item_uid="other", arrival_index=99)
    assert item_signatures(a) == item_signatures(b)
    assert set(item_signatures(a)) == {"odor", "contamination", "physical"}
