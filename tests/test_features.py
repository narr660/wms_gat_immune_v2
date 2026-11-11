import numpy as np

from src.data.catalog import SkuCatalog, sku_key
from src.data.features import (FEATURE_DIM, FEATURE_NAMES, HIDDEN_FEATURES, FeatureNormalizer,
                               N_CONTINUOUS, encode_item, memory_key, observable_mask)
from src.data.graphs import build_datasets

SPLIT = {"train": [0, 1, 2], "val": [3], "test": [4]}


def test_feature_vector_has_26_named_dimensions(sequences):
    assert encode_item(sequences[0][0]).shape == (FEATURE_DIM,) == (26,)
    assert len(FEATURE_NAMES) == 26
    assert "arrival_index" not in FEATURE_NAMES          # 위험과 무관한 값은 넣지 않는다


def test_partial_observability_hides_latent_attributes(sequences):
    mask = observable_mask("partial")
    hidden = [i for i, n in enumerate(FEATURE_NAMES) if n in HIDDEN_FEATURES]
    assert len(hidden) == len(HIDDEN_FEATURES) and mask[hidden].sum() == 0
    datasets, _ = build_datasets(sequences, SPLIT, observability="partial")
    assert np.abs(datasets["train"][0].x.numpy()[:, hidden]).sum() == 0
    full, _ = build_datasets(sequences, SPLIT, observability="full")
    assert np.abs(full["train"][0].x.numpy()[:, hidden]).sum() > 0


def test_normalizer_uses_training_statistics_only(sequences):
    datasets, normalizer = build_datasets(sequences, SPLIT, observability="full")
    train_x = np.concatenate([g.x.numpy() for g in datasets["train"]])
    assert np.allclose(train_x[:, :N_CONTINUOUS].mean(0), 0, atol=1e-4)
    restored = FeatureNormalizer.from_dict(normalizer.to_dict())
    assert np.allclose(restored.mean, normalizer.mean)


def test_same_sku_has_same_attributes_everywhere(raw_sequences, sequences):
    catalog = SkuCatalog(raw_sequences)
    seen = {}
    for raw_seq, seq in zip(raw_sequences, sequences):
        for raw, item in zip(raw_seq, seq):
            key = sku_key(raw)
            attrs = {k: v for k, v in item.items() if k not in ("item_uid", "arrival_index")}
            assert seen.setdefault(key, attrs) == attrs
            assert memory_key(item) == (catalog.ids[key],)
    assert len(seen) == len(catalog) < sum(len(s) for s in sequences)   # SKU가 반복된다
