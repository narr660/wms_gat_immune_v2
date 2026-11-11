"""시퀀스 -> 방향성 완전 그래프 (자기 간선 제외)."""
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from src.data.features import FeatureNormalizer, encode_item, memory_key, observable_mask
from src.data.labels import pair_labels
from src.schema import RELATIONS


@dataclass
class GraphSample:
    sequence_id: int
    x: torch.Tensor                 # [N, 26] 정규화된 특징
    edge_index: torch.Tensor        # [2, E], E = N(N-1)
    y: Dict[str, torch.Tensor]      # 관계별 [E] 등급
    keys: List[tuple]               # 노드별 aiNet 기억 키 (SKU)


def load_sequences(path) -> list:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise TypeError(f"시퀀스 리스트가 필요합니다: {type(data)}")
    return data


def load_split(path) -> Dict[str, List[int]]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    indices = raw.get("indices", raw)
    split = {k: [int(i) for i in indices[k]] for k in ("train", "val", "test")}
    assert set(split["train"]).isdisjoint(split["val"])
    assert set(split["train"]).isdisjoint(split["test"])
    assert set(split["val"]).isdisjoint(split["test"])
    return split


def _ordered_items(sequence, max_nodes: Optional[int]) -> list:
    items = sorted(sequence, key=lambda it: int(it.get("arrival_index", 0)))
    return items[:max_nodes] if max_nodes else items


def complete_edges(n: int) -> np.ndarray:
    src, dst = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    keep = src != dst
    return np.stack([src[keep], dst[keep]])


def raw_features(sequence, max_nodes=None) -> np.ndarray:
    return np.stack([encode_item(it) for it in _ordered_items(sequence, max_nodes)])


def build_graph(sequence, sequence_id: int, normalizer: FeatureNormalizer,
                max_nodes: Optional[int] = None, observability: str = "partial") -> GraphSample:
    items = _ordered_items(sequence, max_nodes)
    if len(items) < 2:
        raise ValueError(f"시퀀스 {sequence_id}: 상품이 2개 이상 필요합니다.")
    edges = complete_edges(len(items))
    labels = pair_labels(items, edges[0], edges[1])        # 등급은 모든 속성으로 정해진다
    x = normalizer.transform(np.stack([encode_item(it) for it in items]))
    return GraphSample(
        sequence_id=sequence_id,
        x=torch.from_numpy(x * observable_mask(observability)),   # 숨은 속성은 0
        edge_index=torch.from_numpy(edges).long(),
        y={r: torch.from_numpy(labels[r]) for r in RELATIONS},
        keys=[memory_key(it) for it in items],
    )


def build_datasets(sequences, split: Dict[str, Sequence[int]], max_nodes=None,
                   limits: Optional[Dict[str, Optional[int]]] = None,
                   observability: str = "partial"):
    """정규화 통계는 학습셋에서만 계산한다."""
    limits = limits or {}
    ids = {k: list(v)[: limits.get(k) or None] for k, v in split.items()}
    normalizer = FeatureNormalizer.fit(
        [raw_features(sequences[i], max_nodes) for i in ids["train"]])
    datasets = {
        name: [build_graph(sequences[i], i, normalizer, max_nodes, observability)
               for i in id_list]
        for name, id_list in ids.items()
    }
    return datasets, normalizer
