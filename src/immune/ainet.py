"""aiNet 면역 기억.

SKU 쌍 (i -> j) x 관계별로 다음을 기억한다.
  - 클론        : 관측된 위험 노출(항원)에 가까워지도록 선택·변이·억제되는 기억 세포
  - 관측 신뢰도 C: n / (n + kappa), n = 관측 횟수
  - 기억 강도 M : 친화도 x C. 친화도 = 1 - |가장 가까운 클론 - 평균 관측 노출|
                  (기억이 경험과 얼마나 일치하는가). 경험이 일관되고 반복될수록 1에 가깝다
  - 보정 방향 s : clip((평균 관측 노출 - 평균 예측 반발강도) / tau, -1, 1)

모델은 이 값으로 반발강도를 보정한다 (src/models/immune_gat.py):
  rho_m4 = clip(rho_dca + gamma * (M * max(s, 0) + C * min(s, 0)), 0, 1)
"""
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch

from src.config import AiNetConfig
from src.models.outputs import EdgeMemory
from src.schema import RELATIONS, max_grade

_BASE = 1 << 31


class _RelationStore:
    def __init__(self, clone_count: int):
        self.rows: Dict[int, int] = {}
        self.clones = np.full((1024, clone_count), np.nan)
        self.count = np.zeros(1024)
        self.sum_observed = np.zeros(1024)
        self.sum_expected = np.zeros(1024)

    def rows_for(self, keys: np.ndarray, create: bool) -> np.ndarray:
        out = np.empty(len(keys), dtype=np.int64)
        for i, key in enumerate(keys.tolist()):
            row = self.rows.get(key, -1)
            if row < 0 and create:
                row = len(self.rows)
                self.rows[key] = row
                if row >= len(self.count):
                    self._grow()
            out[i] = row
        return out

    def _grow(self):
        n = len(self.count)
        self.clones = np.concatenate([self.clones, np.full_like(self.clones, np.nan)])
        for name in ("count", "sum_observed", "sum_expected"):
            setattr(self, name, np.concatenate([getattr(self, name), np.zeros(n)]))


class AiNetMemory:
    def __init__(self, cfg: Optional[AiNetConfig] = None, seed: int = 42):
        self.cfg = cfg or AiNetConfig()
        self.rng = np.random.default_rng(seed)
        self.node_ids: Dict[tuple, int] = {}
        self.stores = {r: _RelationStore(self.cfg.clone_count) for r in RELATIONS}

    # ------------------------------------------------------------ 키 계산
    def _edge_keys(self, keys: List[tuple], edge_index, create: bool):
        """간선 (i -> j)의 기억 키 = (출발 SKU, 도착 SKU)."""
        ids = np.empty(len(keys), dtype=np.int64)
        for i, key in enumerate(keys):
            node = self.node_ids.get(key, -1)
            if node < 0 and create:
                node = len(self.node_ids)
                self.node_ids[key] = node
            ids[i] = node
        edges = edge_index.cpu().numpy() if torch.is_tensor(edge_index) else np.asarray(edge_index)
        src, dst = ids[edges[0]], ids[edges[1]]
        known = (src >= 0) & (dst >= 0)
        return np.where(known, src * _BASE + dst, -1), known

    # ------------------------------------------------------------ 기억 갱신
    def observe(self, keys: List[tuple], edge_index,
                expected: Dict[str, np.ndarray], exposure: Dict[str, np.ndarray]):
        """한 그래프(또는 적재 한 단계)의 관측으로 기억을 갱신한다.

        expected: 관계별 [E] 모델이 예측한 반발강도 (0~1)
        exposure: 관계별 [E] 관측된 노출 (0~1) — 항원
        """
        edge_keys, _ = self._edge_keys(keys, edge_index, create=True)
        unique, inverse = np.unique(edge_keys, return_inverse=True)
        n = np.bincount(inverse, minlength=len(unique)).astype(float)
        for r in RELATIONS:
            store = self.stores[r]
            rows = store.rows_for(unique, create=True)
            obs_sum = np.bincount(inverse, weights=exposure[r], minlength=len(unique))
            exp_sum = np.bincount(inverse, weights=expected[r], minlength=len(unique))
            store.count[rows] += n
            store.sum_observed[rows] += obs_sum
            store.sum_expected[rows] += exp_sum
            antigen = np.clip(obs_sum / n, 0.0, 1.0)     # 항원 = 실제로 겪은 위험 노출
            store.clones[rows] = self._clonal_update(store.clones[rows], antigen)

    def _clonal_update(self, clones, antigen):
        """클론 선택 -> 변이 -> 억제. clones: [U, K] (빈 칸은 NaN)."""
        cfg, k = self.cfg, clones.shape[1]
        clones = clones.copy()
        empty = np.isnan(clones).all(axis=1)
        clones[empty, 0] = antigen[empty]

        distance = np.where(np.isnan(clones), np.inf, np.abs(clones - antigen[:, None]))
        selected = clones[np.arange(len(clones)), distance.argmin(axis=1)]
        mutation = self.rng.uniform(-cfg.mutation_rate, cfg.mutation_rate, clones.shape)
        offspring = np.clip(
            selected[:, None] + cfg.learning_rate * (antigen - selected)[:, None]
            + mutation, 0.0, 1.0)

        candidates = np.concatenate([clones, offspring], axis=1)
        distance = np.where(np.isnan(candidates), np.inf,
                            np.abs(candidates - antigen[:, None]))
        candidates = np.take_along_axis(candidates, distance.argsort(axis=1), axis=1)

        kept = np.full_like(clones, np.nan)
        kept_n = np.zeros(len(clones), dtype=np.int64)
        for c in range(candidates.shape[1]):          # 항원에 가까운 순서로 억제 적용
            value = candidates[:, c]
            far = np.all(np.isnan(kept)
                         | (np.abs(kept - value[:, None]) >= cfg.suppression_distance), axis=1)
            take = ~np.isnan(value) & far & (kept_n < k)
            kept[take, kept_n[take]] = value[take]
            kept_n[take] += 1
        return kept

    # ------------------------------------------------------------ 기억 조회
    def lookup(self, keys: List[tuple], edge_index, shared: bool = False):
        """간선별 기억 -> ({relation: EdgeMemory}, 기억 적용률). 텐서는 CPU에 둔다.

        shared=True 는 관계를 구분하지 않는 기억(세 관계의 평균)을 돌려준다.
        관계 분리가 없을 때의 면역을 재현하는 비교 조건이다.
        """
        edge_keys, known = self._edge_keys(keys, edge_index, create=False)
        unique, inverse = (np.unique(edge_keys[known], return_inverse=True)
                           if known.any() else (np.empty(0, dtype=np.int64), None))
        values, seen_any = {}, np.zeros(len(edge_keys), dtype=bool)
        for r in RELATIONS:
            store = self.stores[r]
            rows = np.full(len(edge_keys), -1, dtype=np.int64)
            if inverse is not None:
                rows[known] = store.rows_for(unique, create=False)[inverse]
            seen = rows >= 0
            safe_rows = np.where(seen, rows, 0)
            n = np.where(seen, store.count[safe_rows], 0.0)
            mean_obs = np.divide(store.sum_observed[safe_rows], n, out=np.zeros_like(n), where=n > 0)
            mean_exp = np.divide(store.sum_expected[safe_rows], n, out=np.zeros_like(n), where=n > 0)
            confidence = n / (n + self.cfg.kappa)
            mismatch = np.abs(store.clones[safe_rows] - mean_obs[:, None])
            mismatch = np.where(np.isnan(mismatch), np.inf, mismatch).min(axis=1)
            affinity = np.where(seen, np.clip(1.0 - mismatch, 0.0, 1.0), 0.0)
            direction = np.where(seen, np.clip((mean_obs - mean_exp) / self.cfg.tau, -1, 1), 0.0)
            values[r] = (affinity * confidence, direction, confidence, mean_obs)
            seen_any |= seen
        if shared:
            mean = tuple(np.mean([values[r][i] for r in RELATIONS], axis=0) for i in range(4))
            values = {r: mean for r in RELATIONS}

        def tensor(a):
            return torch.as_tensor(a, dtype=torch.float32).view(-1, 1)
        memory = {r: EdgeMemory(*(tensor(v) for v in values[r])) for r in RELATIONS}
        return memory, float(seen_any.mean()) if len(edge_keys) else 0.0

    def size(self) -> int:
        return len(self.stores[RELATIONS[0]].rows)

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "AiNetMemory":
        with open(path, "rb") as f:
            return pickle.load(f)


def to_device(memory: Optional[Dict[str, EdgeMemory]], device):
    if memory is None:
        return None
    return {r: EdgeMemory(m.strength.to(device), m.direction.to(device), m.confidence.to(device),
                          None if m.target is None else m.target.to(device))
            for r, m in memory.items()}


@torch.no_grad()
def prequential_pass(model, graphs, memory: AiNetMemory, device,
                     update: bool = True, shared: bool = False):
    """그래프를 순서대로 지나가며 "조회 -> (예측) -> 관측 -> 갱신"을 수행한다.

    각 그래프의 기억은 그 그래프를 관측하기 **전에** 조회하므로, 자기 자신의
    정답이 기억에 들어가지 않는다. update=False 면 조회만 한다 (기억 고정).

    반환: (그래프별 기억 리스트, 그래프별 기억 적용률 리스트)
    """
    model.eval()
    lookups, coverage = [], []
    for g in graphs:
        edge_memory, covered = memory.lookup(g.keys, g.edge_index, shared=shared)
        lookups.append(edge_memory)
        coverage.append(covered)
        if update:
            out = model(g.x.to(device), g.edge_index.to(device))      # 기억 없이 예측
            expected = {r: out.dca_repulsion[r].squeeze(-1).cpu().numpy() for r in RELATIONS}
            exposure = {r: (g.y[r].float() / max_grade(r)).numpy() for r in RELATIONS}  # 관측
            memory.observe(g.keys, g.edge_index, expected, exposure)
    return lookups, coverage
