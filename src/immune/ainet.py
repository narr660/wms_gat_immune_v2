"""aiNet 면역 기억.

상품 쌍(시그니처) x 관계별로 다음을 기억한다.
  - 위험 기억 M : 클론 친화도. 위험 노출 경험(항원)에 가까워지도록 선택·변이·억제
  - 관측 신뢰도 C: n / (n + kappa), n = 관측 횟수
  - 보정 방향 s : clip((평균 관측 노출 - 평균 예측 위험) / tau, -1, 1)

키는 개별 상품이 아니라 관계별 상품 시그니처(그 위험과 관련된 속성 조합)
이므로, 처음 보는 상품에도 같은 종류의 경험을 적용할 수 있다.
"""
import pickle
from typing import Dict, List, Optional

import numpy as np
import torch

from src.config import AiNetConfig
from src.models.outputs import EdgeMemory
from src.schema import RELATIONS

_BASE = 1 << 31


class _RelationStore:
    def __init__(self, clone_count: int):
        self.signature_ids: Dict[tuple, int] = {}
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
        self.stores = {r: _RelationStore(self.cfg.clone_count) for r in RELATIONS}

    # ------------------------------------------------------------ 키 계산
    def _edge_keys(self, relation, signatures: List[tuple], edge_index, create: bool):
        """간선 (i -> j)의 기억 키 = (출발 상품 시그니처, 도착 상품 시그니처)."""
        table = self.stores[relation].signature_ids
        ids = np.empty(len(signatures), dtype=np.int64)
        for i, sig in enumerate(signatures):
            sid = table.get(sig, -1)
            if sid < 0 and create:
                sid = len(table)
                table[sig] = sid
            ids[i] = sid
        edges = edge_index.cpu().numpy() if torch.is_tensor(edge_index) else np.asarray(edge_index)
        src, dst = ids[edges[0]], ids[edges[1]]
        known = (src >= 0) & (dst >= 0)
        return np.where(known, src * _BASE + dst, -1), known

    # ------------------------------------------------------------ 기억 갱신
    def observe(self, signatures: Dict[str, List[tuple]], edge_index,
                expected: Dict[str, np.ndarray], exposure: Dict[str, np.ndarray],
                activation: Optional[Dict[str, np.ndarray]] = None):
        """한 그래프(또는 적재 한 단계)의 관측으로 기억을 갱신한다.

        signatures: 관계별 노드 시그니처
        expected  : 관계별 [E] 모델의 기대 위험 (0~1)
        exposure  : 관계별 [E] 관측된 노출 (0~1)
        activation: 관계별 [E] DCA 활성 (없으면 1)
        """
        for r in RELATIONS:
            store = self.stores[r]
            keys, _ = self._edge_keys(r, signatures[r], edge_index, create=True)
            unique, inverse = np.unique(keys, return_inverse=True)
            n = np.bincount(inverse, minlength=len(unique)).astype(float)
            rows = store.rows_for(unique, create=True)
            obs_sum = np.bincount(inverse, weights=exposure[r], minlength=len(unique))
            exp_sum = np.bincount(inverse, weights=expected[r], minlength=len(unique))
            act = (np.bincount(inverse, weights=activation[r], minlength=len(unique)) / n
                   if activation else np.ones(len(unique)))
            store.count[rows] += n
            store.sum_observed[rows] += obs_sum
            store.sum_expected[rows] += exp_sum
            observed, predicted = obs_sum / n, exp_sum / n
            antigen = np.clip(predicted * observed * act, 0.0, 1.0)
            store.clones[rows] = self._clonal_update(store.clones[rows], antigen, observed * act)

    def _clonal_update(self, clones, antigen, scale):
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
            + mutation * scale[:, None], 0.0, 1.0)

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
    def lookup(self, signatures: Dict[str, List[tuple]], edge_index, device="cpu"):
        """간선별 기억 -> ({relation: EdgeMemory}, {relation: 기억 적용률})."""
        memory, coverage = {}, {}
        for r in RELATIONS:
            store = self.stores[r]
            keys, known = self._edge_keys(r, signatures[r], edge_index, create=False)
            rows = np.full(len(keys), -1, dtype=np.int64)
            if known.any():
                unique, inverse = np.unique(keys[known], return_inverse=True)
                rows[known] = store.rows_for(unique, create=False)[inverse]
            seen = rows >= 0
            safe_rows = np.where(seen, rows, 0)
            n = np.where(seen, store.count[safe_rows], 0.0)
            clones = np.where(np.isnan(store.clones[safe_rows]), -np.inf, store.clones[safe_rows])
            strength = np.where(seen, np.clip(clones.max(axis=1), 0.0, 1.0), 0.0)
            mean_obs = np.divide(store.sum_observed[safe_rows], n, out=np.zeros_like(n), where=n > 0)
            mean_exp = np.divide(store.sum_expected[safe_rows], n, out=np.zeros_like(n), where=n > 0)
            direction = np.where(seen, np.clip((mean_obs - mean_exp) / self.cfg.tau, -1, 1), 0.0)
            confidence = n / (n + self.cfg.kappa)

            def tensor(a):
                return torch.as_tensor(a, dtype=torch.float32, device=device).view(-1, 1)
            memory[r] = EdgeMemory(tensor(strength), tensor(direction), tensor(confidence))
            coverage[r] = float(seen.mean()) if len(keys) else 0.0
        return memory, coverage

    # ------------------------------------------------------------ 저장
    def size(self) -> Dict[str, int]:
        return {r: len(self.stores[r].rows) for r in RELATIONS}

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "AiNetMemory":
        with open(path, "rb") as f:
            return pickle.load(f)


@torch.no_grad()
def build_memory_from_training(model, train_graphs, cfg: AiNetConfig, device,
                               seed: int = 42) -> AiNetMemory:
    """학습 그래프만으로 기억을 만든다. 검증·테스트 정답은 쓰지 않는다.

    관측 노출 O = 실제 등급 / 최고 등급, 예측 위험 R = 등급 확률의 기댓값(0~1).
    """
    from src.schema import max_grade
    memory = AiNetMemory(cfg, seed)
    model.eval()
    for g in train_graphs:
        out = model(g.x.to(device), g.edge_index.to(device))
        expected, exposure, activation = {}, {}, {}
        for r in RELATIONS:
            probs = torch.softmax(out.logits[r], dim=-1)
            grades = torch.arange(probs.shape[1], device=probs.device, dtype=probs.dtype)
            expected[r] = ((probs * grades).sum(-1) / max_grade(r)).cpu().numpy()
            exposure[r] = (g.y[r].float() / max_grade(r)).numpy()
            if out.danger:
                act = 0.5 * out.danger[r] + 0.5 * (1.0 - out.safe[r])
                activation[r] = act.squeeze(-1).clamp(0, 1).cpu().numpy()
        memory.observe(g.signatures, g.edge_index, expected, exposure, activation or None)
    return memory
