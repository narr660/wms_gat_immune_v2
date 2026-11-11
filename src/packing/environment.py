"""온라인 멀티박스 3D 적재 환경.

- 박스: 10 x 10 x 10 (복구된 실험 설정). 박스마다 높이맵(heightmap)으로 적재 상태를 관리한다.
- 상품은 도착 순서대로 하나씩 오고, 되돌릴 수 없다 (온라인).
- 배치: 바닥 좌표 (x, y)를 고르면 z 는 그 밑면 아래의 최고 높이.
- 회전: 축 정렬 회전(세 치수의 순열)을 허용한다. allow_rotation=False 로 끄면
  도착한 방향 그대로만 놓는다 — 회전 효과를 같은 조건에서 비교하기 위한 스위치.
- 치수는 올림 처리해 정수 격자에 맞춘다. 박스보다 큰 상품은 혼자 전용 박스에 담긴 것으로
  간주하고 적재 계산에서 제외한다 (드물어야 정상이며, 결과에 개수를 기록한다).
"""
import math
from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

BOX = 10


def item_dims(item: Dict) -> Tuple[int, int, int]:
    w, d, h = (max(1, math.ceil(float(v))) for v in item["size"])
    return w, d, h


def orientations(dims: Tuple[int, int, int], allow_rotation: bool) -> List[Tuple[int, int, int]]:
    """축 정렬 회전들. 납작하고(h 작고) 바닥이 넓은 순서로 — 탐욕이 동률일 때 먼저 집는다."""
    if not allow_rotation:
        return [dims]
    unique = sorted(set(permutations(dims)), key=lambda o: (o[2], -(o[0] * o[1])))
    return unique


@dataclass
class Placement:
    item_index: int
    x: int
    y: int
    z: int
    dims: Tuple[int, int, int]


@dataclass
class Box:
    heightmap: np.ndarray = field(default_factory=lambda: np.zeros((BOX, BOX), dtype=np.int64))
    placements: List[Placement] = field(default_factory=list)

    def item_indices(self) -> List[int]:
        return [p.item_index for p in self.placements]

    def _z_grid(self, w: int, d: int) -> Optional[np.ndarray]:
        """모든 바닥 좌표의 z (밑면 아래 최고 높이)를 한 번에 계산. [BOX-w+1, BOX-d+1]."""
        if w > BOX or d > BOX:
            return None
        return sliding_window_view(self.heightmap, (w, d)).max(axis=(2, 3))

    def candidates(self, dims) -> List[Tuple[int, int, int]]:
        """놓을 수 있는 (x, y, z) 목록. z = 밑면 아래 최고 높이."""
        w, d, h = dims
        z = self._z_grid(w, d)
        if z is None or h > BOX:
            return []
        xs, ys = np.nonzero(z + h <= BOX)
        return [(int(x), int(y), int(z[x, y])) for x, y in zip(xs, ys)]

    def best_spot(self, dims) -> Optional[Tuple[int, int, int]]:
        """가장 낮은 자리(z, 그다음 x, y 순)를 고른다."""
        w, d, h = dims
        z = self._z_grid(w, d)
        if z is None or h > BOX:
            return None
        masked = np.where(z + h <= BOX, z, BOX + 1)
        x, y = np.unravel_index(int(np.argmin(masked)), masked.shape)   # 행 우선 = (z, x, y) 순 동률 처리
        if masked[x, y] > BOX:
            return None
        return int(x), int(y), int(masked[x, y])

    def place(self, item_index: int, dims, spot) -> Placement:
        w, d, h = dims
        x, y, z = spot
        assert int(self.heightmap[x:x + w, y:y + d].max()) == z, "자리가 더 이상 유효하지 않습니다."
        self.heightmap[x:x + w, y:y + d] = z + h
        placement = Placement(item_index, x, y, z, dims)
        self.placements.append(placement)
        return placement

    def used_volume(self) -> int:
        return sum(w * d * h for _, _, _, _, (w, d, h) in
                   ((p.item_index, p.x, p.y, p.z, p.dims) for p in self.placements))


class PackingEnv:
    """한 시퀀스(상품 100개)를 받아 박스들에 차례로 적재한다."""

    def __init__(self, items: List[Dict], allow_rotation: bool = True):
        self.items = items
        self.allow_rotation = allow_rotation
        self.boxes: List[Box] = []
        self.oversized: List[int] = []           # 박스보다 큰 상품 (전용 박스로 간주)

    def feasible_actions(self, item_index: int):
        """(박스 번호, 자리, 회전된 치수) 목록. 박스 번호 == len(boxes) 는 '새 박스 열기'.

        박스마다 회전들 중 가장 낮은 자리(동률이면 납작한 회전)를 하나씩 올린다.
        """
        base = item_dims(self.items[item_index])
        if max(base) > BOX:                       # 어떤 회전으로도 안 들어감
            return []
        actions = []
        rots = orientations(base, self.allow_rotation)
        for b, box in enumerate(self.boxes):
            best = None
            for dims in rots:
                spot = box.best_spot(dims)
                if spot is not None and (best is None or spot[2] < best[0][2]):
                    best = (spot, dims)
            if best is not None:
                actions.append((b, best[0], best[1]))
        actions.append((len(self.boxes), (0, 0, 0), rots[0]))    # 새 박스: 납작한 회전으로
        return actions

    def step(self, item_index: int, box_id: int, spot, dims=None) -> int:
        """상품을 놓고 실제 박스 번호를 돌려준다. dims 는 feasible_actions 가 고른 회전."""
        if dims is None:
            dims = item_dims(self.items[item_index])
        if max(dims) > BOX:
            self.oversized.append(item_index)
            return -1
        if box_id == len(self.boxes):
            self.boxes.append(Box())
        self.boxes[box_id].place(item_index, dims, spot)
        return box_id

    def utilization(self) -> float:
        if not self.boxes:
            return 0.0
        return sum(b.used_volume() for b in self.boxes) / (len(self.boxes) * BOX ** 3)
