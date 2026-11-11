"""온라인 멀티박스 3D 적재 환경.

- 박스: 10 x 10 x 10 (복구된 실험 설정). 박스마다 높이맵(heightmap)으로 적재 상태를 관리한다.
- 상품은 도착 순서대로 하나씩 오고, 되돌릴 수 없다 (온라인).
- 배치: 바닥 좌표 (x, y)를 고르면 z 는 그 밑면 아래의 최고 높이. 회전은 1단계에서는 없다.
- 치수는 올림 처리해 정수 격자에 맞춘다. 박스보다 큰 상품은 혼자 전용 박스에 담긴 것으로
  간주하고 적재 계산에서 제외한다 (드물어야 정상이며, 결과에 개수를 기록한다).
"""
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

BOX = 10


def item_dims(item: Dict) -> Tuple[int, int, int]:
    w, d, h = (max(1, math.ceil(float(v))) for v in item["size"])
    return w, d, h


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

    def candidates(self, dims) -> List[Tuple[int, int, int]]:
        """놓을 수 있는 (x, y, z) 목록. z = 밑면 아래 최고 높이."""
        w, d, h = dims
        spots = []
        for x in range(BOX - w + 1):
            for y in range(BOX - d + 1):
                z = int(self.heightmap[x:x + w, y:y + d].max())
                if z + h <= BOX:
                    spots.append((x, y, z))
        return spots

    def best_spot(self, dims) -> Optional[Tuple[int, int, int]]:
        """가장 낮은 자리(z, 그다음 x, y 순)를 고른다."""
        spots = self.candidates(dims)
        return min(spots, key=lambda s: (s[2], s[0], s[1])) if spots else None

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

    def __init__(self, items: List[Dict]):
        self.items = items
        self.boxes: List[Box] = []
        self.oversized: List[int] = []           # 박스보다 큰 상품 (전용 박스로 간주)

    def feasible_actions(self, item_index: int) -> List[Tuple[int, Tuple[int, int, int]]]:
        """(박스 번호, 자리) 목록. 박스 번호 == len(boxes) 는 '새 박스 열기'."""
        dims = item_dims(self.items[item_index])
        if max(dims) > BOX:
            return []
        actions = []
        for b, box in enumerate(self.boxes):
            spot = box.best_spot(dims)
            if spot is not None:
                actions.append((b, spot))
        actions.append((len(self.boxes), (0, 0, 0)))     # 새 박스는 항상 가능
        return actions

    def step(self, item_index: int, box_id: int, spot) -> int:
        """상품을 놓고 실제 박스 번호를 돌려준다."""
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
