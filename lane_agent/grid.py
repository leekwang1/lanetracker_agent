from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, Iterable, List, Tuple

import numpy as np


class SpatialGrid:
    def __init__(self, xyz: np.ndarray, cell_size: float):
        self.xyz = xyz
        self.cell_size = float(cell_size)
        self.cells: DefaultDict[Tuple[int, int], List[int]] = defaultdict(list)
        self._build()

    def _cell_xy(self, x: float, y: float) -> Tuple[int, int]:
        return int(np.floor(x / self.cell_size)), int(np.floor(y / self.cell_size))

    def _build(self) -> None:
        for idx, (x, y, _z) in enumerate(self.xyz):
            self.cells[self._cell_xy(x, y)].append(idx)

    def query_radius_xy(self, center_xy: np.ndarray, radius: float) -> np.ndarray:
        cx, cy = float(center_xy[0]), float(center_xy[1])
        r = float(radius)
        min_cell = self._cell_xy(cx - r, cy - r)
        max_cell = self._cell_xy(cx + r, cy + r)
        out: List[int] = []
        r2 = r * r
        for gx in range(min_cell[0], max_cell[0] + 1):
            for gy in range(min_cell[1], max_cell[1] + 1):
                for idx in self.cells.get((gx, gy), []):
                    dx = self.xyz[idx, 0] - cx
                    dy = self.xyz[idx, 1] - cy
                    if dx * dx + dy * dy <= r2:
                        out.append(idx)
        if not out:
            return np.empty((0,), dtype=np.int64)
        return np.asarray(out, dtype=np.int64)
