from __future__ import annotations

from typing import Tuple

import laspy
import numpy as np


class LasData:
    def __init__(self, xyz: np.ndarray, intensity: np.ndarray):
        self.xyz = xyz.astype(np.float64, copy=False)
        self.intensity = intensity.astype(np.float64, copy=False)


def load_las_xyz_intensity(path: str) -> LasData:
    las = laspy.read(path)
    xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64)
    if hasattr(las, "intensity"):
        intensity = np.asarray(las.intensity, dtype=np.float64)
    else:
        raise ValueError("LAS intensity attribute is required.")
    return LasData(xyz=xyz, intensity=intensity)
