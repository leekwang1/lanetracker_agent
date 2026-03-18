from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable

import numpy as np


def save_xyz_csv(path: str | Path, points: np.ndarray) -> None:
    path = Path(path)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y", "z"])
        for row in points:
            writer.writerow([f"{float(row[0]):.6f}", f"{float(row[1]):.6f}", f"{float(row[2]):.6f}"])
