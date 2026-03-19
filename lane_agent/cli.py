from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .agent import LaneTrackerAgent
from .config import load_config
from .csv_io import save_xyz_csv
from .las_io import load_las_xyz_intensity


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Lane tracking agent for MMS LAS data")
    p.add_argument("--las", required=True, help="Input LAS path")
    p.add_argument("--p0", nargs=3, required=True, type=float, metavar=("X", "Y", "Z"))
    p.add_argument("--p1", nargs=3, required=True, type=float, metavar=("X", "Y", "Z"))
    p.add_argument("--config", required=True, help="config.yaml path")
    p.add_argument("--output", required=True, help="Output CSV path")
    return p


def main() -> None:
    args = build_parser().parse_args()
    cfg = load_config(args.config)
    las = load_las_xyz_intensity(args.las)

    p0 = np.asarray(args.p0, dtype=np.float64)
    p1 = np.asarray(args.p1, dtype=np.float64)

    agent = LaneTrackerAgent(las.xyz, las.intensity, cfg)
    output_path = Path(args.output)
    post_output_path = output_path.with_name(f"{output_path.stem}_pc{output_path.suffix}")
    debug_path = output_path.with_suffix(output_path.suffix + ".debug.json")
    result = agent.track(p0, p1, str(debug_path))
    save_xyz_csv(output_path, result.raw_points)
    save_xyz_csv(post_output_path, result.points)

    print(f"Saved raw CSV: {output_path}")
    print(f"Saved post-corrected CSV: {post_output_path}")
    print(f"Stop reason: {result.stop_reason}")
    if cfg.get("save_debug_json", True):
        print(f"Saved debug: {debug_path}")


if __name__ == "__main__":
    main()
