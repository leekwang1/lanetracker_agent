from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .grid import SpatialGrid
from .scoring import SeedProfile, blend_seed_profile, estimate_seed_profile, score_candidate, unit


@dataclass
class TrackResult:
    points: np.ndarray
    scores: List[float]
    stop_reason: str


class LaneTrackerAgent:
    def __init__(self, xyz: np.ndarray, intensity: np.ndarray, cfg: Dict[str, Any]):
        self.xyz = xyz
        self.intensity = intensity
        self.cfg = cfg
        self.grid = SpatialGrid(xyz, float(cfg["grid_size_m"]))

    def _fit_center_z(self, center_xy: np.ndarray, radius: float, fallback_z: float) -> float:
        idx = self.grid.query_radius_xy(center_xy, radius)
        if idx.size == 0:
            return fallback_z
        return float(np.median(self.xyz[idx, 2]))

    def _build_seed_profile(self, p0: np.ndarray) -> SeedProfile:
        idx = self.grid.query_radius_xy(p0[:2], float(self.cfg["seed_profile_radius_m"]))
        return estimate_seed_profile(self.xyz, self.intensity, idx)

    def _refresh_seed_profile(self, center: np.ndarray, current: SeedProfile) -> SeedProfile:
        radius = float(self.cfg.get("profile_update_radius_m", self.cfg["seed_profile_radius_m"]))
        alpha = float(self.cfg.get("profile_update_alpha", 0.12))
        idx = self.grid.query_radius_xy(center[:2], radius)
        if idx.size == 0:
            return current
        updated = estimate_seed_profile(self.xyz, self.intensity, idx)
        return blend_seed_profile(current, updated, alpha)

    def _candidate_centers(self, center: np.ndarray, direction: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        direction = unit(direction)
        normal = np.array([-direction[1], direction[0], 0.0], dtype=np.float64)
        step_m = float(self.cfg["step_m"])
        half_w = float(self.cfg["search_half_width_m"])
        f_count = int(self.cfg["search_forward_samples"])
        l_count = int(self.cfg["search_lateral_samples"])

        out: List[Tuple[np.ndarray, np.ndarray]] = []
        forward_fracs = np.linspace(0.85, 1.15, f_count)
        lateral_offsets = np.linspace(-half_w, half_w, l_count)
        for ff in forward_fracs:
            for lo in lateral_offsets:
                c_xy = center[:2] + direction[:2] * (step_m * ff) + normal[:2] * lo
                out.append((c_xy, direction.copy()))

        # mild heading changes for curve following
        max_delta = np.radians(float(self.cfg["max_heading_change_deg"]))
        for delta in np.linspace(-max_delta, max_delta, 7):
            if abs(delta) < 1e-9:
                continue
            cs = float(np.cos(delta))
            sn = float(np.sin(delta))
            d2 = np.array([
                cs * direction[0] - sn * direction[1],
                sn * direction[0] + cs * direction[1],
                0.0,
            ])
            c_xy = center[:2] + d2[:2] * step_m
            out.append((c_xy, unit(d2)))
        return out

    def track(self, p0: np.ndarray, p1: np.ndarray, debug_json_path: str | None = None) -> TrackResult:
        profile = self._build_seed_profile(p0)
        cur = p0.astype(np.float64).copy()
        cur_dir = unit(p1 - p0)
        cur[2] = self._fit_center_z(cur[:2], float(self.cfg["search_radius_m"]), p0[2])

        points: List[np.ndarray] = [cur.copy()]
        scores: List[float] = [1.0]
        traveled = 0.0
        gap_accum = 0.0
        stop_reason = "max_length_reached"
        debug_steps: List[Dict[str, Any]] = []
        gap_steps = 0

        max_len = float(self.cfg["max_track_length_m"])
        min_score = float(self.cfg["min_score"])
        step_m = float(self.cfg["step_m"])
        max_gap = float(self.cfg["max_gap_m"])
        query_r = float(self.cfg["search_radius_m"])

        while traveled < max_len:
            best_score = -1.0
            best_center = None
            best_dir = None
            candidates_debug: List[Dict[str, Any]] = []

            for c_xy, c_dir in self._candidate_centers(cur, cur_dir):
                idx = self.grid.query_radius_xy(c_xy, query_r)
                c_z = self._fit_center_z(c_xy, query_r, cur[2])
                center3 = np.array([c_xy[0], c_xy[1], c_z], dtype=np.float64)
                sc = score_candidate(
                    candidate_center=center3,
                    candidate_dir=c_dir,
                    prev_center=cur,
                    prev_dir=cur_dir,
                    xyz=self.xyz,
                    intensity=self.intensity,
                    indices=idx,
                    seed_profile=profile,
                    cfg=self.cfg,
                )
                candidates_debug.append({
                    "x": float(center3[0]),
                    "y": float(center3[1]),
                    "z": float(center3[2]),
                    "score": float(sc),
                })
                if sc > best_score:
                    best_score = sc
                    best_center = center3
                    best_dir = c_dir

            if best_center is None or best_dir is None:
                stop_reason = "no_candidate"
                break

            if best_score < min_score:
                gap_accum += step_m
                gap_steps += 1
                if gap_accum > max_gap:
                    stop_reason = "score_below_threshold"
                    break
                # bridge forward but do not store a point yet
                cur = best_center
                cur_dir = best_dir
                traveled += step_m
                debug_steps.append({
                    "mode": "gap_bridge",
                    "best_score": float(best_score),
                    "x": float(cur[0]),
                    "y": float(cur[1]),
                    "z": float(cur[2]),
                    "candidates": candidates_debug,
                })
                continue

            gap_accum = 0.0
            cur = best_center
            cur_dir = best_dir
            profile = self._refresh_seed_profile(cur, profile)
            points.append(cur.copy())
            scores.append(float(best_score))
            traveled += float(np.linalg.norm(points[-1][:2] - points[-2][:2])) if len(points) >= 2 else step_m
            debug_steps.append({
                "mode": "accept",
                "best_score": float(best_score),
                "x": float(cur[0]),
                "y": float(cur[1]),
                "z": float(cur[2]),
                "candidates": candidates_debug,
            })

        arr = np.vstack(points) if points else np.empty((0, 3), dtype=np.float64)
        if arr.shape[0] >= 3 and int(self.cfg.get("smoothing_window", 1)) >= 3:
            arr = self._smooth(arr, int(self.cfg["smoothing_window"]))

        if debug_json_path and bool(self.cfg.get("save_debug_json", True)):
            payload = {
                "stop_reason": stop_reason,
                "num_points": int(arr.shape[0]),
                "num_steps": int(len(debug_steps)),
                "gap_steps": int(gap_steps),
                "scores": scores,
                "steps": debug_steps,
            }
            Path(debug_json_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        return TrackResult(points=arr, scores=scores, stop_reason=stop_reason)

    def _smooth(self, pts: np.ndarray, window: int) -> np.ndarray:
        if window < 3 or pts.shape[0] < window:
            return pts
        half = window // 2
        out = pts.copy()
        for i in range(pts.shape[0]):
            a = max(0, i - half)
            b = min(pts.shape[0], i + half + 1)
            out[i] = np.mean(pts[a:b], axis=0)
        return out
