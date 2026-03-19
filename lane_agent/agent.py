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
    raw_points: np.ndarray
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
        z_gate = float(self.cfg.get("z_fit_window_m", self.cfg.get("max_z_step_m", 0.12) * 1.5))
        local_z = self.xyz[idx, 2]
        mask = np.abs(local_z - fallback_z) <= z_gate
        if np.count_nonzero(mask) >= 3:
            return float(np.median(local_z[mask]))
        return float(np.median(local_z))

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

    def _predict_direction(self, points: List[np.ndarray], fallback_dir: np.ndarray) -> np.ndarray:
        history_len = int(self.cfg.get("direction_history_points", 6))
        if len(points) < 2:
            return unit(fallback_dir)
        hist = np.asarray(points[-history_len:], dtype=np.float64)
        if hist.shape[0] < 2:
            return unit(fallback_dir)
        deltas = hist[1:, :2] - hist[:-1, :2]
        lengths = np.linalg.norm(deltas, axis=1)
        valid = lengths > 1e-9
        if not np.any(valid):
            return unit(fallback_dir)
        deltas = deltas[valid]
        weights = np.linspace(1.0, 2.0, deltas.shape[0], dtype=np.float64)
        pred = np.sum(deltas * weights[:, None], axis=0)
        if float(np.linalg.norm(pred)) < 1e-9:
            pred = unit(fallback_dir[:2])
        pred3 = np.array([pred[0], pred[1], 0.0], dtype=np.float64)
        return unit(pred3)

    def _candidate_is_allowed(self, candidate_center: np.ndarray, prev_center: np.ndarray, prev_dir: np.ndarray) -> bool:
        step_ref = max(float(self.cfg["step_m"]), 1e-6)
        prev_dir_xy = unit(prev_dir[:2])
        pred_xy = prev_center[:2] + prev_dir_xy * step_ref
        normal_xy = np.array([-prev_dir_xy[1], prev_dir_xy[0]], dtype=np.float64)
        lateral_offset = abs(float(np.dot(candidate_center[:2] - pred_xy, normal_xy)))
        hard_limit = float(self.cfg.get("hard_center_offset_limit_m", self.cfg.get("center_offset_tolerance_m", 0.18) * 1.4))
        if lateral_offset > hard_limit:
            return False

        max_z_step = float(self.cfg.get("hard_max_z_step_m", self.cfg.get("max_z_step_m", 0.12) * 1.2))
        if abs(float(candidate_center[2] - prev_center[2])) > max_z_step:
            return False
        return True

    def _lane_loyalty_term(self, points: List[np.ndarray], candidate_center: np.ndarray, pred_dir: np.ndarray) -> float:
        history_len = int(self.cfg.get("lane_loyalty_history_points", 8))
        if len(points) < 3:
            return 1.0

        hist = np.asarray(points[-history_len:], dtype=np.float64)
        if hist.shape[0] < 2:
            return 1.0

        anchor = hist[-1, :2]
        dir_xy = unit(pred_dir[:2])
        normal_xy = np.array([-dir_xy[1], dir_xy[0]], dtype=np.float64)
        lateral_hist = (hist[:, :2] - anchor) @ normal_xy
        center_bias = float(np.median(lateral_hist))
        candidate_lateral = float(np.dot(candidate_center[:2] - anchor, normal_xy))
        loyalty_offset = abs(candidate_lateral - center_bias)

        tolerance = max(float(self.cfg.get("lane_loyalty_tolerance_m", self.cfg.get("center_offset_tolerance_m", 0.18) * 0.75)), 1e-3)
        return float(np.clip(1.0 - loyalty_offset / tolerance, 0.0, 1.0))

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

    def _refine_center_xy(self, center: np.ndarray, direction: np.ndarray) -> np.ndarray:
        radius = float(self.cfg.get("center_refine_radius_m", self.cfg["search_radius_m"]))
        idx = self.grid.query_radius_xy(center[:2], radius)
        if idx.size < 6:
            return center

        max_z_step = float(self.cfg.get("max_z_step_m", 0.12))
        local_xyz = self.xyz[idx]
        local_i = self.intensity[idx]
        z_mask = np.abs(local_xyz[:, 2] - center[2]) <= max_z_step
        if np.count_nonzero(z_mask) < 4:
            return center

        pts = local_xyz[z_mask]
        weights = local_i[z_mask].astype(np.float64)
        if weights.size == 0:
            return center

        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        deltas = pts[:, :2] - center[:2]
        lateral_offsets = deltas @ normal

        shift_limit = float(self.cfg.get("center_refine_max_shift_m", 0.08))
        close_term = np.clip(1.0 - np.abs(lateral_offsets) / max(shift_limit * 2.0, 1e-3), 0.0, 1.0)
        weights = np.maximum(weights - float(np.min(weights)), 0.0) + 1e-3
        weights *= 0.35 + 0.65 * close_term
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            return center

        shift = float(np.sum(lateral_offsets * weights) / weight_sum)
        shift = float(np.clip(shift, -shift_limit, shift_limit))
        refined = center.copy()
        refined[:2] = refined[:2] + normal * shift
        return refined

    def _refine_centerline_cross_section(self, center: np.ndarray, direction: np.ndarray) -> np.ndarray:
        radius = float(self.cfg.get("cross_section_radius_m", max(self.cfg["search_half_width_m"], self.cfg["search_radius_m"])))
        idx = self.grid.query_radius_xy(center[:2], radius)
        if idx.size < 10:
            return center

        local_xyz = self.xyz[idx]
        local_i = self.intensity[idx].astype(np.float64)
        max_z_step = float(self.cfg.get("max_z_step_m", 0.12))
        z_mask = np.abs(local_xyz[:, 2] - center[2]) <= max_z_step
        if np.count_nonzero(z_mask) < 6:
            return center

        pts = local_xyz[z_mask]
        vals = local_i[z_mask]
        dir_xy = unit(direction[:2])
        normal = np.array([-dir_xy[1], dir_xy[0]], dtype=np.float64)
        deltas = pts[:, :2] - center[:2]
        along = deltas @ dir_xy
        lateral = deltas @ normal

        along_half = float(self.cfg.get("cross_section_forward_window_m", max(float(self.cfg["step_m"]) * 0.75, 0.12)))
        lane_half = float(self.cfg.get("lane_half_width_m", 0.09))
        mask = (np.abs(along) <= along_half) & (np.abs(lateral) <= lane_half * 2.5)
        if np.count_nonzero(mask) < 6:
            return center

        lateral = lateral[mask]
        vals = vals[mask]
        val_floor = float(np.quantile(vals, 0.35))
        weights = np.clip(vals - val_floor, 0.0, None)
        if float(np.sum(weights)) <= 1e-9:
            return center

        bin_size = float(self.cfg.get("cross_section_bin_size_m", 0.02))
        bin_size = max(bin_size, 1e-3)
        bins = np.arange(-lane_half * 2.5, lane_half * 2.5 + bin_size, bin_size, dtype=np.float64)
        if bins.size < 4:
            return center

        hist, edges = np.histogram(lateral, bins=bins, weights=weights)
        if hist.size < 3:
            return center

        smooth_hist = hist.copy()
        smooth_hist[1:-1] = 0.25 * hist[:-2] + 0.5 * hist[1:-1] + 0.25 * hist[2:]
        peak_idx = int(np.argmax(smooth_hist))
        peak_value = float(smooth_hist[peak_idx])
        if peak_value <= 1e-9:
            return center

        stripe_ratio = float(self.cfg.get("cross_section_stripe_threshold_ratio", 0.35))
        active_threshold = peak_value * stripe_ratio
        left_idx = peak_idx
        right_idx = peak_idx
        while left_idx > 0 and smooth_hist[left_idx - 1] >= active_threshold:
            left_idx -= 1
        while right_idx < smooth_hist.size - 1 and smooth_hist[right_idx + 1] >= active_threshold:
            right_idx += 1

        stripe_left = float(edges[left_idx])
        stripe_right = float(edges[right_idx + 1])
        stripe_center = 0.5 * (stripe_left + stripe_right)

        stripe_half_width_limit = float(self.cfg.get("cross_section_max_lane_half_width_m", lane_half * 1.6))
        stripe_half_width = 0.5 * (stripe_right - stripe_left)
        if stripe_half_width > stripe_half_width_limit:
            stripe_left = stripe_center - stripe_half_width_limit
            stripe_right = stripe_center + stripe_half_width_limit

        in_stripe = (lateral >= stripe_left) & (lateral <= stripe_right)
        if np.count_nonzero(in_stripe) < 4:
            return center

        stripe_weights = weights[in_stripe]
        stripe_lateral = lateral[in_stripe]
        weighted_center = float(np.sum(stripe_lateral * stripe_weights) / np.sum(stripe_weights))

        center_mix = float(self.cfg.get("cross_section_center_mix", 0.65))
        refined_shift = (center_mix * stripe_center) + ((1.0 - center_mix) * weighted_center)
        shift_limit = float(self.cfg.get("center_refine_max_shift_m", 0.08))
        refined_shift = float(np.clip(refined_shift, -shift_limit, shift_limit))

        refined = center.copy()
        refined[:2] = center[:2] + normal * refined_shift
        return refined

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
            pred_dir = self._predict_direction(points, cur_dir)
            best_score = -1.0
            best_center = None
            best_dir = None
            candidates_debug: List[Dict[str, Any]] = []

            for c_xy, c_dir in self._candidate_centers(cur, pred_dir):
                idx = self.grid.query_radius_xy(c_xy, query_r)
                c_z = self._fit_center_z(c_xy, query_r, cur[2])
                center3 = np.array([c_xy[0], c_xy[1], c_z], dtype=np.float64)
                allowed = self._candidate_is_allowed(center3, cur, pred_dir)
                if not allowed:
                    candidates_debug.append({
                        "x": float(center3[0]),
                        "y": float(center3[1]),
                        "z": float(center3[2]),
                        "score": -1.0,
                        "rejected": "hard_gate",
                    })
                    continue
                sc = score_candidate(
                    candidate_center=center3,
                    candidate_dir=c_dir,
                    prev_center=cur,
                    prev_dir=pred_dir,
                    xyz=self.xyz,
                    intensity=self.intensity,
                    indices=idx,
                    seed_profile=profile,
                    cfg=self.cfg,
                )
                loyalty_term = self._lane_loyalty_term(points, center3, pred_dir)
                loyalty_weight = float(self.cfg.get("lane_loyalty_weight", 0.0))
                sc *= (1.0 - loyalty_weight) + loyalty_weight * loyalty_term
                candidates_debug.append({
                    "x": float(center3[0]),
                    "y": float(center3[1]),
                    "z": float(center3[2]),
                    "score": float(sc),
                    "lane_loyalty": float(loyalty_term),
                })
                if sc > best_score:
                    best_score = sc
                    best_center = center3
                    best_dir = c_dir

            if best_center is None or best_dir is None:
                stop_reason = "no_candidate"
                break

            best_center = self._refine_center_xy(best_center, best_dir)
            best_center = self._refine_centerline_cross_section(best_center, best_dir)
            best_center[2] = self._fit_center_z(best_center[:2], query_r, cur[2])

            if best_score < min_score:
                gap_accum += step_m
                gap_steps += 1
                if gap_accum > max_gap:
                    stop_reason = "score_below_threshold"
                    break
                # bridge forward but do not store a point yet
                cur = best_center
                cur_dir = pred_dir if float(self.cfg.get("gap_use_predicted_direction", True)) else best_dir
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
            prev_center_for_update = cur.copy()
            prev_dir_for_update = pred_dir.copy()
            cur = best_center
            cur_dir = best_dir
            update_min = float(self.cfg.get("profile_update_min_score", min_score + 0.12))
            update_limit = float(self.cfg.get("profile_update_max_lateral_offset_m", self.cfg.get("center_offset_tolerance_m", 0.18) * 0.6))
            pred_xy = prev_center_for_update[:2] + unit(prev_dir_for_update[:2]) * step_m
            normal_xy = np.array([-prev_dir_for_update[1], prev_dir_for_update[0]], dtype=np.float64)
            cur_lateral = abs(float(np.dot(cur[:2] - pred_xy, normal_xy)))
            if best_score >= update_min and cur_lateral <= update_limit:
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

        raw_arr = np.vstack(points) if points else np.empty((0, 3), dtype=np.float64)
        arr = raw_arr.copy()
        if arr.shape[0] >= 7 and bool(self.cfg.get("enable_post_correction", True)):
            arr = self._post_correct_detours(arr)
        if arr.shape[0] >= 3 and int(self.cfg.get("smoothing_window", 1)) >= 3:
            arr = self._smooth(arr, int(self.cfg["smoothing_window"]))

        if debug_json_path and bool(self.cfg.get("save_debug_json", True)):
            payload = {
                "stop_reason": stop_reason,
                "num_points": int(arr.shape[0]),
                "num_raw_points": int(raw_arr.shape[0]),
                "num_steps": int(len(debug_steps)),
                "gap_steps": int(gap_steps),
                "scores": scores,
                "steps": debug_steps,
            }
            Path(debug_json_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        return TrackResult(points=arr, raw_points=raw_arr, scores=scores, stop_reason=stop_reason)

    def _post_correct_detours(self, pts: np.ndarray) -> np.ndarray:
        window = int(self.cfg.get("post_correction_window", 4))
        min_len = int(self.cfg.get("post_correction_min_segment_points", 3))
        max_len = int(self.cfg.get("post_correction_max_segment_points", 20))
        lateral_thr = float(self.cfg.get("post_correction_lateral_threshold_m", 0.06))
        return_thr = float(self.cfg.get("post_correction_return_threshold_m", lateral_thr * 0.5))
        if pts.shape[0] < (window * 2 + 1):
            return pts

        residuals = np.zeros(pts.shape[0], dtype=np.float64)
        signs = np.zeros(pts.shape[0], dtype=np.int32)

        for i in range(window, pts.shape[0] - window):
            prev_pt = pts[i - window]
            next_pt = pts[i + window]
            seg = next_pt[:2] - prev_pt[:2]
            seg_len = float(np.linalg.norm(seg))
            if seg_len < 1e-9:
                continue
            dir_xy = seg / seg_len
            normal_xy = np.array([-dir_xy[1], dir_xy[0]], dtype=np.float64)
            offset = float(np.dot(pts[i, :2] - prev_pt[:2], normal_xy))
            residuals[i] = offset
            signs[i] = 1 if offset > 0.0 else (-1 if offset < 0.0 else 0)

        out = pts.copy()
        i = window
        while i < pts.shape[0] - window:
            if abs(residuals[i]) < lateral_thr or signs[i] == 0:
                i += 1
                continue

            start = i
            sign = signs[i]
            peak = abs(residuals[i])
            while i < pts.shape[0] - window and signs[i] == sign:
                peak = max(peak, abs(residuals[i]))
                if abs(residuals[i]) < return_thr and i > start:
                    break
                i += 1
            end = i if i < pts.shape[0] - window and abs(residuals[i]) < return_thr else (i - 1)
            seg_len = end - start + 1

            if seg_len < min_len or seg_len > max_len:
                i = max(i, start + 1)
                continue

            if peak < lateral_thr:
                i = max(i, start + 1)
                continue

            left_idx = start - 1
            right_idx = end + 1
            if left_idx < 0 or right_idx >= pts.shape[0]:
                i = max(i, start + 1)
                continue

            left_pt = out[left_idx]
            right_pt = out[right_idx]
            for j in range(start, end + 1):
                t = (j - left_idx) / max(right_idx - left_idx, 1)
                out[j] = (1.0 - t) * left_pt + t * right_pt

            i = right_idx

        return out

    def _smooth(self, pts: np.ndarray, window: int) -> np.ndarray:
        if window < 3 or pts.shape[0] < window:
            return pts
        half = window // 2
        out = pts.copy()
        for i in range(pts.shape[0]):
            if i == 0 or i == pts.shape[0] - 1:
                out[i] = pts[i]
                continue
            a = max(0, i - half)
            b = min(pts.shape[0], i + half + 1)
            out[i] = np.mean(pts[a:b], axis=0)
        return out
