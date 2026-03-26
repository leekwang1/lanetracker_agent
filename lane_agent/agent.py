from __future__ import annotations

import json
from dataclasses import dataclass, field
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


@dataclass
class TrackStripeDebug:
    left_m: float
    right_m: float
    center_m: float
    weighted_center_m: float
    peak_value: float
    active_threshold: float


@dataclass
class TrackCrossSectionProfileDebug:
    bins_center: np.ndarray
    hist_combined: np.ndarray
    smooth_hist: np.ndarray
    selected_idx: int | None = None
    stripe_candidates: List[TrackStripeDebug] = field(default_factory=list)


@dataclass
class TrackStepDebug:
    step_index: int
    mode: str
    current_center: np.ndarray
    best_score: float
    candidates: List[Dict[str, Any]] = field(default_factory=list)
    selected_center: np.ndarray | None = None
    predicted_direction: np.ndarray | None = None
    stop_reason: str = ""
    traveled_m: float = 0.0
    gap_accum_m: float = 0.0
    accepted_point_count: int = 0
    cross_section_profile: TrackCrossSectionProfileDebug | None = None


@dataclass
class TrackSession:
    profile: SeedProfile
    cur: np.ndarray
    cur_dir: np.ndarray
    points: List[np.ndarray]
    scores: List[float]
    traveled: float
    gap_accum: float
    gap_steps: int
    stop_reason: str
    debug_steps: List[Dict[str, Any]]
    live_steps: List[TrackStepDebug]
    max_len: float
    min_score: float
    step_m: float
    max_gap: float
    query_r: float
    step_index: int = 0
    finished: bool = False
    latest_step: TrackStepDebug | None = None


class LaneTrackerAgent:
    def __init__(self, xyz: np.ndarray, intensity: np.ndarray, cfg: Dict[str, Any]):
        self.xyz = xyz
        self.intensity = intensity
        self.cfg = cfg
        self.grid = SpatialGrid(xyz, float(cfg["grid_size_m"]))

    def initialize_session(self, p0: np.ndarray, p1: np.ndarray) -> TrackSession:
        profile = self._build_seed_profile(p0)
        cur = np.asarray(p0, dtype=np.float64).copy()
        cur_dir = unit(np.asarray(p1, dtype=np.float64) - cur)
        cur[2] = self._fit_center_z(cur[:2], float(self.cfg["search_radius_m"]), float(cur[2]))

        session = TrackSession(
            profile=profile,
            cur=cur,
            cur_dir=cur_dir,
            points=[cur.copy()],
            scores=[1.0],
            traveled=0.0,
            gap_accum=0.0,
            gap_steps=0,
            stop_reason="",
            debug_steps=[],
            live_steps=[],
            max_len=float(self.cfg["max_track_length_m"]),
            min_score=float(self.cfg["min_score"]),
            step_m=float(self.cfg["step_m"]),
            max_gap=float(self.cfg["max_gap_m"]),
            query_r=float(self.cfg["search_radius_m"]),
        )
        if session.max_len <= 0.0:
            session.stop_reason = "max_length_reached"
            session.finished = True
        return session

    def step_session(self, session: TrackSession) -> TrackStepDebug:
        if session.finished:
            raise RuntimeError(f"Tracking session already finished: {session.stop_reason or 'completed'}")

        pred_dir, best_score, best_center, best_dir, candidates_debug = self._evaluate_step_candidates(session)
        if best_center is None or best_dir is None:
            session.stop_reason = "no_candidate"
            session.finished = True
            step = self._make_step_debug(
                session=session,
                mode="stop",
                best_score=best_score,
                current_center=session.cur,
                selected_center=None,
                predicted_direction=pred_dir,
                candidates=candidates_debug,
                stop_reason=session.stop_reason,
            )
            self._record_step(session, step)
            return step

        best_center = self._refine_center_xy(best_center, best_dir)
        best_center = self._refine_centerline_cross_section(best_center, best_dir)
        best_center[2] = self._fit_center_z(best_center[:2], session.query_r, session.cur[2])
        cross_section_profile = self._analyze_cross_section_profile(best_center, best_dir)

        if best_score < session.min_score:
            next_gap_accum = session.gap_accum + session.step_m
            next_gap_steps = session.gap_steps + 1
            if next_gap_accum > session.max_gap:
                session.gap_accum = next_gap_accum
                session.gap_steps = next_gap_steps
                session.stop_reason = "score_below_threshold"
                session.finished = True
                step = self._make_step_debug(
                    session=session,
                    mode="stop",
                    best_score=best_score,
                    current_center=session.cur,
                    selected_center=best_center,
                    predicted_direction=pred_dir,
                    candidates=candidates_debug,
                    stop_reason=session.stop_reason,
                    cross_section_profile=cross_section_profile,
                )
                self._record_step(session, step)
                return step

            session.gap_accum = next_gap_accum
            session.gap_steps = next_gap_steps
            session.cur = best_center
            session.cur_dir = pred_dir if bool(self.cfg.get("gap_use_predicted_direction", True)) else best_dir
            session.traveled += session.step_m
            if session.traveled >= session.max_len:
                session.stop_reason = "max_length_reached"
                session.finished = True
            step = self._make_step_debug(
                session=session,
                mode="gap_bridge",
                best_score=best_score,
                current_center=session.cur,
                selected_center=best_center,
                predicted_direction=pred_dir,
                candidates=candidates_debug,
                stop_reason=session.stop_reason,
                cross_section_profile=cross_section_profile,
            )
            self._record_step(session, step)
            return step

        session.gap_accum = 0.0
        prev_center_for_update = session.cur.copy()
        prev_dir_for_update = pred_dir.copy()
        session.cur = best_center
        session.cur_dir = best_dir
        update_min = float(self.cfg.get("profile_update_min_score", session.min_score + 0.12))
        update_limit = float(
            self.cfg.get(
                "profile_update_max_lateral_offset_m",
                self.cfg.get("center_offset_tolerance_m", 0.18) * 0.6,
            )
        )
        pred_xy = prev_center_for_update[:2] + unit(prev_dir_for_update[:2]) * session.step_m
        normal_xy = np.array([-prev_dir_for_update[1], prev_dir_for_update[0]], dtype=np.float64)
        cur_lateral = abs(float(np.dot(session.cur[:2] - pred_xy, normal_xy)))
        if best_score >= update_min and cur_lateral <= update_limit:
            session.profile = self._refresh_seed_profile(session.cur, session.profile)
        session.points.append(session.cur.copy())
        session.scores.append(float(best_score))
        if len(session.points) >= 2:
            session.traveled += float(np.linalg.norm(session.points[-1][:2] - session.points[-2][:2]))
        else:
            session.traveled += session.step_m
        if session.traveled >= session.max_len:
            session.stop_reason = "max_length_reached"
            session.finished = True
        step = self._make_step_debug(
            session=session,
            mode="accept",
            best_score=best_score,
            current_center=session.cur,
            selected_center=best_center,
            predicted_direction=pred_dir,
            candidates=candidates_debug,
            stop_reason=session.stop_reason,
            cross_section_profile=cross_section_profile,
        )
        self._record_step(session, step)
        return step

    def finalize_session(self, session: TrackSession, debug_json_path: str | None = None) -> TrackResult:
        raw_arr = np.vstack(session.points) if session.points else np.empty((0, 3), dtype=np.float64)
        arr = raw_arr.copy()
        if arr.shape[0] >= 7 and bool(self.cfg.get("enable_post_correction", True)):
            arr = self._post_correct_detours(arr)
        if arr.shape[0] >= 3 and int(self.cfg.get("smoothing_window", 1)) >= 3:
            arr = self._smooth(arr, int(self.cfg["smoothing_window"]))

        stop_reason = session.stop_reason or ("max_length_reached" if session.finished else "in_progress")
        result = TrackResult(
            points=arr,
            raw_points=raw_arr,
            scores=list(session.scores),
            stop_reason=stop_reason,
        )

        if debug_json_path and bool(self.cfg.get("save_debug_json", True)):
            payload = self._build_debug_payload(session, result)
            Path(debug_json_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        return result

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

    def _analyze_cross_section_profile(
        self,
        center: np.ndarray,
        direction: np.ndarray,
    ) -> TrackCrossSectionProfileDebug | None:
        radius = float(self.cfg.get("cross_section_radius_m", max(self.cfg["search_half_width_m"], self.cfg["search_radius_m"])))
        idx = self.grid.query_radius_xy(center[:2], radius)
        if idx.size < 10:
            return None

        local_xyz = self.xyz[idx]
        local_i = self.intensity[idx].astype(np.float64)
        max_z_step = float(self.cfg.get("max_z_step_m", 0.12))
        z_mask = np.abs(local_xyz[:, 2] - center[2]) <= max_z_step
        if np.count_nonzero(z_mask) < 6:
            return None

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
            return None

        lateral = lateral[mask]
        vals = vals[mask]
        val_floor = float(np.quantile(vals, 0.35))
        weights = np.clip(vals - val_floor, 0.0, None)
        if float(np.sum(weights)) <= 1e-9:
            return None

        bin_size = float(self.cfg.get("cross_section_bin_size_m", 0.02))
        bin_size = max(bin_size, 1e-3)
        bins = np.arange(-lane_half * 2.5, lane_half * 2.5 + bin_size, bin_size, dtype=np.float64)
        if bins.size < 4:
            return None

        hist, edges = np.histogram(lateral, bins=bins, weights=weights)
        if hist.size < 3:
            return None

        smooth_hist = hist.copy()
        smooth_hist[1:-1] = 0.25 * hist[:-2] + 0.5 * hist[1:-1] + 0.25 * hist[2:]
        peak_idx = int(np.argmax(smooth_hist))
        peak_value = float(smooth_hist[peak_idx])
        if peak_value <= 1e-9:
            return None

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
            return None

        stripe_weights = weights[in_stripe]
        stripe_lateral = lateral[in_stripe]
        weighted_center = float(np.sum(stripe_lateral * stripe_weights) / np.sum(stripe_weights))

        bins_center = 0.5 * (edges[:-1] + edges[1:])
        stripe = TrackStripeDebug(
            left_m=float(stripe_left),
            right_m=float(stripe_right),
            center_m=float(stripe_center),
            weighted_center_m=float(weighted_center),
            peak_value=float(peak_value),
            active_threshold=float(active_threshold),
        )
        return TrackCrossSectionProfileDebug(
            bins_center=np.asarray(bins_center, dtype=np.float64),
            hist_combined=np.asarray(hist, dtype=np.float64),
            smooth_hist=np.asarray(smooth_hist, dtype=np.float64),
            selected_idx=0,
            stripe_candidates=[stripe],
        )

    def _refine_centerline_cross_section(self, center: np.ndarray, direction: np.ndarray) -> np.ndarray:
        profile = self._analyze_cross_section_profile(center, direction)
        if profile is None or profile.selected_idx is None or not profile.stripe_candidates:
            return center

        stripe = profile.stripe_candidates[profile.selected_idx]

        center_mix = float(self.cfg.get("cross_section_center_mix", 0.65))
        refined_shift = (center_mix * stripe.center_m) + ((1.0 - center_mix) * stripe.weighted_center_m)
        shift_limit = float(self.cfg.get("center_refine_max_shift_m", 0.08))
        refined_shift = float(np.clip(refined_shift, -shift_limit, shift_limit))

        dir_xy = unit(direction[:2])
        normal = np.array([-dir_xy[1], dir_xy[0]], dtype=np.float64)
        refined = center.copy()
        refined[:2] = center[:2] + normal * refined_shift
        return refined

    def _evaluate_step_candidates(
        self,
        session: TrackSession,
    ) -> Tuple[np.ndarray, float, np.ndarray | None, np.ndarray | None, List[Dict[str, Any]]]:
        pred_dir = self._predict_direction(session.points, session.cur_dir)
        best_score = -1.0
        best_center = None
        best_dir = None
        candidates_debug: List[Dict[str, Any]] = []

        for c_xy, c_dir in self._candidate_centers(session.cur, pred_dir):
            idx = self.grid.query_radius_xy(c_xy, session.query_r)
            c_z = self._fit_center_z(c_xy, session.query_r, session.cur[2])
            center3 = np.array([c_xy[0], c_xy[1], c_z], dtype=np.float64)
            if idx.size > 0:
                local_i = self.intensity[idx].astype(np.float64)
                mean_intensity = float(np.mean(local_i))
                high_intensity = float(np.quantile(local_i, 0.9))
            else:
                mean_intensity = 0.0
                high_intensity = 0.0
            allowed = self._candidate_is_allowed(center3, session.cur, pred_dir)
            if not allowed:
                candidates_debug.append(
                    {
                        "x": float(center3[0]),
                        "y": float(center3[1]),
                        "z": float(center3[2]),
                        "score": -1.0,
                        "mean_intensity": mean_intensity,
                        "high_intensity": high_intensity,
                        "rejected": "hard_gate",
                    }
                )
                continue
            sc = score_candidate(
                candidate_center=center3,
                candidate_dir=c_dir,
                prev_center=session.cur,
                prev_dir=pred_dir,
                xyz=self.xyz,
                intensity=self.intensity,
                indices=idx,
                seed_profile=session.profile,
                cfg=self.cfg,
            )
            loyalty_term = self._lane_loyalty_term(session.points, center3, pred_dir)
            loyalty_weight = float(self.cfg.get("lane_loyalty_weight", 0.0))
            sc *= (1.0 - loyalty_weight) + loyalty_weight * loyalty_term
            candidates_debug.append(
                {
                    "x": float(center3[0]),
                    "y": float(center3[1]),
                    "z": float(center3[2]),
                    "score": float(sc),
                    "mean_intensity": mean_intensity,
                    "high_intensity": high_intensity,
                    "lane_loyalty": float(loyalty_term),
                }
            )
            if sc > best_score:
                best_score = sc
                best_center = center3
                best_dir = c_dir

        return pred_dir, float(best_score), best_center, best_dir, candidates_debug

    def _make_step_debug(
        self,
        session: TrackSession,
        mode: str,
        best_score: float,
        current_center: np.ndarray,
        selected_center: np.ndarray | None,
        predicted_direction: np.ndarray | None,
        candidates: List[Dict[str, Any]],
        stop_reason: str = "",
        cross_section_profile: TrackCrossSectionProfileDebug | None = None,
    ) -> TrackStepDebug:
        return TrackStepDebug(
            step_index=session.step_index,
            mode=mode,
            current_center=np.asarray(current_center, dtype=np.float64).copy(),
            best_score=float(best_score),
            candidates=list(candidates),
            selected_center=None if selected_center is None else np.asarray(selected_center, dtype=np.float64).copy(),
            predicted_direction=None
            if predicted_direction is None
            else np.asarray(predicted_direction, dtype=np.float64).copy(),
            stop_reason=stop_reason,
            traveled_m=float(session.traveled),
            gap_accum_m=float(session.gap_accum),
            accepted_point_count=len(session.points),
            cross_section_profile=cross_section_profile,
        )

    def _record_step(self, session: TrackSession, step: TrackStepDebug) -> None:
        session.latest_step = step
        session.live_steps.append(step)
        session.debug_steps.append(self._step_to_payload(step))
        session.step_index += 1

    def _step_to_payload(self, step: TrackStepDebug) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "step_index": int(step.step_index),
            "mode": step.mode,
            "best_score": float(step.best_score),
            "x": float(step.current_center[0]),
            "y": float(step.current_center[1]),
            "z": float(step.current_center[2]),
            "traveled_m": float(step.traveled_m),
            "gap_accum_m": float(step.gap_accum_m),
            "accepted_point_count": int(step.accepted_point_count),
            "candidates": step.candidates,
        }
        if step.selected_center is not None:
            payload["selected_center"] = [float(v) for v in step.selected_center.tolist()]
        if step.predicted_direction is not None:
            payload["predicted_direction"] = [float(v) for v in step.predicted_direction.tolist()]
        if step.stop_reason:
            payload["stop_reason"] = step.stop_reason
        if step.cross_section_profile is not None:
            payload["cross_section_profile"] = self._cross_section_profile_to_payload(step.cross_section_profile)
        return payload

    def _cross_section_profile_to_payload(
        self,
        profile: TrackCrossSectionProfileDebug,
    ) -> Dict[str, Any]:
        return {
            "bins_center": [float(v) for v in profile.bins_center.tolist()],
            "hist_combined": [float(v) for v in profile.hist_combined.tolist()],
            "smooth_hist": [float(v) for v in profile.smooth_hist.tolist()],
            "selected_idx": profile.selected_idx,
            "stripe_candidates": [
                {
                    "left_m": float(stripe.left_m),
                    "right_m": float(stripe.right_m),
                    "center_m": float(stripe.center_m),
                    "weighted_center_m": float(stripe.weighted_center_m),
                    "peak_value": float(stripe.peak_value),
                    "active_threshold": float(stripe.active_threshold),
                }
                for stripe in profile.stripe_candidates
            ],
        }

    def _build_debug_payload(self, session: TrackSession, result: TrackResult) -> Dict[str, Any]:
        return {
            "stop_reason": result.stop_reason,
            "num_points": int(result.points.shape[0]),
            "num_raw_points": int(result.raw_points.shape[0]),
            "num_steps": int(len(session.debug_steps)),
            "gap_steps": int(session.gap_steps),
            "scores": list(session.scores),
            "steps": session.debug_steps,
        }

    def track(self, p0: np.ndarray, p1: np.ndarray, debug_json_path: str | None = None) -> TrackResult:
        session = self.initialize_session(p0, p1)
        while not session.finished:
            self.step_session(session)
        return self.finalize_session(session, debug_json_path)

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
