from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .grid import SpatialGrid
from .scoring import SeedProfile, blend_seed_profile, estimate_seed_profile, score_candidate, signed_angle_deg, unit


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
    mean_intensity: float = 0.0
    high_intensity: float = 0.0
    center_term: float = 1.0
    signal_term: float = 1.0
    peak_term: float = 1.0
    selection_score: float = 1.0


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
    recovery_hold_until_traveled: float = 0.0
    recovery_lock_dir: np.ndarray | None = None
    recovery_release_streak: int = 0


@dataclass
class CandidateProposal:
    center_xy: np.ndarray
    direction: np.ndarray
    kind: str
    step_distance_m: float
    heading_delta_deg: float = 0.0
    lateral_offset_m: float = 0.0
    score_scale: float = 1.0


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

        pred_dir, best_score, best_center, best_dir, best_proposal, candidates_debug = self._evaluate_step_candidates(session)
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
        best_center = self._refine_centerline_cross_section(best_center, best_dir, session.profile)
        best_center[2] = self._fit_center_z(best_center[:2], session.query_r, session.cur[2])
        pre_steer_profile = self._analyze_cross_section_profile(best_center, best_dir, session.profile)
        hook_ready = self._recovery_hook_ready(
            float(best_score),
            None if best_proposal is None else best_proposal.kind,
            pre_steer_profile,
        )
        best_center, best_dir = self._apply_recovery_steering(
            session,
            best_center,
            best_dir,
            best_proposal,
            float(best_score),
            hook_ready,
            pre_steer_profile,
        )
        best_center[2] = self._fit_center_z(best_center[:2], session.query_r, session.cur[2])
        cross_section_profile = self._analyze_cross_section_profile(best_center, best_dir, session.profile)

        move_distance = float(np.linalg.norm(best_center[:2] - session.cur[:2]))
        if best_score < session.min_score:
            next_gap_accum = session.gap_accum + move_distance
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
            prev_gap_center = session.cur.copy()
            session.cur = best_center
            session.cur_dir = pred_dir if bool(self.cfg.get("gap_use_predicted_direction", True)) else best_dir
            session.traveled += float(np.linalg.norm(session.cur[:2] - prev_gap_center[:2]))
            self._update_recovery_hold(
                session,
                accepted_score=None,
                accepted_kind=None if best_proposal is None else best_proposal.kind,
                reacquired=False,
            )
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
        session.gap_steps = 0
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
        accepted_kind = None if best_proposal is None else best_proposal.kind
        reacquired = self._recovery_reacquired(
            float(best_score),
            accepted_kind,
            cross_section_profile,
        )
        self._update_recovery_hold(
            session,
            accepted_score=float(best_score),
            accepted_kind=accepted_kind,
            reacquired=reacquired,
        )
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

    def _candidate_is_allowed(
        self,
        candidate_center: np.ndarray,
        prev_center: np.ndarray,
        prev_dir: np.ndarray,
        step_distance_m: float,
        candidate_kind: str | None = None,
    ) -> bool:
        step_ref = max(float(step_distance_m), 1e-6)
        prev_dir_xy = unit(prev_dir[:2])
        pred_xy = prev_center[:2] + prev_dir_xy * step_ref
        normal_xy = np.array([-prev_dir_xy[1], prev_dir_xy[0]], dtype=np.float64)
        lateral_offset = abs(float(np.dot(candidate_center[:2] - pred_xy, normal_xy)))
        hard_limit = float(self.cfg.get("hard_center_offset_limit_m", self.cfg.get("center_offset_tolerance_m", 0.18) * 1.4))
        if candidate_kind == "recovery":
            hard_limit = max(
                hard_limit,
                float(self.cfg.get("recovery_hard_center_offset_limit_m", hard_limit * 3.5)),
            )
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

    def _rotate_direction(self, direction: np.ndarray, delta_deg: float) -> np.ndarray:
        direction = unit(direction)
        delta = np.radians(float(delta_deg))
        cs = float(np.cos(delta))
        sn = float(np.sin(delta))
        rotated = np.array(
            [
                cs * direction[0] - sn * direction[1],
                sn * direction[0] + cs * direction[1],
                0.0,
            ],
            dtype=np.float64,
        )
        return unit(rotated)

    def _estimate_curve_hint_deg(self, points: List[np.ndarray], fallback_dir: np.ndarray) -> float:
        if len(points) < 4:
            return 0.0
        hist = np.asarray(points[-6:], dtype=np.float64)
        deltas = hist[1:, :2] - hist[:-1, :2]
        lengths = np.linalg.norm(deltas, axis=1)
        valid = lengths > 1e-9
        if np.count_nonzero(valid) < 2:
            return 0.0
        dirs = deltas[valid] / lengths[valid, None]
        heading_steps = []
        for i in range(1, dirs.shape[0]):
            a = np.array([dirs[i - 1, 0], dirs[i - 1, 1], 0.0], dtype=np.float64)
            b = np.array([dirs[i, 0], dirs[i, 1], 0.0], dtype=np.float64)
            heading_steps.append(float(np.radians(np.clip(np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], np.clip(np.dot(a[:2], b[:2]), -1.0, 1.0))), -45.0, 45.0))))
        if not heading_steps:
            return 0.0
        heading_steps = np.asarray(heading_steps, dtype=np.float64)
        weights = np.linspace(1.0, 2.0, heading_steps.size, dtype=np.float64)
        hint_deg = float(np.degrees(np.sum(heading_steps * weights) / np.sum(weights)))
        max_hint = float(self.cfg.get("curve_candidate_heading_max_deg", self.cfg.get("max_heading_change_deg", 5.0)))
        if not np.isfinite(hint_deg):
            return 0.0
        return float(np.clip(hint_deg, -max_hint, max_hint))

    def _recovery_trigger_active(self, session: TrackSession) -> bool:
        if session.gap_accum > 0.0 or session.gap_steps > 0:
            return True
        if not session.scores:
            return False
        recent = np.asarray(session.scores[-min(len(session.scores), 4):], dtype=np.float64)
        last_score = float(recent[-1])
        recent_mean = float(np.mean(recent))
        trigger = float(self.cfg.get("recovery_score_trigger", max(session.min_score + 0.18, 0.55)))
        return last_score < trigger or recent_mean < (trigger + 0.04)

    def _recovery_hold_remaining_m(self, session: TrackSession) -> float:
        return max(float(session.recovery_hold_until_traveled - session.traveled), 0.0)

    def _should_add_recovery_candidates(self, session: TrackSession) -> bool:
        if self._recovery_trigger_active(session):
            return True
        return self._recovery_hold_remaining_m(session) > 1e-9

    def _clear_recovery_hold(self, session: TrackSession) -> None:
        session.recovery_hold_until_traveled = 0.0
        session.recovery_lock_dir = None
        session.recovery_release_streak = 0

    def _update_recovery_hold(
        self,
        session: TrackSession,
        accepted_score: float | None = None,
        accepted_kind: str | None = None,
        reacquired: bool = False,
    ) -> None:
        hold_distance = max(float(self.cfg.get("recovery_hold_distance_m", 0.0)), 0.0)
        if hold_distance <= 0.0:
            self._clear_recovery_hold(session)
            return
        if self._recovery_trigger_active(session):
            session.recovery_hold_until_traveled = max(
                float(session.recovery_hold_until_traveled),
                float(session.traveled + hold_distance),
            )
            session.recovery_release_streak = 0
            return
        release_score = float(self.cfg.get("recovery_release_score", max(self.cfg.get("recovery_score_trigger", 0.55) + 0.07, 0.60)))
        release_steps = max(int(self.cfg.get("recovery_release_steps", 2)), 1)
        if accepted_kind == "recovery":
            if reacquired and bool(self.cfg.get("recovery_release_on_capture", True)):
                self._clear_recovery_hold(session)
                return
            session.recovery_release_streak = 0
        elif reacquired and accepted_score is not None and accepted_score >= release_score:
            session.recovery_release_streak += 1
            if session.recovery_release_streak >= release_steps:
                self._clear_recovery_hold(session)
                return
        else:
            session.recovery_release_streak = 0
        if session.traveled >= session.recovery_hold_until_traveled:
            self._clear_recovery_hold(session)

    def _recovery_reacquired(
        self,
        accepted_score: float,
        accepted_kind: str | None,
        cross_section_profile: TrackCrossSectionProfileDebug | None,
    ) -> bool:
        release_score = float(self.cfg.get("recovery_release_score", max(self.cfg.get("recovery_score_trigger", 0.55) + 0.07, 0.60)))
        if accepted_score < release_score:
            return False
        if cross_section_profile is None or cross_section_profile.selected_idx is None:
            return False
        if not cross_section_profile.stripe_candidates:
            return False
        stripe = cross_section_profile.stripe_candidates[cross_section_profile.selected_idx]
        stripe_release_score = float(self.cfg.get("recovery_release_stripe_score", 0.62))
        stripe_center_limit = float(
            self.cfg.get(
                "recovery_release_center_limit_m",
                self.cfg.get("lane_half_width_m", 0.09) * 0.9,
            )
        )
        return (
            float(stripe.selection_score) >= stripe_release_score
            and abs(float(stripe.center_m)) <= max(stripe_center_limit, 0.03)
        )

    def _recovery_hook_ready(
        self,
        proposal_score: float,
        proposal_kind: str | None,
        cross_section_profile: TrackCrossSectionProfileDebug | None,
    ) -> bool:
        if proposal_kind != "recovery":
            return False
        hook_score = float(
            self.cfg.get(
                "recovery_hook_score",
                max(
                    float(self.cfg.get("recovery_release_score", 0.60)),
                    float(self.cfg.get("recovery_score_trigger", 0.55)) + 0.07,
                ),
            )
        )
        if proposal_score >= hook_score:
            return True
        if cross_section_profile is None or cross_section_profile.selected_idx is None:
            return False
        if not cross_section_profile.stripe_candidates:
            return False
        stripe = cross_section_profile.stripe_candidates[cross_section_profile.selected_idx]
        hook_stripe_score = float(self.cfg.get("recovery_hook_stripe_score", 0.58))
        hook_center_limit = float(
            self.cfg.get(
                "recovery_hook_center_limit_m",
                self.cfg.get("lane_half_width_m", 0.09) * 1.2,
            )
        )
        return (
            float(stripe.selection_score) >= hook_stripe_score
            and abs(float(stripe.center_m)) <= max(hook_center_limit, 0.03)
        )

    def _search_direction_for_session(self, session: TrackSession, predicted_dir: np.ndarray) -> np.ndarray:
        predicted_dir = unit(predicted_dir)
        if not self._should_add_recovery_candidates(session):
            return predicted_dir

        if bool(self.cfg.get("recovery_force_straight_direction", True)):
            if session.recovery_lock_dir is None or float(np.linalg.norm(session.recovery_lock_dir[:2])) <= 1e-9:
                base_dir = session.cur_dir if float(np.linalg.norm(session.cur_dir[:2])) > 1e-9 else predicted_dir
                session.recovery_lock_dir = unit(base_dir).copy()
            return unit(session.recovery_lock_dir)

        return predicted_dir

    def _apply_recovery_steering(
        self,
        session: TrackSession,
        target_center: np.ndarray,
        target_dir: np.ndarray,
        proposal: CandidateProposal | None,
        proposal_score: float,
        hook_ready: bool,
        hook_profile: TrackCrossSectionProfileDebug | None,
    ) -> tuple[np.ndarray, np.ndarray]:
        if proposal is None or proposal.kind != "recovery":
            return target_center, unit(target_dir)

        step_scale = max(float(self.cfg.get("recovery_advance_max_scale", 1.0)), 0.1)
        max_advance = max(float(session.step_m) * step_scale, 1e-6)
        max_turn_deg = max(float(self.cfg.get("recovery_turn_max_deg", 12.0)), 0.5)
        max_lateral_snap = max(float(self.cfg.get("recovery_lateral_snap_max_m", 0.10)), 0.0)
        lateral_snap_ratio = float(np.clip(self.cfg.get("recovery_lateral_snap_ratio", 0.55), 0.0, 1.0))
        hook_turn_deg = max(float(self.cfg.get("recovery_hook_turn_max_deg", max_turn_deg * 1.5)), max_turn_deg)
        hook_lateral_snap = max(float(self.cfg.get("recovery_hook_lateral_snap_max_m", max_lateral_snap * 1.5)), max_lateral_snap)
        hook_lateral_ratio = float(
            np.clip(
                self.cfg.get("recovery_hook_lateral_snap_ratio", max(lateral_snap_ratio, 0.75)),
                0.0,
                1.0,
            )
        )
        capture_snap_distance = max(float(self.cfg.get("recovery_capture_snap_distance_m", max(float(session.step_m) * 3.5, 1.2))), max_advance)
        capture_advance = max(float(self.cfg.get("recovery_capture_advance_scale", 1.75)) * float(session.step_m), max_advance)
        capture_lateral_snap = max(float(self.cfg.get("recovery_capture_lateral_snap_max_m", max(hook_lateral_snap, 0.45))), hook_lateral_snap)
        capture_lateral_ratio = float(np.clip(self.cfg.get("recovery_capture_lateral_snap_ratio", 1.0), 0.0, 1.0))
        delta_xy = np.asarray(target_center[:2] - session.cur[:2], dtype=np.float64)
        distance = float(np.linalg.norm(delta_xy))
        base_dir = session.recovery_lock_dir if session.recovery_lock_dir is not None else session.cur_dir
        base_dir = unit(base_dir if float(np.linalg.norm(base_dir[:2])) > 1e-9 else target_dir)
        base_dir_xy = unit(base_dir[:2])
        base_normal_xy = np.array([-base_dir_xy[1], base_dir_xy[0]], dtype=np.float64)

        # During true gap recovery, keep advancing on the locked straight axis.
        # Only switch into hook mode once the recovery candidate is strong enough.
        if not hook_ready:
            straight_center = np.asarray(target_center, dtype=np.float64).copy()
            straight_center[:2] = session.cur[:2] + base_dir_xy * max_advance
            return straight_center, base_dir

        capture_dir = unit(target_dir if float(np.linalg.norm(target_dir[:2])) > 1e-9 else np.array([delta_xy[0], delta_xy[1], 0.0], dtype=np.float64))
        capture_dir_xy = unit(capture_dir[:2])
        capture_normal_xy = np.array([-capture_dir_xy[1], capture_dir_xy[0]], dtype=np.float64)
        capture_center = np.asarray(target_center, dtype=np.float64).copy()
        if (
            hook_profile is not None
            and hook_profile.selected_idx is not None
            and hook_profile.stripe_candidates
        ):
            stripe = hook_profile.stripe_candidates[hook_profile.selected_idx]
            capture_center_shift_max = max(
                float(
                    self.cfg.get(
                        "recovery_capture_center_shift_max_m",
                        max(float(self.cfg.get("lane_half_width_m", 0.09)) * 2.0, 0.18),
                    )
                ),
                0.03,
            )
            stripe_shift = float(np.clip(stripe.center_m, -capture_center_shift_max, capture_center_shift_max))
            capture_center[:2] = capture_center[:2] + capture_normal_xy * stripe_shift
            delta_xy = np.asarray(capture_center[:2] - session.cur[:2], dtype=np.float64)
            distance = float(np.linalg.norm(delta_xy))
        if distance <= capture_snap_distance + 1e-9:
            return capture_center, capture_dir

        capture_along = float(np.dot(delta_xy, capture_dir_xy))
        capture_lateral_error = float(np.dot(delta_xy, capture_normal_xy))
        capture_forward = float(np.clip(capture_along, 0.0, capture_advance))
        lateral_snap = float(np.clip(capture_lateral_error * capture_lateral_ratio, -capture_lateral_snap, capture_lateral_snap))
        capture_center[:2] = session.cur[:2] + capture_dir_xy * capture_forward + capture_normal_xy * lateral_snap
        return capture_center, capture_dir

    def _candidate_centers(self, session: TrackSession, direction: np.ndarray) -> List[CandidateProposal]:
        direction = unit(direction)
        center = session.cur
        step_m = float(session.step_m)
        hard_limit = float(self.cfg.get("hard_center_offset_limit_m", self.cfg.get("center_offset_tolerance_m", 0.18) * 1.4))
        search_cap = float(self.cfg.get("search_half_width_m", hard_limit))
        recovery_active = self._should_add_recovery_candidates(session)
        recovery_force_straight = bool(self.cfg.get("recovery_force_straight_direction", True))

        center_half = min(float(self.cfg.get("candidate_center_half_width_m", 0.12)), hard_limit, search_cap)
        curve_half = min(float(self.cfg.get("candidate_curve_half_width_m", max(center_half * 0.75, 0.08))), hard_limit, search_cap)
        recovery_half = min(float(self.cfg.get("candidate_recovery_half_width_m", max(center_half * 1.2, 0.15))), hard_limit, search_cap)

        curve_hint_deg = 0.0 if (recovery_active and recovery_force_straight) else self._estimate_curve_hint_deg(session.points, direction)
        curve_heading_step = float(self.cfg.get("curve_candidate_heading_step_deg", max(float(self.cfg.get("max_heading_change_deg", 5.0)) * 0.5, 2.5)))
        curve_heading_max = float(self.cfg.get("curve_candidate_heading_max_deg", self.cfg.get("max_heading_change_deg", 5.0)))
        recovery_heading_step = float(self.cfg.get("recovery_candidate_heading_step_deg", max(curve_heading_step * 1.6, 4.5)))
        recovery_heading_max = float(self.cfg.get("recovery_candidate_heading_max_deg", max(curve_heading_max * 2.0, recovery_heading_step * 3.0)))
        recovery_heading_levels = max(int(self.cfg.get("recovery_heading_levels", 3)), 1)

        center_rows = [
            (0.82, [0.0, -center_half * 0.70, center_half * 0.70, -center_half, center_half]),
            (1.00, [0.0, -center_half * 0.45, center_half * 0.45]),
            (1.18, [0.0]),
        ]

        if abs(curve_hint_deg) < curve_heading_step * 0.5:
            curve_headings = [-curve_heading_step, curve_heading_step]
        else:
            curve_headings = [curve_hint_deg - curve_heading_step, curve_hint_deg, curve_hint_deg + curve_heading_step]
        curve_headings = sorted(
            {
                float(np.clip(delta_deg, -curve_heading_max, curve_heading_max))
                for delta_deg in curve_headings
            }
        )
        curve_rows = [
            (0.98, [0.0, -curve_half * 0.75, curve_half * 0.75]),
            (1.28, [0.0]),
        ]

        recovery_forward_max = float(self.cfg.get("recovery_forward_max_distance_m", max(step_m * 3.0, 2.5)))
        recovery_forward_near = min(max(step_m * 1.2, step_m * 1.0), recovery_forward_max)
        recovery_forward_mid = min(max(step_m * 2.1, recovery_forward_near + step_m * 0.8), recovery_forward_max)
        recovery_forward_far = min(max(step_m * 3.2, recovery_forward_mid + step_m), recovery_forward_max)
        recovery_rows = [
            (recovery_forward_near, [0.0]),
            (recovery_forward_mid, [0.0]),
            (recovery_forward_far, [0.0]),
            (recovery_forward_max, [0.0]),
        ]

        out: List[CandidateProposal] = []
        seen: set[tuple[int, int, int, int]] = set()

        def add_candidate(
            base_direction: np.ndarray,
            step_distance_m: float,
            lateral_offset: float,
            kind: str,
            heading_delta_deg: float,
            score_scale: float,
        ) -> None:
            cand_dir = unit(base_direction)
            normal = np.array([-cand_dir[1], cand_dir[0], 0.0], dtype=np.float64)
            c_xy = center[:2] + cand_dir[:2] * float(step_distance_m) + normal[:2] * float(lateral_offset)
            key = (
                int(round(float(c_xy[0]) * 1000.0)),
                int(round(float(c_xy[1]) * 1000.0)),
                int(round(float(cand_dir[0]) * 10000.0)),
                int(round(float(cand_dir[1]) * 10000.0)),
            )
            if key in seen:
                return
            seen.add(key)
            out.append(
                CandidateProposal(
                    center_xy=np.asarray(c_xy, dtype=np.float64),
                    direction=cand_dir,
                    kind=kind,
                    step_distance_m=float(step_distance_m),
                    heading_delta_deg=float(heading_delta_deg),
                    lateral_offset_m=float(lateral_offset),
                    score_scale=float(score_scale),
                )
            )

        if not recovery_active:
            for forward_scale, lateral_offsets in center_rows:
                step_distance = step_m * float(forward_scale)
                for lateral_offset in lateral_offsets:
                    add_candidate(
                        direction,
                        step_distance,
                        lateral_offset,
                        "center",
                        0.0,
                        float(self.cfg.get("center_candidate_score_scale", 1.0)),
                    )

            for delta_deg in curve_headings:
                curve_dir = self._rotate_direction(direction, delta_deg)
                for forward_scale, lateral_offsets in curve_rows:
                    step_distance = step_m * float(forward_scale)
                    for lateral_offset in lateral_offsets:
                        add_candidate(
                            curve_dir,
                            step_distance,
                            lateral_offset,
                            "curve",
                            delta_deg,
                            float(self.cfg.get("curve_candidate_score_scale", 0.99)),
                        )

        if recovery_active:
            recovery_center = 0.0 if recovery_force_straight else float(np.clip(curve_hint_deg, -recovery_heading_max, recovery_heading_max))
            recovery_headings = [
                recovery_center + recovery_heading_step * level
                for level in range(-recovery_heading_levels, recovery_heading_levels + 1)
            ]
            recovery_headings = sorted(
                {
                    float(np.clip(delta_deg, -recovery_heading_max, recovery_heading_max))
                    for delta_deg in recovery_headings
                }
            )
            for delta_deg in recovery_headings:
                recovery_dir = self._rotate_direction(direction, delta_deg)
                for step_distance, lateral_offsets in recovery_rows:
                    for lateral_offset in lateral_offsets:
                        add_candidate(
                            recovery_dir,
                            step_distance,
                            lateral_offset,
                            "recovery",
                            delta_deg,
                            float(self.cfg.get("recovery_candidate_score_scale", 0.97)),
                        )

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

    def _build_cross_section_stripe(
        self,
        lateral: np.ndarray,
        vals: np.ndarray,
        weights: np.ndarray,
        edges: np.ndarray,
        smooth_hist: np.ndarray,
        peak_idx: int,
        peak_value_max: float,
        lane_half: float,
        seed_profile: SeedProfile | None,
    ) -> TrackStripeDebug | None:
        stripe_ratio = float(self.cfg.get("cross_section_stripe_threshold_ratio", 0.35))
        active_threshold = float(smooth_hist[peak_idx]) * stripe_ratio
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
        stripe_vals = vals[in_stripe]
        weight_sum = float(np.sum(stripe_weights))
        if weight_sum <= 1e-9:
            return None

        weighted_center = float(np.sum(stripe_lateral * stripe_weights) / weight_sum)
        mean_intensity = float(np.mean(stripe_vals))
        high_intensity = float(np.quantile(stripe_vals, 0.90))

        center_mix = float(self.cfg.get("cross_section_center_mix", 0.65))
        refined_center = (center_mix * stripe_center) + ((1.0 - center_mix) * weighted_center)

        loyalty_tolerance = max(
            float(
                self.cfg.get(
                    "cross_section_loyalty_tolerance_m",
                    self.cfg.get("lane_loyalty_tolerance_m", self.cfg.get("center_offset_tolerance_m", 0.18)),
                )
            ),
            1e-3,
        )
        center_term = float(np.clip(1.0 - abs(refined_center) / loyalty_tolerance, 0.0, 1.0))
        peak_term = float(np.clip(float(smooth_hist[peak_idx]) / max(peak_value_max, 1e-9), 0.0, 1.0))

        signal_term = 1.0
        if seed_profile is not None:
            seed_signal = max(seed_profile.target_intensity - seed_profile.background_intensity, 1.0)
            stripe_background = float(np.quantile(stripe_vals, 0.35))
            stripe_signal = max(high_intensity - stripe_background, 0.0)
            target_similarity = np.clip(1.0 - abs(high_intensity - seed_profile.target_intensity) / seed_signal, 0.0, 1.0)
            signal_similarity = np.clip(1.0 - abs(stripe_signal - seed_signal) / seed_signal, 0.0, 1.0)
            mean_similarity = np.clip(1.0 - abs(mean_intensity - seed_profile.target_intensity) / max(seed_signal * 1.25, 1.0), 0.0, 1.0)
            signal_term = float(0.6 * target_similarity + 0.3 * signal_similarity + 0.1 * mean_similarity)

        peak_weight = float(self.cfg.get("cross_section_peak_weight", 0.15))
        center_weight = float(self.cfg.get("cross_section_loyalty_weight", 0.60))
        signal_weight = float(self.cfg.get("cross_section_signal_similarity_weight", 0.25))
        weight_total = max(peak_weight + center_weight + signal_weight, 1e-9)
        selection_score = float(
            (peak_weight * peak_term + center_weight * center_term + signal_weight * signal_term) / weight_total
        )

        return TrackStripeDebug(
            left_m=float(stripe_left),
            right_m=float(stripe_right),
            center_m=float(stripe_center),
            weighted_center_m=float(weighted_center),
            peak_value=float(smooth_hist[peak_idx]),
            active_threshold=float(active_threshold),
            mean_intensity=mean_intensity,
            high_intensity=high_intensity,
            center_term=center_term,
            signal_term=signal_term,
            peak_term=peak_term,
            selection_score=selection_score,
        )

    def _analyze_cross_section_profile(
        self,
        center: np.ndarray,
        direction: np.ndarray,
        seed_profile: SeedProfile | None = None,
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

        peak_min_ratio = float(self.cfg.get("cross_section_peak_min_ratio", 0.20))
        peak_min_separation = float(self.cfg.get("cross_section_peak_min_separation_m", max(bin_size * 2.0, lane_half * 0.5)))
        peak_min_bins = max(int(np.ceil(peak_min_separation / bin_size)), 1)

        peak_indices: list[int] = []
        for idx_peak in range(smooth_hist.size):
            left_val = float(smooth_hist[idx_peak - 1]) if idx_peak > 0 else -np.inf
            right_val = float(smooth_hist[idx_peak + 1]) if idx_peak < smooth_hist.size - 1 else -np.inf
            cur_val = float(smooth_hist[idx_peak])
            if cur_val < peak_value * peak_min_ratio:
                continue
            if cur_val + 1e-12 < left_val or cur_val + 1e-12 < right_val:
                continue
            peak_indices.append(idx_peak)
        if not peak_indices:
            peak_indices = [peak_idx]

        kept_peaks: list[int] = []
        for idx_peak in sorted(peak_indices, key=lambda item: float(smooth_hist[item]), reverse=True):
            if any(abs(idx_peak - existing) < peak_min_bins for existing in kept_peaks):
                continue
            kept_peaks.append(idx_peak)

        stripe_candidates: list[TrackStripeDebug] = []
        seen_spans: set[tuple[int, int]] = set()
        for idx_peak in kept_peaks:
            stripe = self._build_cross_section_stripe(
                lateral=lateral,
                vals=vals,
                weights=weights,
                edges=edges,
                smooth_hist=smooth_hist,
                peak_idx=idx_peak,
                peak_value_max=peak_value,
                lane_half=lane_half,
                seed_profile=seed_profile,
            )
            if stripe is None:
                continue
            span_key = (
                int(round(stripe.left_m / bin_size)),
                int(round(stripe.right_m / bin_size)),
            )
            if span_key in seen_spans:
                continue
            seen_spans.add(span_key)
            stripe_candidates.append(stripe)
        if not stripe_candidates:
            return None

        stripe_candidates.sort(key=lambda stripe: stripe.selection_score, reverse=True)

        bins_center = 0.5 * (edges[:-1] + edges[1:])
        return TrackCrossSectionProfileDebug(
            bins_center=np.asarray(bins_center, dtype=np.float64),
            hist_combined=np.asarray(hist, dtype=np.float64),
            smooth_hist=np.asarray(smooth_hist, dtype=np.float64),
            selected_idx=0,
            stripe_candidates=stripe_candidates,
        )

    def _refine_centerline_cross_section(
        self,
        center: np.ndarray,
        direction: np.ndarray,
        seed_profile: SeedProfile | None = None,
    ) -> np.ndarray:
        profile = self._analyze_cross_section_profile(center, direction, seed_profile)
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
    ) -> Tuple[np.ndarray, float, np.ndarray | None, np.ndarray | None, CandidateProposal | None, List[Dict[str, Any]]]:
        pred_dir = self._search_direction_for_session(session, self._predict_direction(session.points, session.cur_dir))
        best_score = -1.0
        best_center = None
        best_dir = None
        best_proposal = None
        candidates_debug: List[Dict[str, Any]] = []

        for proposal in self._candidate_centers(session, pred_dir):
            c_xy = proposal.center_xy
            c_dir = proposal.direction
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
            allowed = self._candidate_is_allowed(center3, session.cur, pred_dir, proposal.step_distance_m, proposal.kind)
            if not allowed:
                candidates_debug.append(
                    {
                        "x": float(center3[0]),
                        "y": float(center3[1]),
                        "z": float(center3[2]),
                        "kind": proposal.kind,
                        "step_distance_m": float(proposal.step_distance_m),
                        "heading_delta_deg": float(proposal.heading_delta_deg),
                        "lateral_offset_m": float(proposal.lateral_offset_m),
                        "score": -1.0,
                        "mean_intensity": mean_intensity,
                        "high_intensity": high_intensity,
                        "rejected": "hard_gate",
                    }
                )
                continue
            score_cfg = self.cfg
            if proposal.kind == "recovery":
                score_cfg = dict(self.cfg)
                score_cfg["center_pull_weight"] = float(self.cfg.get("recovery_center_pull_weight", 0.15))
                score_cfg["straight_bias_weight"] = float(
                    self.cfg.get(
                        "recovery_straight_bias_weight",
                        float(self.cfg.get("straight_bias_weight", 0.30)) * 0.25,
                    )
                )
                score_cfg["max_heading_change_deg"] = float(
                    self.cfg.get(
                        "recovery_scoring_heading_deg",
                        max(
                            float(self.cfg.get("recovery_candidate_heading_max_deg", 13.5)),
                            float(self.cfg.get("max_heading_change_deg", 5.0)),
                        ),
                    )
                )
            sc = score_candidate(
                candidate_center=center3,
                candidate_dir=c_dir,
                prev_center=session.cur,
                prev_dir=pred_dir,
                xyz=self.xyz,
                intensity=self.intensity,
                indices=idx,
                seed_profile=session.profile,
                cfg=score_cfg,
                step_reference_m=proposal.step_distance_m,
            )
            loyalty_term = self._lane_loyalty_term(session.points, center3, pred_dir)
            loyalty_weight = float(
                self.cfg.get(
                    "recovery_lane_loyalty_weight",
                    0.10,
                )
            ) if proposal.kind == "recovery" else float(self.cfg.get("lane_loyalty_weight", 0.0))
            sc *= (1.0 - loyalty_weight) + loyalty_weight * loyalty_term
            sc *= float(proposal.score_scale)
            candidates_debug.append(
                {
                    "x": float(center3[0]),
                    "y": float(center3[1]),
                    "z": float(center3[2]),
                    "kind": proposal.kind,
                    "step_distance_m": float(proposal.step_distance_m),
                    "heading_delta_deg": float(proposal.heading_delta_deg),
                    "lateral_offset_m": float(proposal.lateral_offset_m),
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
                best_proposal = proposal

        return pred_dir, float(best_score), best_center, best_dir, best_proposal, candidates_debug

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
                    "mean_intensity": float(stripe.mean_intensity),
                    "high_intensity": float(stripe.high_intensity),
                    "center_term": float(stripe.center_term),
                    "signal_term": float(stripe.signal_term),
                    "peak_term": float(stripe.peak_term),
                    "selection_score": float(stripe.selection_score),
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
