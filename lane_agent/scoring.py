from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass
class SeedProfile:
    target_intensity: float
    background_intensity: float
    z_ref: float


def blend_seed_profile(base: SeedProfile, new: SeedProfile, alpha: float) -> SeedProfile:
    a = float(np.clip(alpha, 0.0, 1.0))
    if a <= 0.0:
        return base
    if a >= 1.0:
        return new
    keep = 1.0 - a
    return SeedProfile(
        target_intensity=keep * base.target_intensity + a * new.target_intensity,
        background_intensity=keep * base.background_intensity + a * new.background_intensity,
        z_ref=keep * base.z_ref + a * new.z_ref,
    )


def unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return v.copy()
    return v / n


def signed_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a2 = unit(a[:2])
    b2 = unit(b[:2])
    dot = float(np.clip(np.dot(a2, b2), -1.0, 1.0))
    det = float(a2[0] * b2[1] - a2[1] * b2[0])
    return float(np.degrees(np.arctan2(det, dot)))


def estimate_seed_profile(xyz: np.ndarray, intensity: np.ndarray, indices: np.ndarray) -> SeedProfile:
    if indices.size == 0:
        raise ValueError("No seed neighborhood points found.")
    local_i = intensity[indices]
    local_z = xyz[indices, 2]
    target = float(np.quantile(local_i, 0.90))
    background = float(np.quantile(local_i, 0.35))
    z_ref = float(np.median(local_z))
    return SeedProfile(target_intensity=target, background_intensity=background, z_ref=z_ref)


def score_candidate(
    candidate_center: np.ndarray,
    candidate_dir: np.ndarray,
    prev_center: np.ndarray,
    prev_dir: np.ndarray,
    xyz: np.ndarray,
    intensity: np.ndarray,
    indices: np.ndarray,
    seed_profile: SeedProfile,
    cfg: Dict[str, float],
) -> float:
    if indices.size < 6:
        return -1.0

    pts = xyz[indices]
    local_i = intensity[indices]
    local_z = pts[:, 2]

    mean_i = float(np.mean(local_i))
    high_i = float(np.quantile(local_i, 0.85))
    z_med = float(np.median(local_z))

    intensity_span = max(seed_profile.target_intensity - seed_profile.background_intensity, 1.0)
    intensity_term = np.clip((high_i - seed_profile.background_intensity) / intensity_span, 0.0, 1.2)

    # local contrast: center neighborhood should be brighter than nearby seed background
    contrast_term = np.clip((mean_i - seed_profile.background_intensity) / intensity_span, -0.2, 1.0)

    continuity_dist = float(np.linalg.norm(candidate_center[:2] - prev_center[:2]))
    step_ref = max(float(cfg["step_m"]), 1e-6)
    continuity_term = np.clip(1.0 - abs(continuity_dist - step_ref) / max(step_ref, 0.15), 0.0, 1.0)

    heading_change = abs(signed_angle_deg(prev_dir, candidate_dir))
    max_heading = max(float(cfg["max_heading_change_deg"]), 1.0)
    straight_term = np.clip(1.0 - heading_change / max_heading, 0.0, 1.0)

    z_tolerance = max(float(cfg.get("z_tolerance_m", 0.45)), 1e-3)
    z_term = np.clip(1.0 - abs(z_med - seed_profile.z_ref) / z_tolerance, 0.0, 1.0)

    max_z_step = max(float(cfg.get("max_z_step_m", 0.12)), 1e-3)
    z_step_term = np.clip(1.0 - abs(candidate_center[2] - prev_center[2]) / max_z_step, 0.0, 1.0)

    prev_dir_xy = unit(prev_dir[:2])
    pred_xy = prev_center[:2] + prev_dir_xy * step_ref
    normal_xy = np.array([-prev_dir_xy[1], prev_dir_xy[0]], dtype=np.float64)
    lateral_offset = abs(float(np.dot(candidate_center[:2] - pred_xy, normal_xy)))
    lateral_tolerance = max(float(cfg.get("center_offset_tolerance_m", 0.18)), 1e-3)
    center_term = np.clip(1.0 - lateral_offset / lateral_tolerance, 0.0, 1.0)

    score = (
        float(cfg["intensity_weight"]) * intensity_term
        + float(cfg["contrast_weight"]) * contrast_term
        + float(cfg["continuity_weight"]) * continuity_term
        + float(cfg["straight_bias_weight"]) * straight_term
    )
    score *= z_term
    score *= z_step_term
    score *= (1.0 - float(cfg.get("center_pull_weight", 0.30))) + float(cfg.get("center_pull_weight", 0.30)) * center_term
    return float(score)
