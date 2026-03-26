from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PySide6 import QtCore

from .agent import LaneTrackerAgent, TrackSession, TrackStepDebug
from .config import load_config
from .csv_io import save_xyz_csv
from .las_io import LasData, load_las_xyz_intensity


@dataclass
class LiveDebugModel:
    xyz: np.ndarray | None = None
    intensity: np.ndarray | None = None
    p0: np.ndarray | None = None
    p1: np.ndarray | None = None
    track_points: np.ndarray | None = None
    current_point: np.ndarray | None = None
    predicted_points: np.ndarray | None = None
    search_box_points: np.ndarray | None = None
    status_text: str = "Ready"
    point_cloud_revision: int = 0
    candidate_rows: list[dict[str, Any]] = field(default_factory=list)
    score_history: list[float] = field(default_factory=list)
    latest_step: TrackStepDebug | None = None
    cross_section_profile: Any | None = None


class LiveDebugController(QtCore.QObject):
    changed = QtCore.Signal()
    log_message = QtCore.Signal(str)

    def __init__(self) -> None:
        super().__init__()
        self.model = LiveDebugModel()
        self.las: LasData | None = None
        self.cfg: dict[str, Any] | None = None
        self.cfg_path: str = ""
        self.las_path: str = ""
        self.output_path: str = ""
        self.agent: LaneTrackerAgent | None = None
        self.session: TrackSession | None = None

    def load_las(self, path: str) -> None:
        self.las = load_las_xyz_intensity(path)
        self.las_path = str(path)
        self.agent = None
        self.session = None
        self.model.xyz = self.las.xyz
        self.model.intensity = self.las.intensity
        self.model.track_points = None
        self.model.current_point = None
        self.model.predicted_points = None
        self.model.search_box_points = None
        self.model.candidate_rows = []
        self.model.score_history = []
        self.model.latest_step = None
        self.model.cross_section_profile = None
        self.model.point_cloud_revision += 1
        self.model.status_text = f"Loaded LAS | points={len(self.las.xyz):,}"
        self.log_message.emit(f"Loaded LAS: {path}")
        self.changed.emit()

    def load_tracker_config(self, path: str) -> None:
        self.cfg = load_config(path)
        self.cfg_path = str(path)
        self.agent = None
        self.session = None
        self.model.track_points = None
        self.model.current_point = None
        self.model.predicted_points = None
        self.model.search_box_points = None
        self.model.candidate_rows = []
        self.model.score_history = []
        self.model.latest_step = None
        self.model.cross_section_profile = None
        self.model.status_text = f"Loaded config | {Path(path).name}"
        self.log_message.emit(f"Loaded config: {path}")
        self.changed.emit()

    def set_output_path(self, path: str) -> None:
        self.output_path = str(path)

    def set_p0(self, x: float, y: float, z: float) -> None:
        self.model.p0 = np.array([x, y, z], dtype=np.float64)
        self.changed.emit()

    def set_p1(self, x: float, y: float, z: float) -> None:
        self.model.p1 = np.array([x, y, z], dtype=np.float64)
        self.changed.emit()

    def initialize_tracker(self) -> None:
        if self.las is None:
            raise RuntimeError("LAS must be loaded first.")
        if self.cfg is None:
            raise RuntimeError("Config must be loaded first.")
        if self.model.p0 is None or self.model.p1 is None:
            raise RuntimeError("P0 and P1 are required.")
        self.agent = LaneTrackerAgent(self.las.xyz, self.las.intensity, self.cfg)
        self.session = self.agent.initialize_session(self.model.p0, self.model.p1)
        self.model.track_points = np.asarray(self.session.points, dtype=np.float64)
        self.model.current_point = self.session.cur.copy()
        self.model.predicted_points = None
        self.model.search_box_points = None
        self.model.candidate_rows = []
        self.model.score_history = list(self.session.scores)
        self.model.latest_step = None
        self.model.cross_section_profile = self.agent._analyze_cross_section_profile(self.session.cur, self.session.cur_dir)
        mode = "finished" if self.session.finished else "ready"
        self.model.status_text = f"Initialized | mode={mode} | accepted={len(self.session.points)}"
        self.log_message.emit("Tracker initialized")
        self.changed.emit()

    def run_step(self) -> TrackStepDebug:
        if self.agent is None or self.session is None:
            raise RuntimeError("Tracker is not initialized.")
        if self.session.finished:
            stop_reason = self.session.stop_reason or "finished"
            self.model.status_text = f"Stopped | reason={stop_reason}"
            self.log_message.emit(f"Step blocked: already stopped ({stop_reason})")
            self.changed.emit()
            if self.session.latest_step is None:
                raise RuntimeError(f"Tracker already finished: {stop_reason}")
            return self.session.latest_step
        step = self.agent.step_session(self.session)
        self._sync_model_from_step(step)
        return step

    def run_full(self) -> None:
        if self.agent is None or self.session is None:
            raise RuntimeError("Tracker is not initialized.")
        if self.session.finished:
            stop_reason = self.session.stop_reason or "finished"
            self.model.status_text = f"Stopped | reason={stop_reason}"
            self.changed.emit()
            return
        last_step: TrackStepDebug | None = None
        while not self.session.finished:
            last_step = self.agent.step_session(self.session)
        if last_step is not None:
            self._sync_model_from_step(last_step)
        else:
            self.changed.emit()

    def save_outputs(self, output_path: str | None = None) -> None:
        if self.agent is None or self.session is None:
            raise RuntimeError("Tracker is not initialized.")
        output_str = str(output_path or self.output_path).strip()
        if not output_str:
            if not self.las_path:
                raise RuntimeError("Output path is empty.")
            output_str = str(Path(self.las_path).with_suffix(".live.csv"))
        output = Path(output_str)
        output.parent.mkdir(parents=True, exist_ok=True)
        post_output = output.with_name(f"{output.stem}_pc{output.suffix}")
        debug_path = output.with_suffix(output.suffix + ".debug.json")
        result = self.agent.finalize_session(self.session, str(debug_path))
        save_xyz_csv(output, result.raw_points)
        save_xyz_csv(post_output, result.points)
        self.output_path = str(output)
        debug_saved = bool(self.cfg and self.cfg.get("save_debug_json", True))
        self.model.status_text = f"Saved outputs | raw={output.name} | stop={result.stop_reason}"
        self.log_message.emit(f"Saved raw CSV: {output}")
        self.log_message.emit(f"Saved post CSV: {post_output}")
        if debug_saved:
            self.log_message.emit(f"Saved debug JSON: {debug_path}")
        self.changed.emit()

    def reset(self) -> None:
        self.agent = None
        self.session = None
        self.model.track_points = None
        self.model.current_point = None
        self.model.predicted_points = None
        self.model.search_box_points = None
        self.model.candidate_rows = []
        self.model.score_history = []
        self.model.latest_step = None
        self.model.cross_section_profile = None
        self.model.status_text = "Reset"
        self.log_message.emit("Reset")
        self.changed.emit()

    def _sync_model_from_step(self, step: TrackStepDebug) -> None:
        assert self.session is not None
        self.model.track_points = np.asarray(self.session.points, dtype=np.float64)
        self.model.current_point = self.session.cur.copy()
        self.model.predicted_points = self._build_candidate_points(step.candidates)
        self.model.search_box_points = self._build_candidate_box(step.current_center, step.predicted_direction, step.candidates)
        self.model.candidate_rows = self._sort_candidate_rows(step.candidates)
        self.model.score_history = list(self.session.scores)
        self.model.latest_step = step
        self.model.cross_section_profile = step.cross_section_profile
        stop_reason = step.stop_reason or self.session.stop_reason or "none"
        self.model.status_text = (
            f"Step {step.step_index} | mode={step.mode} | best={step.best_score:.3f} | "
            f"accepted={len(self.session.points)} | stop={stop_reason}"
        )
        self._emit_step_log(step)
        self.changed.emit()

    def _emit_step_log(self, step: TrackStepDebug) -> None:
        stop_text = step.stop_reason or "none"
        rows = self._sort_candidate_rows(step.candidates)
        top_rows = rows[:3]
        top_text = " || ".join(
            [
                (
                    f"score={float(row.get('score', -1.0)):.3f}, "
                    f"xy=({float(row.get('x', 0.0)):.3f}, {float(row.get('y', 0.0)):.3f}), "
                    f"reject={row.get('rejected', '-')}"
                )
                for row in top_rows
            ]
        )
        self.log_message.emit(
            (
                f"STEP {step.step_index} | mode={step.mode} | best={step.best_score:.3f} | "
                f"cand={len(step.candidates)} | accepted={len(self.session.points) if self.session is not None else 0} | "
                f"gap={step.gap_accum_m:.2f} | stop={stop_text}"
            )
        )
        if top_text:
            self.log_message.emit(f"top_candidates={top_text}")

    def _build_candidate_points(self, rows: list[dict[str, Any]]) -> np.ndarray | None:
        if not rows:
            return None
        pts = np.array([[float(row["x"]), float(row["y"]), float(row["z"])] for row in rows], dtype=np.float64)
        return pts if pts.size else None

    def _build_candidate_box(
        self,
        current_center: np.ndarray,
        predicted_direction: np.ndarray | None,
        rows: list[dict[str, Any]],
    ) -> np.ndarray | None:
        if predicted_direction is None or not rows:
            return None
        dir_xy = np.asarray(predicted_direction[:2], dtype=np.float64)
        norm = float(np.linalg.norm(dir_xy))
        if norm <= 1e-9:
            return None
        dir_xy = dir_xy / norm
        normal_xy = np.array([-dir_xy[1], dir_xy[0]], dtype=np.float64)
        pts = np.array([[float(row["x"]), float(row["y"])] for row in rows], dtype=np.float64)
        along = pts @ dir_xy
        lateral = pts @ normal_xy
        along_min = float(np.min(along))
        along_max = float(np.max(along))
        lat_min = float(np.min(lateral))
        lat_max = float(np.max(lateral))
        z = float(current_center[2])

        def make_point(along_val: float, lat_val: float) -> np.ndarray:
            xy = dir_xy * along_val + normal_xy * lat_val
            return np.array([xy[0], xy[1], z], dtype=np.float64)

        corners = np.vstack(
            [
                make_point(along_max, lat_max),
                make_point(along_max, lat_min),
                make_point(along_min, lat_min),
                make_point(along_min, lat_max),
                make_point(along_max, lat_max),
            ]
        )
        return corners

    def _sort_candidate_rows(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        ranked = list(rows)
        ranked.sort(
            key=lambda row: (
                1 if row.get("rejected") else 0,
                -float(row.get("score", -1.0)),
            )
        )
        return ranked
