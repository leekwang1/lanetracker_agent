from __future__ import annotations

import time

from PySide6 import QtCore, QtWidgets
import numpy as np

try:
    import pyvista as pv
    from pyvistaqt import QtInteractor
except Exception:  # pragma: no cover - runtime fallback when dependency is unavailable
    pv = None
    QtInteractor = None


class PointCloudViewWidget(QtWidgets.QWidget):
    point_context_menu_requested = QtCore.Signal(object, object)
    debug_message = QtCore.Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._cloud_revision = -1
        self._xyz_world: np.ndarray | None = None
        self._xy_world: np.ndarray | None = None
        self._xyz_local: np.ndarray | None = None
        self._xy_local: np.ndarray | None = None
        self._intensity: np.ndarray | None = None
        self._origin_xyz = np.zeros(3, dtype=np.float64)
        self._pick_xyz_world: np.ndarray | None = None
        self._pick_xy_local: np.ndarray | None = None
        self._display_xyz_local: np.ndarray | None = None
        self._detail_xyz_local: np.ndarray | None = None
        self._detail_xy_local: np.ndarray | None = None
        self._detail_intensity: np.ndarray | None = None
        self._display_poly: object | None = None
        self._detail_poly: object | None = None
        self._seed_line_poly: object | None = None
        self._seed_p0_poly: object | None = None
        self._seed_p1_poly: object | None = None
        self._track_poly: object | None = None
        self._current_poly: object | None = None
        self._pred_poly: object | None = None
        self._trajectory_line_poly: object | None = None
        self._profile_line_poly: object | None = None
        self._stripe_segment_poly: object | None = None
        self._stripe_edge_poly: object | None = None
        self._search_box_poly: object | None = None
        self._display_actor = None
        self._detail_actor = None
        self._seed_line_actor = None
        self._seed_p0_actor = None
        self._seed_p1_actor = None
        self._track_actor = None
        self._current_actor = None
        self._pred_actor = None
        self._trajectory_line_actor = None
        self._profile_line_actor = None
        self._stripe_segment_actor = None
        self._stripe_edge_actor = None
        self._search_box_actor = None
        self._display_limit = 100_000
        self._pick_limit = 1_500_000
        self._detail_source_limit = 4_000_000
        self._visible_limit = 320_000
        self._visible_mid_limit = 1_500_000
        self._visible_full_limit = 1_500_000
        self._raw_full_limit = 1_200_000
        self._detail_mid_area_ratio = 1.0
        self._detail_full_area_ratio = 1.0
        self._raw_full_area_ratio = 1.0
        self._log_visible_refresh = False
        self._view_rect_cache: tuple[float, float, float, float] | None = None

        self._viewport_timer = QtCore.QTimer(self)
        self._viewport_timer.setInterval(180)
        self._viewport_timer.timeout.connect(self._refresh_visible_points_if_needed)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self._status_label = QtWidgets.QLabel()
        self._status_label.setAlignment(QtCore.Qt.AlignCenter)
        self._status_label.setWordWrap(True)
        layout.addWidget(self._status_label)

        self.plotter = None
        if pv is None or QtInteractor is None:
            self._status_label.setText("PyVistaQt is not installed.\nInstall `pyvista` and `pyvistaqt` in the active environment.")
            self.debug_message.emit("PyVistaQt unavailable: install `pyvista` and `pyvistaqt` in the active environment.")
            return

        self._status_label.hide()
        pv.global_theme.background = "#050816"
        pv.global_theme.font.color = "white"

        self.plotter = QtInteractor(self)
        self.plotter.set_background("#050816")
        self.plotter.enable_parallel_projection()
        self.plotter.view_xy()
        self.plotter.show_axes()
        self.plotter.add_axes()

        self.plotter.interactor.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.plotter.interactor.customContextMenuRequested.connect(self._open_context_menu)
        layout.addWidget(self.plotter.interactor)
        self.debug_message.emit("PyVistaQt interactor initialized")

    def set_point_cloud(self, xyz: np.ndarray, intensity: np.ndarray, revision: int | None = None) -> None:
        if self.plotter is None:
            self.debug_message.emit("set_point_cloud skipped: plotter is not available")
            return
        if xyz is None or xyz.size == 0:
            self._clear_scene()
            self.debug_message.emit("Point cloud cleared")
            return
        if revision is not None and revision == self._cloud_revision:
            if self._log_visible_refresh:
                self.debug_message.emit(f"set_point_cloud skipped: revision {revision} already rendered")
            return

        t0 = time.perf_counter()
        self._xyz_world = np.asarray(xyz, dtype=np.float64)
        self._xy_world = self._xyz_world[:, :2]
        self._intensity = np.asarray(intensity, dtype=np.float32)
        mins = self._xyz_world.min(axis=0)
        maxs = self._xyz_world.max(axis=0)
        self._origin_xyz = ((mins + maxs) * 0.5).astype(np.float64)
        self._xyz_local = self._xyz_world - self._origin_xyz[None, :]
        self._xy_local = self._xyz_local[:, :2]

        display_idx = self._grid_sample_indices(self._xy_local, target_count=self._display_limit)
        pick_idx = self._grid_sample_indices(self._xy_local, target_count=self._pick_limit)
        detail_idx = self._grid_sample_indices(self._xy_local, target_count=self._detail_source_limit)
        self._display_xyz_local = self._xyz_local[display_idx]
        self._pick_xyz_world = self._xyz_world[pick_idx]
        self._pick_xy_local = self._pick_xyz_world[:, :2] - self._origin_xyz[None, :2]
        self._detail_xyz_local = self._xyz_local[detail_idx]
        self._detail_xy_local = self._detail_xyz_local[:, :2]
        self._detail_intensity = self._intensity[detail_idx]
        self._cloud_revision = revision if revision is not None else self._cloud_revision + 1
        self._view_rect_cache = None

        self._display_poly = pv.PolyData(self._display_xyz_local.astype(np.float32, copy=False))
        self._display_poly["intensity"] = self._intensity[display_idx]

        prep_elapsed = time.perf_counter() - t0
        self.debug_message.emit(
            "Point cloud prepared: "
            f"total={len(self._xyz_world):,}, display={len(display_idx):,}, detail_source={len(detail_idx):,}, "
            f"pick={len(pick_idx):,}, prep={prep_elapsed:.3f}s"
        )
        self.debug_message.emit(
            f"Bounds x=({mins[0]:.3f}, {maxs[0]:.3f}) y=({mins[1]:.3f}, {maxs[1]:.3f}) z=({mins[2]:.3f}, {maxs[2]:.3f})"
        )
        self.debug_message.emit(
            f"Viewer origin shift=({self._origin_xyz[0]:.3f}, {self._origin_xyz[1]:.3f}, {self._origin_xyz[2]:.3f})"
        )

        if self._display_actor is not None:
            self.plotter.remove_actor(self._display_actor, render=False)
        render_t0 = time.perf_counter()
        self._display_actor = self.plotter.add_points(
            self._display_poly,
            scalars="intensity",
            cmap="gray",
            render_points_as_spheres=True,
            point_size=4,
            opacity=1,
            reset_camera=False,
            show_scalar_bar=False,
        )
        render_elapsed = time.perf_counter() - render_t0
        self.debug_message.emit(f"Point cloud uploaded: n={len(display_idx):,}, t={render_elapsed:.3f}s")
        self.reset_view()
        self._refresh_visible_points(force=True)
        self._viewport_timer.start()
        self.plotter.render()

    def reset_view(self) -> None:
        if self.plotter is None or self._xyz_local is None or len(self._xyz_local) == 0:
            self.debug_message.emit("reset_view skipped: plotter or xyz missing")
            return
        self.plotter.view_xy()
        self.plotter.enable_parallel_projection()
        self.plotter.reset_camera()
        self._view_rect_cache = None
        self.debug_message.emit(f"reset_view completed: camera_position={self.plotter.camera_position}")

    def render(self) -> None:
        if self.plotter is not None:
            self.plotter.render()

    def focus_on_point(self, point_xyz: np.ndarray | None, view_width_m: float = 9.0) -> None:
        if self.plotter is None or point_xyz is None:
            return
        point = np.asarray(point_xyz, dtype=np.float64)
        if point.shape[0] == 2:
            point = np.array([point[0], point[1], 0.0], dtype=np.float64)
        local = point - self._origin_xyz
        try:
            camera = self.plotter.camera
            current = list(camera.GetPosition())
            distance = float(current[2] - camera.GetFocalPoint()[2])
            if abs(distance) < 1e-6:
                distance = max(view_width_m * 1.5, 10.0)
            camera.SetFocalPoint(float(local[0]), float(local[1]), float(local[2]))
            camera.SetPosition(float(local[0]), float(local[1]), float(local[2] + distance))
            camera.SetViewUp(0.0, 1.0, 0.0)
            camera.SetParallelScale(max(float(view_width_m) * 0.5, 1.0))
            self._view_rect_cache = None
            self._refresh_visible_points(force=True)
            self.plotter.render()
        except Exception as exc:
            self.debug_message.emit(f"focus_on_point failed: {exc!r}")

    def set_track(self, points: np.ndarray | None, render: bool = True):
        if self.plotter is None:
            return
        if self._track_actor is not None:
            self.plotter.remove_actor(self._track_actor, render=False)
            self._track_actor = None
        if points is None or points.size == 0:
            if render:
                self.plotter.render()
            return
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape[1] == 2:
            pts = np.column_stack([pts, np.zeros(len(pts), dtype=np.float64)])
        pts = pts - self._origin_xyz[None, :]
        self._track_poly = pv.lines_from_points(pts.astype(np.float32, copy=False), close=False)
        self._track_actor = self.plotter.add_mesh(
            self._track_poly,
            color="#38bdf8",
            line_width=3,
            render_lines_as_tubes=False,
            reset_camera=False,
        )
        if render:
            self.plotter.render()

    def set_seed_points(self, p0: np.ndarray | None, p1: np.ndarray | None, render: bool = True):
        if self.plotter is None:
            return
        for actor_name in ["_seed_line_actor", "_seed_p0_actor", "_seed_p1_actor"]:
            actor = getattr(self, actor_name)
            if actor is not None:
                self.plotter.remove_actor(actor, render=False)
                setattr(self, actor_name, None)
        if p0 is None or p1 is None:
            if render:
                self.plotter.render()
            return

        p0a = np.asarray(p0, dtype=np.float64)
        p1a = np.asarray(p1, dtype=np.float64)
        if p0a.shape[0] == 2:
            p0a = np.array([p0a[0], p0a[1], 0.0], dtype=np.float64)
        if p1a.shape[0] == 2:
            p1a = np.array([p1a[0], p1a[1], 0.0], dtype=np.float64)
        pts = np.vstack([p0a, p1a]) - self._origin_xyz[None, :]

        self._seed_line_poly = pv.lines_from_points(pts.astype(np.float32, copy=False), close=False)
        self._seed_line_actor = self.plotter.add_mesh(
            self._seed_line_poly,
            color="#e5e7eb",
            line_width=2,
            render_lines_as_tubes=False,
            reset_camera=False,
        )
        self._seed_p0_poly = pv.PolyData(pts[0:1].astype(np.float32, copy=False))
        self._seed_p1_poly = pv.PolyData(pts[1:2].astype(np.float32, copy=False))
        self._seed_p0_actor = self.plotter.add_points(
            self._seed_p0_poly,
            color="#22c55e",
            render_points_as_spheres=True,
            point_size=12,
            reset_camera=False,
        )
        self._seed_p1_actor = self.plotter.add_points(
            self._seed_p1_poly,
            color="#3b82f6",
            render_points_as_spheres=True,
            point_size=12,
            reset_camera=False,
        )
        if render:
            self.plotter.render()

    def set_current(self, p: np.ndarray | None, render: bool = True):
        if self.plotter is None:
            return
        if self._current_actor is not None:
            self.plotter.remove_actor(self._current_actor, render=False)
            self._current_actor = None
        if p is None:
            if render:
                self.plotter.render()
            return
        point = np.asarray(p, dtype=np.float64)
        if point.shape[0] == 2:
            point = np.array([point[0], point[1], 0.0], dtype=np.float64)
        point = point - self._origin_xyz
        self._current_poly = pv.PolyData(point[None, :].astype(np.float32, copy=False))
        self._current_actor = self.plotter.add_points(
            self._current_poly,
            color="#ef4444",
            render_points_as_spheres=True,
            point_size=12,
            reset_camera=False,
        )
        if render:
            self.plotter.render()

    def set_predicted(self, pts: np.ndarray | None, render: bool = True):
        if self.plotter is None:
            return
        if self._pred_actor is not None:
            self.plotter.remove_actor(self._pred_actor, render=False)
            self._pred_actor = None
        if pts is None or pts.size == 0:
            if render:
                self.plotter.render()
            return
        points = np.asarray(pts, dtype=np.float64)
        if points.shape[1] == 2:
            points = np.column_stack([points, np.zeros(len(points), dtype=np.float64)])
        points = points - self._origin_xyz[None, :]
        self._pred_poly = pv.PolyData(points.astype(np.float32, copy=False))
        self._pred_actor = self.plotter.add_points(
            self._pred_poly,
            color="#f59e0b",
            render_points_as_spheres=True,
            point_size=8,
            opacity=0.7,
            reset_camera=False,
        )
        if render:
            self.plotter.render()

    def set_profile_overlay(
        self,
        profile_line: np.ndarray | None,
        stripe_segment: np.ndarray | None,
        stripe_edges: np.ndarray | None,
        render: bool = True,
    ) -> None:
        if self.plotter is None:
            return

        for actor_name in ["_profile_line_actor", "_stripe_segment_actor", "_stripe_edge_actor"]:
            actor = getattr(self, actor_name)
            if actor is not None:
                self.plotter.remove_actor(actor, render=False)
                setattr(self, actor_name, None)

        if profile_line is not None and np.asarray(profile_line).size:
            pts = np.asarray(profile_line, dtype=np.float64)
            if pts.shape[1] == 2:
                pts = np.column_stack([pts, np.zeros(len(pts), dtype=np.float64)])
            pts = pts - self._origin_xyz[None, :]
            self._profile_line_poly = pv.lines_from_points(pts.astype(np.float32, copy=False), close=False)
            self._profile_line_actor = self.plotter.add_mesh(
                self._profile_line_poly,
                color="#22d3ee",
                line_width=1.5,
                render_lines_as_tubes=False,
                reset_camera=False,
            )

        if stripe_segment is not None and np.asarray(stripe_segment).size:
            pts = np.asarray(stripe_segment, dtype=np.float64)
            if pts.shape[1] == 2:
                pts = np.column_stack([pts, np.zeros(len(pts), dtype=np.float64)])
            pts = pts - self._origin_xyz[None, :]
            self._stripe_segment_poly = pv.lines_from_points(pts.astype(np.float32, copy=False), close=False)
            self._stripe_segment_actor = self.plotter.add_mesh(
                self._stripe_segment_poly,
                color="#f472b6",
                line_width=4,
                render_lines_as_tubes=False,
                reset_camera=False,
            )

        if stripe_edges is not None and np.asarray(stripe_edges).size:
            pts = np.asarray(stripe_edges, dtype=np.float64)
            if pts.shape[1] == 2:
                pts = np.column_stack([pts, np.zeros(len(pts), dtype=np.float64)])
            pts = pts - self._origin_xyz[None, :]
            self._stripe_edge_poly = pv.PolyData(pts.astype(np.float32, copy=False))
            self._stripe_edge_actor = self.plotter.add_points(
                self._stripe_edge_poly,
                color="#fb7185",
                render_points_as_spheres=True,
                point_size=10,
                reset_camera=False,
            )

        if render:
            self.plotter.render()

    def set_trajectory_line(self, line_points: np.ndarray | None, render: bool = True) -> None:
        if self.plotter is None:
            return
        if self._trajectory_line_actor is not None:
            self.plotter.remove_actor(self._trajectory_line_actor, render=False)
            self._trajectory_line_actor = None
        if line_points is None or np.asarray(line_points).size == 0:
            if render:
                self.plotter.render()
            return
        pts = np.asarray(line_points, dtype=np.float64)
        if pts.shape[1] == 2:
            pts = np.column_stack([pts, np.zeros(len(pts), dtype=np.float64)])
        pts = pts - self._origin_xyz[None, :]
        self._trajectory_line_poly = pv.lines_from_points(pts.astype(np.float32, copy=False), close=False)
        self._trajectory_line_actor = self.plotter.add_mesh(
            self._trajectory_line_poly,
            color="#fde047",
            line_width=2.5,
            opacity=0.95,
            render_lines_as_tubes=False,
            reset_camera=False,
        )
        if render:
            self.plotter.render()

    def set_search_box(self, box_points: np.ndarray | None, render: bool = True) -> None:
        if self.plotter is None:
            return
        if self._search_box_actor is not None:
            self.plotter.remove_actor(self._search_box_actor, render=False)
            self._search_box_actor = None
        if box_points is None or np.asarray(box_points).size == 0:
            if render:
                self.plotter.render()
            return
        pts = np.asarray(box_points, dtype=np.float64)
        if pts.shape[1] == 2:
            pts = np.column_stack([pts, np.zeros(len(pts), dtype=np.float64)])
        pts = pts - self._origin_xyz[None, :]
        self._search_box_poly = pv.lines_from_points(pts.astype(np.float32, copy=False), close=False)
        self._search_box_actor = self.plotter.add_mesh(
            self._search_box_poly,
            color="#a78bfa",
            line_width=1.5,
            opacity=0.85,
            render_lines_as_tubes=False,
            reset_camera=False,
        )
        if render:
            self.plotter.render()

    def _open_context_menu(self, pos) -> None:
        point = self._point_from_widget_pos(pos)
        global_pos = self.plotter.interactor.mapToGlobal(pos)
        self.debug_message.emit(
            f"context menu requested at widget=({pos.x()}, {pos.y()}) point={None if point is None else point.tolist()}"
        )
        self.point_context_menu_requested.emit(point, global_pos)

    def _point_from_widget_pos(self, pos) -> np.ndarray | None:
        if self.plotter is None or self._pick_xyz_world is None or self._pick_xy_local is None:
            return None
        anchor_xy = self._widget_pos_to_local_xy(pos)
        if anchor_xy is None:
            world = None
            try:
                if hasattr(self.plotter, "pick_mouse_position"):
                    world = self.plotter.pick_mouse_position()
            except Exception as exc:
                self.debug_message.emit(f"pick_mouse_position failed: {exc!r}")
            if world is None:
                self.debug_message.emit("pick_mouse_position returned None")
                return None
            world = np.asarray(world, dtype=np.float64)
            anchor_xy = world[:2]
        return self._find_nearest_point(anchor_xy)

    def _find_nearest_point(self, anchor_xy: np.ndarray) -> np.ndarray | None:
        if self._xyz_world is None or self._xy_local is None:
            return None
        rect = self._current_view_rect_xy()
        if rect is not None:
            view_w = max(float(rect[1] - rect[0]), 1e-6)
            view_h = max(float(rect[3] - rect[2]), 1e-6)
            radius = max(min(max(view_w, view_h) * 0.02, 1.0), 0.03)
        else:
            span = float(np.max(self._xy_local[:, 0]) - np.min(self._xy_local[:, 0]))
            radius = max(min(span * 0.005, 2.0), 0.10)

        local_mask = (
            (self._xy_local[:, 0] >= anchor_xy[0] - radius)
            & (self._xy_local[:, 0] <= anchor_xy[0] + radius)
            & (self._xy_local[:, 1] >= anchor_xy[1] - radius)
            & (self._xy_local[:, 1] <= anchor_xy[1] + radius)
        )
        candidate_idx = np.flatnonzero(local_mask)
        source_label = "full-local"
        candidate_xy = self._xy_local[candidate_idx] if candidate_idx.size else None

        if candidate_idx.size == 0 and self._pick_xy_local is not None and self._pick_xyz_world is not None:
            source_label = "pick-fallback"
            d2 = np.sum((self._pick_xy_local - anchor_xy[None, :]) ** 2, axis=1)
            nearest = int(np.argmin(d2))
            nearest_dist = float(np.sqrt(d2[nearest]))
            if nearest_dist > radius:
                self.debug_message.emit(
                    f"_find_nearest_point miss: source={source_label}, radius={radius:.3f}, nearest_dist={nearest_dist:.3f}"
                )
                return None
            self.debug_message.emit(
                f"_find_nearest_point hit: source={source_label}, radius={radius:.3f}, nearest_dist={nearest_dist:.3f}"
            )
            return self._pick_xyz_world[nearest].astype(float, copy=True)

        if candidate_idx.size == 0:
            self.debug_message.emit(f"_find_nearest_point miss: source={source_label}, radius={radius:.3f}, no candidates")
            return None

        d2 = np.sum((candidate_xy - anchor_xy[None, :]) ** 2, axis=1)
        nearest_local = int(np.argmin(d2))
        nearest_dist = float(np.sqrt(d2[nearest_local]))
        if nearest_dist > radius:
            self.debug_message.emit(
                f"_find_nearest_point miss: source={source_label}, radius={radius:.3f}, nearest_dist={nearest_dist:.3f}, candidates={candidate_idx.size:,}"
            )
            return None
        world_idx = int(candidate_idx[nearest_local])
        self.debug_message.emit(
            f"_find_nearest_point hit: source={source_label}, radius={radius:.3f}, nearest_dist={nearest_dist:.3f}, candidates={candidate_idx.size:,}"
        )
        return self._xyz_world[world_idx].astype(float, copy=True)

    def _widget_pos_to_local_xy(self, pos) -> np.ndarray | None:
        rect = self._current_pick_rect_xy()
        if rect is None:
            return None
        if self.plotter is None:
            return None
        try:
            size = self.plotter.interactor.GetRenderWindow().GetSize()
            width_px = max(int(size[0]), 1)
            height_px = max(int(size[1]), 1)
        except Exception as exc:
            self.debug_message.emit(f"_widget_pos_to_local_xy size failed: {exc!r}")
            return None
        x0, x1, y0, y1 = rect
        px = float(np.clip(pos.x(), 0, width_px - 1))
        py = float(np.clip(pos.y(), 0, height_px - 1))
        x = x0 + (px / max(width_px - 1, 1)) * (x1 - x0)
        y = y1 - (py / max(height_px - 1, 1)) * (y1 - y0)
        return np.array([x, y], dtype=np.float64)

    def _clear_scene(self) -> None:
        self._xyz_world = None
        self._xy_world = None
        self._xyz_local = None
        self._xy_local = None
        self._intensity = None
        self._origin_xyz[:] = 0.0
        self._pick_xyz_world = None
        self._pick_xy_local = None
        self._display_xyz_local = None
        self._display_poly = None
        self._track_poly = None
        self._current_poly = None
        self._pred_poly = None
        self._trajectory_line_poly = None
        self._profile_line_poly = None
        self._stripe_segment_poly = None
        self._stripe_edge_poly = None
        self._search_box_poly = None
        self._cloud_revision = -1
        self._detail_xyz_local = None
        self._detail_xy_local = None
        self._detail_intensity = None
        self._detail_poly = None
        self._seed_line_poly = None
        self._seed_p0_poly = None
        self._seed_p1_poly = None
        self._view_rect_cache = None
        self._viewport_timer.stop()
        if self.plotter is None:
            return
        for actor_name in ["_display_actor", "_detail_actor", "_seed_line_actor", "_seed_p0_actor", "_seed_p1_actor", "_track_actor", "_current_actor", "_pred_actor", "_trajectory_line_actor", "_profile_line_actor", "_stripe_segment_actor", "_stripe_edge_actor", "_search_box_actor"]:
            actor = getattr(self, actor_name)
            if actor is not None:
                self.plotter.remove_actor(actor, render=False)
                setattr(self, actor_name, None)
        self.plotter.render()

    def _refresh_visible_points_if_needed(self) -> None:
        self._refresh_visible_points(force=False)

    def _refresh_visible_points(self, force: bool) -> None:
        if self.plotter is None or self._xy_local is None or self._xyz_local is None or self._intensity is None:
            return
        rect = self._current_view_rect_xy()
        if rect is None:
            self.debug_message.emit("Visible refresh skipped: could not compute current view rect")
            return
        rect_key = tuple(round(v, 3) for v in rect)
        if not force and rect_key == self._view_rect_cache:
            return
        self._view_rect_cache = rect_key

        x0, x1, y0, y1 = rect
        t0 = time.perf_counter()
        area_ratio = self._view_area_ratio(rect)
        raw_mask = (
            (self._xy_local[:, 0] >= x0)
            & (self._xy_local[:, 0] <= x1)
            & (self._xy_local[:, 1] >= y0)
            & (self._xy_local[:, 1] <= y1)
        )
        raw_idx = np.flatnonzero(raw_mask)
        raw_count = int(raw_idx.size)

        if raw_count <= self._raw_full_limit and area_ratio <= self._raw_full_area_ratio:
            points_local = self._xyz_local[raw_idx]
            intensities = self._intensity[raw_idx]
            sample_mode = "raw-full"
        else:
            if self._detail_xyz_local is None or self._detail_xy_local is None or self._detail_intensity is None:
                return
            mask = (
                (self._detail_xy_local[:, 0] >= x0)
                & (self._detail_xy_local[:, 0] <= x1)
                & (self._detail_xy_local[:, 1] >= y0)
                & (self._detail_xy_local[:, 1] <= y1)
            )
            idx = np.flatnonzero(mask)
            if idx.size <= self._visible_full_limit and area_ratio <= self._detail_full_area_ratio:
                sample_mode = "detail-full"
            elif idx.size <= self._visible_mid_limit and area_ratio <= self._detail_mid_area_ratio:
                local_xy = self._detail_xy_local[idx]
                keep = self._grid_sample_indices(local_xy, target_count=self._visible_mid_limit)
                idx = idx[keep]
                sample_mode = "detail-mid"
            else:
                local_xy = self._detail_xy_local[idx]
                keep = self._grid_sample_indices(local_xy, target_count=self._visible_limit)
                idx = idx[keep]
                sample_mode = "grid-sampled"
            points_local = self._detail_xyz_local[idx]
            intensities = self._detail_intensity[idx]

        if self._detail_actor is not None:
            self.plotter.remove_actor(self._detail_actor, render=False)
            self._detail_actor = None
        if len(points_local) > 0:
            self._detail_poly = pv.PolyData(points_local.astype(np.float32, copy=False))
            self._detail_poly["intensity"] = intensities
            self._detail_actor = self.plotter.add_points(
                self._detail_poly,
                scalars="intensity",
                cmap="gray",
                render_points_as_spheres=True,
                point_size=4,
                opacity=1,
                reset_camera=False,
                show_scalar_bar=False,
            )
        elapsed = time.perf_counter() - t0
        if self._log_visible_refresh:
            self.debug_message.emit(
                f"Visible points refreshed: n={len(points_local):,}, raw={raw_count:,}, "
                f"area_ratio={area_ratio:.5f}, mode={sample_mode}, "
                f"rect=({x0:.3f}, {x1:.3f}, {y0:.3f}, {y1:.3f}), t={elapsed:.3f}s"
            )
        self.plotter.render()

    def _current_view_rect_xy(self) -> tuple[float, float, float, float] | None:
        if self.plotter is None:
            return None
        try:
            camera = self.plotter.camera
            focal = camera.GetFocalPoint()
            half_h = float(camera.GetParallelScale())
            size = self.plotter.interactor.GetRenderWindow().GetSize()
            width_px = max(int(size[0]), 1)
            height_px = max(int(size[1]), 1)
            aspect = width_px / max(height_px, 1)
            half_w = half_h * aspect
            margin_x = half_w * 0.30
            margin_y = half_h * 0.30
            return (
                float(focal[0]) - half_w - margin_x,
                float(focal[0]) + half_w + margin_x,
                float(focal[1]) - half_h - margin_y,
                float(focal[1]) + half_h + margin_y,
            )
        except Exception as exc:
            self.debug_message.emit(f"_current_view_rect_xy failed: {exc!r}")
            return None

    def _current_pick_rect_xy(self) -> tuple[float, float, float, float] | None:
        if self.plotter is None:
            return None
        try:
            camera = self.plotter.camera
            focal = camera.GetFocalPoint()
            half_h = float(camera.GetParallelScale())
            size = self.plotter.interactor.GetRenderWindow().GetSize()
            width_px = max(int(size[0]), 1)
            height_px = max(int(size[1]), 1)
            aspect = width_px / max(height_px, 1)
            half_w = half_h * aspect
            return (
                float(focal[0]) - half_w,
                float(focal[0]) + half_w,
                float(focal[1]) - half_h,
                float(focal[1]) + half_h,
            )
        except Exception as exc:
            self.debug_message.emit(f"_current_pick_rect_xy failed: {exc!r}")
            return None

    def _grid_sample_indices(self, xy: np.ndarray, target_count: int) -> np.ndarray:
        count = len(xy)
        if count == 0:
            return np.empty((0,), dtype=np.int32)
        if count <= target_count:
            return np.arange(count, dtype=np.int32)

        x_min = float(np.min(xy[:, 0]))
        x_max = float(np.max(xy[:, 0]))
        y_min = float(np.min(xy[:, 1]))
        y_max = float(np.max(xy[:, 1]))
        width = max(x_max - x_min, 1e-6)
        height = max(y_max - y_min, 1e-6)
        area = max(width * height, 1e-6)
        cell = max((area / float(target_count)) ** 0.5, 1e-6)

        gx = np.floor((xy[:, 0] - x_min) / cell).astype(np.int64)
        gy = np.floor((xy[:, 1] - y_min) / cell).astype(np.int64)
        cell_ids = gx * 73856093 + gy * 19349663

        order = np.argsort(cell_ids, kind="mergesort")
        sorted_ids = cell_ids[order]
        keep_mask = np.ones(len(order), dtype=bool)
        keep_mask[1:] = sorted_ids[1:] != sorted_ids[:-1]
        keep = np.sort(order[keep_mask].astype(np.int32))

        if keep.size > target_count:
            step = int(np.ceil(keep.size / target_count))
            keep = keep[::step]
        return keep

    def _view_area_ratio(self, rect: tuple[float, float, float, float]) -> float:
        if self._xy_local is None or len(self._xy_local) == 0:
            return 1.0
        full_width = max(float(np.max(self._xy_local[:, 0]) - np.min(self._xy_local[:, 0])), 1e-9)
        full_height = max(float(np.max(self._xy_local[:, 1]) - np.min(self._xy_local[:, 1])), 1e-9)
        rect_width = max(float(rect[1] - rect[0]), 1e-9)
        rect_height = max(float(rect[3] - rect[2]), 1e-9)
        return float((rect_width * rect_height) / (full_width * full_height))

    def set_view_log_enabled(self, enabled: bool) -> None:
        self._log_visible_refresh = bool(enabled)
        self.debug_message.emit(f"View log {'enabled' if enabled else 'disabled'}")
