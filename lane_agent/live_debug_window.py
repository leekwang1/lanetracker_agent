from __future__ import annotations

import csv
from pathlib import Path

from PySide6 import QtCore, QtGui, QtWidgets

try:
    import pyqtgraph as pg
except Exception:  # pragma: no cover - runtime fallback when dependency is unavailable
    pg = None

from _V2.ui.pointcloud_view_widget import PointCloudViewWidget

from .live_debug_controller import LiveDebugController


class CrossSectionProfileWidget(QtWidgets.QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        if pg is None:
            self._label = QtWidgets.QLabel("pyqtgraph is not installed.\n단면 프로파일 그래프를 표시할 수 없습니다.")
            self._label.setAlignment(QtCore.Qt.AlignCenter)
            self._label.setWordWrap(True)
            layout.addWidget(self._label)
            self.plot = None
            self.curve_raw = None
            self.curve_smooth = None
            self.vline_left = None
            self.vline_right = None
            self.vline_center = None
            return

        self.plot = pg.PlotWidget()
        self.plot.showGrid(x=True, y=True, alpha=0.2)
        self.plot.setLabel("left", "가중 히스토그램")
        self.plot.setLabel("bottom", "횡방향 오프셋 (m)")
        self.curve_raw = self.plot.plot(pen=pg.mkPen("#94a3b8", width=1))
        self.curve_smooth = self.plot.plot(pen=pg.mkPen("#f97316", width=2))
        self.vline_left = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ef4444", width=1))
        self.vline_right = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#ef4444", width=1))
        self.vline_center = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen("#22c55e", width=1.5))
        self.plot.addItem(self.vline_left)
        self.plot.addItem(self.vline_right)
        self.plot.addItem(self.vline_center)
        layout.addWidget(self.plot)

    def update_profile(self, profile) -> None:
        if self.plot is None or self.curve_raw is None or self.curve_smooth is None:
            if hasattr(self, "_label"):
                self._label.setText(
                    "pyqtgraph is not installed.\n"
                    "단면 프로파일 데이터를 텍스트로만 표시합니다."
                )
            return
        if profile is None or profile.bins_center.size == 0:
            self.curve_raw.setData([], [])
            self.curve_smooth.setData([], [])
            self.vline_left.hide()
            self.vline_right.hide()
            self.vline_center.hide()
            return
        self.curve_raw.setData(profile.bins_center, profile.hist_combined)
        self.curve_smooth.setData(profile.bins_center, profile.smooth_hist)
        if profile.selected_idx is not None and profile.stripe_candidates:
            stripe = profile.stripe_candidates[profile.selected_idx]
            self.vline_left.setValue(float(stripe.left_m))
            self.vline_right.setValue(float(stripe.right_m))
            self.vline_center.setValue(float(stripe.center_m))
            self.vline_left.show()
            self.vline_right.show()
            self.vline_center.show()
        else:
            self.vline_left.hide()
            self.vline_right.hide()
            self.vline_center.hide()


class CandidateTableWidget(QtWidgets.QTableWidget):
    HEADERS = ["Rank", "Score", "X", "Y", "Z", "Mean I", "High I", "Loyalty", "Reject"]

    def __init__(self, parent=None) -> None:
        super().__init__(0, len(self.HEADERS), parent)
        self.setHorizontalHeaderLabels(self.HEADERS)
        self.verticalHeader().setVisible(False)
        self.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.setAlternatingRowColors(True)
        self.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

    def update_rows(self, rows: list[dict]) -> None:
        self.setRowCount(len(rows))
        for row_idx, row in enumerate(rows):
            values = [
                str(row_idx + 1),
                f"{float(row.get('score', -1.0)):.3f}",
                f"{float(row.get('x', 0.0)):.3f}",
                f"{float(row.get('y', 0.0)):.3f}",
                f"{float(row.get('z', 0.0)):.3f}",
                f"{float(row.get('mean_intensity', 0.0)):.1f}",
                f"{float(row.get('high_intensity', 0.0)):.1f}",
                f"{float(row.get('lane_loyalty', 0.0)):.3f}",
                str(row.get("rejected", "")),
            ]
            for col_idx, value in enumerate(values):
                item = QtWidgets.QTableWidgetItem(value)
                if row.get("rejected"):
                    item.setForeground(QtGui.QBrush(QtGui.QColor("#ef4444")))
                elif row_idx == 0:
                    item.setForeground(QtGui.QBrush(QtGui.QColor("#22c55e")))
                self.setItem(row_idx, col_idx, item)


class MainWindow(QtWidgets.QMainWindow):
    def __init__(
        self,
        las_path: str | None = None,
        config_path: str | None = None,
        output_path: str | None = None,
        p0: list[float] | tuple[float, ...] | None = None,
        p1: list[float] | tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        self.setWindowTitle("Lane Agent Live Debugger")
        self.controller = LiveDebugController()
        self.view = PointCloudViewWidget()
        self.profile_plot = CrossSectionProfileWidget()
        self.candidate_table = CandidateTableWidget()
        self.log = QtWidgets.QPlainTextEdit()
        self.log.setReadOnly(True)
        self.view_log_check = QtWidgets.QCheckBox("View Log")
        self.view_log_check.setChecked(False)
        self.status = QtWidgets.QLabel("Ready")
        self.status.setWordWrap(True)
        self.las_edit = QtWidgets.QLineEdit()
        self.cfg_edit = QtWidgets.QLineEdit()
        self.output_edit = QtWidgets.QLineEdit()
        self.p0_edit = QtWidgets.QLineEdit("0 0 0")
        self.p1_edit = QtWidgets.QLineEdit("1 0 0")
        self.legend = QtWidgets.QLabel(
            "\n".join(
                [
                    "[메인 화면]",
                    "P0: 초록색, P1: 파란색",
                    "수락된 경로: 하늘색 선",
                    "현재 중심: 빨간 점",
                    "후보들: 주황 점",
                    "탐색 박스: 보라색 상자",
                    "",
                    "[디버그 패널]",
                    "단면 프로파일: 현재 중심 기준 횡단 히스토그램",
                    "빨간 선: stripe 좌우 경계, 초록 선: 중심",
                    "후보 테이블: 현재 스텝 후보 목록",
                    "로그: 스텝별 요약 정보",
                ]
            )
        )
        self.legend.setWordWrap(True)
        self.legend.setStyleSheet(
            "QLabel {"
            "background: #111827;"
            "color: #e5e7eb;"
            "border: 1px solid #374151;"
            "border-radius: 6px;"
            "padding: 8px;"
            "}"
        )

        btn_load = QtWidgets.QPushButton("Load LAS")
        btn_init = QtWidgets.QPushButton("Initialize")
        btn_step = QtWidgets.QPushButton("Run One Step")
        btn_full = QtWidgets.QPushButton("Run Full")
        btn_save = QtWidgets.QPushButton("Save Result")
        btn_reset = QtWidgets.QPushButton("Reset")
        btn_las_browse = QtWidgets.QPushButton("Browse...")
        btn_cfg_browse = QtWidgets.QPushButton("Browse...")
        btn_output_browse = QtWidgets.QPushButton("Browse...")

        form = QtWidgets.QFormLayout()
        las_row = QtWidgets.QHBoxLayout()
        las_row.addWidget(self.las_edit)
        las_row.addWidget(btn_las_browse)
        cfg_row = QtWidgets.QHBoxLayout()
        cfg_row.addWidget(self.cfg_edit)
        cfg_row.addWidget(btn_cfg_browse)
        output_row = QtWidgets.QHBoxLayout()
        output_row.addWidget(self.output_edit)
        output_row.addWidget(btn_output_browse)
        form.addRow("LAS", las_row)
        form.addRow("Config", cfg_row)
        form.addRow("Output", output_row)
        form.addRow("P0", self.p0_edit)
        form.addRow("P1", self.p1_edit)

        btns = QtWidgets.QHBoxLayout()
        for widget in [btn_load, btn_init, btn_step, btn_full, btn_save, btn_reset]:
            btns.addWidget(widget)

        side = QtWidgets.QVBoxLayout()
        side.addLayout(form)
        side.addLayout(btns)
        side.addWidget(self.status)
        side.addWidget(self.legend)
        side.addWidget(QtWidgets.QLabel("단면 프로파일"))
        side.addWidget(self.profile_plot)
        side.addWidget(QtWidgets.QLabel("현재 스텝 후보"))
        side.addWidget(self.candidate_table)
        side.addWidget(self.log)
        log_opts = QtWidgets.QHBoxLayout()
        log_opts.addStretch(1)
        log_opts.addWidget(self.view_log_check)
        side.addLayout(log_opts)

        right = QtWidgets.QWidget()
        right.setLayout(side)

        splitter = QtWidgets.QSplitter()
        splitter.addWidget(self.view)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([960, 640])
        self.setCentralWidget(splitter)

        self._set_default_paths(las_path, config_path, output_path)
        if p0 is not None:
            self.p0_edit.setText(" ".join(f"{float(v):.10f}" for v in p0))
        if p1 is not None:
            self.p1_edit.setText(" ".join(f"{float(v):.10f}" for v in p1))

        btn_las_browse.clicked.connect(self.on_browse_las)
        btn_cfg_browse.clicked.connect(self.on_browse_config)
        btn_output_browse.clicked.connect(self.on_browse_output)
        btn_load.clicked.connect(self.on_load)
        btn_init.clicked.connect(self.on_init)
        btn_step.clicked.connect(self.on_step)
        btn_full.clicked.connect(self.on_full)
        btn_save.clicked.connect(self.on_save)
        btn_reset.clicked.connect(self.controller.reset)
        self.controller.changed.connect(self.refresh)
        self.controller.log_message.connect(self.log.appendPlainText)
        self.view.point_context_menu_requested.connect(self._open_point_context_menu)
        self.view.debug_message.connect(lambda msg: self.log.appendPlainText(f"VIEW: {msg}"))
        self.view_log_check.toggled.connect(self.view.set_view_log_enabled)

    def _set_default_paths(
        self,
        las_path: str | None,
        config_path: str | None,
        output_path: str | None,
    ) -> None:
        root = Path(__file__).resolve().parents[1]
        if las_path:
            self.las_edit.setText(las_path)
        if config_path:
            self.cfg_edit.setText(config_path)
        else:
            solid_cfg = root / "config_solid.yaml"
            default_cfg = solid_cfg if solid_cfg.exists() else (root / "config.yaml")
            if default_cfg.exists():
                self.cfg_edit.setText(str(default_cfg))
        if output_path:
            self.output_edit.setText(output_path)
        elif las_path:
            self.output_edit.setText(str(Path(las_path).with_suffix(".live.csv")))

    def on_browse_las(self) -> None:
        start_dir = str(Path(self.las_edit.text()).parent) if self.las_edit.text().strip() else str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select LAS", start_dir, "LAS Files (*.las);;All Files (*)")
        if path:
            self.las_edit.setText(path)
            if not self.output_edit.text().strip():
                self.output_edit.setText(str(Path(path).with_suffix(".live.csv")))

    def on_browse_config(self) -> None:
        start_dir = str(Path(self.cfg_edit.text()).parent) if self.cfg_edit.text().strip() else str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Config", start_dir, "YAML Files (*.yaml *.yml);;All Files (*)")
        if path:
            self.cfg_edit.setText(path)

    def on_browse_output(self) -> None:
        start_dir = str(Path(self.output_edit.text()).parent) if self.output_edit.text().strip() else str(Path.cwd())
        path, _ = QtWidgets.QFileDialog.getSaveFileName(self, "Select Output CSV", start_dir, "CSV Files (*.csv);;All Files (*)")
        if path:
            self.output_edit.setText(path)

    def on_load(self) -> None:
        las_path = self.las_edit.text().strip()
        if not las_path:
            self._show_error("LAS path is empty.")
            return
        if not Path(las_path).exists():
            self._show_error(f"LAS file not found:\n{las_path}")
            return
        if self._run_action("Loading LAS...", lambda: self.controller.load_las(las_path)):
            if not self.output_edit.text().strip():
                self.output_edit.setText(str(Path(las_path).with_suffix(".live.csv")))
            self._populate_seed_points_from_csv(Path(las_path))

    def on_init(self) -> None:
        cfg_path = self.cfg_edit.text().strip()
        if not cfg_path:
            self._show_error("Config path is empty.")
            return
        if not Path(cfg_path).exists():
            self._show_error(f"Config file not found:\n{cfg_path}")
            return
        try:
            p0 = self._parse_xyz_text(self.p0_edit.text())
            p1 = self._parse_xyz_text(self.p1_edit.text())
        except ValueError as exc:
            self._show_error(f"Invalid P0/P1 format.\nUse: x y z\n\n{exc}")
            return
        if len(p0) != 3 or len(p1) != 3:
            self._show_error("P0 and P1 must each contain exactly 3 numbers.")
            return

        def init_action() -> None:
            self.controller.load_tracker_config(cfg_path)
            self.controller.set_p0(*p0)
            self.controller.set_p1(*p1)
            self.controller.set_output_path(self.output_edit.text().strip())
            self.controller.initialize_tracker()

        if self._run_action("Initializing tracker...", init_action):
            self.view.focus_on_point(self.controller.model.current_point)

    def on_step(self) -> None:
        if self._run_action("Running one step...", self.controller.run_step):
            self.view.focus_on_point(self.controller.model.current_point)

    def on_full(self) -> None:
        if self._run_action("Running full tracker...", self.controller.run_full):
            self.view.focus_on_point(self.controller.model.current_point)

    def on_save(self) -> None:
        self.controller.set_output_path(self.output_edit.text().strip())
        self._run_action("Saving outputs...", lambda: self.controller.save_outputs(self.output_edit.text().strip()))

    def _parse_xyz_text(self, text: str) -> list[float]:
        parts = text.replace(",", " ").split()
        return [float(x) for x in parts]

    def _populate_seed_points_from_csv(self, las_path: Path) -> None:
        csv_path = las_path.with_suffix(".csv")
        if not csv_path.exists():
            return
        try:
            with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                rows = []
                for row in reader:
                    rows.append(row)
                    if len(rows) >= 2:
                        break
            if len(rows) < 2:
                return
            p0 = [float(rows[0][k]) for k in ("x", "y", "z")]
            p1 = [float(rows[1][k]) for k in ("x", "y", "z")]
        except Exception as exc:  # pragma: no cover - best effort helper
            self.log.appendPlainText(f"WARN: failed to read seed CSV {csv_path}: {exc}")
            return
        self.p0_edit.setText(" ".join(f"{v:.6f}" for v in p0))
        self.p1_edit.setText(" ".join(f"{v:.6f}" for v in p1))
        self.log.appendPlainText(f"Loaded seed points from {csv_path.name}")

    def _run_action(self, status_text: str, fn) -> bool:
        self.status.setText(status_text)
        QtWidgets.QApplication.setOverrideCursor(QtGui.QCursor(QtCore.Qt.WaitCursor))
        try:
            QtWidgets.QApplication.processEvents()
            fn()
            return True
        except Exception as exc:
            self._show_error(str(exc))
            return False
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _show_error(self, message: str) -> None:
        self.log.appendPlainText(f"ERROR: {message}")
        self.status.setText("Error")
        QtWidgets.QMessageBox.critical(self, "Lane Agent Live Debugger", message)

    def _open_point_context_menu(self, point_xyz, global_pos) -> None:
        menu = QtWidgets.QMenu(self)
        if point_xyz is not None:
            xyz_text = ", ".join(f"{float(v):.6f}" for v in point_xyz)
            title = QtGui.QAction(f"Point: {xyz_text}", menu)
            title.setEnabled(False)
            menu.addAction(title)
            menu.addSeparator()
            act_set_p0 = menu.addAction("Set P0 Here")
            act_set_p1 = menu.addAction("Set P1 Here")
            act_copy_xyz = menu.addAction("Copy XYZ")
            menu.addSeparator()
            act_init = menu.addAction("Initialize From P0 / P1")
        else:
            disabled = QtGui.QAction("No nearby point", menu)
            disabled.setEnabled(False)
            menu.addAction(disabled)
            menu.addSeparator()
            act_set_p0 = None
            act_set_p1 = None
            act_copy_xyz = None
            act_init = menu.addAction("Initialize From P0 / P1")

        menu.addSeparator()
        act_reset_view = menu.addAction("Reset Camera / View")
        chosen = menu.exec(global_pos)
        if chosen is None:
            return
        if chosen == act_set_p0 and point_xyz is not None:
            self._apply_seed_point(self.p0_edit, point_xyz, "P0")
        elif chosen == act_set_p1 and point_xyz is not None:
            self._apply_seed_point(self.p1_edit, point_xyz, "P1")
        elif chosen == act_copy_xyz and point_xyz is not None:
            QtWidgets.QApplication.clipboard().setText(" ".join(f"{float(v):.10f}" for v in point_xyz))
            self.status.setText("Copied point XYZ")
        elif chosen == act_init:
            self.on_init()
        elif chosen == act_reset_view:
            self.view.reset_view()

    def _apply_seed_point(self, target: QtWidgets.QLineEdit, point_xyz, label: str) -> None:
        vals = [float(v) for v in point_xyz]
        target.setText(" ".join(f"{v:.10f}" for v in vals))
        if label == "P0":
            self.controller.set_p0(*vals)
        elif label == "P1":
            self.controller.set_p1(*vals)
        self.status.setText(f"{label} updated from point cloud")
        self.log.appendPlainText(f"{label} <- {vals[0]:.6f}, {vals[1]:.6f}, {vals[2]:.6f}")

    def refresh(self) -> None:
        model = self.controller.model
        if model.xyz is not None and model.intensity is not None:
            self.view.set_point_cloud(model.xyz, model.intensity, revision=model.point_cloud_revision)
        self.view.set_seed_points(model.p0, model.p1, render=False)
        self.view.set_track(model.track_points, render=False)
        self.view.set_current(model.current_point, render=False)
        self.view.set_predicted(model.predicted_points, render=False)
        self.view.set_profile_overlay(None, None, None, render=False)
        self.view.set_search_box(model.search_box_points, render=False)
        self.view.render()
        self.profile_plot.update_profile(model.cross_section_profile)
        self.candidate_table.update_rows(model.candidate_rows)
        self.status.setText(model.status_text or "Ready")
