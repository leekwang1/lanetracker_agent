from __future__ import annotations

import argparse
import sys

from PySide6 import QtWidgets

from .live_debug_window import MainWindow


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Lane Agent live-step debugger")
    parser.add_argument("--las", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--output", default="")
    parser.add_argument("--p0", nargs=3, type=float, metavar=("X", "Y", "Z"))
    parser.add_argument("--p1", nargs=3, type=float, metavar=("X", "Y", "Z"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow(
        las_path=args.las or None,
        config_path=args.config or None,
        output_path=args.output or None,
        p0=args.p0,
        p1=args.p1,
    )
    window.resize(1680, 940)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
