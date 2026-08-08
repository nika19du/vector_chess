from __future__ import annotations

import sys

from PySide6.QtGui import QSurfaceFormat
from PySide6.QtWidgets import QApplication

from desktop_app.main_window import MainWindow

CANVAS_SIZE_PX = 896  # 8x8 grid at 112px/square -- a reasonable default window size


def configure_gl_surface_format() -> None:
    """
    Requests an explicit OpenGL 3.3 core profile before any widget is
    created, so `#version 330 core` shaders (desktop_app/gl_canvas.py)
    compile deterministically rather than depending on whatever profile the
    platform's default context happens to provide.
    """
    surface_format = QSurfaceFormat()
    surface_format.setVersion(3, 3)
    surface_format.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    QSurfaceFormat.setDefaultFormat(surface_format)


def main() -> None:
    configure_gl_surface_format()
    application = QApplication(sys.argv)
    window = MainWindow()
    window.resize(CANVAS_SIZE_PX, CANVAS_SIZE_PX)
    window.show()
    sys.exit(application.exec())


if __name__ == "__main__":
    main()
