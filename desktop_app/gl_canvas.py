from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from OpenGL import GL
from OpenGL.GL import shaders as gl_shaders
from PySide6.QtOpenGLWidgets import QOpenGLWidget

GRID_SIZE = 8
VERTICES_PER_QUAD = 6
VERTEX_COUNT = GRID_SIZE * GRID_SIZE * VERTICES_PER_QUAD

# Part 9's neutral slate board colors (docs/interactive_ui.md) -- deliberately
# not the wood tones visualization/board_plot.py uses for its static plots.
# Only chrome; not new colors.
BOARD_LIGHT_SQUARE = (0.796, 0.835, 0.882, 1.0)  # #cbd5e1
BOARD_DARK_SQUARE = (0.278, 0.333, 0.412, 1.0)  # #475569
CHROME_BACKGROUND = (0.059, 0.090, 0.165, 1.0)  # #0f172a

_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec2 in_position;
layout(location = 1) in vec4 in_color;
out vec4 vertex_color;
void main() {
    gl_Position = vec4(in_position, 0.0, 1.0);
    vertex_color = in_color;
}
"""

_FRAGMENT_SHADER = """
#version 330 core
in vec4 vertex_color;
out vec4 fragment_color;
uniform float u_opacity;
void main() {
    fragment_color = vec4(vertex_color.rgb, vertex_color.a * u_opacity);
}
"""


def build_grid_positions() -> np.ndarray:
    """
    Static 8x8 grid of quads (2 triangles / 6 vertices each) in normalized
    device coordinates, row-major with row 0 = rank 8 at the top of the canvas
    and column 0 = file a at the left -- matching analysis/geometry.py's
    matrix convention exactly, so a per-cell color array in row-major
    (row, column) order aligns index-for-index with these vertices.
    """
    step = 2.0 / GRID_SIZE
    positions = np.empty((VERTEX_COUNT, 2), dtype=np.float32)
    index = 0
    for row in range(GRID_SIZE):
        y_top = 1.0 - row * step
        y_bottom = y_top - step
        for column in range(GRID_SIZE):
            x_left = -1.0 + column * step
            x_right = x_left + step
            quad = (
                (x_left, y_bottom),
                (x_right, y_bottom),
                (x_right, y_top),
                (x_left, y_bottom),
                (x_right, y_top),
                (x_left, y_top),
            )
            for vertex in quad:
                positions[index] = vertex
                index += 1
    return positions


def build_checkerboard_colors() -> np.ndarray:
    """Static board-square chrome -- always on, underneath any layer overlay."""
    colors = np.empty((VERTEX_COUNT, 4), dtype=np.float32)
    index = 0
    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            is_light = (row + column) % 2 == 0
            color = BOARD_LIGHT_SQUARE if is_light else BOARD_DARK_SQUARE
            colors[index : index + VERTICES_PER_QUAD] = color
            index += VERTICES_PER_QUAD
    return colors


@dataclass(frozen=True)
class LayerGeometry:
    """
    One drawable piece of geometry for a Phase 5c mathematical layer.

    `positions`/`colors` are plain, equal-length vertex arrays (NDC space,
    same convention `build_grid_positions` already uses) -- not tied to the
    board's fixed `VERTEX_COUNT`, since each layer's vertex count genuinely
    differs by layer and by position (a critical-points layer might emit 6
    points on one position and 2 on the next). `primitive` is a plain GL
    enum (`GL.GL_POINTS`/`GL.GL_LINES`/`GL.GL_TRIANGLES`) -- the shader is
    already primitive-agnostic (position in, color in, color out), so no new
    shader is needed for any of these.

    A layer's `renderer` may return more than one `LayerGeometry` -- e.g. a
    Morse-Smale cell layer needs a filled-triangle interior and a separate
    line boundary, two different primitives for one layer.
    """

    positions: np.ndarray  # (N, 2) float32, NDC space
    colors: np.ndarray  # (N, 4) float32
    primitive: int  # GL.GL_POINTS | GL.GL_LINES | GL.GL_TRIANGLES


class MathCanvas(QOpenGLWidget):
    """
    The mathematical canvas (docs/interactive_ui.md Part 2, Part 5).

    Phase 5a/5b's board-square chrome + Attack Influence overlay path
    (`set_overlay_colors`, the fixed `VERTEX_COUNT` 8x8 quad buffer) is
    unchanged below. Phase 5c adds a second, generic mechanism alongside it
    for the five new layers, whose geometry (points, lines, arbitrary
    polygons) doesn't fit that fixed 8x8-quad shape: `set_layer_order`,
    `set_layer_geometry`, `set_layer_enabled`. Both mechanisms share the same
    program/VAO/shader -- no per-primitive-type shader is needed.

    No animation, no camera yet (Phase 5d/5i).
    """

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._program = None
        self._vao = None
        self._position_vbo = None
        self._board_color_vbo = None
        self._overlay_color_vbo = None
        self._overlay_colors: np.ndarray | None = None
        self._opacity_uniform_location = None
        # Attack Influence's visibility/opacity (SessionState.layer_state,
        # keyed "attack_influence") is applied here, not through
        # _layer_enabled/_layer_opacity below -- it draws via the fixed
        # 8x8-quad overlay path (set_overlay_colors), never through
        # _layer_gpu_buffers, so it needs its own flag/value.
        self._overlay_enabled: bool = True
        self._overlay_opacity: float = 1.0

        # Phase 5c: fixed draw order (registration order), per-layer stored
        # geometry (CPU-side, always kept even while disabled), per-layer
        # enabled flag, and the uploaded GPU buffers per layer.
        self._layer_order: list[str] = []
        self._layer_geometries: dict[str, list[LayerGeometry]] = {}
        self._layer_enabled: dict[str, bool] = {}
        self._layer_opacity: dict[str, float] = {}
        self._layer_gpu_buffers: dict[str, list[tuple[int, int, int, int]]] = {}
        # Each tuple: (position_vbo, color_vbo, vertex_count, primitive).

    def set_overlay_colors(self, colors: np.ndarray) -> None:
        """`colors` must be shaped (VERTEX_COUNT, 4), matching `build_grid_positions`'s vertex order."""
        colors = np.asarray(colors, dtype=np.float32)
        if colors.shape != (VERTEX_COUNT, 4):
            raise ValueError(f"expected shape ({VERTEX_COUNT}, 4), got {colors.shape}")
        self._overlay_colors = colors
        if self._vao is not None:
            self._upload_overlay_colors()
        self.update()

    def set_layer_order(self, layer_ids: list[str]) -> None:
        """
        The fixed, deterministic draw sequence -- set once at startup from
        `LayerRegistry.all()`'s registration order, never reordered per
        frame. Board chrome + Attack Influence always draw first (via
        `set_overlay_colors`, above), then these layers in this order.
        """
        self._layer_order = list(layer_ids)
        self.update()

    def set_layer_enabled(self, layer_id: str, enabled: bool) -> None:
        """
        Toggling this never touches `_layer_geometries` or re-uploads
        anything -- only which already-uploaded GPU buffers get drawn this
        frame. This is what makes a visibility change never trigger
        recomputation: nothing here calls back into the cache or a layer's
        `data_source`/`renderer`.
        """
        self._layer_enabled[layer_id] = enabled
        self.update()

    def set_layer_opacity(self, layer_id: str, opacity: float) -> None:
        """
        Like `set_layer_enabled`, this never touches `_layer_geometries` or
        re-uploads anything -- it only changes the `u_opacity` value the
        fragment shader multiplies into each vertex's alpha on the next
        frame's draw call for this layer's already-uploaded buffers.
        """
        self._layer_opacity[layer_id] = opacity
        self.update()

    def set_overlay_enabled(self, enabled: bool) -> None:
        """Visibility for the Attack Influence overlay -- see `_overlay_enabled`'s docstring in __init__."""
        self._overlay_enabled = enabled
        self.update()

    def set_overlay_opacity(self, opacity: float) -> None:
        """Opacity for the Attack Influence overlay -- see `_overlay_enabled`'s docstring in __init__."""
        self._overlay_opacity = opacity
        self.update()

    def set_layer_geometry(self, layer_id: str, geometries: list[LayerGeometry]) -> None:
        self._layer_geometries[layer_id] = geometries
        if self._vao is not None:
            self._upload_layer_geometry(layer_id)
        self.update()

    def initializeGL(self) -> None:
        vertex_shader = gl_shaders.compileShader(_VERTEX_SHADER, GL.GL_VERTEX_SHADER)
        fragment_shader = gl_shaders.compileShader(_FRAGMENT_SHADER, GL.GL_FRAGMENT_SHADER)
        self._program = gl_shaders.compileProgram(vertex_shader, fragment_shader)
        self._opacity_uniform_location = GL.glGetUniformLocation(self._program, "u_opacity")

        self._vao = GL.glGenVertexArrays(1)
        GL.glBindVertexArray(self._vao)

        self._position_vbo = self._upload_static_buffer(build_grid_positions(), location=0, components=2)
        self._board_color_vbo = self._upload_static_buffer(build_checkerboard_colors(), location=1, components=4)

        GL.glBindVertexArray(0)

        GL.glClearColor(*CHROME_BACKGROUND)
        GL.glEnable(GL.GL_BLEND)
        # Separate RGB and alpha blend equations -- discovered necessary
        # during Phase 5c's real-position visual smoke test, not present
        # since Phase 5a: QOpenGLWidget composites its *own* framebuffer's
        # alpha channel against whatever is behind the widget in the Qt
        # window. A plain glBlendFunc(SRC_ALPHA, ONE_MINUS_SRC_ALPHA) lets
        # every translucent draw (Attack Influence's own alpha=0.72 overlay
        # already did this even before Phase 5c) leave the framebuffer's
        # alpha below 1.0, so Qt then blends the whole widget against
        # whatever is behind it -- producing unrelated colors with no
        # relationship to the intended RGB values (verified directly: the
        # exact same RGB values render correctly at alpha=1.0 and
        # incorrectly at alpha=0.72). RGB blending is unchanged (still the
        # standard "over" formula); the alpha channel separately
        # accumulates toward 1.0 as soon as anything opaque is drawn
        # beneath it (true from the very first, always-opaque board-chrome
        # draw), so the framebuffer Qt sees is always fully opaque
        # regardless of how translucent any individual layer is.
        GL.glBlendFuncSeparate(
            GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA,
            GL.GL_ONE, GL.GL_ONE_MINUS_SRC_ALPHA,
        )

        if self._overlay_colors is not None:
            self._upload_overlay_colors()

        for layer_id in self._layer_geometries:
            self._upload_layer_geometry(layer_id)

    def resizeGL(self, width: int, height: int) -> None:
        GL.glViewport(0, 0, max(width, 1), max(height, 1))

    def paintGL(self) -> None:
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        GL.glUseProgram(self._program)
        GL.glBindVertexArray(self._vao)

        # Board chrome is always fully opaque, regardless of any layer's
        # opacity setting -- reset the uniform explicitly every frame rather
        # than relying on whatever the previous frame's last draw left it at.
        GL.glUniform1f(self._opacity_uniform_location, 1.0)

        # Location 0 (position) is re-bound explicitly here, every frame,
        # rather than relying on whatever it was last left pointing to --
        # the per-layer draws below rebind location 0 to each layer's own
        # position buffer, so the board/overlay draws can no longer assume
        # it's still pointing at the static grid from initializeGL.
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._position_vbo)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._board_color_vbo)
        GL.glVertexAttribPointer(1, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, VERTEX_COUNT)

        # `_overlay_enabled` gates this draw -- previously missing, so
        # toggling Attack Influence's visibility off had no visual effect
        # even though SessionState.layer_visible("attack_influence")
        # correctly reported False (found and fixed while wiring the
        # layer-strip UI to this layer).
        if self._overlay_color_vbo is not None and self._overlay_enabled:
            GL.glUniform1f(self._opacity_uniform_location, self._overlay_opacity)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._overlay_color_vbo)
            GL.glVertexAttribPointer(1, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
            GL.glDrawArrays(GL.GL_TRIANGLES, 0, VERTEX_COUNT)

        for layer_id in self._layer_order:
            if not self._layer_enabled.get(layer_id, True):
                continue
            GL.glUniform1f(self._opacity_uniform_location, self._layer_opacity.get(layer_id, 1.0))
            for position_vbo, color_vbo, vertex_count, primitive in self._layer_gpu_buffers.get(layer_id, []):
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, position_vbo)
                GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
                GL.glBindBuffer(GL.GL_ARRAY_BUFFER, color_vbo)
                GL.glVertexAttribPointer(1, 4, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
                GL.glDrawArrays(primitive, 0, vertex_count)

        GL.glBindVertexArray(0)

    def _upload_overlay_colors(self) -> None:
        if self._overlay_color_vbo is None:
            self._overlay_color_vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._overlay_color_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, self._overlay_colors.nbytes, self._overlay_colors, GL.GL_DYNAMIC_DRAW)

    def _upload_layer_geometry(self, layer_id: str) -> None:
        for position_vbo, color_vbo, _vertex_count, _primitive in self._layer_gpu_buffers.get(layer_id, []):
            GL.glDeleteBuffers(1, [position_vbo])
            GL.glDeleteBuffers(1, [color_vbo])

        new_buffers: list[tuple[int, int, int, int]] = []
        for geometry in self._layer_geometries.get(layer_id, []):
            positions = np.asarray(geometry.positions, dtype=np.float32)
            colors = np.asarray(geometry.colors, dtype=np.float32)
            vertex_count = positions.shape[0]
            if vertex_count == 0:
                continue

            position_vbo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, position_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, positions.nbytes, positions, GL.GL_DYNAMIC_DRAW)

            color_vbo = GL.glGenBuffers(1)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, color_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, colors.nbytes, colors, GL.GL_DYNAMIC_DRAW)

            new_buffers.append((position_vbo, color_vbo, vertex_count, geometry.primitive))

        self._layer_gpu_buffers[layer_id] = new_buffers

    def _upload_static_buffer(self, data: np.ndarray, location: int, components: int) -> int:
        vbo = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, data.nbytes, data, GL.GL_STATIC_DRAW)
        GL.glVertexAttribPointer(location, components, GL.GL_FLOAT, GL.GL_FALSE, 0, None)
        GL.glEnableVertexAttribArray(location)
        return vbo
