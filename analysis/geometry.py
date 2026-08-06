import chess

"""
Единна конвенция за координати, споделена от analysis/ и visualization/.

mathematical (x, y):
    x = файл, 0..7 -> a..h.
    y = ранг, 0..7 -> rank 1..rank 8 (расте нагоре).

matrix (row, column):
    row 0 = rank 8, row 7 = rank 1 (за да изглежда като шахматна дъска
    при отпечатване).
    column 0..7 = файл a..h, същото като mathematical x.

plot/screen (x, y):
    Използвана от Matplotlib (imshow extent, scatter, text) с
    axes.set_ylim(8, 0) — top-down, т.е. същата ориентация като matrix
    row/column: plot_x = mathematical x, plot_y = 7 - mathematical y.
"""


def square_to_math_coords(square: chess.Square) -> tuple[int, int]:
    return chess.square_file(square), chess.square_rank(square)


def math_coords_to_square(x: int, y: int) -> chess.Square:
    return chess.square(x, y)


def math_to_plot_coords(x: float, y: float) -> tuple[float, float]:
    return x, 7 - y


def plot_to_math_coords(x: float, y: float) -> tuple[float, float]:
    return x, 7 - y


def square_to_plot_coords(square: chess.Square) -> tuple[float, float]:
    x, y = square_to_math_coords(square)
    return math_to_plot_coords(x, y)


def plot_coords_to_square(x: float, y: float) -> chess.Square:
    math_x, math_y = plot_to_math_coords(x, y)
    return math_coords_to_square(int(math_x), int(math_y))


def matrix_rc_to_math_coords(row: int, column: int) -> tuple[int, int]:
    return column, 7 - row


def math_coords_to_matrix_rc(x: int, y: int) -> tuple[int, int]:
    return 7 - y, x


def square_to_matrix_rc(square: chess.Square) -> tuple[int, int]:
    x, y = square_to_math_coords(square)
    return math_coords_to_matrix_rc(x, y)


def matrix_rc_to_square(row: int, column: int) -> chess.Square:
    x, y = matrix_rc_to_math_coords(row, column)
    return math_coords_to_square(x, y)


def matrix_row_to_rank(row: int) -> int:
    return 8 - row
