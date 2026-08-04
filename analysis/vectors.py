import math
import chess

from analysis.center_of_mass import calculate_control_center
from chess_engine.models import FieldVector

def calculate_control_vector(board: chess.Board) -> FieldVector | None:
    # Геометричните центрове на контрола за двата цвята.
    white_center = calculate_control_center(board, chess.WHITE)
    black_center = calculate_control_center(board, chess.BLACK)

    if white_center is None or black_center is None:
        return None

    # Начална и крайна точка на вектора.
    start_x, start_y = white_center

    # Крайна точка на вектора.
    end_x, end_y = black_center

    """
    delta_x,y - получавам посоката на вектора между двете точки
    Векторът от центъра на белия контрол към центъра на черния контрол.
    """
    delta_x = end_x - start_x
    delta_y = end_y - start_y

    # Евклидово разстояние между двата центъра.Дължина (големина) на вектора
    magnitude = math.sqrt(delta_x ** 2 + delta_y ** 2)

    return FieldVector(
        start_x=start_x,
        start_y=start_y,
        end_x=end_x,
        end_y=end_y,
        delta_x=delta_x,
        delta_y=delta_y,
        magnitude=magnitude
    )