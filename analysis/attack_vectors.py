import chess

from chess_engine.models import AttackVector

def generate_attack_vectors(board: chess.Board)->list[AttackVector]:
    """
    Създава по един вектор за всяка атака на всяка фигурал

    Прим. фиг. на f3 атакува d4 => създава се вектор от f3 към d4
    """

    vectors: list[AttackVector] = []

    # обхожда всяка фигура на дъската
    for from_square, piece in board.piece_map().items():
        # взима всички полета, атакувани от тази фигура
        attacked_squares = board.attacks(from_square)

        start_x = chess.square_file(from_square)
        start_y = chess.square_rank(from_square)

        # прави отделен вектор към всяко атакувано поле
        for to_square in attacked_squares:
            end_x = chess.square_file(to_square)
            end_y = chess.square_rank(to_square)

            delta_x = end_x - start_x
            delta_y = end_y - start_y

            vector = AttackVector(
                piece_name = chess.piece_name(piece.piece_type),
                color=(
                    "white"
                    if piece.color == chess.WHITE
                    else "black"
                ),
                from_square = chess.square_name(from_square),
                to_square = chess.square_name(to_square),
                start_x = start_x,
                start_y = start_y,
                end_x = end_x,
                end_y = end_y,
                delta_x = delta_x,
                delta_y = delta_y
            )

            vectors.append(vector)

    return  vectors