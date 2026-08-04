import chess
from analysis.control import build_control_map

def build_heatmap(board: chess.Board) ->list[list[int]]:
    """
    Връща матрица 8x8

    Положително число:
    повече бели фигури атакуват полето.

    Отриц. число:
    повече черни фигури атакуват полето.

    Нула:
    равен контрол или липса на контрол
    """

    control_map = build_control_map(board)

    values_by_square = {
        item.square: item.difference
        for item in control_map
    }

    matrix: list[list[int]] = []

    # започваме от 8ред за да изглежка като шахматна дъска

    for rank in range(8, 0, -1):
        row: list[int] = []

        for file_name in "abcdefgh":
            square_name = f"{file_name}{rank}"
            row.append(values_by_square[square_name])

        matrix.append(row)

    return matrix

def print_heatmap(matrix: list[list[int]]) -> None:
    print("\nControl heatmap")
    print("     a   b   c   d   e   f   g   h")

    for index, row in enumerate(matrix):
        rank = 8 - index
        formatted_row = " ".join(f"{value:>3}" for value in row)
        print(f"{rank}  {formatted_row}")