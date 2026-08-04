import chess
import matplotlib.pyplot as plt


UNICODE_PIECES = {
    "P": "♙",
    "N": "♘",
    "B": "♗",
    "R": "♖",
    "Q": "♕",
    "K": "♔",

    "p": "♟",
    "n": "♞",
    "b": "♝",
    "r": "♜",
    "q": "♛",
    "k": "♚",
}


def plot_board(board: chess.Board):

    fig, ax = plt.subplots(figsize=(8, 8))

    # квадратите
    for rank in range(8):
        for file in range(8):

            color = "#f0d9b5" if (rank + file) % 2 == 0 else "#b58863"

            ax.add_patch(
                plt.Rectangle(
                    (file, rank),
                    1,
                    1,
                    color=color,
                )
            )

    # фигурите
    for square, piece in board.piece_map().items():

        x = chess.square_file(square)
        y = 7 - chess.square_rank(square)

        symbol = UNICODE_PIECES[piece.symbol()]

        ax.text(
            x + 0.5,
            y + 0.5,
            symbol,
            fontsize=30,
            ha="center",
            va="center",
        )

    ax.set_xlim(0, 8)
    ax.set_ylim(8, 0)

    ax.set_xticks(range(8))
    ax.set_xticklabels(list("abcdefgh"))

    ax.set_yticks(range(8))
    ax.set_yticklabels(range(8, 0, -1))

    ax.set_aspect("equal")

    plt.show()