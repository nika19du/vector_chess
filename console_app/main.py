import chess


def main() -> None:
    board = chess.Board()

    print("VectorChess — терминална версия")
    print("Въвеждай ходове като: e2e4, e7e5, g1f3")
    print("За изход напиши: quit")
    print()
    print(board)

    while not board.is_game_over():
        user_input = input("\nХод: ").strip().lower()

        if user_input == "quit":
            print("Играта беше прекратена.")
            return

        try:
            move = chess.Move.from_uci(user_input)
        except ValueError:
            print("Невалиден формат. Пример за ход: e2e4")
            continue

        if move not in board.legal_moves:
            print("Този ход не е позволен в текущата позиция.")
            continue

        moved_piece = board.piece_at(move.from_square)

        board.push(move)

        if board.turn == chess.WHITE:
            print("Ред на белите")
        else:
            print("Ред на черните")

        print()
        print(board)

        if moved_piece is not None:
            print(
                f"Преместена фигура: "
                f"{chess.piece_name(moved_piece.piece_type)}"
            )

        print(
            f"От {chess.square_name(move.from_square)} "
            f"до {chess.square_name(move.to_square)}"
        )

    print("\nИграта приключи.")
    print(f"Резултат: {board.result()}")


if __name__ == "__main__":
    main()