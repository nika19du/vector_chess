import math

from chess_engine.models import (
    GradientField,
    GradientVector,
)


def calculate_x_gradient(
    matrix: list[list[float]],
    row: int,
    column: int,
) -> float:
    """
    Изчислява промяната на потенциала по хоризонтала.

    Положителен резултат:
        потенциалът расте надясно.

    Отрицателен резултат:
        потенциалът расте наляво.
    """

    # Лява граница: няма клетка отляво.
    if column == 0:
        return (
            matrix[row][column + 1]
            - matrix[row][column]
        )

    # Дясна граница: няма клетка отдясно.
    if column == 7:
        return (
            matrix[row][column]
            - matrix[row][column - 1]
        )

    # Вътрешна клетка:
    # използваме централна крайна разлика.
    return (
        matrix[row][column + 1]
        - matrix[row][column - 1]
    ) / 2.0


def calculate_y_gradient(
    matrix: list[list[float]],
    row: int,
    column: int,
) -> float:
    """
    Изчислява промяната на потенциала по вертикала.

    Матрицата е подредена така:
        row 0 -> rank 8
        row 7 -> rank 1

    В математическата координатна система Y расте нагоре.
    Затова горната клетка е row - 1,
    а долната клетка е row + 1.

    Положителен резултат:
        потенциалът расте нагоре.

    Отрицателен резултат:
        потенциалът расте надолу.
    """

    # Горна граница, rank 8.
    if row == 0:
        return (
            matrix[row][column]
            - matrix[row + 1][column]
        )

    # Долна граница, rank 1.
    if row == 7:
        return (
            matrix[row - 1][column]
            - matrix[row][column]
        )

    # Централна крайна разлика:
    # потенциал отгоре минус потенциал отдолу.
    return (
        matrix[row - 1][column]
        - matrix[row + 1][column]
    ) / 2.0


def calculate_gradient_magnitude(
    delta_x: float,
    delta_y: float,
) -> float:
    """
    Изчислява големината на градиентния вектор:

        |∇F| = sqrt(dx² + dy²)
    """

    return math.sqrt(
        delta_x ** 2
        + delta_y ** 2
    )


def build_gradient_field(
    potential_matrix: list[list[float]],
) -> GradientField:
    """
    Изгражда градиентно поле от Potential Field.

    За всяка клетка създава вектор:

        ∇F = (dF/dx, dF/dy)

    Този вектор сочи към посоката,
    в която потенциалът на Белите спрямо
    Черните расте най-бързо.
    """

    if len(potential_matrix) != 8:
        raise ValueError(
            "Potential matrix must contain 8 rows."
        )

    if any(len(row) != 8 for row in potential_matrix):
        raise ValueError(
            "Each potential matrix row must contain 8 values."
        )

    vectors: list[GradientVector] = []

    x_matrix: list[list[float]] = []
    y_matrix: list[list[float]] = []
    magnitude_matrix: list[list[float]] = []

    max_magnitude = 0.0

    for row in range(8):
        x_row: list[float] = []
        y_row: list[float] = []
        magnitude_row: list[float] = []

        for column in range(8):
            delta_x = calculate_x_gradient(
                matrix=potential_matrix,
                row=row,
                column=column,
            )

            delta_y = calculate_y_gradient(
                matrix=potential_matrix,
                row=row,
                column=column,
            )

            magnitude = calculate_gradient_magnitude(
                delta_x=delta_x,
                delta_y=delta_y,
            )

            # column 0..7 съответства на a..h.
            mathematical_x = column

            # row 0 е rank 8, затова математическото
            # Y е 7 - row.
            mathematical_y = 7 - row

            file_name = chr(
                ord("a") + column
            )
            rank_name = 8 - row
            square_name = (
                f"{file_name}{rank_name}"
            )

            vectors.append(
                GradientVector(
                    square=square_name,
                    x=mathematical_x,
                    y=mathematical_y,
                    delta_x=delta_x,
                    delta_y=delta_y,
                    magnitude=magnitude,
                )
            )

            x_row.append(delta_x)
            y_row.append(delta_y)
            magnitude_row.append(magnitude)

            max_magnitude = max(
                max_magnitude,
                magnitude,
            )

        x_matrix.append(x_row)
        y_matrix.append(y_row)
        magnitude_matrix.append(
            magnitude_row
        )

    return GradientField(
        vectors=vectors,
        x_matrix=x_matrix,
        y_matrix=y_matrix,
        magnitude_matrix=magnitude_matrix,
        max_magnitude=max_magnitude,
    )


def print_gradient_field(
    gradient_field: GradientField,
) -> None:
    """Отпечатва градиентните вектори в терминала."""

    print("\n--- Gradient Field ---")
    print(
        f"Maximum magnitude: "
        f"{gradient_field.max_magnitude:.2f}"
    )

    for vector in gradient_field.vectors:
        print(
            f"{vector.square}: "
            f"gradient=("
            f"{vector.delta_x:+.2f}, "
            f"{vector.delta_y:+.2f}), "
            f"magnitude={vector.magnitude:.2f}"
        )