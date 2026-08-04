from dataclasses import dataclass

@dataclass
class ChessVector:
    from_square: str
    to_square: str
    piece_name: str
    color: str
    strength: float

@dataclass
class FieldVector:
    start_x: float
    start_y: float
    end_x: float
    end_y: float
    delta_x: float
    delta_y: float
    magnitude: float

@dataclass
class AttackVector:
    piece_name: str
    color: str

    from_square: str
    to_square: str

    start_x: float
    start_y: float

    end_x: float
    end_y: float

    delta_x: int
    delta_y: int

@dataclass
class ControlSummary:
    """ store total influence """
    white_controlled_squares: int
    black_controlled_squares: int
    difference: int

@dataclass
class SquareControl:
    square: str
    white_attackers: int
    black_attackers: int
    difference: int

@dataclass
class MoveDetails:
    move: str
    piece_name: str
    color: str
    from_square: str
    to_square: str
    is_capture: bool
    is_check: bool

@dataclass
class MoveAnalysis:
    move: str
    piece_name: str
    color: str
    from_square: str
    to_square: str
    is_capture: bool
    is_check: bool
    control: ControlSummary
    vectors: list[ChessVector]