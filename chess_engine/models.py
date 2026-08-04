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
    heatmap: list[list[int]]

    white_center: tuple[float, float ]|None
    black_center: tuple[float, float ]|None

    control_vector: FieldVector | None
    attack_vectors: list[AttackVector]

@dataclass
class DynamicsAnalysis:
    previous_force: int
    current_force: int
    delta_force: int

    white_control_delta: int
    black_control_delta: int

    attack_vectors_delta: int
    heatmap_change: int

    intensity: float
    label: str

# force = white control - black control
# prev position: 29 -22 = 7
# current position: 29-29=0
# deltaF = 0 - 7 = -7