"""
Shared real-game chess position pool used by both the standalone
PositionCache crash-reproduction harness (`scripts/repro_position_cache_crash.py`)
and the pytest regression/stress suite
(`tests/test_desktop_app_position_cache_stress.py`), so crash/behavior rates
measured inside and outside pytest are directly comparable against identical
input positions.

~8 real openings sampled every couple of plies: the 8x8 attack-influence
matrices feeding RectBivariateSpline vary in density/asymmetry this way,
rather than exercising only the symmetric start position (which
under-exercises locate_critical_points/check_self_intersection degenerate
cases).
"""

from __future__ import annotations

import chess

OPENING_LINES = [
    # Ruy Lopez, Breyer setup
    "e4 e5 Nf3 Nc6 Bb5 a6 Ba4 Nf6 O-O Be7 Re1 b5 Bb3 d6 c3 O-O h3 Nb8 d4 Nbd7".split(),
    # Sicilian Najdorf, classical setup
    "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6 Nc3 a6 Be2 e5 Nb3 Be7 O-O O-O Be3 Be6 Nd5 Bxd5".split(),
    # Queen's Gambit Declined, classical
    "d4 d5 c4 e6 Nc3 Nf6 Bg5 Be7 e3 O-O Nf3 h6 Bh4 b6 Rc1 Bb7 Bxf6 Bxf6 cxd5 exd5".split(),
    # King's Indian Defense, classical
    "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6 Nf3 O-O Be2 e5 O-O Nc6 d5 Ne7 Ne1 Nd7 Be3 f5".split(),
    # Italian Game
    "e4 e5 Nf3 Nc6 Bc4 Bc5 c3 Nf6 d3 d6 O-O O-O Re1 a6 Nbd2 Ba7 h3 Be6 Bb3 Qd7".split(),
    # Caro-Kann
    "e4 c6 d4 d5 Nc3 dxe4 Nxe4 Bf5 Ng3 Bg6 h4 h6 Nf3 Nd7 h5 Bh7 Bd3 Bxd3 Qxd3 e6".split(),
    # English Opening
    "c4 e5 Nc3 Nf6 Nf3 Nc6 g3 d5 cxd5 Nxd5 Bg2 Nb6 O-O Be7 d3 O-O a3 Be6 Rb1 f6".split(),
    # French Defense, Tarrasch
    "e4 e6 d4 d5 Nd2 Nf6 e5 Nfd7 Bd3 c5 c3 Nc6 Ne2 cxd4 cxd4 f6 exf6 Nxf6 Nf3 Bd6".split(),
]


def build_position_pool() -> list[chess.Board]:
    positions: list[chess.Board] = []
    for line in OPENING_LINES:
        board = chess.Board()
        for ply_index, san in enumerate(line):
            board.push_san(san)
            if ply_index >= 5 and ply_index % 2 == 0:
                positions.append(board.copy())
    return positions
