"""Board and move encoding shared by training and the runtime agent.

Positions are always oriented so the side to move plays "up" the board:
when it is black's turn the board is mirrored vertically with colors
swapped (python-chess Board.mirror()), so the network always sees the
position from the mover's point of view.

Planes (19 x 8 x 8), indexed [plane, rank, file] with rank 0 = mover's
back rank:
  0-5   mover's P N B R Q K
  6-11  opponent's P N B R Q K
  12-15 castling rights: mover K-side, mover Q-side, opp K-side, opp Q-side
  16    en-passant file (whole file set)
  17    halfmove clock / 100 (clamped)
  18    constant ones
"""
import chess
import numpy as np

NUM_PLANES = 19
POLICY_SIZE = 73 * 64  # AlphaZero-style move encoding, defined further down.

_PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                chess.ROOK, chess.QUEEN, chess.KING]


def orient(board: chess.Board) -> chess.Board:
    """Return the position from the mover's point of view (mover = white)."""
    return board if board.turn == chess.WHITE else board.mirror()


def board_to_array(board: chess.Board):
    """Compact storage form of the oriented position.

    Returns (board64, meta): board64 is 64 uint8 piece codes (0 empty,
    1-6 mover's P..K, 7-12 opponent's P..K); meta is
    [castling bits, ep file or 255, halfmove clamped to 100].
    """
    b = orient(board)
    board64 = np.zeros(64, dtype=np.uint8)
    for square, piece in b.piece_map().items():
        code = _PIECE_ORDER.index(piece.piece_type) + 1
        if piece.color != chess.WHITE:
            code += 6
        board64[square] = code
    castling = (
        (1 if b.has_kingside_castling_rights(chess.WHITE) else 0)
        | (2 if b.has_queenside_castling_rights(chess.WHITE) else 0)
        | (4 if b.has_kingside_castling_rights(chess.BLACK) else 0)
        | (8 if b.has_queenside_castling_rights(chess.BLACK) else 0)
    )
    ep_file = b.ep_square % 8 if b.ep_square is not None else 255
    halfmove = min(b.halfmove_clock, 100)
    return board64, np.array([castling, ep_file, halfmove], dtype=np.uint8)


def array_to_planes(board64, meta):
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    grid = board64.reshape(8, 8)
    for code in range(1, 13):
        planes[code - 1][grid == code] = 1.0
    castling, ep_file, halfmove = int(meta[0]), int(meta[1]), int(meta[2])
    for bit in range(4):
        if castling & (1 << bit):
            planes[12 + bit][:] = 1.0
    if ep_file != 255:
        planes[16][:, ep_file] = 1.0
    planes[17][:] = halfmove / 100.0
    planes[18][:] = 1.0
    return planes


def board_to_planes(board: chess.Board):
    return array_to_planes(*board_to_array(board))
