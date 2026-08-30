import chess
import numpy as np

import encoding


def test_startpos_planes():
    planes = encoding.board_to_planes(chess.Board())
    assert planes.shape == (19, 8, 8)
    assert planes.dtype == np.float32
    # Own pawns (plane 0) on rank index 1
    assert planes[0].sum() == 8 and planes[0][1].sum() == 8
    # Opponent pawns (plane 6) on rank index 6
    assert planes[6].sum() == 8 and planes[6][6].sum() == 8
    # Own king (plane 5) on e1
    assert planes[5][0][4] == 1.0 and planes[5].sum() == 1
    # All four castling planes full
    for p in range(12, 16):
        assert planes[p].sum() == 64
    # Constant plane
    assert planes[18].sum() == 64


def test_black_to_move_is_mirrored():
    board = chess.Board()
    board.push_uci("e2e4")
    planes = encoding.board_to_planes(board)  # black to move
    # Black is the mover: their pawns appear as "own" pawns on rank index 1.
    assert planes[0][1].sum() == 8
    # Mirroring flips ranks: white's e4 pawn (rank idx 3) shows as an
    # opponent pawn at rank idx 4, file e.
    assert planes[6][4][4] == 1.0


def test_en_passant_and_halfmove():
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
    planes = encoding.board_to_planes(board)
    assert planes[16][:, 3].sum() == 8  # d-file marked
    board2 = chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 40 90")
    planes2 = encoding.board_to_planes(board2)
    assert abs(planes2[17][0][0] - 0.4) < 1e-6
    assert planes2[16].sum() == 0


def test_array_roundtrip_matches_direct():
    board = chess.Board("r3k2r/pppq1ppp/2n2n2/3pp3/3PP3/2N2N2/PPPQ1PPP/R3K2R b KQkq - 4 8")
    board64, meta = encoding.board_to_array(board)
    assert board64.dtype == np.uint8 and meta.dtype == np.uint8
    np.testing.assert_array_equal(
        encoding.array_to_planes(board64, meta), encoding.board_to_planes(board)
    )


def test_black_to_move_asymmetric_castling():
    # Black to move with only white kingside (K) and black queenside (q) castling.
    # After orientation/mirroring, black becomes the mover.
    board = chess.Board("r3k3/pppppppp/8/8/8/8/PPPPPPPP/4K2R b Kq - 0 1")
    planes = encoding.board_to_planes(board)
    # Black's q (queenside) becomes mover's Q-side (plane 13)
    assert planes[13].sum() == 64
    # White's K (kingside) becomes opponent's K-side (plane 14)
    assert planes[14].sum() == 64
    # Black's k (kingside, absent) -> plane 12 (mover's K-side) should be 0
    assert planes[12].sum() == 0
    # White's Q (queenside, absent) -> plane 15 (opponent's Q-side) should be 0
    assert planes[15].sum() == 0


def test_black_to_move_en_passant():
    # Black to move with en-passant square f3 set.
    # f-file survives mirroring since mirror() flips ranks, not files.
    board = chess.Board("rnbqkbnr/pppp1ppp/8/8/3PpP2/8/PPP1P1PP/RNBQKBNR b KQkq f3 0 3")
    planes = encoding.board_to_planes(board)
    # f-file is index 5 (a=0, b=1, c=2, d=3, e=4, f=5)
    assert planes[16][:, 5].sum() == 8
    # Rest of plane 16 should be 0
    planes[16][:, 5] = 0
    assert planes[16].sum() == 0
