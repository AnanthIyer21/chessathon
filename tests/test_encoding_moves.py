import random

import chess

import encoding


def random_board(plies, seed):
    rng = random.Random(seed)
    board = chess.Board()
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    return board


def test_known_encodings():
    # e2e4: from e2 (sq 12), direction N distance 2 -> movetype 0*7+1=1
    assert encoding.encode_move(chess.Move.from_uci("e2e4")) == 1 * 64 + 12
    # g1f3: knight (2,-1)... from g1 (sq 6) delta (dr,dc)=(2,-1) -> knight idx 7
    assert encoding.encode_move(chess.Move.from_uci("g1f3")) == (56 + 7) * 64 + 6
    # e7e8q: queen promo encoded as plain N distance 1 -> movetype 0
    assert encoding.encode_move(chess.Move.from_uci("e7e8q")) == 0 * 64 + 52
    # e7d8n: underpromotion capture-left to knight -> 64 + (dc+1)*3 + 0 = 64
    assert encoding.encode_move(chess.Move.from_uci("e7d8n")) == 64 * 64 + 52


def test_roundtrip_all_legal_moves_many_positions():
    for seed in range(60):
        board = random_board(plies=seed % 90, seed=seed)
        oriented = encoding.orient(board)
        for move in oriented.legal_moves:
            idx = encoding.encode_move(move)
            assert 0 <= idx < encoding.POLICY_SIZE
            assert encoding.decode_move(idx, oriented) == move


def test_mirror_move_maps_between_perspectives():
    board = chess.Board()
    board.push_uci("e2e4")  # black to move
    oriented = encoding.orient(board)
    real = {m.uci() for m in board.legal_moves}
    mapped = {encoding.mirror_move(m).uci() for m in oriented.legal_moves}
    assert mapped == real


def test_decode_underpromotion():
    """Verify decode_move correctly handles underpromotion moves and restores promotion piece."""
    # Build a position with white pawn on g7 able to capture on h8 (underpromotion)
    fen = "k5bq/6P1/8/8/8/8/8/6K1 w - - 0 1"
    board = chess.Board(fen)
    assert board.is_valid(), f"FEN {fen} is not legal"

    # Verify the underpromotion move exists
    move_g7h8n = chess.Move.from_uci("g7h8n")
    assert move_g7h8n in board.legal_moves, f"Move g7h8n not in legal moves for position {fen}"

    # Test encode and decode roundtrip for underpromotion
    idx = encoding.encode_move(move_g7h8n)
    decoded = encoding.decode_move(idx, board)
    assert decoded == move_g7h8n, f"Roundtrip failed: {move_g7h8n} -> {idx} -> {decoded}"
    assert decoded.promotion == chess.KNIGHT, f"Expected knight promotion, got {decoded.promotion}"


def test_decode_auto_queen_promotion():
    """Verify decode_move re-adds queen promotion for pawn moves to rank 8."""
    # Build a position with white pawn on g7 able to move to g8 (queen promo)
    fen = "k6b/6P1/8/8/8/8/8/6K1 w - - 0 1"
    board = chess.Board(fen)
    assert board.is_valid(), f"FEN {fen} is not legal"

    # Verify the queen promotion move exists
    move_g7g8q = chess.Move.from_uci("g7g8q")
    assert move_g7g8q in board.legal_moves, f"Move g7g8q not in legal moves for position {fen}"

    # Test encode and decode roundtrip for queen promotion
    idx = encoding.encode_move(move_g7g8q)
    decoded = encoding.decode_move(idx, board)
    assert decoded == move_g7g8q, f"Roundtrip failed: {move_g7g8q} -> {idx} -> {decoded}"
    assert decoded.promotion == chess.QUEEN, f"Expected queen promotion, got {decoded.promotion}"


def test_castling_roundtrip():
    """Verify castling moves roundtrip through encode/decode."""
    # Build a position with castling available
    fen = "r3k2r/pppppppp/8/8/8/8/PPPPPPPP/R3K2R w KQkq - 0 1"
    board = chess.Board(fen)
    assert board.is_valid(), f"FEN {fen} is not legal"

    # Test kingside castling (e1g1)
    move_e1g1 = chess.Move.from_uci("e1g1")
    assert move_e1g1 in board.legal_moves, f"Move e1g1 not in legal moves for position {fen}"
    idx = encoding.encode_move(move_e1g1)
    decoded = encoding.decode_move(idx, board)
    assert decoded == move_e1g1, f"Kingside castling roundtrip failed: {move_e1g1} -> {idx} -> {decoded}"

    # Test queenside castling (e1c1)
    move_e1c1 = chess.Move.from_uci("e1c1")
    assert move_e1c1 in board.legal_moves, f"Move e1c1 not in legal moves for position {fen}"
    idx = encoding.encode_move(move_e1c1)
    decoded = encoding.decode_move(idx, board)
    assert decoded == move_e1c1, f"Queenside castling roundtrip failed: {move_e1c1} -> {idx} -> {decoded}"


def test_en_passant_roundtrip():
    """Verify en passant capture moves roundtrip through encode/decode."""
    # Build a position with a legal en passant capture available
    fen = "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3"
    board = chess.Board(fen)
    assert board.is_valid(), f"FEN {fen} is not legal"

    # Verify the en passant move exists
    move_e5f6 = chess.Move.from_uci("e5f6")
    assert move_e5f6 in board.legal_moves, f"Move e5f6 (en passant) not in legal moves for position {fen}"

    # Test encode and decode roundtrip for en passant
    idx = encoding.encode_move(move_e5f6)
    decoded = encoding.decode_move(idx, board)
    assert decoded == move_e5f6, f"En passant roundtrip failed: {move_e5f6} -> {idx} -> {decoded}"
