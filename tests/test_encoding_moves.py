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
