import time

import chess

import agent


def test_budget_shape():
    assert agent._move_budget(5_000) == 0.0
    assert agent._move_budget(9_999) == 0.0
    b = agent._move_budget(120_000)
    assert 3.0 < b <= 4.0
    assert agent._move_budget(20_000) <= 5.0


def test_uses_search_and_respects_wall_clock():
    board = chess.Board()
    start = time.perf_counter()
    uci = agent.get_move(board.fen(), 30_000)
    elapsed = time.perf_counter() - start
    assert chess.Move.from_uci(uci) in board.legal_moves
    assert elapsed < agent._move_budget(30_000) + 1.0


def test_low_clock_stays_instant():
    board = chess.Board()
    start = time.perf_counter()
    agent.get_move(board.fen(), 3_000)
    assert time.perf_counter() - start < 0.4


def test_search_path_legality_fuzz():
    # Small fuzz THROUGH the search (12s clock -> ~0.75s budget per call).
    import random

    for seed in range(10):
        rng = random.Random(seed)
        board = chess.Board()
        for _ in range(seed * 11 % 80):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over():
            continue
        uci = agent.get_move(board.fen(), 12_000)
        assert chess.Move.from_uci(uci) in board.legal_moves
