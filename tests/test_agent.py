import random
import time

import chess

import agent


def random_board(plies, seed):
    rng = random.Random(seed)
    board = chess.Board()
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    return board


def test_returns_legal_moves_fuzz():
    # 5000 ms is below the search threshold (added in a later task), so this
    # stays a fast single-inference fuzz even after MCTS lands. The search
    # path gets its own (smaller) legality fuzz in test_time_management.py.
    for seed in range(120):
        board = random_board(plies=seed % 120, seed=seed)
        if board.is_game_over():
            continue
        uci = agent.get_move(board.fen(), time_left_ms=5000)
        assert chess.Move.from_uci(uci) in board.legal_moves


def test_promotion_position():
    board = chess.Board("8/2P5/8/8/8/1k6/8/1K6 w - - 0 1")
    uci = agent.get_move(board.fen(), 5000)
    assert chess.Move.from_uci(uci) in board.legal_moves


def test_never_raises_on_garbage():
    assert agent.get_move("not a fen", 5000) == "0000"


def test_low_clock_is_fast():
    board = chess.Board()
    start = time.perf_counter()
    agent.get_move(board.fen(), time_left_ms=800)
    assert time.perf_counter() - start < 0.5


def test_evaluate_priors_sum_to_one():
    priors, values = agent.evaluate([chess.Board()])
    assert abs(sum(priors[0].values()) - 1.0) < 1e-4
    assert -1.0 <= float(values[0]) <= 1.0
    assert set(priors[0]) == set(chess.Board().legal_moves)
