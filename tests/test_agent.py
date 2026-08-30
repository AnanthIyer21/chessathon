import random
import subprocess
import sys
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


def test_evaluate_black_to_move():
    # Start position after e2e4 to get black to move.
    board = chess.Board()
    board.push_san("e4")
    priors, values = agent.evaluate([board])
    # Verify priors are over the ACTUAL board's legal moves (not mirrored).
    assert set(priors[0]) == set(board.legal_moves)
    # Verify they sum to ~1.0.
    assert abs(sum(priors[0].values()) - 1.0) < 1e-4
    # Verify value is in range.
    assert -1.0 <= float(values[0]) <= 1.0


def test_evaluate_terminal_board():
    # Fool's mate: 1.e4 e5 2.f4?? Qh4#
    board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3")
    assert board.is_checkmate()
    priors, values = agent.evaluate([board])
    # Terminal board has no legal moves, so priors should be empty dict.
    assert priors[0] == {}
    # Values still produced by network.
    assert -1.0 <= float(values[0]) <= 1.0


def test_wire_protocol_survives_garbage():
    # Spawn agent.py as a subprocess and test the JSON-lines wire protocol.
    proc = subprocess.Popen(
        [sys.executable, "submission/agent.py"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=".",
    )
    # Send three lines: two malformed, one valid.
    requests = (
        "not json\n"
        '{"wrong": 1}\n'
        '{"fen": "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "time_left_ms": 5000}\n'
    )
    stdout, stderr = proc.communicate(input=requests, timeout=10)

    # Parse responses.
    lines = [line for line in stdout.strip().split("\n") if line]
    assert len(lines) == 3, f"Expected 3 responses, got {len(lines)}: {lines}"

    import json
    responses = [json.loads(line) for line in lines]

    # First two should be "0000" (error responses).
    assert responses[0] == {"move": "0000"}
    assert responses[1] == {"move": "0000"}

    # Third should be a legal startpos move.
    move_uci = responses[2]["move"]
    startpos_board = chess.Board()
    assert chess.Move.from_uci(move_uci) in startpos_board.legal_moves

    # Process should exit cleanly.
    assert proc.returncode == 0
