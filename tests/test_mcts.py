import time

import chess

import agent
import mcts


def make_searcher():
    return mcts.Searcher(agent.evaluate, batch_size=8)


def test_finds_mate_in_one():
    # Ra8# is the only mate; terminal backup must dominate any net prior.
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R3K3 w Q - 0 1")
    move = make_searcher().search(board, time.perf_counter() + 2.0)
    assert move == chess.Move.from_uci("a1a8")


def test_respects_deadline():
    board = chess.Board()
    start = time.perf_counter()
    make_searcher().search(board, start + 0.5)
    assert time.perf_counter() - start < 1.0


def test_returns_legal_in_forced_positions():
    # Only one legal move: king must take.
    board = chess.Board("7k/8/8/8/8/8/6q1/7K w - - 0 1")
    move = make_searcher().search(board, time.perf_counter() + 0.3)
    assert move in board.legal_moves


def test_board_unmodified_after_search():
    # The searcher pushes/pops moves on the caller's board; it must leave
    # it exactly as it found it (same FEN, same move stack).
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R3K3 w Q - 0 1")
    fen_before = board.fen()
    stack_before = list(board.move_stack)
    make_searcher().search(board, time.perf_counter() + 0.3)
    assert board.fen() == fen_before
    assert list(board.move_stack) == stack_before
