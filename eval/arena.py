import argparse
import os
import random
import sys
import time

import chess

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "submission"))
import agent  # noqa: E402
import mcts  # noqa: E402

_VALS = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material_player(rng):
    def choose(board, movetime):
        def score(m):
            b = board.copy(stack=False)
            b.push(m)
            s = sum(v * (len(b.pieces(p, board.turn)) - len(b.pieces(p, not board.turn)))
                    for p, v in _VALS.items())
            return s + rng.random() * 0.1
        return max(board.legal_moves, key=score)
    return choose


def policy_player(board, movetime):
    return agent.policy_move(board)


def mcts_player(board, movetime):
    return mcts.Searcher(agent.evaluate).search(board, time.perf_counter() + movetime)


PLAYERS = {"policy": lambda rng: policy_player,
           "mcts": lambda rng: mcts_player,
           "material": material_player}


def play(white, black, movetime, max_plies=300, opening_plies=0, opening_rng=None):
    board = chess.Board()
    if opening_plies and opening_rng is not None:
        for _ in range(opening_plies):
            if board.is_game_over(claim_draw=True):
                break
            board.push(opening_rng.choice(list(board.legal_moves)))
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        mover = white if board.turn == chess.WHITE else black
        board.push(mover(board, movetime))
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == chess.WHITE else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, choices=PLAYERS)
    ap.add_argument("--b", required=True, choices=PLAYERS)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--movetime", type=float, default=1.0)
    ap.add_argument("--random-opening-plies", type=int, default=4,
                     help="play this many random legal moves (game-seeded) before the "
                          "two players take over, so games don't collapse to a handful "
                          "of deterministic replays")
    args = ap.parse_args()
    w = d = losses = 0
    for g in range(args.games):
        rng = random.Random(g)
        pa, pb = PLAYERS[args.a](rng), PLAYERS[args.b](rng)
        op = args.random_opening_plies
        if g % 2 == 0:
            s = play(pa, pb, args.movetime, opening_plies=op, opening_rng=random.Random(g))
        else:
            s = 1 - play(pb, pa, args.movetime, opening_plies=op, opening_rng=random.Random(g))
        w += s == 1.0
        d += s == 0.5
        losses += s == 0.0
        print(f"game {g + 1}: {args.a} {'WDL'[int(2 - 2 * s)]}  running {w}-{d}-{losses}")
    print(f"{args.a} vs {args.b}: +{w} ={d} -{losses}")


if __name__ == "__main__":
    main()
