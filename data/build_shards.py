import argparse
import glob
import os
import sys

import chess.pgn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "submission"))
import encoding  # noqa: E402

_RESULTS = {"1-0": 1, "0-1": -1, "1/2-1/2": 0}


def process_pgn(stream, min_elo=2400, min_plies=10):
    """Yield (board64, meta, move_idx, outcome) for every position of every
    qualifying game. outcome is from the mover's perspective."""
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            return
        h = game.headers
        result = _RESULTS.get(h.get("Result"))
        if result is None:
            continue
        try:
            if int(h.get("WhiteElo", 0)) < min_elo or int(h.get("BlackElo", 0)) < min_elo:
                continue
        except ValueError:
            continue
        board = game.board()
        rows = []
        for move in game.mainline_moves():
            board64, meta = encoding.board_to_array(board)
            oriented = move if board.turn == chess.WHITE else encoding.mirror_move(move)
            outcome = result if board.turn == chess.WHITE else -result
            rows.append((board64, meta, encoding.encode_move(oriented), outcome))
            board.push(move)
        if len(rows) >= min_plies:
            yield from rows


class ShardWriter:
    def __init__(self, out_dir, shard_size):
        self.out_dir, self.shard_size = out_dir, shard_size
        self.buf, self.count = [], 0
        os.makedirs(out_dir, exist_ok=True)

    def add(self, row):
        self.buf.append(row)
        if len(self.buf) >= self.shard_size:
            self.flush()

    def flush(self):
        if not self.buf:
            return
        boards, metas, moves, outcomes = zip(*self.buf)
        path = os.path.join(self.out_dir, f"shard_{self.count:05d}.npz")
        np.savez_compressed(
            path,
            boards=np.array(boards, dtype=np.uint8),
            metas=np.array(metas, dtype=np.uint8),
            moves=np.array(moves, dtype=np.uint16),
            outcomes=np.array(outcomes, dtype=np.int8),
        )
        print(f"wrote {path} ({len(self.buf)} positions)")
        self.buf, self.count = [], self.count + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard-size", type=int, default=500_000)
    ap.add_argument("--min-elo", type=int, default=2400)
    args = ap.parse_args()
    writer = ShardWriter(args.out, args.shard_size)
    total = 0
    for path in sorted(glob.glob(os.path.join(args.pgn_dir, "*.pgn"))):
        print(f"parsing {path}")
        with open(path, encoding="utf-8", errors="replace") as f:
            for row in process_pgn(f, min_elo=args.min_elo):
                writer.add(row)
                total += 1
                if total % 1_000_000 == 0:
                    print(f"{total:,} positions")
    writer.flush()
    print(f"done: {total:,} positions")


if __name__ == "__main__":
    main()
