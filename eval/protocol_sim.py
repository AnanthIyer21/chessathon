"""Play a self-game against an extracted submission ZIP over the exact
JSON wire protocol, enforcing the contest clock (120s + 0.5s/move)."""
import argparse
import json
import queue
import subprocess
import sys
import tempfile
import threading
import time
import zipfile

import chess


def _reader_thread(proc, q):
    """Daemon thread that reads stdout lines and puts them into a queue."""
    try:
        for line in proc.stdout:
            q.put(line)
    except Exception:
        pass
    finally:
        q.put(None)  # Signal EOF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--plies", type=int, default=40)
    ap.add_argument("--base-ms", type=int, default=120_000)
    ap.add_argument("--inc-ms", type=int, default=500)
    args = ap.parse_args()

    with tempfile.TemporaryDirectory() as tmp:
        zipfile.ZipFile(args.zip).extractall(tmp)
        proc = subprocess.Popen(
            [sys.executable, "-u", "agent.py"], cwd=tmp,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        board = chess.Board()
        # Note: process startup (model load) is counted against move 1 here;
        # the real harness gives a separate 60 s init budget, so this is a
        # strictly more conservative test.
        clocks = {chess.WHITE: float(args.base_ms), chess.BLACK: float(args.base_ms)}

        # Start reader thread with queue
        q = queue.Queue()
        reader = threading.Thread(target=_reader_thread, args=(proc, q), daemon=True)
        reader.start()

        try:
            for ply in range(args.plies):
                if board.is_game_over():
                    break
                side = board.turn
                req = json.dumps({"fen": board.fen(), "time_left_ms": clocks[side]})
                t0 = time.perf_counter()
                proc.stdin.write(req + "\n")
                proc.stdin.flush()

                # Wait for response with timeout based on remaining clock + 10s buffer
                timeout = clocks[side] / 1000.0 + 10.0
                try:
                    line = q.get(timeout=timeout)
                except queue.Empty:
                    proc.kill()
                    print(f"FAIL: move timed out after {timeout:.1f} s (agent hung)", file=sys.stderr)
                    sys.exit(1)

                if line is None:
                    proc.kill()
                    print(f"FAIL: agent stdout closed unexpectedly at ply {ply}", file=sys.stderr)
                    sys.exit(1)

                elapsed_ms = (time.perf_counter() - t0) * 1000
                assert len(line.encode()) <= 4096, "reply over 4096 bytes"
                move = chess.Move.from_uci(json.loads(line)["move"])
                assert move in board.legal_moves, f"illegal {move} at {board.fen()}"
                clocks[side] -= elapsed_ms
                assert clocks[side] > 0, f"flagged on ply {ply} ({elapsed_ms:.0f} ms)"
                clocks[side] += args.inc_ms
                board.push(move)
                print(f"ply {ply:3d} {move.uci()} {elapsed_ms:6.0f} ms "
                      f"clock {clocks[side] / 1000:6.1f} s")
        finally:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            reader.join(timeout=1)
        print(f"OK: {board.fen()}")


if __name__ == "__main__":
    main()
