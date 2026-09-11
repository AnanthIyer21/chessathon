import contextlib, io, re, sys, time
import chess
sys.path.insert(0, sys.argv[1]); import agent
FENS = ["r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",
        "r2q1rk1/pp2bppp/2n1pn2/3p4/2PP4/2N1PN2/PPQ2PPP/R3KB1R w KQ - 0 9",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1"]
tot_nodes = tot_time = 0.0
for fen in FENS:
    agent.new_game(); buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        agent.get_move(fen, 72000)
    last = [l for l in buf.getvalue().splitlines() if l.startswith("depth")][-1]
    m = re.search(r"depth (\d+)/\d+ .* nodes (\d+) time ([\d.]+)s", last)
    d, nodes, t = int(m.group(1)), int(m.group(2)), float(m.group(3))
    tot_nodes += nodes; tot_time += t
    print(f"  {fen[:24]:24s} depth {d:2d} {nodes/t/1e6:.2f} Mnps")
print(f"[{sys.argv[2]}] overall {tot_nodes/tot_time/1e6:.2f} Mnps")
