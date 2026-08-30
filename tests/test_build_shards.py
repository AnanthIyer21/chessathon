import io
import sys

import chess
import numpy as np

sys.path.insert(0, ".")
import encoding
from data.build_shards import process_pgn

PGN = """[Event "t"]
[White "a"]
[Black "b"]
[Result "1-0"]
[WhiteElo "2500"]
[BlackElo "2450"]

1. e4 e5 2. Nf3 Nc6 3. Bb5 a6 4. Ba4 Nf6 5. O-O Be7 6. Re1 b5 1-0

[Event "t2"]
[White "c"]
[Black "d"]
[Result "0-1"]
[WhiteElo "2000"]
[BlackElo "2600"]

1. d4 d5 2. c4 e6 3. Nc3 Nf6 4. Bg5 Be7 5. e3 O-O 6. Nf3 h6 0-1
"""


def test_process_pgn_filters_and_labels():
    rows = list(process_pgn(io.StringIO(PGN), min_elo=2400, min_plies=10))
    # Second game excluded (White 2000 < 2400); first has 12 plies.
    assert len(rows) == 12
    board64, meta, move_idx, outcome = rows[0]
    # First position: startpos, white played e2e4, white won -> +1
    b, m = encoding.board_to_array(chess.Board())
    np.testing.assert_array_equal(board64, b)
    assert move_idx == encoding.encode_move(chess.Move.from_uci("e2e4"))
    assert outcome == 1
    # Second position: black to move, black lost -> -1 from mover's view
    assert rows[1][3] == -1


def test_draw_outcome():
    pgn = PGN.replace('[Result "1-0"]', '[Result "1/2-1/2"]').replace("b5 1-0", "b5 1/2-1/2")
    rows = list(process_pgn(io.StringIO(pgn), min_elo=2400, min_plies=10))
    assert rows and all(r[3] == 0 for r in rows)
