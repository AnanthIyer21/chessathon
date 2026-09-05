# AI Chessathon agent

`agent.py` is a complete submission for [AI Chessathon](https://aichessathon.com): a bitboard
chess engine compiled with numba. No model, no external engine, no opening book. The whole
entry is one readable Python file, which is what a judge reads if a game gets flagged.

Copied from the `aichessathon-starter` fork where it was developed. The local harness that
plays it under the contest clock lives there, not here (see "Running it").

## What the engine does

- **Move generation.** Magic bitboards for sliders, precomputed knight, king and pawn attack
  tables, copy-make on a flat int64 position array, Zobrist hashing. The magic constants were
  found with `tools/find_magics.py`.
- **Evaluation.** Tapered material plus piece-square tables, passed pawns, mobility, king
  safety (pawn shield and attack units), tempo.
- **Search.** Iterative deepening with aspiration windows, principal-variation search, a
  transposition table of 2^22 entries, killer and history ordering, MVV-LVA captures, null-move
  pruning, late-move reductions, futility and reverse-futility pruning, quiescence with delta
  pruning, repetition and insufficient-material detection, draw contempt.
- **Time.** Per-move budget is `min(t/24 + 0.4 s, t/4) - 150 ms`. The clock is checked every
  2048 nodes and deepening stops once 45% of the budget is spent.
- **Pondering.** After replying, a thread keeps searching the position handed to the opponent,
  sharing the transposition table and history. The next `get_move` stops it first.
- **Safety.** python-chess parses the FEN and validates the returned move. Any exception falls
  back to the best capture, then the first legal move. `get_move` never raises.
- **Init.** Every jitted function is compiled at import by a warm-up search, so compilation
  lands inside the 60 s init budget and never on the clock.

## Files

| Path | What it is |
|---|---|
| `agent.py` | The submission. Zip it alone at the root. |
| `tools/check_engine.py` | Perft on reference positions, random-playout cross-check of moves and hashes against python-chess, mate conversion in won endgames. |
| `tools/find_magics.py` | Regenerates the magic multipliers pasted into `agent.py`. |
| `snapshots/stage1/` | Pure-Python negamax over python-chess with material eval. |
| `snapshots/stage4/` | Same, plus numba-jitted tapered eval, TT, killers, quiescence. |
| `snapshots/stage5/` | The bitboard engine, identical to the current `agent.py`. |
| `docs/` | The design spec and plan for the neural-network approach on `feature/agent`. |

## Running it

Needs Python 3.12+, `chess`, `numpy`, `numba`. The platform pins chess 1.11.2, numpy 2.5.2,
numba 0.67.0.

```
python -c "import agent, chess; print(agent.get_move(chess.STARTING_FEN, 120000))"
python tools/check_engine.py
```

Build the upload with `agent.py` at the zip root:

```
zip submission.zip agent.py
```

For full games under the contest clock, use the harness in
[AnanthIyer21/aichessathon-starter](https://github.com/AnanthIyer21/aichessathon-starter)
(`make play`, `make arena`, `make gate`). It mirrors the platform's protocol and clock.
