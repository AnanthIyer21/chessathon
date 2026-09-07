# AI Chessathon agent

Team entry for [AI Chessathon](https://aichessathon.com). `agent.py` at the repo root is the
submission: a bitboard chess engine compiled with numba. No model, no external engine, no
opening book. `submission.zip` is built from it and is the file to upload.

This branch carries the full starter layout (harness, baselines, tooling) so games can be played
under the contest clock from here.

```
make setup     # uv sync
make play      # one game against a baseline, real time control
make arena     # 20 fast games, prints a score
make zip       # rebuild submission.zip with agent.py at the root
make gate      # ruff, mypy, and two games that have to finish cleanly
```

## The engine

- **Move generation.** Magic bitboards for sliders, precomputed knight, king and pawn attack
  tables, copy-make on a flat int64 position array, Zobrist hashing. The magic constants were
  found with `tools/find_magics.py`.
- **Evaluation.** Tapered material plus piece-square tables, passed, isolated and doubled
  pawns, mobility, king safety (pawn shield and attack units), bishop pair, rooks on open
  files, tempo. In won pawnless endings a mop-up term drives the losing king to an edge (a
  corner of the bishop's colour for bishop and knight) and rewards closing the net around it.
- **Search.** Iterative deepening with aspiration windows, principal-variation search, a
  transposition table of 2^22 entries, killer and history ordering with a malus for quiet
  moves that failed to cut, MVV-LVA captures, internal iterative reduction, null-move
  pruning, late-move reductions, futility and reverse-futility pruning, quiescence with delta
  pruning, repetition and insufficient-material detection, draw contempt.
- **Time.** Per-move budget is `min(t/24 + 0.4 s, t/4) - 150 ms`. The clock is checked every
  2048 nodes and deepening stops once 45% of the budget is spent.
- **No pondering.** The platform suspends the process while the opponent thinks, so searching
  on their time earns nothing there. The code stays behind `PONDER = False`.
- **Safety.** python-chess parses the FEN and validates the returned move. Any exception falls
  back to the best capture, then the first legal move. `get_move` never raises.
- **Init.** Every jitted function is compiled at import by a warm-up search, about 7 s on an
  M2, well inside the 90 s init budget.

Correctness checks, independent of any search result:

```
uv run python tools/check_engine.py
```

Bare-king wins still fail to convert about one time in four at a short clock (the
evaluation goes flat while the rook shuffles). Three and four man Syzygy tablebases are
allowed by the rules and would close that gap.

## Measuring a change

Keep the previous build as the opponent. `snapshots/stage6` is the build packaged on
7 September; `snapshots/stage5` is the one that played the ladder before it.

```
uv run python -m harness.arena --opponent snapshots/stage6 --games 16
```

Sixteen games from the eight seeded openings give a 95% interval of about +-18%, so only
a large change shows in one run. Batch 1 (this build) scored 65.6% against stage5. A second
batch (an "improving" pruning guard, Manhattan mop-up distance, a longer iteration rule)
scored 50.0% against this build and was dropped.

## What's here

```
agent.py             the submission
submission.zip       agent.py zipped at the root, ready to upload
tools/check_engine.py  perft, random-playout cross-check against python-chess, endgame conversion
tools/find_magics.py   regenerates the magic multipliers in agent.py
snapshots/stage1/    pure-Python negamax over python-chess, material eval
snapshots/stage4/    same plus numba-jitted tapered eval, TT, killers, quiescence (the 152nd-place agent)
snapshots/stage5/    the bitboard engine as it first played the ladder (6 Sep)
snapshots/stage6/    the bitboard engine with batch 1, identical to the current agent.py
baselines/           random, greedy, minimax, numba; each is a directory with an agent.py
harness/runner.py    the process the platform runs your agent in
harness/referee.py   the clock, legality, draw and adjudication rules
harness/rules.py     the event constants the harness enforces
harness/sandbox.py   the one process, spoken to as the platform speaks to a container
harness/play.py      one game between two agent directories
harness/arena.py     many games, with a score
harness/package.py   builds submission.zip with agent.py at the root
docs/IDEAS.md        where the strength actually comes from
docs/superpowers/    design spec and plan for the neural-network approach on feature/agent
AGENTS.md            the contract, the footguns, and how to work in this repo
```

## Running games

```
make play FEN="<fen>"                              # start from a given position
uv run python -m harness.play --black baselines/minimax --pgn game.pgn
uv run python -m harness.arena --opponent snapshots/stage6 --games 16
```

Anything the agent writes to stdout or stderr shows up under the result, so `print` debugging
works. The platform discards it during rated games and shows it in the validation log.

Local games start from the normal position unless you pass `--fen`. Rated games start from
curated neutral positions.

The harness is here so local games are honest, not so you can pre-validate an upload. Acceptance
happens on the platform, and the validation log on the dashboard is the authority on it.

## The rules

[aichessathon.com/docs](https://aichessathon.com/docs) is canonical and changes. Read it before
you upload.
