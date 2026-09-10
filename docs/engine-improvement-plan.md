# Engine improvement plan

Written 10 September 2026, the day before the submission lock, from a survey of engine
development logs (Stockfish 8, Ethereal, Weiss, Berserk, Alexandria, int0x80's blog) and
chessprogramming.org. Elo figures are self-play SPRT results those projects recorded; the
"expected here" column scales them for an engine at ~2M nodes/s and depth 15-19. Status is
against the `ananth` branch.

## Where the submission stood on 10 September (snapshots/stage6)

Magic-bitboard movegen, tapered Michniewski piece-square tables plus pawn structure,
mobility, king safety, bishop pair, rook files and a mop-up term; PVS with aspiration
windows, a 2^22-entry transposition table, MVV-LVA, killers, butterfly history with malus,
check extension, internal iterative reduction, reverse futility, null move (R = 3 + d/6),
futility, late-move pruning, late-move reductions, capture-only quiescence with delta
pruning; time budget t/24 + 0.4 s, new iteration only under 45% of it. Rated record with
this build: 15 wins, 11 draws, 20 losses over rounds 54-99, rank 109 of about 400.

## Ranked list

| # | Technique | Measured elsewhere | Expected here | Status |
|---|---|---|---|---|
| 1 | PeSTO piece-square tables and material | TSCP +200 | +80 to +150 | in test (v7) |
| 2 | Static exchange evaluation: prune losing captures in quiescence, order good captures first, losing captures last, prune losing captures at depth <= 3 | Weiss +36/+25, int0x80 +44 | +40 to +60 | done (batch 3) |
| 3 | 1-ply continuation history and counter-move, gravity-bounded scores | Weiss +45/+34, Ethereal +24/+35 | +30 to +50 | done (batch 3) |
| 4 | Time manager: soft and hard limits, best-move stability, node share of the best move | Weiss +57/+36, Berserk +31/+28 | +20 to +40 | in test (v8) |
| 5 | History-steered late-move reductions | Weiss +14/+14 | +10 to +20 | done (batch 3, +-1 ply) |
| 6 | Use the table score in place of static eval when its bound agrees | Weiss +11/+12 | +8 to +12 | in test (v8) |
| 7 | Null move R = 4 + d/5 + min(3, (eval - beta)/191) | int0x80 +10, Berserk +12 | +10 to +20 | in test (v8) |
| 8 | History pruning of quiet moves at depth <= 3 | Weiss +9/+11, int0x80 +21 | +5 to +15 | in test (v8) |
| 9 | 3-4 man Syzygy tables at the root (4.3 MB) | conversion only | small, but wins that were draws | done |
| 10 | Quiescence: per-move futility with endgame values, table probe and store, prune quiet check evasions | Weiss +34/+6, Ethereal +11/+3 | +15 to +25 | todo |
| 11 | 2-ply continuation history | Weiss +14/+4 | +5 to +10 | todo |
| 12 | Improving flag on reverse futility and late-move pruning counts | Berserk +12 | +5 to +12 | tested on LMR/LMP/futility: neutral; retry on RFP only |
| 13 | Razoring at depth <= 3 | int0x80 +31, Ethereal +9 | +5 to +15 | todo |
| 14 | Aspiration: delta 9 + score^2/16384, widen by a third | Weiss +14/+8 | +5 to +10 | todo |
| 15 | Transposition table buckets with age | Weiss +17/+7 | +5 to +10 | todo |
| 16 | Singular extensions and multi-cut | Weiss +12/+24 | +10 to +20 | todo, bug-prone |
| 17 | Correction history (pawn and non-pawn) | int0x80 +8 and +17 | +10 to +20 | todo, needs pawn key |
| 18 | ProbCut | Weiss +6/+7 | +5 to +8 | todo |
| 19 | Evaluation: pawn and minor threats, passed-pawn king distance by rank, safe checks in king safety, rook on 7th, outposts | Weiss/Ethereal +3 to +13 each | +10 to +30 total | todo |
| 20 | Reduce captures too; reduce less for checking moves | Weiss +12/+5, Berserk +14 | +5 to +15 | todo |
| 21 | Checks in the first quiescence ply | Berserk +5/+7 | 0 to +5 | skip |

Not worth it on this platform: pondering (the process is frozen between moves), opening
books (rated games start from curated positions), null-move verification and multi-cut
alone (neutral in Berserk).

## How each step is judged

`uv run python -m harness.arena --agent <candidate dir> --opponent <previous build>` at
10 s + 0.1 s over the eight seeded openings, 16 games, giving about +-18% on the score.
Anything inside that interval is noise; batch 2 (an "improving" guard, Manhattan mop-up
distance, longer iterations) landed at exactly 50% and was dropped. Correctness gates
before any arena: `tools/check_engine.py` (perft, movegen and hash cross-check against
python-chess, won-ending conversion) and, for the exchange evaluator, attack sets compared
with python-chess on 9,600 squares plus ten hand-built exchange positions.

## Platform facts that shaped the list

One EPYC core slower than a laptop, 90 s init budget, 120 s + 0.5 s, process frozen
between moves, 600-ply draw, opponents' rated games start from curated openings, 50 MB
unzipped cap, tablebases and books allowed, nothing engine-derived shipped.
