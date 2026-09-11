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
| 1 | PeSTO piece-square tables and material | TSCP +200 | +80 to +150 | done: 68.8% (+10 =2 -4) vs batch 3 |
| 2 | Static exchange evaluation: prune losing captures in quiescence, order good captures first, losing captures last, prune losing captures at depth <= 3 | Weiss +36/+25, int0x80 +44 | +40 to +60 | done (batch 3) |
| 3 | 1-ply continuation history and counter-move, gravity-bounded scores | Weiss +45/+34, Ethereal +24/+35 | +30 to +50 | done (batch 3) |
| 4 | Time manager: soft and hard limits, best-move stability, node share of the best move | Weiss +57/+36, Berserk +31/+28 | +20 to +40 | Ethereal-style ideal t/20 + 1.25 inc with hard t/5 measured 28% on top of PeSTO (drains the clock, then starves); kept only the stability and node-share factors over the old t/24 budget |
| 5 | History-steered late-move reductions | Weiss +14/+14 | +10 to +20 | done (batch 3, +-1 ply) |
| 6 | Use the table score in place of static eval when its bound agrees | Weiss +11/+12 | +8 to +12 | tested in the v11 bundle: 50% over 32 games vs PeSTO, not shipped |
| 7 | Null move R = 4 + d/5 + min(3, (eval - beta)/191) | int0x80 +10, Berserk +12 | +10 to +20 | tested in the v11 bundle, not shipped |
| 8 | History pruning of quiet moves at depth <= 3 | Weiss +9/+11, int0x80 +21 | +5 to +15 | tested in the v11 bundle, not shipped |
| 9 | 3-4 man Syzygy tables at the root (4.3 MB) | conversion only | small, but wins that were draws | done |
| 10 | Quiescence: per-move futility with endgame values, table probe and store, prune quiet check evasions | Weiss +34/+6, Ethereal +11/+3 | +15 to +25 | table probe tested in the v11 bundle; futility and evasion pruning todo |
| 11 | 2-ply continuation history | Weiss +14/+4 | +5 to +10 | todo |
| 12 | Improving flag on reverse futility and late-move pruning counts | Berserk +12 | +5 to +12 | tested on LMR/LMP/futility: neutral; retry on RFP only |
| 13 | Razoring at depth <= 3 | int0x80 +31, Ethereal +9 | +5 to +15 | tested in the v11 bundle (250/depth, depth <= 2), not shipped |
| 14 | Aspiration: delta 9 + score^2/16384, widen by a third | Weiss +14/+8 | +5 to +10 | tested in the v11 bundle (12 + score^2/16384), not shipped |
| 15 | Transposition table buckets with age | Weiss +17/+7 | +5 to +10 | todo |
| 16 | Singular extensions and multi-cut | Weiss +12/+24 | +10 to +20 | todo, bug-prone |
| 17 | Correction history (pawn and non-pawn) | int0x80 +8 and +17 | +10 to +20 | todo, needs pawn key |
| 18 | ProbCut | Weiss +6/+7 | +5 to +8 | todo |
| 19 | Evaluation: pawn and minor threats, passed-pawn king distance by rank, safe checks in king safety, rook on 7th, outposts | Weiss/Ethereal +3 to +13 each | +10 to +30 total | todo |
| 20 | Reduce captures too; reduce less for checking moves | Weiss +12/+5, Berserk +14 | +5 to +15 | losing-capture reductions tested in the v11 bundle, not shipped |
| 21 | Checks in the first quiescence ply | Berserk +5/+7 | 0 to +5 | skip |

Not worth it on this platform: pondering (the process is frozen between moves), opening
books (rated games start from curated positions), null-move verification and multi-cut
alone (neutral in Berserk).

## The v11 bundle

Items 6, 7, 8, 13, 14, the quiescence table probe from 10 and the losing-capture
reductions from 20 were tried together on top of PeSTO, first with an Ethereal-style clock
(28%, the clock drains) and then with the conservative t/24 budget scaled by best-move
stability and node share (v11): +5 =7 -4 and +4 =7 -5, 50.0% over 32 games. Every one of
them is a measured gain elsewhere, so the likely reading is that they are small here and
need hundreds of games each; that testing is the next step, one item at a time. The
patches live in the session scratch area and are described in the commit history.

## How each step is judged

`uv run python -m harness.arena --agent <candidate dir> --opponent <previous build>` at
10 s + 0.1 s over the eight seeded openings, 16 games, giving about +-18% on the score.
Anything inside that interval is noise; batch 2 (an "improving" guard, Manhattan mop-up
distance, longer iterations) landed at exactly 50% and was dropped. Correctness gates
before any arena: `tools/check_engine.py` (perft, movegen and hash cross-check against
python-chess, won-ending conversion) and, for the exchange evaluator, attack sets compared
with python-chess on 9,600 squares plus ten hand-built exchange positions.

## What comes after the qualifier

`docs/research-strength-path.md` is the deeper survey (papers, engine logs, measured numba
and PyTorch numbers) with a one-to-three-week path: an SPRT harness, Texel tuning of the
hand-written terms, then a 768 -> 128x2 -> 1 NNUE trained on self-play and lichess data.

## Platform facts that shaped the list

One EPYC core slower than a laptop, 90 s init budget, 120 s + 0.5 s, process frozen
between moves, 600-ply draw, opponents' rated games start from curated openings, 50 MB
unzipped cap, tablebases and books allowed, nothing engine-derived shipped.
