# Strength path after the 2026 qualifier

Research report written 11 September 2026, the day of the locked-build Swiss. It surveys
papers and engine-development logs for the largest strength gains available to this engine
(numba bitboards, ~2M nodes/s, PeSTO evaluation, rank 81 of about 400 on the ladder) over
one to three weeks of part-time work on an M2 laptop. Numbers marked "measured" were run
on that laptop with the scripts in `tools/bench/`.

## Platform finding

The docs say uploads closed 11 September 11:00, "Your last valid build then freezes", the
13-round Swiss runs over locked builds that afternoon, and the live final is 12 September at
Encode Club, London, as knockout ties. Nothing provides for a new upload before the final,
so this plan targets a later event or the post-event ladder unless the organisers say
otherwise.

## Recommended path

Two measured facts drive it. A 768 -> 128x2 -> 1 int16 net costs about 213 ns per node in
numba (copy-make of the accumulator, two adds, two subtracts, squared-ReLU output); at the
engine's current ~500 ns per node that is ~1.4M nodes/s (1.1M for 256x2). PyTorch on the
M2's MPS trains that net at 0.66M positions/s, so 100M positions for 10 epochs is about 25
minutes; generating the data, not training, is the bottleneck.

Week 1, measurement, data, cheap Elo (+40 to +80):

- Day 1: an SPRT harness (fastchess or cutechess, 8 s + 0.08 s, elo0 = 0, elo1 = 10). The
  32-game arenas used so far have a 95% error of about +-120 Elo, so every "neutral" search
  result (razoring, history pruning, quiescence table probe, and the rest) is unresolved and
  should be re-run; each is +5 to +20 elsewhere.
- Days 2-3: gradient (Texel-style) tuning of the ~30 untuned hand-written terms in PyTorch,
  expected +30 to +80.
- Days 2-7, in the background on all cores: self-play data generation, 8 random plies then
  5k-node searches, recording FEN, score and result; ~3k positions/s gives 100M in ~9 h.
  The lichess database's 409.7M Stockfish-evaluated positions (CC0) may be added: the rules
  allow training on engine-annotated positions and ban only shipping a net someone else
  trained.

Week 2, NNUE (+150 to +300 net of the slowdown):

- Days 8-10: numba int16 inference for 768 -> 128x2 -> 1 with squared ReLU (QA = 255,
  QB = 64), the accumulator living in the copy-make stack. Check quantised against float
  output on 10k positions.
- Days 10-12: train generation 1 on 100M positions, SPRT against the hand-written
  evaluation at equal time. Days 12-14: regenerate data with the net engine, train
  generation 2; try 256x2 once data exceeds 300M positions.

Week 3, search on top of the net (+50 to +100): singular extensions, correction history,
improving-flag margins, 2-ply continuation history, time-management stop and extend rules,
then ProbCut and a small book from the known openings, each SPRT-tested.

Expected cumulative: +250 to +400 Elo; pessimistic +150 if the data pipeline stalls, and
the tuning branch alone still yields +40 to +80.

## Ranked options

| # | Option | Expected Elo | Effort (days) | Risk | Evidence |
|---|---|---|---|---|---|
| 1 | SPRT harness (prerequisite) | unlocks the rest | 0.5-1 | low | dogeystamp.com/chess3 |
| 2 | NNUE 768 -> 128x2 -> 1 on self-play + lichess data | +150 to +300 | 6-9 | med-high | Rudim +200 with 768 -> 32x2 (talkchess t=85845); Altair 6.0 +374; Stormphrax 1.0 +357 vs its HCE; Ethereal 13 +120 |
| 3 | Gradient/Texel tuning of the hand-written terms | +30 to +80 | 1-2 | low | Texel +99.6 cumulative (CPW); PeSTO +200 on TSCP; Grant +10/+3.4/+2.3 on an already tuned Ethereal |
| 4 | Singular extensions | +15 to +40 | 0.5-1 | low-med | +20 (talkchess t=83513); 7-10 then 35-40 after tuning; Stockfish comment +60 |
| 5 | Correction history | +10 to +25 | 0.5 | low | Viridithas 13 +12.25; grows with time control (CPW) |
| 6 | Time management: easy-move stop, fail-low extend | +5 to +20 | 0.5 | low | CPW Time Management; Solak and Vuckovic 2009 |
| 7 | Improving flag on RFP/LMP/LMR margins | +5 to +15 | 0.25 | low | CPW Improving |
| 8 | 2-ply continuation history | +5 to +15 | 0.5 | low | CPW History Heuristic |
| 9 | Bigger net 256x2 (needs >= 300M positions) | +30 to +60 over 128x2, -20% nps | 1 + data | med | Bobak 2024 NNUE research notes; Leorik release notes |
| 10 | ProbCut | 0 to +15 | 0.5 | med | overlaps null move (Jiang 2003 via CPW) |
| 11 | Opening book from the known positions | 0 to +10 | 0.5 | low | between similar engines openings matter less (talkchess t=71089) |
| 12 | Staged move generation | 0 to +10 | 1 | low | TT-move-first is the real gain and is already in |
| 13 | Incremental PST evaluation | +10 to +20 (15-25% nps) | 0.5 | low | moot once NNUE replaces the hand-written eval |
| 14 | Quiescence futility beyond delta pruning | 0 to +5 | 0.25 | low | no published number |

Measured on 11 September: numba `error_model="numpy"` on every jitted function gave
2.25 -> 2.30M nodes/s, within noise; not shipped.

## Notes

### NNUE

Net size and speed. Nasu's original is HalfKP 256x2-32-32-1 with int16 accumulators. Small
engines use 768 inputs; bullet's guide says to start there and grow the first layer before
adding layers. Rudim shipped (768 -> 32)x2 -> 1; Leorik 3.0 (768 -> 256)x2 -> 1 on 622M
positions; Altair 6.0 (768 -> 768)x2 -> 1. No published numba NNUE figures exist, so one was
measured here (numba 0.67, M2, one core):

| Hidden | copy-make + 2 add / 2 sub + squared-ReLU out | ReLU out alone | full refresh (32 features) | nodes/s at 500 ns search cost |
|---|---|---|---|---|
| 64 | 114 ns | 46 ns | 275 ns | 1.63M |
| 128 | 213 ns | 92 ns | 379 ns | 1.40M |
| 256 | 397 ns | 211 ns | 559 ns | 1.12M |
| 512 | 812 ns | 385 ns | 1786 ns | 0.76M |

Dropping the hand-written evaluation (100-150 ns per node) offsets part of this. Other
non-C reports: Numbfish (numpy HalfKP-256) fell from 54k to 14k nodes/s and still beat
Sunfish 90%; black_numba reaches 1.5M search nodes/s with a hand-written evaluation. This
engine's 2M is already the top of the published Python range.

Data. The datagen thread on talkchess (t=84811) gives: a few million labelled positions for
tiny nets, about a billion for good 256-hidden nets, 100M positions "in some hours" for
768x32, ~4k positions/s at a 5k-node limit with 7-8 random opening moves. Tan and Watkinson
(arXiv 2412.17948) got +100 Elo for a 256-hidden net from 44k master games plus 30k self-play
games, filtered to quiet positions. Target 50-200M positions for 128x2. Allowed inputs under
the rules: own self-play, the lichess database (409.7M Stockfish-evaluated positions), the
Stockfish master binpacks, the Lc0 training data; only loading or fine-tuning a published
network is banned. Stockfish-depth labels are distillation from a far stronger evaluator;
blend with game results and filter quiet positions.

Training time, measured on the M2 (EmbeddingBag sparse input, batch 16384, forward,
backward and Adam): MPS 0.66M positions/s at 128 hidden, 0.33M at 256, 0.17M at 512; CPU
0.26/0.17/0.09M. 100M positions for 10 epochs is about 25 minutes at 128 hidden.

Gain at equal time, reported by engines that switched: Rudim +200, Altair 6.0 +374,
Stormphrax 1.0 +357, Ethereal 13 up to +120 over a 3000+ hand-written evaluation, Weiss 2.0
+110, Shredder 13 +250. For a PeSTO-class evaluation around 2000, +150 to +300 after two
data generations is the realistic band. These are fixed-time results, so the slowdown is
already paid: Heinz's self-play figures put a 30% nodes/s loss at roughly 30-50 Elo, and
the net must beat the hand-written evaluation by more than that at fixed nodes. Use squared
ReLU (plain ReLU needs about 1.5x the neurons for the same loss) and accumulate the squared
activations in int32.

### Texel and gradient tuning

Osterlund's method minimises the squared error between game results and a sigmoid of the
evaluation over millions of positions; Texel 1.03 gained about 100 Elo from it in stages.
Grant's AdaGrad with exact gradients gave +10 linear, +3.4 king safety, +2.3 in an already
tuned Ethereal; Weiss gained about 65 Elo in a month largely from tuned evaluation. PeSTO's
tables are themselves Texel-tuned, so material and piece-square are done and the other ~30
terms are not: expect +30 to +80. Implementation: dump per-position feature coefficients
(mg and eg) from the numba evaluation, fit the tapered linear model in PyTorch, re-inject;
one to two days. Data: the self-play dump or lichess elite games.

### Search

At 32 games the standard error of the score is about 8.8%, roughly +-60 Elo at one sigma;
every "neutral" feature so far is unresolved. Evidence for what remains: singular
extensions +20 in a hobby engine with multi-cut, 35-40 after tuning; correction history
+12 in Viridithas 13, more at longer controls; 2-ply continuation history and the improving
flag are standard everywhere with no isolated public number; ProbCut overlaps null move and
is low priority; staged move generation is worth only a few Elo once the table move is tried
first. Calibration: null move is worth 50-100 in most engines, reverse futility +146 in a
young engine, late-move reductions at least +100; all three are already in.

### numba speed

Published Python figures: black_numba 1.5M search nodes/s; Numbfish 14k; a 2025 student
report measured numba at 4.5k nodes/s when a Python board object stayed in the loop
(object residue is the killer); Antares needs 20-45 s of compilation. Flags: `error_model`
gave 2% here; `boundscheck=False` is the default; `cache=True` built on arm64 is useless on
the x86 match machine; `fastmath` only matters for floats. Incremental middlegame and
endgame table sums would save 100-150 ns of the 500 ns node, about +15-25% nodes/s, but are
moot once NNUE lands.

### Time management

CPW's baseline base/20 + increment/2 is "very competitive with advanced schemes"; the
engine's t/24 + 0.4 s with a hard cap of t/4 is equivalent. Hyatt 1984 gives extra time
early; Markovitch and Sella 1996 learn allocation strategies from self-play; Solak and
Vuckovic 2009 test six models and report gains for Rybka and Shredder; Rheude 2021's neural
time manager found no significant gain. Concrete: stop after an iteration when the best
move has held for three or more iterations, extend up to the hard cap on a fail-low or a
best-move change, and add a node-count hard stop. An Ethereal-style budget of t/20 + 1.25
inc with hard t/5 was measured here at 28% because it drains the clock; do not ship a
budget whose steady state exceeds the increment.

### Opening preparation

The rules allow a shipped table that answers the opening (move 20 or lower) or the endgame,
counting against the 50 MB cap; a table that answers middlegame positions counts as an
engine. Rated games start from an unpublished curated set; eight are known. A book of deep
offline lines for those eight and their top replies to four plies is legal and worth 0 to
+10 Elo; do it last.

## Bibliography

- Nasu, Y. (2018). NNUE: Efficiently Updatable Neural-Network-based Evaluation Functions
  for Computer Shogi. https://github.com/asdfjkl/nnue and
  https://www.chessprogramming.org/NNUE
- Stockfish nnue-pytorch documentation.
  https://raw.githubusercontent.com/official-stockfish/nnue-pytorch/master/docs/nnue.md
- Whiting, J. bullet trainer docs. https://github.com/jw1912/bullet/blob/main/docs/1-basics.md
- Bobak, C. (2024). NNUE research notes. https://asteri.sm/files/2024-06-01-nnue.html,
  https://asteri.sm/files/2024-06-25-nnue-research-01.html,
  https://asteri.sm/files/2024-07-15-nnue-research-02.html
- Tan, D. and Watkinson Medina, N. (2024). Study of the Proper NNUE Dataset. arXiv:2412.17948.
- Osterlund, P. (2014). Texel's Tuning Method.
  https://www.chessprogramming.org/Texel's_Tuning_Method
- Grant, A. (2020). Evaluation and Tuning in Chess Engines.
  https://www.talkchess.com/forum3/viewtopic.php?t=74877
- Friederich, R. PeSTO evaluation. https://www.chessprogramming.org/PeSTO's_Evaluation_Function
- Anantharaman, T., Campbell, M. and Hsu, F. (1988/1990). Singular Extensions.
  https://www.chessprogramming.org/Singular_Extensions
- Buro, M. (1995). ProbCut; Jiang, A.X. (2003). https://www.chessprogramming.org/ProbCut
- Schaeffer, J. (1989). The History Heuristic and Alpha-Beta Search Enhancements in Practice.
  IEEE PAMI 11(11). Marsland, T.A. (1986). A Review of Game-Tree Pruning. ICCA J. 9(1).
- Donninger, C. (1993). Null Move and Deep Search. ICCA J. 16(3). Heinz, E.A. (1999).
  Adaptive Null-Move Pruning. ICCA J. 22(3). Heinz, E.A. (2001). Self-Play, Deep Search and
  Diminishing Returns. ICGA J. 24(2).
- Stockfish correction history commit (2024).
  https://github.com/official-stockfish/Stockfish/commit/b4d995d0d910044cf4ea2ad3ee30fd1d21070cd8
- Hyatt, R. (1984). Using Time Wisely. ICCA J. 7(1). Markovitch, S. and Sella, Y. (1996).
  Learning of Resource Allocation Strategies for Game Playing. Computational Intelligence
  12(1). Solak, R. and Vuckovic, V. (2009). Time Management During a Chess Game. ICGA J. 32(4).
- Rheude, T. (2021). Time Management in Chess with Neural Networks and Human Data.
  https://ml-research.github.io/papers/rheude2021time.pdf
- Hennecke, F. (2025). Chess Engine Optimization in Python.
  https://hps.vi4io.org/_media/teaching/autumn_term_2024/stud/nthpda/frederick_hennecke.pdf
- Engine releases: Ethereal 13, Stormphrax 1.0, Altair, Leorik, Viridithas, Weiss 2.0
  (GitHub release pages); Rudim notes at https://talkchess.com/viewtopic.php?t=85845&start=150
- Python engines: https://github.com/Avo-k/black_numba, https://github.com/dimdano/numbfish,
  https://github.com/Alex2262/Antares
- numba docs: https://numba.readthedocs.io/en/stable/reference/jit-compilation.html
- Data: https://database.lichess.org/, https://huggingface.co/official-stockfish/master-binpacks,
  https://storage.lczero.org/files/
- AI Chessathon rules: https://aichessathon.com/docs/rules.md and https://aichessathon.com/docs
- Testing: https://www.dogeystamp.com/chess3/; datagen thread
  https://talkchess.com/viewtopic.php?t=84811
