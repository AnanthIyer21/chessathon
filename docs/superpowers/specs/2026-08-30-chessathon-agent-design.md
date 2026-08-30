# AI Chessathon Agent — Design Spec

Date: 2026-08-30
Competition: https://aichessathon.com/docs — submission lock 2026-09-11 12:00 (uploads close 11:00).

## Goal

A competitive entry for the AI Chessathon: a policy+value neural network trained
from scratch on strong human games, played through a small batched MCTS at game
time. Target: qualify for the top-48 finals via the ladder and the 11-round
Swiss.

## Competition constraints (verified against the docs)

- Submission: ZIP with `agent.py` at root exposing
  `get_move(fen: str, time_left_ms: int) -> str` (UCI move). Optional
  `requirements.txt` (PyPI wheels only). Model files ≤ 200 MB total.
- Runtime: Python 3.12; pre-installed torch (CPU), numpy, python-chess,
  onnxruntime. **1 dedicated CPU core, 2 GB RAM, no network, read-only
  filesystem + 256 MB scratch at /tmp.**
- Clock: 120 s per side + 0.5 s/move increment; 60 s init budget before the
  clock starts; response over 4096 bytes loses the game.
- Wire protocol: JSON over stdin/stdout (`{"fen": ..., "time_left_ms": ...}` →
  `{"move": ...}`); stderr captured up to 8 KB.
- Format: rated ladder rounds every 2 h (08:00–22:00) ranked by Elo, then an
  11-round Swiss over locked submissions decides the top 48 finalists.
- Rate limit: 6 uploads per team per 24 h.

## Compliance (disqualification risk is the top constraint)

Rules: "Prohibited: Stockfish, Lc0, or wrappers around any existing engine."
"A learned model must materially drive move selection." "Obfuscated or opaque
agents are disqualified." Analysis can be retroactive.

Design decisions made specifically for compliance:

1. **Nothing engine-derived anywhere.** The network is trained from scratch.
   Training data is human games (Lichess Elite database) labeled only with the
   move played and the game outcome — no Stockfish evaluations, no Maia or Lc0
   weights, no engine code.
2. **The learned model is the only source of chess knowledge.** No handcrafted
   evaluation terms, no opening book, no tablebases. MCTS priors and leaf values
   come exclusively from the network (AlphaZero recipe), so "materially drives
   move selection" is satisfied in the strongest possible sense.
3. **Readable source.** `agent.py` and its helpers stay short, plainly written,
   and commented; weights ship as `.onnx` (explicitly allowed by the docs).

## Architecture

### Model (`train/`)

- Input: 8×8×19 planes — 12 piece planes, side to move, 4 castling rights,
  en-passant file, halfmove-clock (scaled). Board always oriented from the
  side to move.
- Tower: residual CNN, ~6 blocks × 96–128 filters (~2–3 M params). Final width
  chosen from a measured single-core onnxruntime benchmark, not assumed.
- Heads: policy over the 8×8×73 AlphaZero move encoding (4672 logits, illegal
  moves masked at play time); value head → tanh scalar in [−1, 1].
- Export: ONNX (~10–15 MB), verified numerically against the PyTorch model.

### Training data (`data/`)

- Source: Lichess Elite database (both players ≥ 2400). Parse PGNs with
  python-chess into (planes, move-index, outcome) tuples; store as compressed
  numpy shards. Target 20–40 M positions.
- Loss: cross-entropy on the played move + MSE on outcome, standard weighting.
- Hardware: RTX 3060 Laptop (6 GB). Mixed precision; an epoch over the shard
  set in hours, full run overnight at most.

### Runtime agent (`agent/`)

- `agent.py` exposes `get_move`; also runnable as a stdin/stdout JSON loop for
  local protocol testing. Loads the ONNX session once at import (well inside
  the 60 s init budget), pinned to 1 intra-op thread.
- Search: small batched PUCT MCTS. Collect ~16 leaves with virtual loss,
  evaluate in one onnxruntime batch call, repeat until the move's time budget
  is spent. Expected ~300–800 node visits/move on one core.
- Time manager: budget ≈ `time_left/30 + 0.35 s`, hard cap per move; below
  10 s on the clock, degrade to pure policy (single inference, ~30 ms) so the
  agent cannot flag.
- Safety: the returned move is always checked against python-chess legal moves.
  Any exception falls back to highest-prior legal move, then to any legal move.
  `get_move` can never raise or return an illegal move.

### Evaluation (`eval/`) and packaging

- Match harness: agent vs. pure-policy vs. a simple material baseline; tracks
  results across model checkpoints so strength changes are measured, not
  guessed.
- Wire-protocol simulator that speaks the exact JSON contract and enforces the
  clock, run against every candidate ZIP before any upload (uploads are
  rate-limited to 6/day — none get burned on a broken package).
- `package.ps1` builds the submission ZIP: `agent.py`, helper modules,
  `model.onnx`, optional `requirements.txt`.

## Order of work

1. Scaffold + agent running with random-weight model → a valid, submittable
   ZIP exists from day one.
2. Data pipeline (download, parse, shard).
3. Training run + ONNX export + single-core benchmark → final model size.
4. MCTS tuning and time management against the match harness.
5. Package, simulate, upload; iterate on strength until lock.

## Testing

- Unit: FEN→planes encoder round-trips and symmetry; move-encoding round-trip
  over every legal move in random positions.
- Fuzz: agent returns a legal move across hundreds of random positions,
  including edge cases (promotions, en passant, underpromotion, stalemate-adjacent).
- Timing: per-move budget respected at various `time_left_ms`, including the
  low-clock degradation path.
- Integration: full game over the JSON wire protocol under a simulated clock.
