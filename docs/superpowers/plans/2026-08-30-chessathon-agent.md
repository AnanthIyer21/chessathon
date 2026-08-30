# AI Chessathon Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A compliant, competitive AI Chessathon entry: a policy+value residual CNN trained from scratch on strong human games, played through a batched PUCT search, shipped as a ZIP whose root `agent.py` exposes `get_move(fen, time_left_ms) -> str`.

**Architecture:** A `submission/` directory holds exactly the files that go in the contest ZIP (`agent.py`, `encoding.py`, `mcts.py`, `model.onnx`) using flat imports so they work at ZIP root. Training (`train/`), data pipeline (`data/`), and evaluation (`eval/`) live outside the submission and import the submission modules by adding `submission/` to `sys.path`.

**Tech Stack:** Python 3.12, PyTorch (CUDA locally for training), numpy, python-chess, onnxruntime, pytest.

**Spec:** `docs/superpowers/specs/2026-08-30-chessathon-agent-design.md`

## Global Constraints

- Contest runtime: Python 3.12; only torch (CPU), numpy, python-chess, onnxruntime available; **1 CPU core, 2 GB RAM, no network, read-only FS**. `submission/` files may import ONLY these libraries plus each other.
- `submission/` files use flat imports (`import encoding`, `import mcts`) — they must work when placed at ZIP root.
- Clock: 120 s + 0.5 s/move; 60 s init budget; submission ZIP ≤ 200 MB.
- Compliance: nothing engine-derived anywhere (no Stockfish/Lc0/Maia code, weights, or evaluations — training data is human games only); no handcrafted evaluation, book, or tablebase in the agent; all `submission/` code plainly written and commented (judges must be able to read it).
- `get_move` must NEVER raise and NEVER return an illegal move.
- Windows host; venv at `.venv`; run tests with `.venv\Scripts\python -m pytest`.
- Large data lives in `C:\chessdata` (NOT the OneDrive-synced repo directory).

---

### Task 1: Scaffold and dev environment

**Files:**
- Create: `.gitignore`, `requirements-dev.txt`, `pytest.ini`, `tests/conftest.py`, `README.md` (note: `submission/` gets NO `__init__.py` — it is not a package; its files are used as top-level modules, matching ZIP root)

**Interfaces:**
- Produces: a venv at `.venv` with all dev deps; `tests/conftest.py` that puts `submission/` on `sys.path` so tests can `import encoding`.

- [ ] **Step 1: Create files**

`.gitignore`:
```
.venv/
__pycache__/
*.pyc
data/raw/
data/shards/
train/checkpoints/
dist/
*.onnx
!submission/model.onnx
```

`requirements-dev.txt`:
```
numpy
python-chess
onnxruntime
pytest
tqdm
requests
```

`pytest.ini`:
```ini
[pytest]
testpaths = tests
```

`tests/conftest.py`:
```python
import os
import sys

# Submission files use flat imports because they sit at ZIP root in the
# contest environment. Mirror that here.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "submission"))
```

`README.md`:
```markdown
# AI Chessathon agent

Policy+value residual CNN trained from scratch on high-rated human games
(Lichess Elite database), played through a small batched PUCT search.
No external engine (Stockfish, Lc0, ...) is involved at any stage:
not in the agent, not in training labels, not in the data pipeline.

- `submission/` — exactly what ships in the contest ZIP
- `train/` — model definition, training, ONNX export
- `data/` — download + PGN-to-shard pipeline
- `eval/` — arena, wire-protocol simulator, inference benchmark
- `scripts/` — packaging
```

Also create empty dirs used later: `submission/`, `train/`, `data/`, `eval/`, `scripts/`, `tests/`.

- [ ] **Step 2: Create venv and install deps**

Run (PowerShell):
```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu121
```
If `py -3.12` is missing, install Python 3.12 first (winget: `winget install Python.Python.3.12`). Verify CUDA: `.venv\Scripts\python -c "import torch; print(torch.cuda.is_available())"` → `True` (if False, training still works via CPU but slowly — flag it to the user rather than silently proceeding).

- [ ] **Step 3: Verify pytest runs**

Run: `.venv\Scripts\python -m pytest`
Expected: "no tests ran" (exit code 5) — that's success at this stage.

- [ ] **Step 4: Commit**

```bash
git add -A && git commit -m "chore: scaffold project and dev environment"
```

---

### Task 2: Board-to-planes encoding

**Files:**
- Create: `submission/encoding.py`
- Test: `tests/test_encoding_planes.py`

**Interfaces:**
- Produces:
  - `encoding.NUM_PLANES == 19`, `encoding.POLICY_SIZE == 4672`
  - `encoding.orient(board: chess.Board) -> chess.Board` (white-to-move view)
  - `encoding.board_to_array(board) -> (np.uint8[64], np.uint8[3])` (compact storage form of the oriented position: piece codes; [castling bits, ep file or 255, halfmove clamped 100])
  - `encoding.array_to_planes(board64, meta) -> np.float32[19,8,8]`
  - `encoding.board_to_planes(board) -> np.float32[19,8,8]`

- [ ] **Step 1: Write the failing tests**

`tests/test_encoding_planes.py`:
```python
import chess
import numpy as np

import encoding


def test_startpos_planes():
    planes = encoding.board_to_planes(chess.Board())
    assert planes.shape == (19, 8, 8)
    assert planes.dtype == np.float32
    # Own pawns (plane 0) on rank index 1
    assert planes[0].sum() == 8 and planes[0][1].sum() == 8
    # Opponent pawns (plane 6) on rank index 6
    assert planes[6].sum() == 8 and planes[6][6].sum() == 8
    # Own king (plane 5) on e1
    assert planes[5][0][4] == 1.0 and planes[5].sum() == 1
    # All four castling planes full
    for p in range(12, 16):
        assert planes[p].sum() == 64
    # Constant plane
    assert planes[18].sum() == 64


def test_black_to_move_is_mirrored():
    board = chess.Board()
    board.push_uci("e2e4")
    planes = encoding.board_to_planes(board)  # black to move
    # Black is the mover: their pawns appear as "own" pawns on rank index 1.
    assert planes[0][1].sum() == 8
    # Mirroring flips ranks: white's e4 pawn (rank idx 3) shows as an
    # opponent pawn at rank idx 4, file e.
    assert planes[6][4][4] == 1.0


def test_en_passant_and_halfmove():
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 3")
    planes = encoding.board_to_planes(board)
    assert planes[16][:, 3].sum() == 8  # d-file marked
    board2 = chess.Board("8/8/8/4k3/8/8/4K3/8 w - - 40 90")
    planes2 = encoding.board_to_planes(board2)
    assert abs(planes2[17][0][0] - 0.4) < 1e-6
    assert planes2[16].sum() == 0


def test_array_roundtrip_matches_direct():
    board = chess.Board("r3k2r/pppq1ppp/2n2n2/3pp3/3PP3/2N2N2/PPPQ1PPP/R3K2R b KQkq - 4 8")
    board64, meta = encoding.board_to_array(board)
    assert board64.dtype == np.uint8 and meta.dtype == np.uint8
    np.testing.assert_array_equal(
        encoding.array_to_planes(board64, meta), encoding.board_to_planes(board)
    )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_encoding_planes.py -v`
Expected: FAIL (ModuleNotFoundError: encoding)

- [ ] **Step 3: Implement `submission/encoding.py` (planes half)**

```python
"""Board and move encoding shared by training and the runtime agent.

Positions are always oriented so the side to move plays "up" the board:
when it is black's turn the board is mirrored vertically with colors
swapped (python-chess Board.mirror()), so the network always sees the
position from the mover's point of view.

Planes (19 x 8 x 8), indexed [plane, rank, file] with rank 0 = mover's
back rank:
  0-5   mover's P N B R Q K
  6-11  opponent's P N B R Q K
  12-15 castling rights: mover K-side, mover Q-side, opp K-side, opp Q-side
  16    en-passant file (whole file set)
  17    halfmove clock / 100 (clamped)
  18    constant ones
"""
import chess
import numpy as np

NUM_PLANES = 19
POLICY_SIZE = 73 * 64  # AlphaZero-style move encoding, defined further down.

_PIECE_ORDER = [chess.PAWN, chess.KNIGHT, chess.BISHOP,
                chess.ROOK, chess.QUEEN, chess.KING]


def orient(board: chess.Board) -> chess.Board:
    """Return the position from the mover's point of view (mover = white)."""
    return board if board.turn == chess.WHITE else board.mirror()


def board_to_array(board: chess.Board):
    """Compact storage form of the oriented position.

    Returns (board64, meta): board64 is 64 uint8 piece codes (0 empty,
    1-6 mover's P..K, 7-12 opponent's P..K); meta is
    [castling bits, ep file or 255, halfmove clamped to 100].
    """
    b = orient(board)
    board64 = np.zeros(64, dtype=np.uint8)
    for square, piece in b.piece_map().items():
        code = _PIECE_ORDER.index(piece.piece_type) + 1
        if piece.color != chess.WHITE:
            code += 6
        board64[square] = code
    castling = (
        (1 if b.has_kingside_castling_rights(chess.WHITE) else 0)
        | (2 if b.has_queenside_castling_rights(chess.WHITE) else 0)
        | (4 if b.has_kingside_castling_rights(chess.BLACK) else 0)
        | (8 if b.has_queenside_castling_rights(chess.BLACK) else 0)
    )
    ep_file = b.ep_square % 8 if b.ep_square is not None else 255
    halfmove = min(b.halfmove_clock, 100)
    return board64, np.array([castling, ep_file, halfmove], dtype=np.uint8)


def array_to_planes(board64, meta):
    planes = np.zeros((NUM_PLANES, 8, 8), dtype=np.float32)
    grid = board64.reshape(8, 8)
    for code in range(1, 13):
        planes[code - 1][grid == code] = 1.0
    castling, ep_file, halfmove = int(meta[0]), int(meta[1]), int(meta[2])
    for bit in range(4):
        if castling & (1 << bit):
            planes[12 + bit][:] = 1.0
    if ep_file != 255:
        planes[16][:, ep_file] = 1.0
    planes[17][:] = halfmove / 100.0
    planes[18][:] = 1.0
    return planes


def board_to_planes(board: chess.Board):
    return array_to_planes(*board_to_array(board))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_encoding_planes.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add submission/encoding.py tests/test_encoding_planes.py
git commit -m "feat: board-to-planes encoding"
```

---

### Task 3: Move encoding (8x8x73)

**Files:**
- Modify: `submission/encoding.py` (append)
- Test: `tests/test_encoding_moves.py`

**Interfaces:**
- Produces:
  - `encoding.mirror_move(move: chess.Move) -> chess.Move`
  - `encoding.encode_move(move: chess.Move) -> int` — move must already be oriented (mover plays up); returns index in [0, 4672)
  - `encoding.decode_move(index: int, board: chess.Board) -> chess.Move` — board must be oriented; adds queen promotion automatically

- [ ] **Step 1: Write the failing tests**

`tests/test_encoding_moves.py`:
```python
import random

import chess

import encoding


def random_board(plies, seed):
    rng = random.Random(seed)
    board = chess.Board()
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    return board


def test_known_encodings():
    # e2e4: from e2 (sq 12), direction N distance 2 -> movetype 0*7+1=1
    assert encoding.encode_move(chess.Move.from_uci("e2e4")) == 1 * 64 + 12
    # g1f3: knight (2,-1)... from g1 (sq 6) delta (dr,dc)=(2,-1) -> knight idx 7
    assert encoding.encode_move(chess.Move.from_uci("g1f3")) == (56 + 7) * 64 + 6
    # e7e8q: queen promo encoded as plain N distance 1 -> movetype 0
    assert encoding.encode_move(chess.Move.from_uci("e7e8q")) == 0 * 64 + 52
    # e7d8n: underpromotion capture-left to knight -> 64 + (dc+1)*3 + 0 = 64
    assert encoding.encode_move(chess.Move.from_uci("e7d8n")) == 64 * 64 + 52


def test_roundtrip_all_legal_moves_many_positions():
    for seed in range(60):
        board = random_board(plies=seed % 90, seed=seed)
        oriented = encoding.orient(board)
        for move in oriented.legal_moves:
            idx = encoding.encode_move(move)
            assert 0 <= idx < encoding.POLICY_SIZE
            assert encoding.decode_move(idx, oriented) == move


def test_mirror_move_maps_between_perspectives():
    board = chess.Board()
    board.push_uci("e2e4")  # black to move
    oriented = encoding.orient(board)
    real = {m.uci() for m in board.legal_moves}
    mapped = {encoding.mirror_move(m).uci() for m in oriented.legal_moves}
    assert mapped == real
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_encoding_moves.py -v`
Expected: FAIL (AttributeError: encode_move)

- [ ] **Step 3: Append move encoding to `submission/encoding.py`**

```python
# ---------------------------------------------------------------------------
# Move encoding: AlphaZero's 8x8x73 scheme, flattened to movetype*64 + from_sq
# (matching how the network's (73,8,8) policy output flattens).
#   movetype 0-55:  "queen" moves, 8 directions x 7 distances
#   movetype 56-63: knight moves
#   movetype 64-72: underpromotions (3 directions x N/B/R); queen
#                   promotions are encoded as plain forward/diagonal moves.
# Moves must be oriented (mover plays up the board) before encoding.
# ---------------------------------------------------------------------------

_DIRS = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
_DIR_IDX = {d: i for i, d in enumerate(_DIRS)}
_KNIGHT = [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]
_KNIGHT_IDX = {d: i for i, d in enumerate(_KNIGHT)}
_UNDERPROMO = [chess.KNIGHT, chess.BISHOP, chess.ROOK]


def mirror_move(move: chess.Move) -> chess.Move:
    """Map a move between the real board and its mirrored orientation."""
    return chess.Move(chess.square_mirror(move.from_square),
                      chess.square_mirror(move.to_square),
                      promotion=move.promotion)


def encode_move(move: chess.Move) -> int:
    fr, fc = divmod(move.from_square, 8)
    tr, tc = divmod(move.to_square, 8)
    dr, dc = tr - fr, tc - fc
    if move.promotion is not None and move.promotion != chess.QUEEN:
        movetype = 64 + (dc + 1) * 3 + _UNDERPROMO.index(move.promotion)
    elif (dr, dc) in _KNIGHT_IDX:
        movetype = 56 + _KNIGHT_IDX[(dr, dc)]
    else:
        dist = max(abs(dr), abs(dc))
        step = (dr // dist if dr else 0, dc // dist if dc else 0)
        movetype = _DIR_IDX[step] * 7 + dist - 1
    return movetype * 64 + move.from_square


def decode_move(index: int, board: chess.Board) -> chess.Move:
    movetype, from_sq = divmod(index, 64)
    fr, fc = divmod(from_sq, 8)
    if movetype >= 64:
        u = movetype - 64
        dc, piece = u // 3 - 1, _UNDERPROMO[u % 3]
        return chess.Move(from_sq, (fr + 1) * 8 + fc + dc, promotion=piece)
    if movetype >= 56:
        dr, dc = _KNIGHT[movetype - 56]
    else:
        d, dist = divmod(movetype, 7)
        dr, dc = _DIRS[d][0] * (dist + 1), _DIRS[d][1] * (dist + 1)
    to_sq = (fr + dr) * 8 + fc + dc
    promotion = None
    if board.piece_type_at(from_sq) == chess.PAWN and to_sq >= 56:
        promotion = chess.QUEEN
    return chess.Move(from_sq, to_sq, promotion=promotion)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_encoding_moves.py tests/test_encoding_planes.py -v`
Expected: all passed

- [ ] **Step 5: Commit**

```bash
git add submission/encoding.py tests/test_encoding_moves.py
git commit -m "feat: AlphaZero-style 4672 move encoding"
```

---

### Task 4: Model definition

**Files:**
- Create: `train/model.py`, `train/__init__.py` (empty)
- Test: `tests/test_model.py`

**Interfaces:**
- Produces: `train.model.PolicyValueNet(channels=128, blocks=6)`; `forward(x: float32[N,19,8,8]) -> (policy_logits[N,4672], value[N] in [-1,1])`. Policy flatten order matches `encoding.encode_move` (movetype*64 + square).

- [ ] **Step 1: Write the failing test**

`tests/test_model.py`:
```python
import sys

import torch

sys.path.insert(0, ".")  # repo root, for train package
from train.model import PolicyValueNet


def test_forward_shapes_and_ranges():
    net = PolicyValueNet(channels=32, blocks=2)
    x = torch.randn(4, 19, 8, 8)
    policy, value = net(x)
    assert policy.shape == (4, 4672)
    assert value.shape == (4,)
    assert value.abs().max() <= 1.0


def test_param_count_default():
    net = PolicyValueNet()
    n = sum(p.numel() for p in net.parameters())
    assert 1_000_000 < n < 5_000_000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_model.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `train/model.py`**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + y)


class PolicyValueNet(nn.Module):
    """Residual CNN with a 73-plane policy head (flattens to movetype*64+sq,
    matching encoding.encode_move) and a scalar tanh value head."""

    def __init__(self, channels=128, blocks=6):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(19, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
        )
        self.tower = nn.Sequential(*[ResBlock(channels) for _ in range(blocks)])
        self.policy_head = nn.Conv2d(channels, 73, 1)
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(8 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.tower(self.stem(x))
        return self.policy_head(x).flatten(1), self.value_head(x).squeeze(1)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_model.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add train/ tests/test_model.py
git commit -m "feat: policy+value residual CNN"
```

---

### Task 5: ONNX export with parity check

**Files:**
- Create: `train/export_onnx.py`
- Test: `tests/test_export.py`

**Interfaces:**
- Produces: `train.export_onnx.export(net, out_path)` writing an ONNX model with input name `"planes"` (dynamic batch), outputs `["policy", "value"]`; CLI `python -m train.export_onnx --checkpoint <pt-or-none> --out submission/model.onnx [--channels 128 --blocks 6]`. With no checkpoint it exports random weights (used for the day-one valid submission).

- [ ] **Step 1: Write the failing test**

`tests/test_export.py`:
```python
import sys

import numpy as np
import onnxruntime as ort
import torch

sys.path.insert(0, ".")
from train.export_onnx import export
from train.model import PolicyValueNet


def test_onnx_matches_torch(tmp_path):
    net = PolicyValueNet(channels=32, blocks=2).eval()
    path = tmp_path / "m.onnx"
    export(net, str(path))
    x = np.random.randn(5, 19, 8, 8).astype(np.float32)
    with torch.no_grad():
        tp, tv = net(torch.from_numpy(x))
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    op, ov = sess.run(None, {"planes": x})
    np.testing.assert_allclose(op, tp.numpy(), atol=1e-4)
    np.testing.assert_allclose(ov, tv.numpy(), atol=1e-4)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_export.py -v`
Expected: FAIL (ModuleNotFoundError)

- [ ] **Step 3: Implement `train/export_onnx.py`**

```python
import argparse

import torch

from train.model import PolicyValueNet


def export(net, out_path):
    net = net.eval().cpu()
    dummy = torch.zeros(1, 19, 8, 8)
    torch.onnx.export(
        net, dummy, out_path,
        input_names=["planes"], output_names=["policy", "value"],
        dynamic_axes={"planes": {0: "batch"}, "policy": {0: "batch"}, "value": {0: "batch"}},
        opset_version=17,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=6)
    args = ap.parse_args()
    channels, blocks = args.channels, args.blocks
    state = None
    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        # Trained checkpoints record their architecture; trust that over flags.
        channels = state.get("channels", channels)
        blocks = state.get("blocks", blocks)
    net = PolicyValueNet(channels=channels, blocks=blocks)
    if state is not None:
        net.load_state_dict(state["model"] if "model" in state else state)
    export(net, args.out)
    print(f"exported to {args.out} (channels={channels}, blocks={blocks})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test, then export the random-weights placeholder model**

Run: `.venv\Scripts\python -m pytest tests/test_export.py -v` → PASS
Run: `.venv\Scripts\python -m train.export_onnx --out submission/model.onnx`
Expected: `submission/model.onnx` exists, roughly 10-15 MB.

- [ ] **Step 5: Commit**

```bash
git add train/export_onnx.py tests/test_export.py submission/model.onnx
git commit -m "feat: ONNX export with torch parity test; placeholder model"
```

---

### Task 6: Policy-only agent with hard safety guarantees

**Files:**
- Create: `submission/agent.py`
- Test: `tests/test_agent.py`

**Interfaces:**
- Consumes: `encoding.*` (Task 2/3), `submission/model.onnx` (Task 5)
- Produces:
  - `agent.get_move(fen: str, time_left_ms: int) -> str` (never raises, always legal UCI)
  - `agent.evaluate(boards: list[chess.Board]) -> (list[dict[chess.Move, float]], np.ndarray)` — legal-move priors (softmax over legal moves only) and values, one batched ONNX call. Used by MCTS in Task 11.
  - `agent.policy_move(board) -> chess.Move`
  - `python agent.py` runs a JSON-lines stdin/stdout loop: `{"fen":..., "time_left_ms":...}` → `{"move": "..."}`

- [ ] **Step 1: Write the failing tests**

`tests/test_agent.py`:
```python
import random
import time

import chess

import agent


def random_board(plies, seed):
    rng = random.Random(seed)
    board = chess.Board()
    for _ in range(plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    return board


def test_returns_legal_moves_fuzz():
    # 5000 ms is below the search threshold (added in a later task), so this
    # stays a fast single-inference fuzz even after MCTS lands. The search
    # path gets its own (smaller) legality fuzz in test_time_management.py.
    for seed in range(120):
        board = random_board(plies=seed % 120, seed=seed)
        if board.is_game_over():
            continue
        uci = agent.get_move(board.fen(), time_left_ms=5000)
        assert chess.Move.from_uci(uci) in board.legal_moves


def test_promotion_position():
    board = chess.Board("8/2P5/8/8/8/1k6/8/1K6 w - - 0 1")
    uci = agent.get_move(board.fen(), 5000)
    assert chess.Move.from_uci(uci) in board.legal_moves


def test_never_raises_on_garbage():
    assert agent.get_move("not a fen", 5000) == "0000"


def test_low_clock_is_fast():
    board = chess.Board()
    start = time.perf_counter()
    agent.get_move(board.fen(), time_left_ms=800)
    assert time.perf_counter() - start < 0.5


def test_evaluate_priors_sum_to_one():
    priors, values = agent.evaluate([chess.Board()])
    assert abs(sum(priors[0].values()) - 1.0) < 1e-4
    assert -1.0 <= float(values[0]) <= 1.0
    assert set(priors[0]) == set(chess.Board().legal_moves)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_agent.py -v`
Expected: FAIL (ModuleNotFoundError: agent)

- [ ] **Step 3: Implement `submission/agent.py`**

```python
"""AI Chessathon agent.

All chess knowledge comes from model.onnx: a policy+value residual CNN
trained from scratch on high-rated human games (Lichess Elite database).
No external engine is involved at any stage - not in this agent and not
in the training pipeline. Move selection = the network's policy/value
outputs, optionally refined by a small PUCT search (mcts.py) whose
priors and leaf evaluations also come exclusively from the network.
"""
import json
import os
import sys
import time

import chess
import numpy as np
import onnxruntime as ort

import encoding

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model.onnx")

# The contest runtime provides one dedicated core; keep onnxruntime on it.
_opts = ort.SessionOptions()
_opts.intra_op_num_threads = 1
_opts.inter_op_num_threads = 1
_SESSION = ort.InferenceSession(_MODEL_PATH, _opts, providers=["CPUExecutionProvider"])


def evaluate(boards):
    """Batched network evaluation.

    Returns (priors, values): per board, a dict {legal move -> prior}
    (softmax over legal moves only) and the value in [-1, 1] from the
    mover's perspective.
    """
    planes = np.stack([encoding.board_to_planes(b) for b in boards])
    logits, values = _SESSION.run(None, {"planes": planes})
    priors = []
    for board, lg in zip(boards, logits):
        moves = list(board.legal_moves)
        if board.turn == chess.WHITE:
            idx = [encoding.encode_move(m) for m in moves]
        else:
            idx = [encoding.encode_move(encoding.mirror_move(m)) for m in moves]
        raw = lg[idx].astype(np.float64)
        raw = np.exp(raw - raw.max())
        raw /= raw.sum()
        priors.append(dict(zip(moves, raw)))
    return priors, values


def policy_move(board):
    """Single network call; play the highest-prior legal move."""
    priors, _ = evaluate([board])
    return max(priors[0], key=priors[0].get)


def _choose(board, time_left_ms):
    return policy_move(board)  # search is added in a later task


def get_move(fen: str, time_left_ms: int) -> str:
    """Contest entry point. Never raises; always returns a legal move
    (or 0000 if the position itself is unusable)."""
    try:
        board = chess.Board(fen)
    except Exception:
        return "0000"
    try:
        move = _choose(board, time_left_ms)
        if move in board.legal_moves:
            return move.uci()
    except Exception as exc:  # noqa: BLE001 - losing on error is worse
        print(f"agent error: {exc!r}", file=sys.stderr)
    moves = list(board.legal_moves)
    return moves[0].uci() if moves else "0000"


def main():
    """JSON-lines wire protocol over stdin/stdout, for local simulation."""
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        req = json.loads(line)
        move = get_move(req["fen"], int(req["time_left_ms"]))
        print(json.dumps({"move": move}), flush=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_agent.py -v`
Expected: 5 passed (random weights are fine — legality and speed are what's under test)

- [ ] **Step 5: Commit**

```bash
git add submission/agent.py tests/test_agent.py
git commit -m "feat: policy-only agent with legality and crash guarantees"
```

---

### Task 7: Protocol simulator + packaging → first valid ZIP

**Files:**
- Create: `eval/protocol_sim.py`, `scripts/package.py`
- Test: `tests/test_package.py`

**Interfaces:**
- Consumes: `submission/` contents
- Produces:
  - `scripts/package.py` → builds `dist/submission.zip` (agent.py, encoding.py, mcts.py if present, model.onnx at ZIP ROOT), fails if > 200 MB
  - `eval/protocol_sim.py --zip dist/submission.zip [--plies 40]` → extracts to a temp dir, spawns `python agent.py`, plays a self-game over the JSON protocol with a real 120s+0.5s clock, asserts every reply is legal and in time; exit code 0 on success

- [ ] **Step 1: Write the failing test**

`tests/test_package.py`:
```python
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_package_and_simulate(tmp_path):
    zip_path = tmp_path / "submission.zip"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "package.py"), "--out", str(zip_path)],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    names = set(zipfile.ZipFile(zip_path).namelist())
    assert "agent.py" in names and "model.onnx" in names and "encoding.py" in names
    r2 = subprocess.run(
        [sys.executable, str(ROOT / "eval" / "protocol_sim.py"),
         "--zip", str(zip_path), "--plies", "10"],
        capture_output=True, text=True, timeout=300,
    )
    assert r2.returncode == 0, r2.stdout + r2.stderr
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_package.py -v`
Expected: FAIL (package.py not found)

- [ ] **Step 3: Implement `scripts/package.py`**

```python
import argparse
import os
import zipfile

SUBMISSION_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "submission")
MAX_BYTES = 200 * 1024 * 1024


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join("dist", "submission.zip"))
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    files = sorted(
        f for f in os.listdir(SUBMISSION_DIR)
        if f.endswith((".py", ".onnx")) and not f.startswith("test")
    )
    assert "agent.py" in files and "model.onnx" in files, files
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(os.path.join(SUBMISSION_DIR, f), arcname=f)
    size = os.path.getsize(args.out)
    assert size <= MAX_BYTES, f"zip too large: {size}"
    print(f"wrote {args.out} ({size / 1e6:.1f} MB): {files}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Implement `eval/protocol_sim.py`**

```python
"""Play a self-game against an extracted submission ZIP over the exact
JSON wire protocol, enforcing the contest clock (120s + 0.5s/move)."""
import argparse
import json
import subprocess
import sys
import tempfile
import time
import zipfile

import chess


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
            [sys.executable, "agent.py"], cwd=tmp,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
        )
        board = chess.Board()
        # Note: process startup (model load) is counted against move 1 here;
        # the real harness gives a separate 60 s init budget, so this is a
        # strictly more conservative test.
        clocks = {chess.WHITE: float(args.base_ms), chess.BLACK: float(args.base_ms)}
        try:
            for ply in range(args.plies):
                if board.is_game_over():
                    break
                side = board.turn
                req = json.dumps({"fen": board.fen(), "time_left_ms": clocks[side]})
                t0 = time.perf_counter()
                proc.stdin.write(req + "\n")
                proc.stdin.flush()
                line = proc.stdout.readline()
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
        print(f"OK: {board.fen()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `.venv\Scripts\python -m pytest tests/test_package.py -v`
Expected: PASS. Also run the full thing once by hand:
`.venv\Scripts\python scripts/package.py` then
`.venv\Scripts\python eval/protocol_sim.py --zip dist/submission.zip`
Expected: 40 plies, every move legal and fast. **We now have a valid submittable ZIP.**

- [ ] **Step 6: Commit**

```bash
git add scripts/package.py eval/protocol_sim.py tests/test_package.py
git commit -m "feat: packaging and wire-protocol simulator; first valid ZIP"
```

---

### Task 8: Data pipeline (download + PGN → shards)

**Files:**
- Create: `data/download.py`, `data/build_shards.py`
- Test: `tests/test_build_shards.py`

**Interfaces:**
- Produces:
  - `data/download.py --months 2024-01 2024-02 --dest C:\chessdata\raw` — downloads Lichess Elite DB monthly zips from `https://database.nikonoel.fr/lichess_elite_{month}.zip` and extracts the PGNs. (If that mirror is down, fall back to manual download from the Lichess Elite Database page and place PGNs in the dest dir — the script must say so in its error message.)
  - `data.build_shards.process_pgn(pgn_stream, min_elo=2400, min_plies=10) -> iterator of (board64, meta, move_idx, outcome)` — outcome is from the mover's perspective (+1/0/-1)
  - `data/build_shards.py --pgn-dir C:\chessdata\raw --out C:\chessdata\shards --shard-size 500000` → `shard_00000.npz` files with arrays `boards (N,64) u8`, `metas (N,3) u8`, `moves (N,) u16`, `outcomes (N,) i8`

- [ ] **Step 1: Write the failing test**

`tests/test_build_shards.py`:
```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_build_shards.py -v`
Expected: FAIL (ModuleNotFoundError: data.build_shards)

- [ ] **Step 3: Implement `data/build_shards.py` (and empty `data/__init__.py`)**

```python
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
```

- [ ] **Step 4: Implement `data/download.py`**

```python
import argparse
import os
import zipfile

import requests

URL = "https://database.nikonoel.fr/lichess_elite_{month}.zip"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--months", nargs="+", required=True, help="e.g. 2024-01 2024-02")
    ap.add_argument("--dest", default=r"C:\chessdata\raw")
    args = ap.parse_args()
    os.makedirs(args.dest, exist_ok=True)
    for month in args.months:
        url = URL.format(month=month)
        zip_path = os.path.join(args.dest, f"elite_{month}.zip")
        if not os.path.exists(zip_path):
            print(f"downloading {url}")
            r = requests.get(url, stream=True, timeout=60)
            if r.status_code != 200:
                raise SystemExit(
                    f"{url} -> HTTP {r.status_code}. If the mirror is down, download "
                    f"the Lichess Elite Database manually and put the .pgn files in "
                    f"{args.dest}, then skip this script."
                )
            with open(zip_path, "wb") as f:
                for chunk in r.iter_content(1 << 20):
                    f.write(chunk)
        print(f"extracting {zip_path}")
        zipfile.ZipFile(zip_path).extractall(args.dest)
    print("done")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests, then run the real pipeline**

Run: `.venv\Scripts\python -m pytest tests/test_build_shards.py -v` → PASS

Then start the real data run **in the background** (it's long; don't block):
```powershell
.venv\Scripts\python data/download.py --months 2024-01 2024-02 2024-03 --dest C:\chessdata\raw
.venv\Scripts\python data/build_shards.py --pgn-dir C:\chessdata\raw --out C:\chessdata\shards
```
Target: ≥ 20M positions (add more months if short). If the mirror URL 404s, tell the user and ask them to download manually — do not silently substitute a different data source.

- [ ] **Step 6: Commit**

```bash
git add data/ tests/test_build_shards.py
git commit -m "feat: Lichess Elite download and PGN-to-shard pipeline"
```

---

### Task 9: Training loop

**Files:**
- Create: `train/train.py`, `train/dataset.py`
- Test: `tests/test_train_smoke.py`

**Interfaces:**
- Consumes: shards from Task 8; `PolicyValueNet` from Task 4
- Produces:
  - `train.dataset.ShardDataset(shard_paths, shuffle=True)` — IterableDataset yielding `(planes f32[19,8,8], move_idx i64, outcome f32)`
  - `python -m train.train --shards C:\chessdata\shards --epochs 1 [--channels 128 --blocks 6 --batch 512 --lr 1e-3 --resume <ckpt>]` — writes `train/checkpoints/ckpt_<step>.pt` (dict with keys `model`, `optim`, `step`, `channels`, `blocks`) every 2000 steps and `ckpt_final.pt` at the end; prints policy/value loss and top-1 policy accuracy every 100 steps

- [ ] **Step 1: Write the failing test**

`tests/test_train_smoke.py`:
```python
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def make_fake_shard(path, n=2000):
    rng = np.random.default_rng(0)
    boards = np.zeros((n, 64), dtype=np.uint8)
    boards[:, rng.integers(0, 64, size=n)] = rng.integers(1, 13, size=n).astype(np.uint8)
    np.savez_compressed(
        path,
        boards=boards,
        metas=rng.integers(0, 3, size=(n, 3)).astype(np.uint8),
        moves=rng.integers(0, 4672, size=n).astype(np.uint16),
        outcomes=rng.integers(-1, 2, size=n).astype(np.int8),
    )


def test_train_smoke(tmp_path):
    make_fake_shard(tmp_path / "shard_00000.npz")
    out = tmp_path / "ckpt"
    r = subprocess.run(
        [sys.executable, "-m", "train.train", "--shards", str(tmp_path),
         "--epochs", "1", "--batch", "64", "--channels", "16", "--blocks", "1",
         "--max-steps", "5", "--ckpt-dir", str(out), "--device", "cpu"],
        capture_output=True, text=True, cwd=ROOT, timeout=300,
    )
    assert r.returncode == 0, r.stderr
    assert (out / "ckpt_final.pt").exists()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv\Scripts\python -m pytest tests/test_train_smoke.py -v`
Expected: FAIL

- [ ] **Step 3: Implement `train/dataset.py`**

```python
import random

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "submission"))
import encoding  # noqa: E402


class ShardDataset(IterableDataset):
    """Streams (planes, move_idx, outcome) from npz shards, one shard in
    memory at a time, shuffled within shard and across shard order."""

    def __init__(self, shard_paths, shuffle=True, seed=0):
        self.shard_paths = list(shard_paths)
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        info = get_worker_info()
        paths = self.shard_paths
        if info is not None:
            paths = paths[info.id::info.num_workers]
        rng = random.Random(self.seed + (info.id if info else 0))
        if self.shuffle:
            paths = paths[:]
            rng.shuffle(paths)
        for path in paths:
            with np.load(path) as z:
                boards, metas = z["boards"], z["metas"]
                moves, outcomes = z["moves"], z["outcomes"]
            order = np.arange(len(moves))
            if self.shuffle:
                np.random.default_rng(rng.randrange(1 << 30)).shuffle(order)
            for i in order:
                planes = encoding.array_to_planes(boards[i], metas[i])
                yield (torch.from_numpy(planes),
                       torch.tensor(int(moves[i]), dtype=torch.long),
                       torch.tensor(float(outcomes[i])))
```

- [ ] **Step 4: Implement `train/train.py`**

```python
import argparse
import glob
import os
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from train.dataset import ShardDataset
from train.model import PolicyValueNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shards", required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--blocks", type=int, default=6)
    ap.add_argument("--value-weight", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--ckpt-dir", default=os.path.join("train", "checkpoints"))
    ap.add_argument("--resume", default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.shards, "*.npz")))
    assert paths, f"no shards in {args.shards}"
    os.makedirs(args.ckpt_dir, exist_ok=True)

    net = PolicyValueNet(args.channels, args.blocks).to(args.device)
    optim = torch.optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(enabled=args.device == "cuda")
    step = 0
    if args.resume:
        state = torch.load(args.resume, map_location=args.device)
        net.load_state_dict(state["model"])
        optim.load_state_dict(state["optim"])
        step = state["step"]

    def save(name):
        torch.save({"model": net.state_dict(), "optim": optim.state_dict(),
                    "step": step, "channels": args.channels, "blocks": args.blocks},
                   os.path.join(args.ckpt_dir, name))

    t0 = time.time()
    for epoch in range(args.epochs):
        loader = DataLoader(ShardDataset(paths, seed=epoch), batch_size=args.batch,
                            num_workers=args.workers, pin_memory=args.device == "cuda")
        for planes, moves, outcomes in loader:
            planes = planes.to(args.device, non_blocking=True)
            moves = moves.to(args.device, non_blocking=True)
            outcomes = outcomes.to(args.device, non_blocking=True)
            with torch.amp.autocast(args.device, enabled=args.device == "cuda"):
                policy, value = net(planes)
                ploss = F.cross_entropy(policy, moves)
                vloss = F.mse_loss(value, outcomes)
                loss = ploss + args.value_weight * vloss
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optim)
            scaler.update()
            step += 1
            if step % 100 == 0:
                acc = (policy.argmax(1) == moves).float().mean().item()
                rate = step * args.batch / max(time.time() - t0, 1)
                print(f"step {step:7d} ploss {ploss.item():.3f} vloss {vloss.item():.3f} "
                      f"top1 {acc:.3f} ({rate:,.0f} pos/s)", flush=True)
            if step % 2000 == 0:
                save(f"ckpt_{step}.pt")
            if args.max_steps and step >= args.max_steps:
                save("ckpt_final.pt")
                return
    save("ckpt_final.pt")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run smoke test**

Run: `.venv\Scripts\python -m pytest tests/test_train_smoke.py -v`
Expected: PASS

- [ ] **Step 6: Launch the real training run (background, long)**

```powershell
.venv\Scripts\python -m train.train --shards C:\chessdata\shards --epochs 2 --batch 512 --workers 2
```
Run in background; check the log every so often. Expect top-1 policy accuracy to climb past ~0.35 within the first epoch (0.40-0.50 by the end is normal for this setup). If it plateaus near random (~0.001) something is wrong with move-index alignment — stop and debug rather than continuing the run.

- [ ] **Step 7: Commit**

```bash
git add train/train.py train/dataset.py tests/test_train_smoke.py
git commit -m "feat: supervised training loop with checkpointing"
```

---

### Task 10: Trained export + single-core benchmark

**Files:**
- Create: `eval/bench_inference.py`

**Interfaces:**
- Consumes: `train/checkpoints/ckpt_final.pt`, `train.export_onnx`
- Produces: `submission/model.onnx` replaced with trained weights; a measured evals/sec figure at batch 16 on one thread that Task 11 uses to sanity-check search speed.

- [ ] **Step 1: Implement `eval/bench_inference.py`**

```python
import argparse
import time

import numpy as np
import onnxruntime as ort


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="submission/model.onnx")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    sess = ort.InferenceSession(args.model, opts, providers=["CPUExecutionProvider"])
    x = np.random.randn(args.batch, 19, 8, 8).astype(np.float32)
    for _ in range(5):
        sess.run(None, {"planes": x})
    t0 = time.perf_counter()
    n = 50
    for _ in range(n):
        sess.run(None, {"planes": x})
    dt = (time.perf_counter() - t0) / n
    print(f"batch {args.batch}: {dt * 1000:.1f} ms/call, "
          f"{args.batch / dt:,.0f} positions/s (single thread)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Export the trained model and benchmark**

```powershell
.venv\Scripts\python -m train.export_onnx --checkpoint train/checkpoints/ckpt_final.pt --out submission/model.onnx
.venv\Scripts\python -m pytest tests/  # everything still green with real weights
.venv\Scripts\python eval/bench_inference.py
```
Decision rule: at batch 16 we want ≥ 300 positions/s single-threaded (≈ 300+ MCTS visits in a 1 s move). If well below, retrain narrower (`--channels 96` or `--blocks 5`); if far above, a wider net is affordable. Record the number in the commit message.

- [ ] **Step 3: Quick strength sanity check**

Interim check before the arena exists — policy-only agent vs. random mover, 20 games, expect ≥ 95% score. Snippet (run in repl or scratch file, not committed):
```python
import sys, random, chess
sys.path.insert(0, "submission")
import agent
wins = 0
for g in range(20):
    board = chess.Board(); rng = random.Random(g)
    while not board.is_game_over(claim_draw=True):
        if board.turn == (chess.WHITE if g % 2 == 0 else chess.BLACK):
            board.push(chess.Move.from_uci(agent.get_move(board.fen(), 60000)))
        else:
            board.push(rng.choice(list(board.legal_moves)))
    o = board.outcome(claim_draw=True)
    if o.winner == (chess.WHITE if g % 2 == 0 else chess.BLACK):
        wins += 1
print(wins, "/ 20")
```

- [ ] **Step 4: Commit**

```bash
git add submission/model.onnx eval/bench_inference.py
git commit -m "feat: trained model export (<N> pos/s single-core at batch 16)"
```

---

### Task 11: Batched PUCT search

**Files:**
- Create: `submission/mcts.py`
- Test: `tests/test_mcts.py`

**Interfaces:**
- Consumes: `agent.evaluate` signature — `evaluate(list[chess.Board]) -> (list[dict[chess.Move, float]], values)`
- Produces: `mcts.Searcher(evaluate_fn, c_puct=1.5, batch_size=16)` with `search(board: chess.Board, deadline: float) -> chess.Move` (deadline = `time.perf_counter()` value; always returns a legal move; does at least one batch even if the deadline already passed)

- [ ] **Step 1: Write the failing tests**

`tests/test_mcts.py`:
```python
import time

import chess

import agent
import mcts


def make_searcher():
    return mcts.Searcher(agent.evaluate, batch_size=8)


def test_finds_mate_in_one():
    # Ra8# is the only mate; terminal backup must dominate any net prior.
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R3K3 w Q - 0 1")
    move = make_searcher().search(board, time.perf_counter() + 2.0)
    assert move == chess.Move.from_uci("a1a8")


def test_respects_deadline():
    board = chess.Board()
    start = time.perf_counter()
    make_searcher().search(board, start + 0.5)
    assert time.perf_counter() - start < 1.0


def test_returns_legal_in_forced_positions():
    # Only one legal move: king must take.
    board = chess.Board("7k/8/8/8/8/8/6q1/7K w - - 0 1")
    move = make_searcher().search(board, time.perf_counter() + 0.3)
    assert move in board.legal_moves
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_mcts.py -v`
Expected: FAIL (ModuleNotFoundError: mcts)

- [ ] **Step 3: Implement `submission/mcts.py`**

```python
"""Small batched PUCT search (the AlphaZero recipe, sized for one CPU core).

Every prior and every leaf evaluation comes from the network via the
evaluate() callable - there is no handcrafted evaluation anywhere.
Leaves are collected in small batches using virtual loss so the ONNX
runtime evaluates several positions per call.
"""
import time

import chess

_VIRTUAL_LOSS = 1.0


class _Node:
    __slots__ = ("prior", "visits", "value_sum", "children")

    def __init__(self, prior):
        self.prior = prior
        self.visits = 0
        self.value_sum = 0.0
        self.children = None  # dict[chess.Move, _Node] once expanded


class Searcher:
    def __init__(self, evaluate, c_puct=1.5, batch_size=16):
        self.evaluate = evaluate
        self.c_puct = c_puct
        self.batch_size = batch_size

    def search(self, board, deadline):
        root = _Node(1.0)
        priors, _ = self.evaluate([board])
        root.children = {m: _Node(p) for m, p in priors[0].items()}
        if len(root.children) == 1:
            return next(iter(root.children))

        while True:
            leaves, terminal_backups = self._run_batch(board, root)
            if not leaves and not terminal_backups:
                break  # no progress possible
            if time.perf_counter() >= deadline:
                break
        return max(root.children, key=lambda m: root.children[m].visits)

    def _run_batch(self, board, root):
        """Collect up to batch_size distinct leaves with virtual loss,
        evaluate them in one network call, back up the results.

        Visit accounting: a child's visits are incremented exactly once per
        traversal, at selection time (that increment doubles as the virtual
        loss); backup only settles value_sum. A parent's visit total is the
        sum of its children's visits, computed in _select_child."""
        leaves, terminal_backups = [], 0
        seen = set()
        for _ in range(self.batch_size):
            node, path = root, []
            while node.children:
                node = self._select_child(node, path)
                board.push(path[-1][0])
            outcome = board.outcome()
            stop = False
            if outcome is not None:
                # Terminal: exact value, no net call. Mated mover scores -1.
                self._backup(path, -1.0 if outcome.winner is not None else 0.0)
                terminal_backups += 1
            elif id(node) in seen:
                # Same unexpanded leaf selected twice: abandon this path and
                # evaluate what we have (more selections would repeat it).
                self._undo_virtual(path)
                stop = True
            else:
                seen.add(id(node))
                leaves.append((node, path))
            for _ in range(len(path)):
                board.pop()
            if stop:
                break

        if leaves:
            boards = []
            for _, path in leaves:
                b = board.copy(stack=False)
                for mv, _ in path:
                    b.push(mv)
                boards.append(b)
            priors, values = self.evaluate(boards)
            for (node, path), pri, val in zip(leaves, priors, values):
                node.children = {m: _Node(p) for m, p in pri.items()}
                self._backup(path, float(val))
        return leaves, terminal_backups

    def _select_child(self, node, path):
        total = sum(c.visits for c in node.children.values())
        sqrt_n = (total + 1) ** 0.5
        best, best_score, best_move = None, -1e9, None
        for move, child in node.children.items():
            q = child.value_sum / child.visits if child.visits else 0.0
            score = q + self.c_puct * child.prior * sqrt_n / (1 + child.visits)
            if score > best_score:
                best, best_score, best_move = child, score, move
        # Selection increments the visit and applies virtual loss so other
        # batch members are steered elsewhere until backup settles the value.
        best.visits += 1
        best.value_sum -= _VIRTUAL_LOSS
        path.append((best_move, best))
        return best

    def _backup(self, path, leaf_value):
        # leaf_value is from the leaf mover's perspective; each edge above
        # stores value from ITS mover's perspective, so the sign alternates.
        value = -leaf_value
        for _, node in reversed(path):
            node.value_sum += value + _VIRTUAL_LOSS  # undo virtual loss, add result
            value = -value

    def _undo_virtual(self, path):
        for _, node in path:
            node.visits -= 1
            node.value_sum += _VIRTUAL_LOSS
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv\Scripts\python -m pytest tests/test_mcts.py -v`
Expected: 3 passed. If mate-in-one fails, debug the sign convention in `_backup` first — that is the classic bug here.

- [ ] **Step 5: Commit**

```bash
git add submission/mcts.py tests/test_mcts.py
git commit -m "feat: batched PUCT search over network priors and values"
```

---

### Task 12: Time management + search integration

**Files:**
- Modify: `submission/agent.py` (`_choose`, add `_move_budget`, module-level `_SEARCHER`)
- Test: `tests/test_time_management.py`

**Interfaces:**
- Produces: `agent._move_budget(time_left_ms) -> float` (seconds; 0.0 means policy-only); `get_move` uses MCTS when budget > 0, single policy call otherwise. Behavior contract: `time_left_ms < 10_000` → policy only; budget never exceeds `min(t/30 + 0.35, t/4, 4.0)` seconds.

- [ ] **Step 1: Write the failing tests**

`tests/test_time_management.py`:
```python
import time

import chess

import agent


def test_budget_shape():
    assert agent._move_budget(5_000) == 0.0
    assert agent._move_budget(9_999) == 0.0
    b = agent._move_budget(120_000)
    assert 3.0 < b <= 4.0
    assert agent._move_budget(20_000) <= 5.0


def test_uses_search_and_respects_wall_clock():
    board = chess.Board()
    start = time.perf_counter()
    uci = agent.get_move(board.fen(), 30_000)
    elapsed = time.perf_counter() - start
    assert chess.Move.from_uci(uci) in board.legal_moves
    assert elapsed < agent._move_budget(30_000) + 1.0


def test_low_clock_stays_instant():
    board = chess.Board()
    start = time.perf_counter()
    agent.get_move(board.fen(), 3_000)
    assert time.perf_counter() - start < 0.4


def test_search_path_legality_fuzz():
    # Small fuzz THROUGH the search (12s clock -> ~0.75s budget per call).
    import random

    for seed in range(10):
        rng = random.Random(seed)
        board = chess.Board()
        for _ in range(seed * 11 % 80):
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
        if board.is_game_over():
            continue
        uci = agent.get_move(board.fen(), 12_000)
        assert chess.Move.from_uci(uci) in board.legal_moves
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv\Scripts\python -m pytest tests/test_time_management.py -v`
Expected: FAIL (`_move_budget` missing)

- [ ] **Step 3: Modify `submission/agent.py`**

Add after the `_SESSION` setup and `evaluate`/`policy_move` definitions (mcts import goes at the top with the others: `import mcts`):

```python
_SEARCHER = mcts.Searcher(evaluate, c_puct=1.5, batch_size=16)


def _move_budget(time_left_ms: int) -> float:
    """Seconds to spend on this move. 0.0 = single policy call.

    Conservative by design: flagging loses outright, a slightly shallower
    search only costs a little strength."""
    t = time_left_ms / 1000.0
    if t < 10.0:
        return 0.0
    return min(t / 30.0 + 0.35, t / 4.0, 4.0)


def _choose(board, time_left_ms):
    budget = _move_budget(time_left_ms)
    if budget <= 0.0:
        return policy_move(board)
    return _SEARCHER.search(board, time.perf_counter() + budget)
```

(Remove the old `_choose` stub from Task 6.)

- [ ] **Step 4: Run the full suite**

Run: `.venv\Scripts\python -m pytest -v`
Expected: all passed. (Task 6's big fuzz stays on the fast policy-only path by design; `test_search_path_legality_fuzz` covers the search path.)

- [ ] **Step 5: Commit**

```bash
git add submission/agent.py tests/test_time_management.py
git commit -m "feat: time-managed search with low-clock degradation"
```

---

### Task 13: Arena (strength measurement)

**Files:**
- Create: `eval/arena.py`

**Interfaces:**
- Consumes: `agent.evaluate`, `agent.policy_move`, `mcts.Searcher`
- Produces: `python eval/arena.py --a mcts --b policy --games 30 [--movetime 1.0]` printing `A wins/draws/losses` with colors alternating. Players: `mcts` (searcher with fixed movetime), `policy` (single call), `material` (depth-1 material count with random tie-break — baseline only, lives in eval/, never in submission/).

- [ ] **Step 1: Implement `eval/arena.py`**

```python
import argparse
import os
import random
import sys
import time

import chess

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "submission"))
import agent  # noqa: E402
import mcts  # noqa: E402

_VALS = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3, chess.ROOK: 5, chess.QUEEN: 9}


def material_player(rng):
    def choose(board, movetime):
        def score(m):
            b = board.copy(stack=False)
            b.push(m)
            s = sum(v * (len(b.pieces(p, board.turn)) - len(b.pieces(p, not board.turn)))
                    for p, v in _VALS.items())
            return s + rng.random() * 0.1
        return max(board.legal_moves, key=score)
    return choose


def policy_player(board, movetime):
    return agent.policy_move(board)


def mcts_player(board, movetime):
    return mcts.Searcher(agent.evaluate).search(board, time.perf_counter() + movetime)


PLAYERS = {"policy": lambda rng: policy_player,
           "mcts": lambda rng: mcts_player,
           "material": material_player}


def play(white, black, movetime, max_plies=300):
    board = chess.Board()
    while not board.is_game_over(claim_draw=True) and board.ply() < max_plies:
        mover = white if board.turn == chess.WHITE else black
        board.push(mover(board, movetime))
    outcome = board.outcome(claim_draw=True)
    if outcome is None or outcome.winner is None:
        return 0.5
    return 1.0 if outcome.winner == chess.WHITE else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, choices=PLAYERS)
    ap.add_argument("--b", required=True, choices=PLAYERS)
    ap.add_argument("--games", type=int, default=30)
    ap.add_argument("--movetime", type=float, default=1.0)
    args = ap.parse_args()
    w = d = losses = 0
    for g in range(args.games):
        rng = random.Random(g)
        pa, pb = PLAYERS[args.a](rng), PLAYERS[args.b](rng)
        s = play(pa, pb, args.movetime) if g % 2 == 0 else 1 - play(pb, pa, args.movetime)
        w += s == 1.0
        d += s == 0.5
        losses += s == 0.0
        print(f"game {g + 1}: {args.a} {'WDL'[int(2 - 2 * s)]}  running {w}-{d}-{losses}")
    print(f"{args.a} vs {args.b}: +{w} ={d} -{losses}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run baselines and record**

```powershell
.venv\Scripts\python eval/arena.py --a policy --b material --games 20
.venv\Scripts\python eval/arena.py --a mcts --b policy --games 20 --movetime 1.0
```
Expected with the trained net: policy crushes material (≥ 90%); mcts beats policy clearly (≥ 65%). If mcts does NOT beat policy, the search has a bug or too few visits — stop and debug before shipping search (bench from Task 10 tells you visits/move).

- [ ] **Step 3: Commit**

```bash
git add eval/arena.py
git commit -m "feat: arena for measuring checkpoint strength"
```

---

### Task 14: Final package, full-game simulation, submission checklist

**Files:**
- Modify: none (verification only) — plus any README polish

**Interfaces:** none — terminal task.

- [ ] **Step 1: Full verification**

```powershell
.venv\Scripts\python -m pytest -v
.venv\Scripts\python scripts/package.py
.venv\Scripts\python eval/protocol_sim.py --zip dist/submission.zip --plies 200
```
Expected: all tests pass; a full game plays out with no illegal move, no flag, no reply over 4096 bytes. Note peak observed move times.

- [ ] **Step 2: Memory + compliance audit (read, don't assume)**

- Confirm 2 GB is safe: model (~15 MB) + onnxruntime + python-chess tree ≈ well under; spot-check with `Get-Process` RSS while the sim runs.
- Re-read every file in `submission/` top to bottom as a judge would: plain code, honest comments, module docstring stating the training provenance (human games, no engines). No dead code, no obfuscation.
- Confirm the ZIP contains ONLY: `agent.py`, `encoding.py`, `mcts.py`, `model.onnx`. No `requirements.txt` needed (only preinstalled libs are used).

- [ ] **Step 3: Commit and hand to the user for upload**

```bash
git add -A && git commit -m "chore: final submission package"
```
The upload itself is the user's action (account credentials, 6/day rate limit). Present: zip path, size, arena results, and the reminder that uploads close 11 September 11:00.

---

## Post-plan improvement loop (until submission lock)

Not tasks — the iteration menu once the base agent is live, in rough order of expected Elo per effort:
1. More data / more epochs (watch validation top-1).
2. Wider net if the Task 10 benchmark shows headroom; narrower if search visits are too few.
3. `c_puct`, `batch_size`, budget-curve tuning via arena self-play.
4. Root Dirichlet-free determinism vs. tiny root temperature for ladder variety — measure, don't guess.
Every candidate ZIP goes through `protocol_sim.py` before an upload slot is spent.
