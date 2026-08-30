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
