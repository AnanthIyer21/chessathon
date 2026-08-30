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
