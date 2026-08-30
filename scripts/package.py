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
