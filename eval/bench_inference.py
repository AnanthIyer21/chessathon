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
