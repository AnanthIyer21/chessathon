# Benchmarks behind docs/research-strength-path.md

- `nnue_bench.py`: cost per node of a 768 -> Hx2 -> 1 int16 net in numba (copy-make
  accumulator, incremental update, squared-ReLU output, full refresh), for H in 64..512.
- `train_bench.py`: PyTorch training throughput for the same net on CPU and MPS.
- `nps_bench.py <agent dir> <label>`: nodes per second of an agent over four positions at a
  72 s clock. Run from the repo root with `.venv/bin/python`.
