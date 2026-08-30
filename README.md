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
