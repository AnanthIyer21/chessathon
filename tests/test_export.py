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
