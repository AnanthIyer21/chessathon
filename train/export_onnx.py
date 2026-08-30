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
