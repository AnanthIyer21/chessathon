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
