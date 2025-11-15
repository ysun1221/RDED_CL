import argparse
import json
from pathlib import Path
import sys
import torch


def parse_args():
    p = argparse.ArgumentParser(description="RDED bridge CLI (standalone stub)")
    p.add_argument("--inputs", required=True, help="Path to input tensor .pt (float BCHW)")
    p.add_argument("--labels", required=True, help="Path to labels tensor .pt (long [N])")
    p.add_argument("--per-class", type=int, required=True, help="Target samples per class")
    p.add_argument("--cfg-json", required=True, help="Path to minimal RDED cfg JSON")
    p.add_argument("--out-x", required=True, help="Path to save distilled images tensor")
    p.add_argument("--out-y", required=True, help="Path to save distilled labels tensor")
    p.add_argument("--rded-root", default="", help="Path to upstream RDED repo root (optional)")
    return p.parse_args()


def main():
    args = parse_args()

    x = torch.load(args.inputs, map_location="cpu")
    y = torch.load(args.labels, map_location="cpu")

    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise TypeError("inputs/labels must be saved torch.Tensors")
    if x.dim() != 4 or y.dim() != 1 or x.shape[0] != y.shape[0]:
        raise ValueError(f"Bad shapes: x={tuple(x.shape)} y={tuple(y.shape)} (need [N,C,H,W] and [N])")

    cfg_path = Path(args.cfg_json)
    cfg = json.loads(cfg_path.read_text())
    seed = int(cfg.get("seed", 1))

    g = torch.Generator().manual_seed(seed)
    m = max(int(args.per_class), 1)

    classes = torch.unique(y).tolist()
    x_s_list, y_s_list = [], []
    for c in classes:
        idx = (y == c).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        take = idx[torch.randperm(idx.numel(), generator=g)[:min(m, idx.numel())]]
        x_s_list.append(x[take])
        y_s_list.append(y[take])

    if x_s_list:
        x_s = torch.cat(x_s_list, dim=0)
        y_s = torch.cat(y_s_list, dim=0)
    else:
        k = min(m, x.shape[0])
        take = torch.randperm(x.shape[0], generator=g)[:k]
        x_s, y_s = x[take], y[take]

    torch.save(x_s, args.out_x)
    torch.save(y_s, args.out_y)


if __name__ == "__main__":
    main()
