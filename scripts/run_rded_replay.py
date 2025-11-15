from __future__ import annotations

import sys, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MAM = ROOT / "mammoth"
sys.path.insert(0, str(MAM))
from mammoth.models.er import Er
from mammoth.utils.buffer import Buffer
sys.path.pop(0) 

import os
import argparse
import random
from typing import List, Dict, Any

import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from agents.rded_replay_agent import RDEDReplayAgent

def set_seed(seed: int = 1):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(ds: Dataset, bs: int, workers: int = 0, shuffle: bool = True) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
    )


def _resolve_path(p: str) -> pathlib.Path:
    pp = pathlib.Path(p)
    return pp if pp.is_absolute() else (ROOT / pp)


class TinyNet(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 256), nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)



def build_split_cifar10_tasks(num_tasks: int = 5, train: bool = True) -> List[Dataset]:
    try:
        from torchvision import datasets, transforms
    except Exception as e:
        raise RuntimeError("torchvision is required for the Split-CIFAR10 fallback.") from e

    transform = transforms.Compose([transforms.ToTensor()])
    cifar = datasets.CIFAR10(root=str(ROOT / "data"), train=train, download=True, transform=transform)

    # group indices by class
    indices_by_class: Dict[int, List[int]] = {c: [] for c in range(10)}
    for idx, (_, y) in enumerate(cifar):
        indices_by_class[int(y)].append(idx)

    classes_per_task = 10 // num_tasks
    assert classes_per_task == 2,

    tasks: List[Subset] = []
    for t in range(num_tasks):
        cls_start, cls_end = t * classes_per_task, (t + 1) * classes_per_task
        chosen = []
        for c in range(cls_start, cls_end):
            chosen.extend(indices_by_class[c])
        tasks.append(Subset(cifar, chosen))
    return tasks

# return list of CIFAR100 subsets, equally split classes per task
def build_split_cifar100_tasks(num_tasks: int = 10, train: bool = True) -> List[Dataset]:
    try:
        from torchvision import datasets, transforms
    except Exception as e:
        raise RuntimeError("torchvision is required for the Split-CIFAR100 fallback.") from e

    transform = transforms.Compose([transforms.ToTensor()])
    cifar = datasets.CIFAR100(root=str(ROOT / "data"), train=train, download=True, transform=transform)

    # collect indices by class
    from collections import defaultdict
    idx_by_cls: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(cifar)):
        _, y = cifar[i]
        idx_by_cls[int(y)].append(i)

    classes = list(range(100))
    classes_per_task = 100 // num_tasks
    assert classes_per_task * num_tasks == 100, "num_tasks must divide 100."

    tasks: List[Subset] = []
    for t in range(num_tasks):
        cls_start, cls_end = t * classes_per_task, (t + 1) * classes_per_task
        chosen = []
        for c in classes[cls_start:cls_end]:
            chosen.extend(idx_by_cls[c])
        tasks.append(Subset(cifar, chosen))
    return tasks


def infer_num_classes_from_dataset(ds: Dataset) -> int:
    for attr in ("num_classes", "n_classes", "classes"):
        if hasattr(ds, attr):
            val = getattr(ds, attr)
            if isinstance(val, int):
                return val
            if isinstance(val, (list, tuple)):
                return len(val)
    labels = []
    for i in range(min(len(ds), 5000)):
        _, y = ds[i]
        labels.append(int(y))
    return max(labels) + 1 if labels else 10

def train_one_task(
    er,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    epochs: int,
    device: str,
) -> None:
    model.train()
    criterion = nn.CrossEntropyLoss()
    for _ in range(epochs):
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            if hasattr(er, "model"):
                logits = er.model(xb)
            elif hasattr(er, "backbone"):
                logits = er.backbone(xb)
            else:
                logits = model(xb)
            loss = criterion(logits, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()


def validate(model: nn.Module, val_loader: DataLoader, device: str) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            total += yb.numel()
    return correct / max(1, total)


def _build_er_via_args(backbone: nn.Module, buffer_size: int, n_classes: int, device: str):
    """
    Build Er for Mammoth variants with signature:
        Er(backbone, loss, args, transform, dataset=None)

    We synthesize a minimal `args` and a no-op `transform`.
    """
    from types import SimpleNamespace

    args = SimpleNamespace(
        buffer_size=buffer_size,
        n_classes=n_classes,
        device=device,
        lr=0.1,
        momentum=0.9,
        batch_size=64,
    )

    def identity_transform(batch):
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]
        if isinstance(batch, dict) and "x" in batch and "y" in batch:
            return batch["x"], batch["y"]
        return batch

    loss = torch.nn.CrossEntropyLoss()
    return Er(backbone, loss, args, identity_transform, dataset=None)


def main(cfg_path: str):
    cfg_file = _resolve_path(cfg_path).resolve()
    with open(cfg_file, "r") as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["training"]["seed"])
    epochs = int(cfg["training"]["epochs_per_task"])
    batch_sz = int(cfg["training"]["batch_size"])
    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    name = cfg.get("scenario", {}).get("name", "split_cifar10").lower()
    if name in ("split_cifar10", "split-cifar10", "cifar10"):
        train_tasks = build_split_cifar10_tasks(num_tasks=5,  train=True)
        val_tasks   = build_split_cifar10_tasks(num_tasks=5,  train=False)
    elif name in ("split_cifar100", "split-cifar-100", "cifar100"):
        train_tasks = build_split_cifar100_tasks(num_tasks=10, train=True)
        val_tasks   = build_split_cifar100_tasks(num_tasks=10, train=False)
    else:
        raise RuntimeError(f"Scenario '{name}' not supported here. Swap in your Mammoth scenario builder.")

    n_classes = infer_num_classes_from_dataset(train_tasks[0])
    model = TinyNet(num_classes=n_classes).to(device)

    opt_cfg = cfg.get("optimizer", {"name": "sgd", "args": {"lr": 0.1, "momentum": 0.9}})
    lr = opt_cfg.get("args", {}).get("lr", 0.1)
    momentum = opt_cfg.get("args", {}).get("momentum", 0.9)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    agent_args = dict(cfg["agent"]["args"])
    if "rded_cfg_from" in agent_args:
        rded_cfg_file = _resolve_path(agent_args.pop("rded_cfg_from")).resolve()
        with open(rded_cfg_file, "r") as f:
            rded_cfg = yaml.safe_load(f)["rded"]
    else:
        rded_cfg = agent_args.pop("rded_cfg")
    budget_mb = int(agent_args["budget_mb"])

    x0, _ = train_tasks[0][0]
    bytes_per_img = x0.element_size() * x0.numel()
    budget_bytes = budget_mb * 1024 * 1024
    buffer_size = max(1, budget_bytes // bytes_per_img)

    er = _build_er_via_args(model, buffer_size, n_classes, device)

    rded_agent = RDEDReplayAgent(
        er_model=er,
        budget_mb=budget_mb,
        rded_cfg=rded_cfg,
        rded_root=str(ROOT),
        bridge_rel="scripts/rded_bridge.py",
        device=device,
        logger=None,
    )

    gather_workers = int(rded_cfg.get("gather_workers", 0))

    for task_id, (train_ds, val_ds) in enumerate(zip(train_tasks, val_tasks)):
        print(f"\n=== Task {task_id} ===")
        train_loader = build_loader(train_ds, batch_sz, workers=gather_workers, shuffle=True)
        val_loader   = build_loader(val_ds,   batch_sz, workers=0, shuffle=False)

        train_one_task(er, model, optimizer, train_loader, epochs, device)
        acc = validate(model, val_loader, device)
        print(f"  val acc: {acc:.3f}")

        rded_agent.after_task(task_id=task_id, train_dataset=train_ds)

    print("\nAll tasks finished.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/rded_replay.yaml")
    args = ap.parse_args()
    main(args.cfg)
