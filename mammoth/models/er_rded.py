# mammoth/models/er_rded.py
from __future__ import annotations
import os
import sys
from typing import Any, Dict, Optional, List

import torch
from torch.utils.data import DataLoader, Dataset

from .er import Er
from utils.buffer import Buffer

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cl_utils.rded_adapter import RDEDDistiller

@staticmethod
def _extract_label(item):
    """Return int label from a dataset item that can be tuple/list/dict."""
    import torch
    if isinstance(item, (tuple, list)):
        if len(item) >= 2:
            y = item[1]
        else:
            raise ValueError("Tuple/List item has <2 elements.")
    elif isinstance(item, dict):
        for k in ("target", "label", "y"):
            if k in item:
                y = item[k]
                break
        else:
            raise ValueError("Dict item missing target/label/y.")
    else:
        raise ValueError(f"Unsupported item type: {type(item)}")

    if isinstance(y, torch.Tensor):
        y = int(y.item())
    else:
        y = int(y)
    return y

@staticmethod
def _infer_task_labels(ds, cap=5000):
    """Collect a set of labels present in this task (scan up to `cap` samples)."""
    labels = set()
    n = len(ds) if hasattr(ds, "__len__") else cap
    n = min(n, cap)
    for i in range(n):
        try:
            item = ds[i]
            labels.add(_extract_label(item))
        except StopIteration:
            break
        except Exception:
            continue
    if not labels:
        for attr in ("classes", "targets", "labels"):
            if hasattr(ds, attr):
                seq = getattr(ds, attr)
                try:
                    return set(int(v) for v in list(seq))
                except Exception:
                    pass
    return labels


class ErRDED(Er):
    NAME = "er_rded"

    @staticmethod
    def get_parser(parser):
        parser = Er.get_parser(parser)

        g = parser.add_argument_group("RDED")
        g.add_argument("--rded_root", type=str, required=True,
                       help="Absolute path to your RDED_CL (project root) or RDED repo root.")
        g.add_argument("--rded_bridge_rel", type=str, default="scripts/rded_bridge.py",
                       help="Path to bridge script (relative to --rded_root).")
        g.add_argument("--rded_per_class", type=int, default=10,
                       help="Synthetic images per class produced *per task* before merging.")
        g.add_argument("--rded_iters", type=int, default=500)
        g.add_argument("--rded_lr", type=float, default=0.1)
        g.add_argument("--rded_augment", type=int, default=1)
        g.add_argument("--rded_seed", type=int, default=1)
        g.add_argument("--rded_gather_bs", type=int, default=256,
                       help="Batch size used by adapter to gather raw dataset into RAM.")
        g.add_argument("--rded_gather_workers", type=int, default=0,
                       help="Windows-safe: keep 0 on Windows; >0 ok on Linux.")
        return parser

    def __init__(self, backbone, loss, args, transform, dataset=None):
        super().__init__(backbone, loss, args, transform, dataset=dataset)

        assert isinstance(self.buffer, Buffer), "ER should provide a Buffer instance."

        self._buf_limit: int = int(self.args.buffer_size)

        self._rded_cfg: Dict[str, Any] = dict(
            iters=int(args.rded_iters),
            lr=float(args.rded_lr),
            augment=bool(args.rded_augment),
            seed=int(args.rded_seed),
            gather_bs=int(args.rded_gather_bs),
            gather_workers=int(args.rded_gather_workers),
        )
        self._rded = RDEDDistiller(
            cfg=self._rded_cfg,
            rded_root=args.rded_root,
            main_py_rel=args.rded_bridge_rel,
            backend=None,
        )
        self._rded_per_class: int = int(args.rded_per_class)

        self._bank_x: Optional[torch.Tensor] = None
        self._bank_y: Optional[torch.Tensor] = None

    def _extract_indexable_train_ds(self, dataset) -> Dataset:
        
        if hasattr(dataset, "train_loader") and dataset.train_loader is not None:
            tl = dataset.train_loader
            if isinstance(tl, (list, tuple)):
                task_idx = getattr(self, "current_task", 0)
                task_idx = max(0, min(task_idx, len(tl) - 1))
                tl = tl[task_idx]
            cand = getattr(tl, "dataset", None)
            if cand is not None and hasattr(cand, "__len__"):
                return cand

        cand = getattr(dataset, "train_dataset", None)
        if cand is not None:
            base = getattr(cand, "dataset", cand)
            if hasattr(base, "__len__"):
                return base

        raise RuntimeError(
            "ErRDED: could not extract an indexable train dataset for the completed task. "
            "update _extract_indexable_train_ds to match dataset structure."
        )

    def _rebalance_to_budget(
        self,
        bank_x: torch.Tensor,
        bank_y: torch.Tensor,
        device: torch.device,
        seed_offset: int = 0,
    ) -> torch.Tuple[torch.Tensor, torch.Tensor]:
        classes: List[int] = torch.unique(bank_y).tolist()
        m_per = max(self._buf_limit // max(len(classes), 1), 1)

        g = torch.Generator(device=device).manual_seed(int(self.args.seed) + int(seed_offset))

        keep_x, keep_y = [], []
        for c in classes:
            idx = (bank_y == c).nonzero(as_tuple=True)[0]
            if idx.numel() <= m_per:
                take = idx
            else:
                perm = torch.randperm(idx.numel(), generator=g, device=device)
                take = idx[perm[:m_per]]
            keep_x.append(bank_x[take])
            keep_y.append(bank_y[take])

        new_x = torch.cat(keep_x, dim=0) if keep_x else bank_x.new_zeros((0, *bank_x.shape[1:]))
        new_y = torch.cat(keep_y, dim=0) if keep_y else bank_y.new_zeros((0,), dtype=bank_y.dtype)

        if new_x.shape[0] > self._buf_limit:
            perm = torch.randperm(new_x.shape[0], generator=g, device=device)[: self._buf_limit]
            new_x, new_y = new_x[perm], new_y[perm]

        return new_x, new_y

    def end_task(self, dataset) -> None:
        device = next(self.net.parameters()).device

        if hasattr(dataset, "train_loader") and getattr(dataset.train_loader, "dataset", None) is not None:
            raw_ds = dataset.train_loader.dataset
        elif hasattr(dataset, "train_dataloader") and getattr(dataset.train_dataloader, "dataset", None) is not None:
            raw_ds = dataset.train_dataloader.dataset
        elif hasattr(dataset, "train_dataset"):
            raw_ds = dataset.train_dataset
        else:
            raw_ds = dataset

        self._rded.cfg["gather_limit"] = self._rded.cfg.get("gather_limit", None)

        synth_ds = self._rded.distill(raw_ds, per_class=self._rded_per_class, device=str(device))
        xs = torch.stack([synth_ds[i][0] for i in range(len(synth_ds))], dim=0).to(device)
        ys = torch.tensor([int(synth_ds[i][1]) for i in range(len(synth_ds))], device=device)

        self.buffer.empty()
        self.buffer.add_data(examples=xs, labels=ys)

        super().end_task(dataset)

