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
# customized mammoth model, wires experience replay model with rded
# at task bounds


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
    """Infer which labels are present in dataset for the current task,
        Collect a set of labels present in this task."""
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

# extend er model
class ErRDED(Er):
    NAME = "er_rded"

    @staticmethod
    def get_parser(parser):
        """reuse ER argument and add rded argument"""
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
        """parameter:
            backbone: the neural network model
            loss: loss function
            args: parsed cli arguments
            transform: data augmentation/preprocess transform
            dataset: dataset
        """
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
        """extract an indexable train dataset from the dataset object that mammoth passes into endtask"""
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
        """rebalance a big bank of samples (bank_x , bank_y) to fit the replay buffer
            budget while preserving a roughly uniform perclass allocation"""
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

    def _seen_classes(self, dataset, device):
        """Return list of class ids present in the just-seen task dataset."""
        loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
        ys = []

        with torch.no_grad():
            for batch in loader:
                # case 1: tuple/list (x, y) or (x, y, ...)
                if isinstance(batch, (list, tuple)):
                    if len(batch) >= 2:
                        y = batch[1]
                    else:
                        # single element; try dict inside
                        item = batch[0]
                        if isinstance(item, dict):
                            y = item.get("targets") or item.get("labels") or item.get("y")
                        else:
                            continue
                # case 2: dict batch
                elif isinstance(batch, dict):
                    y = batch.get("targets") or batch.get("labels") or batch.get("y")
                else:
                    # unknown structure; skip
                    continue

                if y is None:
                    continue

                # ensure tensor on same device
                if not torch.is_tensor(y):
                    try:
                        y = torch.as_tensor(y)
                    except Exception:
                        continue

                ys.append(y.to(device))

                # safety cap (avoid scanning huge datasets)
                if sum(t.numel() for t in ys) >= 50000:
                    break

        if not ys:
            return []

        y_all = torch.cat(ys)
        return sorted(torch.unique(y_all).tolist())

    def _merge_into_buffer(self, new_x, new_y, device):
        """Merge distilled samples with current buffer, class-balanced to buffer_size."""
        max_n = int(self.args.buffer_size)

        # current contents
        old_x = getattr(self.buffer, "examples", None)
        old_y = getattr(self.buffer, "labels", None)
        if old_x is None or old_y is None or old_x.numel() == 0:
            self.buffer.empty()
            self.buffer.add_data(examples=new_x, labels=new_y)
            return

        X = torch.cat([old_x.to(device), new_x], dim=0)
        Y = torch.cat([old_y.to(device), new_y], dim=0)

        classes = torch.unique(Y).tolist()
        per_class = max(1, max_n // max(1, len(classes)))

        keep_idx = []
        for c in classes:
            idx = (Y == c).nonzero(as_tuple=True)[0]
            if idx.numel() > per_class:
                idx = idx[torch.randperm(idx.numel(), device=idx.device)[:per_class]]
            keep_idx.append(idx)
        keep_idx = torch.cat(keep_idx)
        if keep_idx.numel() > max_n:
            keep_idx = keep_idx[torch.randperm(keep_idx.numel(), device=keep_idx.device)[:max_n]]

        X, Y = X[keep_idx], Y[keep_idx]

        self.buffer.empty()
        self.buffer.add_data(examples=X, labels=Y)

    def end_task(self, dataset) -> None:
        """ At each task boundary:
            estimate classes seen so far,
            distill current-task data with RDED (per-class target based on seen classes),
            merge distilled samples with existing buffer (class-balanced)."""
        device = next(self.net.parameters()).device

        raw_ds = getattr(dataset, "train_dataset", None)
        if raw_ds is None:
            raw_ds = getattr(dataset, "train_loader", None)
            raw_ds = getattr(raw_ds, "dataset", None) if raw_ds is not None else None
        if raw_ds is None:
            raw_ds = dataset

        seen = self._seen_classes(raw_ds, device)
        n_seen = max(1, len(seen))

        per_class_target = max(
            1,
            min(int(self.args.buffer_size) // n_seen, int(self._rded_per_class))
        )

        synth_ds = self._rded.distill(raw_ds, per_class=per_class_target, device=str(device))

        xs = torch.stack([synth_ds[i][0] for i in range(len(synth_ds))], dim=0).to(device)
        ys = torch.tensor([int(synth_ds[i][1]) for i in range(len(synth_ds))], device=device)

        self._merge_into_buffer(xs, ys, device)

        with torch.no_grad():
            ex = getattr(self.buffer, "examples", None)
            lab = getattr(self.buffer, "labels", None)
            if ex is not None and lab is not None and lab.numel() > 0:
                uniq = torch.unique(lab)
                counts = {int(c): int((lab == c).sum()) for c in uniq}
                print(f"[ErRDED] buffer size: {len(lab)} | classes: {len(counts)} "
                    f"| per-class min/max: "
                    f"{min(counts.values())}/{max(counts.values())}")

        super().end_task(dataset)