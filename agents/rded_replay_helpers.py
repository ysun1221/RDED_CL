# agents/rded_replay_helpers.py
from __future__ import annotations
from typing import Optional, Tuple
import torch
from torch.utils.data import ConcatDataset, TensorDataset, DataLoader

from mammoth.models.er import Er
from mammoth.utils.buffer import Buffer
from cl_utils.rded_adapter import RDEDDistiller


# convert mammoth buffer to tensor dataset
def _buffer_as_dataset(buf: Buffer) -> TensorDataset:

    for xs_name, ys_name in [
        ("x", "y"),
        ("xs", "ys"),
        ("data_x", "data_y"),
        ("examples", "labels"),  # mammoth Buffer fields
    ]:
        if hasattr(buf, xs_name) and hasattr(buf, ys_name):
            xs = getattr(buf, xs_name)
            ys = getattr(buf, ys_name)
            if torch.is_tensor(xs) and torch.is_tensor(ys):
                return TensorDataset(xs.detach().cpu(), ys.detach().cpu())

    for storage_name in ["storage", "examples", "items", "memory"]:
        if hasattr(buf, storage_name):
            storage = getattr(buf, storage_name)
            if isinstance(storage, (list, tuple)) and len(storage) > 0:
                xs, ys = zip(*storage)
                X = torch.stack([x.detach().cpu() for x in xs], dim=0)
                Y = torch.tensor([int(y) for y in ys], dtype=torch.long)
                return TensorDataset(X, Y)

    if hasattr(buf, "get_all") and callable(buf.get_all):
        X, Y = buf.get_all()
        return TensorDataset(X.detach().cpu(), Y.detach().cpu())

    get_all_data = getattr(buf, "get_all_data", None)
    if callable(get_all_data):
        data = get_all_data()
        if isinstance(data, (list, tuple)) and len(data) >= 2:
            X, Y = data[0], data[1]
            if torch.is_tensor(X) and torch.is_tensor(Y):
                return TensorDataset(X.detach().cpu(), Y.detach().cpu())

    raise RuntimeError("Could not convert Buffer to dataset; adjust adapters to your Buffer API.")


def _buffer_clear(buf: Buffer) -> None:
    for m in ["clear", "reset", "empty", "flush"]:
        if hasattr(buf, m) and callable(getattr(buf, m)):
            getattr(buf, m)()
            return
    for name in ["x", "y", "xs", "ys", "data_x", "data_y", "storage", "examples", "items", "memory"]:
        if hasattr(buf, name):
            attr = getattr(buf, name)
            if isinstance(attr, list):
                attr.clear()
            elif torch.is_tensor(attr):
                setattr(buf, name, attr.new_zeros((0, *attr.shape[1:])))


def _buffer_add_dataset(buf: Buffer, ds: TensorDataset, batch_size: int = 512) -> None:
    """
    Add a TensorDataset back into the Buffer. Tries common APIs, falls back to mutating fields.
    """
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    add_data = getattr(buf, "add_data", None)
    if callable(add_data):
        for xb, yb in loader:
            add_data(xb, yb)
        return

    add = getattr(buf, "add", None)
    if callable(add):
        for xb, yb in loader:
            add(xb, yb)
        return

    if hasattr(buf, "x") and hasattr(buf, "y") and torch.is_tensor(getattr(buf, "x")):
        xs, ys = [], []
        for xb, yb in loader:
            xs.append(xb); ys.append(yb)
        X = torch.cat(xs, 0)
        Y = torch.cat(ys, 0)
        if getattr(buf, "x").numel() == 0:
            buf.x = X; buf.y = Y
        else:
            buf.x = torch.cat([buf.x, X], 0)
            buf.y = torch.cat([buf.y, Y], 0)
        return

    raise RuntimeError("Could not insert data into Buffer; adjust adapters to your Buffer API.")


def _bytes_per_image(example_tensor: torch.Tensor) -> int:
    C, H, W = example_tensor.shape
    return example_tensor.element_size() * C * H * W


def rded_refresh_buffer(
    er_model: Er,
    train_dataset,
    budget_mb: int,
    rded_cfg: dict,
    rded_root: str,
    bridge_rel: str = "scripts/rded_bridge.py",
    device: Optional[str] = None,
) -> Tuple[int, int, int]:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    try:
        buffer_ds = _buffer_as_dataset(er_model.buffer)
        union = ConcatDataset([train_dataset, buffer_ds])
    except Exception:
        union = train_dataset

    x0, y0 = union[0]
    bpi = _bytes_per_image(x0)
    try:
        n_classes = int(max([y for _, y in union])) + 1
    except Exception:
        n_classes = getattr(train_dataset, "n_classes", None) or getattr(train_dataset, "num_classes", None)
        if n_classes is None:
            raise RuntimeError("Could not infer number of classes from dataset; set it explicitly.")

    max_imgs = (budget_mb * 1024 * 1024) // bpi
    per_class = max(1, max_imgs // n_classes)

    distiller = RDEDDistiller(cfg=rded_cfg, rded_root=rded_root, main_py_rel=bridge_rel)
    distilled = distiller.distill(union, per_class=per_class, device=device)

    _buffer_clear(er_model.buffer)
    _buffer_add_dataset(er_model.buffer, distilled)

    return per_class, n_classes, len(distilled)