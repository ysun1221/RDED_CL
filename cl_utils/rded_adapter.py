from __future__ import annotations
from typing import Any, Callable, Dict, Optional, Tuple
import os, sys, tempfile, subprocess, json, importlib
import torch
from torch.utils.data import DataLoader, TensorDataset, IterableDataset
from pathlib import Path

class RDEDDistiller:
    # cfg: dict of rded hyperparameters
    # backend: optional python callable implementing the rded entry point
    # gather: optional override for the batch size when loading original dataset into memory
    # rded root: optional absolute path to rded repo
    # main py rel: relative path to main.py inside rded root
    def __init__(self, cfg: Dict[str, Any], backend: Optional[Callable] = None,
                 gather_bs: Optional[int] = None, rded_root: Optional[str]=None, main_py_rel: str="main.py"):
        self.cfg = dict(cfg or {})
        if gather_bs is not None:
            self.cfg["gather_bs"] = gather_bs
        self.rded_root = Path(
            rded_root or os.environ.get("RDED_ROOT", "")
        ).resolve() if (rded_root or os.environ.get("RDED_ROOT")) else None
        self.main_py_rel = main_py_rel
        self.backend = backend or self._resolve_backend()

    @torch.no_grad()
    # distill dataset into small class balanced tensordataset
    # dataset: any pytorch dataset, dataset[i] -> (image_tensor, label)
    # per_class: target # of synthetic images per class
    # device: cuda, cpu or none
    # returns tensordataset(images, labels) on cpu, contiguous, ready for replay buffer
    def distill(self, dataset, per_class, device: Optional[str]=None) -> TensorDataset:
        device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        x, y = self._gather(dataset, device)
        if self.backend is not None:
            x_s, y_s = self.backend(x, y, per_class, self.cfg, device)
        else:
            x_s, y_s = self._run_via_subprocess(x, y, per_class,device)
        return TensorDataset(x_s.detach().cpu().contiguous(), y_s.detach().cpu().contiguous())
    
    # load entire dataset into memory efficiently
    def _gather(self, dataset, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    
        if isinstance(dataset, DataLoader):
            loader = dataset

        else:
            loader = None
            for name in ("train_loader", "train_dataloader"):
                if hasattr(dataset, name):
                    cand = getattr(dataset, name)
                    if isinstance(cand, DataLoader):
                        loader = cand
                        break

            if loader is not None and hasattr(loader, "dataset") and hasattr(loader.dataset, "__len__"):
                dataset = loader.dataset
                loader = None

            if loader is None and (isinstance(dataset, IterableDataset) or not hasattr(dataset, "__len__")):
                raise TypeError(
                    "Adapter expects a map-style Dataset or a DataLoader. "
                    "Got an iterable/non-len dataset with no train_loader/train_dataloader to unwrap."
                )   

        if loader is None:
            bs = int(self.cfg.get("gather_bs", 256))
            num_workers = int(self.cfg.get("gather_workers", 0))
            loader = DataLoader(
                dataset,
                batch_size=bs,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=(device.type == "cuda"),
            )

        gather_limit = self.cfg.get("gather_limit", None)
        seen = 0

        xs, ys = [], []
        for batch in loader:
            if isinstance(batch, (list, tuple)) and len(batch) >= 2:
                xb, yb = batch[0], batch[1]
            else:
                raise ValueError("Expected batches like (images, labels).")

            xb = xb.to(device, non_blocking=True)
            if xb.dtype not in (torch.float16, torch.float32, torch.float64):
                xb = xb.float()
            yb = yb.to(device=device, dtype=torch.long, non_blocking=True)

            xs.append(xb)
            ys.append(yb)

            if gather_limit is not None:
                seen += xb.shape[0]
                if seen >= int(gather_limit):
                    break

        x = torch.cat(xs, dim=0)
        y = torch.cat(ys, dim=0)

        if x.dim() != 4:
            raise ValueError(f"Expect images [N,C,H,W], got {tuple(x.shape)}")
        if y.dim() != 1 or y.shape[0] != x.shape[0]:
            raise ValueError("Labels must be [N] and match images")

        return x, y
    

    #
    def _run_via_subprocess(self, x:torch.Tensor, y:torch.Tensor, per_class: int,
                            device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.rded_root is None:
            raise RuntimeError("no rded backend and no root provided, subprocess")
        main_py = (self.rded_root / self.main_py_rel).resolve()
        if not main_py.exists():
            raise FileNotFoundError(f"main.py not found at {main_py}, subprocess")
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            # save input to cpu .pt files 
            in_x = td / "in_x.pt"
            in_y = td / "in_y.pt"
            out_x = td / "out_x.pt"
            out_y = td / "out_y.pt"

            torch.save(x.detach().cpu(), in_x.as_posix())
            torch.save(y.detach().cpu(), in_y.as_posix())

            cfg_min = {
                "iters": int(self.cfg.get("iters", 500)),
                "lr": float(self.cfg.get("lr", 0.1)),
                "augment": bool(self.cfg.get("augment", True)),
                "seed": int(self.cfg.get("seed", 1)),
                "device": str(device),
            }
            cfg_json = (td / "cfg.json")
            cfg_json.write_text(json.dumps(cfg_min))

            # commands
            cmd = [
                sys.executable,
                str(main_py),
                "--inputs", str(in_x),
                "--labels", str(in_y),
                "--per-class", str(int(per_class)),
                "--cfg-json", str(cfg_json),
                "--out-x", str(out_x),
                "--out-y", str(out_y),
                "--rded-root", str(self.rded_root),
            ]

            env = os.environ.copy()
            env["RDED_ROOT"] = str(self.rded_root)
            env["PYTHONPATH"] = f"{env.get('PYTHONPATH','')}{os.pathsep}{str(self.rded_root)}"

            proc = subprocess.run(
                cmd,
                cwd=str(self.rded_root),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                env=env, 
            )
            if proc.returncode != 0:
                raise RuntimeError(f"RDED subprocess failed ({proc.returncode}).\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")

            x_s = torch.load(out_x.as_posix(), map_location=device)
            y_s = torch.load(out_y.as_posix(), map_location=device)
            if not isinstance(x_s, torch.Tensor) or not isinstance(y_s, torch.Tensor):
                raise TypeError("main.py did not produce tensors.")
            if x_s.dim() != 4 or y_s.dim() != 1 or x_s.shape[0] != y_s.shape[0]:
                raise ValueError("main.py must save [M,C,H,W] and [M].")
            return x_s, y_s

    
    def _run_rded(self, x: torch.Tensor, y: torch.Tensor, per_class: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.backend is None:
            raise RuntimeError("rded backend not found, _run_rded")
        # call backend
        x_s, y_s = self.backend(x, y, per_class, self.cfg, device)
        # check output
        if not isinstance(x_s, torch.Tensor) or not isinstance(y_s, torch.Tensor):
            raise TypeError("RDED backend not return (Tensor, Tensor), _run_rded")
        if x_s.dim() != 4 or y_s.dim() != 1 or x_s.shape[0] != y_s.shape[0]:
            raise ValueError("backend not returning shape[m,c,h,w] and [m], _run_rded")
        return x_s, y_s

    
    def _resolve_backend(self) -> Optional[Callable]:
        if self.rded_root is not None:
            r = self.rded_root.as_posix()
            if r not in sys.path:
                sys.path.insert(0, r)

        candidates = [
            # (module path, function name)
            ("synthesize.api", "distill_dataset"),
            ("scripts.api", "distill_dataset"),
            ("api", "distill_dataset"),
            ("runner", "distill"),
            ("main", "distill"), 
        ]
        for path, name in candidates:
            try:
                mod = importlib.import_module(path)
                fn = getattr(mod, name, None)
                if callable(fn):
                    return self._wrap_backend(fn)
            except ModuleNotFoundError:
                continue
        return None
        
    def _wrap_backend(self, fn:Callable) -> Callable:
        def call(x, y, per_class, cfg, device):
            return fn(
                x, y,
                per_class=per_class,
                iters=cfg.get("iters", 500),
                lr=cfg.get("lr", 0.1),
                augment=cfg.get("augment", True),
                seed=cfg.get("seed", 1),
                device=device,
            )
        return call
        
        