# agents/rded_replay_agent.py
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any
import torch

from mammoth.models.er import Er
from mammoth.utils.buffer import Buffer

from agents.rded_replay_helpers import rded_refresh_buffer


class RDEDReplayAgent:

    def __init__(
        self,
        er_model: Er,
        budget_mb: int,
        rded_cfg: Dict[str, Any],
        rded_root: str,
        bridge_rel: str = "scripts/rded_bridge.py",
        device: Optional[str] = None,
        logger: Optional[Any] = None,
    ):
        self.er: Er = er_model
        self.buffer: Buffer = er_model.buffer
        self.budget_mb: int = int(budget_mb)
        self.rded_cfg: Dict[str, Any] = dict(rded_cfg or {})
        self.rded_root: str = rded_root
        self.bridge_rel: str = bridge_rel
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logger

    def after_task(self, task_id: int, train_dataset) -> Tuple[int, int, int]:
        per_class, n_classes, total = rded_refresh_buffer(
            er_model=self.er,
            train_dataset=train_dataset,
            budget_mb=self.budget_mb,
            rded_cfg=self.rded_cfg,
            rded_root=self.rded_root,
            bridge_rel=self.bridge_rel,
            device=self.device,
        )

        if self.logger is not None:
            payload = {
                "buffer/budget_mb": self.budget_mb,
                "buffer/per_class": per_class,
                "buffer/classes": n_classes,
                "buffer/total_synth": total,
            }
            if hasattr(self.logger, "log_metrics"):
                self.logger.log_metrics(payload, step=task_id)
            elif hasattr(self.logger, "log"):
                self.logger.log(payload, step=task_id)

        return per_class, n_classes, total
