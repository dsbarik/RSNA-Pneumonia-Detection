import json
import os
from datetime import datetime

import torch


class CheckpointManager:
    """
    Configuration-aware checkpoint manager.
    Saves the best model based on a chosen metric.
    """

    def __init__(self, config, metric_name, device, mode="max"):
        """
        Args:
            config: config module
            metric_name: key in metrics dict to monitor (e.g. 'roc_auc')
            device: torch device
            mode: 'max' or 'min',
        """
        self.config = config
        self.metric_name = metric_name
        self.device = device
        self.mode = mode

        # Identity
        self.identity = self._build_identity()
        self.config_id = self._compute_config_id()

        # Paths
        self.ckpt_dir = self._prepare_dir()
        self.ckpt_path = os.path.join(self.ckpt_dir, "best.pt")
        self.identity_path = os.path.join(self.ckpt_dir, "identity.json")
        self.state_path = os.path.join(self.ckpt_dir, "state.json")
        self.meta_path = os.path.join(self.ckpt_dir, "meta.txt")

        # State
        self.best_metric = None
        self.best_epoch = 0
        self.last_epoch = 0

        self._load_state()
        self._validate_or_write_identity()
        self._write_metadata()

        print("✔ Checkpoint Manager initialized.\n")

    def _build_identity(self):
        return {key: getattr(self.config, key) for key in self.config.CONFIG_KEYS}

    def _compute_config_id(self):
        parts = []
        parts.append(f"{self.config.CHECKPOINT_METRIC}")
        for key in sorted(self.identity):
            value = self.identity[key]
            parts.append(f"{key.lower()}={self._stringify(value)}")

        return "__".join(parts)

    def _stringify(self, value):
        if isinstance(value, float):
            return format(value, ".6g")

        if isinstance(value, bool):
            return str(value).lower()

        return str(value)

    def _prepare_dir(self):
        path = os.path.join(
            self.config.CHECKPOINT_ROOT,
            self.config.MODEL_NAME,
            self.config_id,
        )
        os.makedirs(path, exist_ok=True)
        return path

    def _validate_or_write_identity(self):
        if os.path.exists(self.identity_path):
            with open(self.identity_path, "r") as f:
                stored = json.load(f)

            if stored != self.identity:
                raise RuntimeError(
                    (
                        "Checkpoint identity mismatch!\n"
                        f"Stored: {stored}\n"
                        f"Current: {self.identity}"
                    )
                )
        else:
            with open(self.identity_path, "w") as f:
                json.dump(self.identity, f, indent=2)

    def _write_metadata(self):
        if os.path.exists(self.meta_path):
            return

        with open(self.meta_path, "w") as f:
            f.write(f"model_name: {self.config.MODEL_NAME}\n")
            f.write(f"metric_name: {self.metric_name}\n")
            f.write(f"mode: {self.mode}\n")

            f.write("# Training parameters\n")
            f.write(f"LR: {self.config.LR}\n")
            f.write(f"BATCH_SIZE: {self.config.TRAIN_BATCH_SIZE}\n")
            f.write(f"EPOCHS: {self.config.EPOCHS}\n")
            f.write(f"DEVICE: {self.config.DEVICE}\n")

    def _load_state(self):
        if not os.path.exists(self.state_path):
            return

        with open(self.state_path, "r") as f:
            state = json.load(f)
            self.best_metric = state.get("best_metric", None)
            self.best_epoch = state.get("best_epoch", 0)
            self.last_epoch = state.get("last_epoch", 0)

    def _write_state(self):
        state = {
            "best_metric": self.best_metric,
            "best_epoch": self.best_epoch,
            "last_epoch": self.last_epoch,
            "metric_name": self.metric_name,
            "mode": self.mode,
            "update_at": datetime.now().isoformat(),
        }

        with open(self.state_path, "w") as f:
            json.dump(state, f, indent=2)

    def _is_better(self, value):
        if self.best_metric is None:
            return True
        return (
            value > self.best_metric if self.mode == "max" else value < self.best_metric
        )

    def save_if_best(self, model, optimizer, metrics, epoch):

        if self.metric_name not in metrics:
            raise KeyError(f"Metric '{self.metric_name}' not found in metrics.")

        metric_value = metrics[self.metric_name]
        self.last_epoch = epoch

        improved = self._is_better(metric_value)

        if improved:
            self.best_metric = metric_value
            self.best_epoch = self.last_epoch

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metric_value": metric_value,
                    "epoch": self.last_epoch,
                },
                self.ckpt_path,
            )

        self._write_state()
        return improved

    def load_best(self, model, optimizer=None):
        if not os.path.exists(self.ckpt_path):
            return 0

        checkpoint = torch.load(self.ckpt_path, map_location=self.device)
        with open(self.state_path, "r") as f:
            state = json.load(f)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        model.to(self.device)

        return state.get("last_epoch", 0)
