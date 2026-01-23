import torch
from tqdm.auto import tqdm


class Trainer:
    """
    Handles model training and validation.
    """

    def __init__(self, model, optimizer, criterion, device, metrics=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.metrics = metrics

        if metrics is not None:
            self.metrics = self.metrics.to(device)

        print(f"✔ Trainer initialized on device: {device}")

    def train_epoch(self, loader):
        """
        Trains the model for one epoch.

        Args:
            loader: PyTorch DataLoader for training data

        Returns:
            Average training loss for the epoch
        """
        self.model.train()
        running_loss = 0.0
        curr_len = 0

        pbar = tqdm(loader, total=len(loader), desc="Training", leave=False)
        for batch in pbar:
            x = batch["image"].to(self.device)
            y = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(x).view(-1)
            loss = self.criterion(logits, y.float())
            loss.backward()
            self.optimizer.step()

            curr_len += x.size(0)
            running_loss += loss.item() * x.size(0)

            pbar.set_postfix({"loss": running_loss / curr_len})

        avg_loss = running_loss / curr_len

        return avg_loss

    @torch.no_grad()
    def validate_epoch(self, loader):
        """
        Validates the model for one epoch.

        Args:
            loader: PyTorch DataLoader for validation data

        Returns:
            Average validation loss for the epoch
        """
        self.model.eval()
        running_loss = 0.0
        cur_len = 0

        if self.metrics is not None:
            self.metrics.reset()

        pbar = tqdm(loader, total=len(loader), desc="Validation", leave=False)

        for batch in pbar:
            x = batch["image"].to(self.device)
            y = batch["label"].to(self.device)

            logits = self.model(x).view(-1)
            loss = self.criterion(logits, y.float())

            running_loss += loss.item() * x.size(0)
            cur_len += x.size(0)

            probs = torch.sigmoid(logits)

            if self.metrics is not None:
                self.metrics.update(probs, y)

            pbar.set_postfix({"loss": running_loss / cur_len})

        avg_loss = running_loss / cur_len

        metric_results = (
            {k: v.item() for k, v in self.metrics.compute().items()}
            if self.metrics is not None
            else {}
        )

        return avg_loss, metric_results
