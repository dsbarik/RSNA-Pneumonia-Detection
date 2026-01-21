import torch


@torch.no_grad()
def evaluate(model, loader, device, metrics) -> dict:
    """
    Evaluates a trained binary classification model.

    Args:
        model: trained PyTorch model
        loader: DataLoader
        device: torch device
        metric_names: list[str] of metric names registered in get_metrics

    Returns:
        dict[str, float]: evaluation metrics
    """
    model.eval()

    metrics = metrics.to(device)
    metrics.reset()

    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device).long()

        logits = model(x).view(-1)      # [B, 1] → [B]
        probs = torch.sigmoid(logits)   # required for binary torchmetrics

        metrics.update(probs, y)

    # torchmetrics → Python floats
    return {k: v.item() for k, v in metrics.compute().items()}
