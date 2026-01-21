from torchmetrics import MetricCollection
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)
from torchmetrics.regression import MeanSquaredError

# All available metrics (single source of truth)
METRIC_REGISTRY = {
    "accuracy": BinaryAccuracy,
    "precision": BinaryPrecision,
    "recall": BinaryRecall,
    "f1-score": BinaryF1Score,
    "roc_auc": BinaryAUROC,
    "pr_auc": BinaryAveragePrecision,
    "brier": MeanSquaredError,  # used as Brier score
}



def get_metrics(metric_names, threshold=0.5):
    """
    Select metrics by name.

    Args:
        metric_names (list[str])

    Returns:
        dict[str, callable]
    """
    metrics = {}

    for name in metric_names:
        metric_cls = METRIC_REGISTRY[name]

        if name in {"accuracy", "precision", "recall", "f1-score"}:
            metrics[name] = metric_cls(threshold=threshold)
        else:
            metrics[name] = metric_cls()

    return MetricCollection(metrics)
