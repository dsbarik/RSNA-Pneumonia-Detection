import warnings
from collections import Counter

import torch

# from torchinfo import summary
import config
from datasets.dataloaders import get_dataloaders
from models.build import build_model
from train.checkpoint_manager import CheckpointManager
from train.evaluator import evaluate
from train.trainer import Trainer
from utils.metrics import get_metrics

warnings.filterwarnings("ignore")

def get_device():
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if config.DEVICE == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    # --------------------------------------------------
    # Device
    # --------------------------------------------------
    device = get_device()
    print(f"Using device: {device}")

    # --------------------------------------------------
    # Data
    # --------------------------------------------------
    loaders = get_dataloaders()

    # --------------------------------------------------
    # Class weights (handle imbalance)
    # --------------------------------------------------
    counts = Counter(loaders["class_counts"])
    # total = sum(counts.values())

    # class_weights = torch.tensor(
    #     [total / counts[0], total / counts[1]],
    #     dtype=torch.float32,
    # ).to(device)
    
    pos_weight = torch.tensor(counts[0]/counts[1], dtype=torch.float32).to(device)

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    torch.manual_seed(config.SEED)
    model = build_model(
        model_name=config.MODEL_NAME, num_outputs=config.NUM_OUTPUTS
    )
    
    # print("\nModel Architecture:")
    # summary(model, input_size=(1, 1, 224, 224))
    # print()
    
    model = model.to(device)

    # --------------------------------------------------
    # Training components
    # --------------------------------------------------
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LR
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        metrics=get_metrics(config.METRIC_NAMES),  # pass metrics dict
    )

    # --------------------------------------------------
    # Checkpoint manager
    # --------------------------------------------------
    ckpt_mgr = CheckpointManager(
        config=config,
        metric_name=config.CHECKPOINT_METRIC,
        device=device,
        mode=config.CHECKPOINT_MODE,
    )
    
    start_epoch = ckpt_mgr.load_best(model, optimizer)

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for epoch in range(start_epoch, config.EPOCHS):
        train_loss = trainer.train_epoch(loaders["train"])
        val_loss, val_metrics = trainer.validate_epoch(loaders["val"])

        saved = ckpt_mgr.save_if_best(
            model=model,
            optimizer=optimizer,
            metrics=val_metrics,
            epoch=epoch+1
        )

        print(
            f"Epoch [{epoch + 1:>{len(str(config.EPOCHS))}}/{config.EPOCHS}] | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f} | "
            f"Val {config.CHECKPOINT_METRIC}: {val_metrics.get(config.CHECKPOINT_METRIC, 0.0):.4f}"
        )

        if saved:
            print("  ✓ Saved new best model")

    # --------------------------------------------------
    # Load best checkpoint before evaluation
    # --------------------------------------------------
    _ = ckpt_mgr.load_best(model)
    model.eval()

    # --------------------------------------------------
    # Evaluation (Test set)
    # --------------------------------------------------
    metrics = evaluate(model, loaders["val"], device, get_metrics(config.METRIC_NAMES))

    print("\n=== Test Results (Best Model) ===")
    print(f"Accuracy      : {metrics['accuracy']:.4f}")
    print(f"Precision     : {metrics['precision']:.4f}")
    print(f"Recall        : {metrics['recall']:.4f}")
    print(f"F1-score      : {metrics['f1-score']:.4f}")
    print(f"ROC-AUC       : {metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
