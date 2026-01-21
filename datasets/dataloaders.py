from torch.utils.data import DataLoader
from .transforms import get_train_transforms, get_eval_transforms
from .chestxray_dataset import RSNADataset
import config


def get_dataloaders(
    train_batch_size: int | None = None, val_batch_size: int | None = None
):
    """
    Creates train and validation dataloaders using config.
    """

    train_ds = RSNADataset(split="train", transform=get_train_transforms())

    val_ds = RSNADataset(split="val", transform=get_eval_transforms())

    train_loader = DataLoader(
        train_ds,
        batch_size=(
            train_batch_size
            if train_batch_size is not None
            else config.TRAIN_BATCH_SIZE
        ),
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=(
            val_batch_size if val_batch_size is not None else config.VAL_BATCH_SIZE
        ),
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=config.PIN_MEMORY,
        persistent_workers=True,
    )

    class_counts = [sample["label"].item() for sample in train_ds]

    return {"train": train_loader, "val": val_loader, "class_counts": class_counts}
