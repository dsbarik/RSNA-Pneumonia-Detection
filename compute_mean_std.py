import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from datasets.cached_dataset import CachedRSNADataset


def compute_stable_stats():
    print("--- Computing Stable Stats ---")

    df = pd.read_csv(config.TRAIN_DF)
    dataset = CachedRSNADataset(df=df, split="all", transform=None)

    loader = DataLoader(
        dataset, batch_size=64, shuffle=False, num_workers=config.NUM_WORKERS
    )

    mean = 0.0
    std = 0.0
    total_samples = 0

    print("Processing all images...")
    for batch in tqdm(loader, desc="Computing"):
        # Get image: [B, H, W] (uint8)
        images = batch["image"]
        batch_samples = images.size(0)

        # Reshape to [B, H*W] and normalize to 0-1 float
        # This mimics the ToTensor() behavior
        images = images.view(batch_samples, -1).float() / 255.0

        # Accumulate
        mean += images.mean(1).sum()
        std += images.std(1).sum()
        total_samples += batch_samples

    # Average over total samples
    final_mean = mean / total_samples
    final_std = std / total_samples

    print("\n" + "=" * 30)
    print(f"STABLE RSNA STATS (Calculated on {total_samples} images)")
    print("=" * 30)
    print(f"Mean: {final_mean:.4f}")
    print(f"Std:  {final_std:.4f}")
    print("=" * 30)


if __name__ == "__main__":
    compute_stable_stats()
