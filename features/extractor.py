import torch


@torch.no_grad()
def extract_features(model, loader, device):
    """
    Extracts deep feature vectors from a trained CNN.

    Args:
        model: trained PyTorch model with `features` and `gap` attributes
        loader: DataLoader (train / val / test)
        device: torch device

    Returns:
        features: torch.Tensor of shape (N, D)
        labels:   torch.Tensor of shape (N,)
    """
    model.eval()

    all_features = []
    all_labels = []

    for x, y in loader:
        x = x.to(device)

        # Forward through feature extractor only
        feats = model.features(x)
        feats = model.gap(feats)
        feats = feats.view(feats.size(0), -1)

        all_features.append(feats.cpu())
        all_labels.append(y)

    features = torch.cat(all_features, dim=0)
    labels = torch.cat(all_labels, dim=0)

    return features, labels
