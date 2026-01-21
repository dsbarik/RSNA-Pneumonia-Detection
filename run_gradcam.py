import matplotlib.pyplot as plt
import numpy as np
import torch
from torchinfo import summary

import config
from analysis.gradcam import GradCAM
from datasets.dataloaders import get_dataloaders
from models.build import build_model
from train.checkpoint_manager import CheckpointManager


def get_device():
    if config.DEVICE == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def normalize_img(img):
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return img

def normalize_cam(cam):
    p5, p95 = np.percentile(cam, [5, 95])
    cam = np.clip(cam, p5, p95)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam


def main():
    # --------------------------------------------------
    # Device
    # --------------------------------------------------
    device = get_device()
    print(f"Using device: {device}")
    
    ckpt_manager = CheckpointManager(
        config, 
        config.CHECKPOINT_METRIC, 
        device,
        config.CHECKPOINT_MODE)

    # --------------------------------------------------
    # Load model
    # --------------------------------------------------
    model = build_model(
        model_name=config.MODEL_NAME, num_classes=config.NUM_CLASSES
    )
    
    ckpt_manager.load_best(model)
    model.eval()

    print("Model Architecture: \n")
    summary(model, input_size=(1, 1, 224, 224))
    print()
    

    # --------------------------------------------------
    # Data (batch_size=1 is mandatory)
    # --------------------------------------------------
    loaders = get_dataloaders(val_batch_size=1)

    # --------------------------------------------------
    # Grad-CAM setup
    # --------------------------------------------------
    gradcam = GradCAM(
        model=model,
        target_layer=model.features[-3]  # last Conv2d
    )
    
    gradcam.to(device)
    # --------------------------------------------------
    # Find 4 cases: TP, TN, FP, FN
    # --------------------------------------------------
    cases = {
        "TP": None,  # GT=1, Pred=1
        "TN": None,  # GT=0, Pred=0
        "FP": None,  # GT=0, Pred=1
        "FN": None,  # GT=1, Pred=0
    }

    with torch.no_grad():
        for batch in loaders["val"]:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)
            preds = logits.argmax(dim=1)

            gt = labels.item()
            pred = preds.item()

            if gt == 1 and pred == 1 and cases["TP"] is None:
                cases["TP"] = (images, gt, pred)

            elif gt == 0 and pred == 0 and cases["TN"] is None:
                cases["TN"] = (images, gt, pred)

            elif gt == 0 and pred == 1 and cases["FP"] is None:
                cases["FP"] = (images, gt, pred)

            elif gt == 1 and pred == 0 and cases["FN"] is None:
                cases["FN"] = (images, gt, pred)

            if all(v is not None for v in cases.values()):
                break

    missing = [k for k, v in cases.items() if v is None]
    if missing:
        raise RuntimeError(f"Could not find cases: {missing}")

    print("Found all cases: TP, TN, FP, FN")

    # --------------------------------------------------
    # Grad-CAM visualization (2x2 grid)
    # --------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    fig.suptitle("Grad-CAM Analysis (Baseline CNN)", fontsize=14)

    case_order = ["TP", "TN", "FP", "FN"]
    titles = {
        "TP": "True Positive (Pneumonia)",
        "TN": "True Negative (Normal)",
        "FP": "False Positive",
        "FN": "False Negative",
    }

    for ax, key in zip(axes.flatten(), case_order):
        x, gt, pred = cases[key]

        cam = gradcam.generate(x, class_idx=gt)
        cam = normalize_cam(cam)

        img = x.squeeze().cpu().numpy()
        img = normalize_img(img)

        ax.imshow(img, cmap="gray")
        ax.imshow(cam, cmap="jet", vmin=0.0, vmax=1.0, alpha=0.4)
        ax.set_title(f"{titles[key]}\nGT={gt}, Pred={pred}")
        ax.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
