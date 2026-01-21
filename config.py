import os

# ===============================
# Dataset
# ===============================
DATA_ROOT = "rsna_dataset" # Root directory for dataset

TRAIN_DF = os.path.join(DATA_ROOT, "image_labels.csv")  # CSV file with image labels
DICOM_DIR = os.path.join(DATA_ROOT, "images")      # Original DICOM files
PNG_DIR = os.path.join(DATA_ROOT, "images_png")    # Converted PNG files

USE_PNG = True  # Whether to use PNG images instead of DICOM

# ===============================
# Augmentation / Transforms
# ===============================

IMG_SIZE = 224

TRANSFORM_MODE = "standard" # options: "standard", "imagenet"

NORMALIZATION_STATS = {
    "standard": {
        "mean": (0.5,),
        "std": (0.5,)
    },
    "imagenet": {
        "mean": (0.485, 0.456, 0.406),
        "std": (0.229, 0.224, 0.225)
    }
}

TRAIN_CROP_SCALE = (0.75, 1.0) 
HFLIP_PROB = 0.5
ROTATE_LIMIT = 15

# ===============================
# Training
# ===============================
TRAIN_BATCH_SIZE = 8
VAL_BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-3

# ===============================
# Model
# ===============================
# Options: "baseline", "small_resnet", "hybrid_cnn", "cnn_v2", "resnet18", "resnet50"
MODEL_NAME = "baseline"
NUM_OUTPUTS = 1

# ===============================
# Device
# ===============================
# Options: "mps", "cpu"
DEVICE = "mps"

# ===============================
# Validation split
# ===============================
VAL_SPLIT = 0.2

# ===============================
# DataLoader
# ===============================
NUM_WORKERS = 8
PIN_MEMORY = False

# ===============================
# Reproducibility
# ===============================
SEED = 42

# ===============================
# Metrics
# ===============================
METRIC_NAMES = [
    "accuracy",
    "precision",
    "recall",
    "f1-score",
    "roc_auc",
]

CHECKPOINT_METRIC = "f1-score"
CHECKPOINT_MODE = "max"


# ===============================
# Checkpointing
# ===============================
CHECKPOINT_ROOT = "checkpoints"

# Which config values define a unique experiment
CONFIG_KEYS = [
    "LR",
    "TRAIN_BATCH_SIZE",
]

# ===============================
# TensorBoard (optional)
# ===============================
USE_TENSORBOARD = False
TB_LOG_DIR = "runs"
TB_EXPERIMENT_NAME = "baseline_cnn"
