import os

import cv2
import dicomsdl
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import config


class RSNADataset(Dataset):
    CLASS_MAP = {0: "NORMAL", 1: "PNEUMONIA"}

    def __init__(self, split="all", transform=None):
        self.transform = transform
        df = pd.read_csv(config.TRAIN_DF)

        if split in ["train", "val"]:
            train_df, val_df = train_test_split(
                df,
                test_size=config.VAL_SPLIT,
                stratify=df["Target"],
                random_state=config.SEED,
            )
            self.df = train_df if split == "train" else val_df
        else:
            self.df = df

        self.df = self.df.reset_index(drop=True)
        self.patient_ids = self.df["patientId"].values
        self.labels = self.df["Target"].values

    def __len__(self):
        return len(self.patient_ids)

    def __getitem__(self, idx):
        patient_id = self.patient_ids[idx]
        label = self.labels[idx]

        if config.USE_PNG:
            image_path = os.path.join(config.PNG_DIR, f"{patient_id}.png")
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

            if img is None:
                raise FileNotFoundError(f"PNG image not found: {image_path}")
            img = img.astype(np.float32)

        else:
            dicom_path = os.path.join(config.DICOM_DIR, f"{patient_id}.dcm")
            dicom = dicomsdl.open(dicom_path)
            img = dicom.pixelData().astype(np.float32)

        mn, mx = img.min(), img.max()
        if mx - mn > 0:
            img = (img - mn) / (mx - mn) * 255.0
        else:
            img = img * 0.0

        if self.transform:
            aug = self.transform(image=img)
            img = aug["image"]

        return {
            "image": img,
            "label": torch.tensor(label, dtype=torch.long),
            "patient_id": patient_id
        }
