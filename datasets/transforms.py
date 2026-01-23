import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

import config


def to_3_channels(image, **kwargs):
    """
    Helper to convert 1-channel grayscale to 3-channel RGB
    by duplicating the channel 3 times.
    """
    if image.ndim == 2:  # Use simple duplicate if just 2D
        return cv2.merge([image, image, image])
    elif image.shape[2] == 1:  # If (H, W, 1)
        return cv2.merge([image, image, image])
    return image


def get_train_transforms():
    stats = config.NORMALIZATION_STATS[config.TRANSFORM_MODE]
    transforms_list = []

    # 1. Adapt Channels if necessary
    # If using ImageNet mode on grayscale data, force conversion to RGB
    if config.TRANSFORM_MODE == "imagenet":
        transforms_list.append(A.Lambda(name="GrayToRGB", image=to_3_channels))

    # 2. Geometric Augmentations
    transforms_list.extend([
        A.Resize(
            height=config.IMG_SIZE,
            width=config.IMG_SIZE,
        ),
        A.HorizontalFlip(p=config.HFLIP_PROB),
        # A.ShiftScaleRotate(
        #     shift_limit=0.05,
        #     scale_limit=0.05,
        #     rotate_limit=config.ROTATE_LIMIT,
        #     p=0.5
        # ),
    ])

    # 3. Normalize & Convert
    transforms_list.extend([
        A.Normalize(mean=stats["mean"], std=stats["std"], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    return A.Compose(transforms_list)


def get_eval_transforms():
    stats = config.NORMALIZATION_STATS[config.TRANSFORM_MODE]
    transforms_list = []

    if config.TRANSFORM_MODE == "imagenet":
        transforms_list.append(A.Lambda(name="GrayToRGB", image=to_3_channels))

    transforms_list.extend([
        A.Resize(height=config.IMG_SIZE, width=config.IMG_SIZE),
        A.Normalize(mean=stats["mean"], std=stats["std"], max_pixel_value=255.0),
        ToTensorV2(),
    ])

    return A.Compose(transforms_list)
