"""
Crack Segmentation Dataset (PyTorch Dataset)
============================================

Datasets directory convention (crack_seg_models_6/dataset):
    train/image/*.<ext>   train/target/*.<ext>
    test/image/*.<ext>    test/target/*.<ext>
    eval/image/*.<ext>    eval/target/*.<ext>

Each image and target folder holds files with the SAME basename; image
corresponds to the RGB input image and target to the binary crack mask
(grayscale, 0 = background, 255 = crack).

This object inherits from torch.utils.data.Dataset.

Training-time augmentation (applied jointly to BOTH image and mask for
geometric transforms and to the image only for photometric transforms):
    * random rotations of 0 / 90 / 180 / 270 degrees
    * random horizontal and vertical flips
    * random circular shifts up to +/- 0.5 x the image dimensions
    * Gaussian noise with sigma = 0.01 (image only)
    * brightness and contrast jittered by up to +/- 10% (image only)

Input images are uniformly resized to a multiple of 32 pixels.
"""
from __future__ import annotations

import os
import random
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# Supported raster extensions.
IMG_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Augmentation configuration (from the reported protocol).
_NOISE_SIGMA = 0.01      # Gaussian noise sigma
_JITTER = 0.10           # brightness / contrast +/- 10%
_SHIFT_FRAC = 0.50       # circular shift up to +/- 0.5 x image dimension
_MULTIPLE = 32           # resize to a multiple of 32 pixels


def _round_to_multiple(value: int, multiple: int = _MULTIPLE) -> int:
    """Round value UP to the nearest multiple of multiple (default 32)."""
    return int(np.ceil(value / multiple) * multiple)


def _find_target(target_dir: str, image_name: str) -> Optional[str]:
    """Return the target path matching image_name (exact, then by stem)."""
    direct = os.path.join(target_dir, image_name)
    if os.path.isfile(direct):
        return direct
    stem = os.path.splitext(image_name)[0]
    for entry in os.listdir(target_dir):
        if os.path.splitext(entry)[0] == stem and entry.lower().endswith(IMG_EXTS):
            return os.path.join(target_dir, entry)
    return None


def compute_pos_weight(target_dir: str) -> float:
    """Compute pos_weight = (num_neg / num_pos) / 4 over target_dir.

    Masks are binarised at 128/255 (strictly greater than 127 is a crack
    pixel). The result is the per-class weight for the positive (crack) class
    used by torch.nn.BCEWithLogitsLoss.
    """
    total_pos = 0.0
    total_neg = 0.0
    for entry in sorted(os.listdir(target_dir)):
        if not entry.lower().endswith(IMG_EXTS):
            continue
        path = os.path.join(target_dir, entry)
        if not os.path.isfile(path):
            continue
        arr = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        pos = float((arr > 127.0).sum())
        total_pos += pos
        total_neg += float(arr.size) - pos
    if total_pos <= 0.0:
        raise ValueError(
            f"No crack pixels found in {target_dir!r}; cannot compute pos_weight."
        )
    return (total_neg / total_pos) / 4.0


class CrackDataset(Dataset):
    """PyTorch dataset for binary crack segmentation.

    Parameters
    ----------
    image_dir : str
        Directory containing the input RGB images.
    target_dir : str
        Directory containing the binary crack masks (same basenames).
    img_size : int
        Square size to resize to (rounded up to a multiple of 32).
    train : bool
        Whether this is training data (drives the default augment).
    augment : bool, optional
        Explicit augmentation switch; if None the value of train is used.
    """

    def __init__(
        self,
        image_dir: str,
        target_dir: str,
        img_size: int = 512,
        train: bool = True,
        augment: Optional[bool] = None,
    ) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.target_dir = target_dir
        self.img_size = _round_to_multiple(int(img_size), _MULTIPLE)
        self.train = bool(train)
        self.augment = bool(train) if augment is None else bool(augment)

        self.files: List[str] = []
        for entry in sorted(os.listdir(image_dir)):
            path = os.path.join(image_dir, entry)
            if not (entry.lower().endswith(IMG_EXTS) and os.path.isfile(path)):
                continue
            if _find_target(target_dir, entry) is not None:
                self.files.append(entry)

        if not self.files:
            raise FileNotFoundError(
                f"No matching image/target pairs in {image_dir!r} / {target_dir!r}."
            )

    def __len__(self) -> int:
        return len(self.files)

    # ------------------------------------------------------------------ #
    def _load(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Load image + mask, resize to a multiple of 32, return float tensors."""
        name = self.files[idx]
        img = Image.open(os.path.join(self.image_dir, name)).convert("RGB")
        tgt_path = _find_target(self.target_dir, name)
        if tgt_path is None:
            raise FileNotFoundError(f"Missing target for {name!r}.")
        mask = Image.open(tgt_path).convert("L")

        size = (self.img_size, self.img_size)
        img = img.resize(size, resample=Image.BILINEAR)
        mask = mask.resize(size, resample=Image.NEAREST)

        img = torch.from_numpy(
            np.asarray(img, dtype=np.float32) / 255.0
        ).permute(2, 0, 1).contiguous()
        mask = torch.from_numpy(
            (np.asarray(mask, dtype=np.float32) > 127.0).astype(np.float32)
        ).unsqueeze(0).contiguous()
        return img, mask

    def _augment(self, img: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply the reported augmentation to image (and mask where geometric)."""
        s = self.img_size

        # --- random rotation of 0/90/180/270 degrees (both) ---
        k = random.randint(0, 3)
        if k:
            img = torch.rot90(img, k, dims=(1, 2))
            mask = torch.rot90(mask, k, dims=(1, 2))

        # --- random horizontal & vertical flips (both) ---
        if random.random() < 0.5:
            img = torch.flip(img, dims=(2,))
            mask = torch.flip(mask, dims=(2,))
        if random.random() < 0.5:
            img = torch.flip(img, dims=(1,))
            mask = torch.flip(mask, dims=(1,))

        # --- random circular shift up to +/- 0.5 x dims (both) ---
        half = int(_SHIFT_FRAC * s)
        dy = random.randint(-half, half)
        dx = random.randint(-half, half)
        if dy or dx:
            img = torch.roll(img, shifts=(dy, dx), dims=(1, 2))
            mask = torch.roll(mask, shifts=(dy, dx), dims=(1, 2))

        # --- Gaussian noise, sigma = 0.01 (image only) ---
        img = img + torch.randn_like(img) * _NOISE_SIGMA

        # --- brightness & contrast jitter, +/- 10% (image only) ---
        img = img + random.uniform(-_JITTER, _JITTER)          # brightness
        contrast = 1.0 + random.uniform(-_JITTER, _JITTER)     # contrast
        img = 0.5 + (img - 0.5) * contrast
        img = img.clamp(0.0, 1.0)

        return img, mask

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img, mask = self._load(idx)
        if self.augment:
            img, mask = self._augment(img, mask)
        # Input kept in [0, 1]; GroupNorm in the model normalises internally.
        return img, mask
