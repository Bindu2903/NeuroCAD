"""System configuration module for NeuroCAD.

Resolves local dataset roots, output directories, compute accelerators,
and baseline experimental parameters.
"""
from __future__ import annotations

import os
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

for m in ("method1", "method2", "method3", "method4", "inspections"):
    (OUTPUTS_DIR / m).mkdir(parents=True, exist_ok=True)

HOME_DOWNLOADS = Path.home() / "Downloads"
KAGGLE_DATASET_DIR = Path(os.environ.get("KAGGLE_DIR", PROJECT_ROOT / "data" / "kaggle" if (PROJECT_ROOT / "data" / "kaggle").exists() else HOME_DOWNLOADS / "kaggle"))
BRATS2023_DATASET_DIR = Path(os.environ.get("BRATS_DIR", PROJECT_ROOT / "data" / "BraTs 2023" if (PROJECT_ROOT / "data" / "BraTs 2023").exists() else HOME_DOWNLOADS / "BraTs 2023"))
SPECT_MRI_DATASET_DIR = Path(os.environ.get("SPECT_DIR", PROJECT_ROOT / "data" / "SPECT-MRI" / "SPECT-MRI" if (PROJECT_ROOT / "data" / "SPECT-MRI" / "SPECT-MRI").exists() else HOME_DOWNLOADS / "SPECT-MRI" / "SPECT-MRI"))

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

IMAGE_SIZE_2D = 224
RANDOM_SEED = 42
BATCH_SIZE = 16
NUM_WORKERS = 0
