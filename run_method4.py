"""Execution entrypoint for Method 4.

Multimodal Structural MRI and Functional SPECT Co-Registration
with Multi-Class U-Net Segmentation and DenseNet Classification.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.method4_mri_spect_densenet.pipeline import run_method4_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Method 4: Multi-Modal MRI-SPECT Fusion + Multi-Class U-Net + DenseNet")
    parser.add_argument("--full", action="store_true", help="Process full local SPECT-MRI dataset")
    parser.add_argument("--samples", type=int, default=15, help="Paired cases in demo mode")
    parser.add_argument("--epochs", type=int, default=5, help="Epochs for DenseNet classifier head")
    parser.add_argument("--class", dest="target_class", type=str, default=None,
                        help="Target class to visualize (e.g. 'glioma', 'meningioma', 'pituitary')")
    parser.add_argument("--pair-idx", type=int, default=0, help="Index of paired scan to visualize (default: 0)")
    parser.add_argument("--random", action="store_true", help="Randomly pick a pair each time")
    parser.add_argument("--mri", type=str, default=None, help="Path to custom MRI slice")
    parser.add_argument("--spect", type=str, default=None, help="Path to custom SPECT slice")
    parser.add_argument("--open", action="store_true", help="Auto-open high-res 300 DPI graphs and reports in VS Code editor tabs")
    parser.add_argument("--summary", action="store_true", help="Display model architectural summary and exit")
    args = parser.parse_args()

    if args.summary:
        from methods.method4_mri_spect_densenet.multiclass_unet import MultiClassUNet
        from methods.method4_mri_spect_densenet.densenet_classifier import DenseNetSPECTClassifier
        print("\n" + "=" * 80)
        print(" METHOD 4 (Part A) ARCHITECTURAL SUMMARY: Multi-Class 2D U-Net (MRI + SPECT)")
        print("=" * 80)
        MultiClassUNet(in_channels=2, num_classes=4, base=32).summary(depth=2)
        print("\n" + "=" * 80)
        print(" METHOD 4 (Part B) ARCHITECTURAL SUMMARY: DenseNet SPECT ROI Classifier")
        print("=" * 80)
        DenseNetSPECTClassifier(num_classes=3, pretrained=False).summary(depth=2)
        print("=" * 80 + "\n")
        sys.exit(0)

    samples = None if args.full else args.samples

    run_method4_pipeline(
        samples_limit=samples,
        epochs=args.epochs,
        target_class=args.target_class,
        pair_index=args.pair_idx,
        random_sample=args.random,
        custom_mri=args.mri,
        custom_spect=args.spect,
        auto_open=args.open,
    )

