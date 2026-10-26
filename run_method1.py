"""Execution entrypoint for Method 1.

Deep Transfer Learning Ensemble (GoogLeNet, VGG16, ResNet50, AlexNet)
with Bayesian Fuzzy OTSU Segmentation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.method1_transfer_learning.pipeline import run_method1_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Method 1: Transfer Learning Ensemble + Bayesian Fuzzy OTSU")
    parser.add_argument("--full", action="store_true", help="Process full local dataset")
    parser.add_argument("--dataset", choices=["kaggle", "brats"], default="kaggle", help="Dataset target")
    parser.add_argument("--samples", type=int, default=15, help="Samples per class in demo mode")
    parser.add_argument("--epochs", type=int, default=35, help="Epochs for classification head calibration")
    parser.add_argument("--image", type=str, default=None, help="Path to specific custom image to test & visualize")
    parser.add_argument("--class", dest="target_class", type=str, default=None,
                        help="Target class to visualize (e.g., 'tumor', 'healthy', 'glioma', 'meningioma', 'pituitary')")
    parser.add_argument("--sample-idx", type=int, default=0, help="Index of sample to visualize (default: 0)")
    parser.add_argument("--random", action="store_true", help="Randomly pick a sample each time")
    parser.add_argument("--open", action="store_true", help="Auto-open high-res 300 DPI graphs and reports in VS Code editor tabs")
    parser.add_argument("--summary", action="store_true", help="Display model architectural summary and exit")
    args = parser.parse_args()

    if args.summary:
        from methods.method1_transfer_learning.feature_extractors import MultiBackboneFusionNetwork
        print("\n" + "=" * 80)
        print(" METHOD 1 ARCHITECTURAL SUMMARY: Multi-Backbone Feature Fusion Network")
        print("=" * 80)
        MultiBackboneFusionNetwork(num_classes=2, pretrained=False).summary(depth=2)
        print("=" * 80 + "\n")
        sys.exit(0)

    samples = None if args.full else args.samples

    run_method1_pipeline(
        dataset_type=args.dataset,
        samples_per_class=samples,
        epochs=args.epochs,
        target_image=args.image,
        target_class=args.target_class,
        sample_index=args.sample_idx,
        random_sample=args.random,
        auto_open=args.open,
    )

