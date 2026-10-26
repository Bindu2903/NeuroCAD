"""Execution entrypoint for Method 2.

ZFNet Deep Feature Extraction with RedFox Metaheuristic Optimization
and Support Vector Machine (SVM) Classification.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.method2_zfnet_redfox.pipeline import run_method2_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Method 2: ZFNet + RedFox Optimizer + SVM")
    parser.add_argument("--full", action="store_true", help="Process full local dataset")
    parser.add_argument("--dataset", choices=["kaggle", "brats"], default="kaggle", help="Dataset target")
    parser.add_argument("--samples", type=int, default=15, help="Samples per class in demo mode")
    parser.add_argument("--image", type=str, default=None, help="Path to specific custom image to test & visualize")
    parser.add_argument("--class", dest="target_class", type=str, default=None,
                        help="Target class to visualize (e.g., 'tumor', 'healthy', 'glioma', 'meningioma', 'pituitary')")
    parser.add_argument("--sample-idx", type=int, default=0, help="Index of sample to visualize (default: 0)")
    parser.add_argument("--random", action="store_true", help="Randomly pick a sample each time")
    parser.add_argument("--open", action="store_true", help="Auto-open high-res 300 DPI graphs and reports in VS Code editor tabs")
    parser.add_argument("--summary", action="store_true", help="Display model architectural summary and exit")
    args = parser.parse_args()

    if args.summary:
        from methods.method2_zfnet_redfox.zfnet_model import ZFNet
        print("\n" + "=" * 80)
        print(" METHOD 2 ARCHITECTURAL SUMMARY: ZFNet Convolutional Deep Feature Extractor")
        print("=" * 80)
        ZFNet(in_channels=1, feature_dim=512).summary()
        print("=" * 80 + "\n")
        sys.exit(0)

    samples = None if args.full else args.samples

    run_method2_pipeline(
        dataset_type=args.dataset,
        samples_per_class=samples,
        target_image=args.image,
        target_class=args.target_class,
        sample_index=args.sample_idx,
        random_sample=args.random,
        auto_open=args.open,
    )

