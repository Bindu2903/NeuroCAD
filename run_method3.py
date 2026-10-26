"""Execution entrypoint for Method 3.

Volumetric 3D U-Net Segmentation with 3D ConvLSTM Spatio-Temporal
Modeling and Shuffled Frog Leaping Algorithm (SFLA) Optimization.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.method3_unet3d_convlstm_sfla.pipeline import run_method3_pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Method 3: 3D U-Net + 3D ConvLSTM + SFLA")
    parser.add_argument("--full", action="store_true", help="Process full local BraTS volumetric dataset")
    parser.add_argument("--volumes", type=int, default=15, help="Volumetric sequences in demo mode")
    parser.add_argument("--depth", type=int, default=8, help="Slices per 3D volume sequence")
    parser.add_argument("--class", dest="target_class", type=str, default=None,
                        help="Target class to visualize (e.g. 'tumor', 'abnormal', 'normal', 'healthy')")
    parser.add_argument("--volume-idx", type=int, default=0, help="Index of volume to visualize (default: 0)")
    parser.add_argument("--random", action="store_true", help="Randomly pick a volume each time")
    parser.add_argument("--open", action="store_true", help="Auto-open high-res 300 DPI graphs and reports in VS Code editor tabs")
    parser.add_argument("--summary", action="store_true", help="Display model architectural summary and exit")
    args = parser.parse_args()

    if args.summary:
        from methods.method3_unet3d_convlstm_sfla.unet_3d import UNet3D
        from methods.method3_unet3d_convlstm_sfla.convlstm_3d import VolumetricConvLSTM
        print("\n" + "=" * 80)
        print(" METHOD 3 (Part A) ARCHITECTURAL SUMMARY: Volumetric 3D U-Net")
        print("=" * 80)
        UNet3D(in_channels=1, out_channels=1, base_filters=16).summary()
        print("\n" + "=" * 80)
        print(" METHOD 3 (Part B) ARCHITECTURAL SUMMARY: Volumetric 3D ConvLSTM")
        print("=" * 80)
        VolumetricConvLSTM(in_channels=1, hidden_dim=64, num_classes=2).summary()
        print("=" * 80 + "\n")
        sys.exit(0)

    vols = None if args.full else args.volumes

    run_method3_pipeline(
        depth_slices=args.depth,
        num_volumes=vols,
        target_class=args.target_class,
        volume_index=args.volume_idx,
        random_sample=args.random,
        auto_open=args.open,
    )

