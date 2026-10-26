"""Architectural Model Summary Viewer for NeuroCAD.

Generates complete layer-by-layer architectural breakdown, input/output tensor shapes,
and parameter statistics for all deep learning models across all 4 methods.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from methods.method1_transfer_learning.feature_extractors import MultiBackboneFusionNetwork
from methods.method2_zfnet_redfox.zfnet_model import ZFNet
from methods.method3_unet3d_convlstm_sfla.unet_3d import UNet3D
from methods.method3_unet3d_convlstm_sfla.convlstm_3d import VolumetricConvLSTM
from methods.method4_mri_spect_densenet.multiclass_unet import MultiClassUNet
from methods.method4_mri_spect_densenet.densenet_classifier import DenseNetSPECTClassifier


def show_summary_method1():
    print("\n" + "#" * 90)
    print(" METHOD 1: MULTI-BACKBONE FEATURE FUSION NETWORK (GoogLeNet + VGG16 + ResNet50 + AlexNet)")
    print("#" * 90)
    model = MultiBackboneFusionNetwork(num_classes=2, pretrained=False)
    model.summary(depth=2)


def show_summary_method2():
    print("\n" + "#" * 90)
    print(" METHOD 2: ZFNET CONVOLUTIONAL DEEP FEATURE EXTRACTOR")
    print("#" * 90)
    model = ZFNet(in_channels=1, feature_dim=512)
    model.summary()


def show_summary_method3():
    print("\n" + "#" * 90)
    print(" METHOD 3 (Part A): VOLUMETRIC 3D U-NET SEGMENTATION MODEL")
    print("#" * 90)
    unet3d = UNet3D(in_channels=1, out_channels=1, base_filters=16)
    unet3d.summary()

    print("\n" + "#" * 90)
    print(" METHOD 3 (Part B): VOLUMETRIC 3D CONVLSTM SPATIO-TEMPORAL SEQUENCE MODEL")
    print("#" * 90)
    convlstm = VolumetricConvLSTM(in_channels=1, hidden_dim=64, num_classes=2)
    convlstm.summary()


def show_summary_method4():
    print("\n" + "#" * 90)
    print(" METHOD 4 (Part A): MULTI-CLASS 2D U-NET (TWO-CHANNEL INPUT: MRI + SPECT)")
    print("#" * 90)
    m_unet = MultiClassUNet(in_channels=2, num_classes=4, base=32)
    m_unet.summary(depth=2)

    print("\n" + "#" * 90)
    print(" METHOD 4 (Part B): DENSENET SPECT ROI CLASSIFICATION MODEL")
    print("#" * 90)
    densenet = DenseNetSPECTClassifier(num_classes=3, pretrained=False)
    densenet.summary(depth=2)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Display PyTorch Neural Network model.summary() across methods.")
    parser.add_argument("--method", type=int, choices=[1, 2, 3, 4], default=None, help="Specific method number (1-4). Defaults to all.")
    args = parser.parse_args(argv)

    dispatch = {
        1: [show_summary_method1],
        2: [show_summary_method2],
        3: [show_summary_method3],
        4: [show_summary_method4],
    }

    if args.method in dispatch:
        for fn in dispatch[args.method]:
            fn()
    else:
        show_summary_method1()
        show_summary_method2()
        show_summary_method3()
        show_summary_method4()


if __name__ == "__main__":
    main()
