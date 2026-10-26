"""Comparative benchmark runner executing all 4 CAD architectures sequentially."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from methods.method1_transfer_learning.pipeline import run_method1_pipeline
from methods.method2_zfnet_redfox.pipeline import run_method2_pipeline
from methods.method3_unet3d_convlstm_sfla.pipeline import run_method3_pipeline
from methods.method4_mri_spect_densenet.pipeline import run_method4_pipeline


def run_all_methods_benchmark(full: bool = False) -> dict:
    t_start = time.perf_counter()
    mode_str = "Full Local Datasets" if full else "Demo Mode (~12 samples/method)"
    print("\n" + "=" * 85)
    print(f" NEUROCAD: MASTER 4-METHOD BENCHMARK & COMPARATIVE EVALUATION [{mode_str}]")
    print("=" * 85)

    all_results = {}
    s_per_class = None if full else 12
    vols = None if full else 10
    s_limit = None if full else 12

    m1_metrics = run_method1_pipeline(dataset_type="kaggle", samples_per_class=s_per_class, epochs=4)
    all_results["Method 1 (Transfer Learning + Fuzzy-OTSU)"] = m1_metrics

    m2_metrics = run_method2_pipeline(dataset_type="kaggle", samples_per_class=s_per_class)
    all_results["Method 2 (ZFNet + RedFox + SVM)"] = m2_metrics

    m3_metrics = run_method3_pipeline(depth_slices=8, num_volumes=vols)
    all_results["Method 3 (3D U-Net + ConvLSTM + SFLA)"] = m3_metrics

    m4_metrics = run_method4_pipeline(samples_limit=s_limit, epochs=4)
    all_results["Method 4 (MRI-SPECT Fusion + DenseNet)"] = m4_metrics

    elapsed_total = time.perf_counter() - t_start

    print("\n\n" + "=" * 85)
    print(" ARCHITECTURAL PERFORMANCE METRICS COMPARISON")
    print("=" * 85)
    header = f"{'Architecture Method':<44} | {'Accuracy':<9} | {'Precision':<9} | {'Recall':<9} | {'F1-Score':<9}"
    print(header)
    print("-" * len(header))

    for name, res in all_results.items():
        acc = f"{res.get('accuracy', 0.0)}%"
        prec = f"{res.get('precision', 0.0)}%"
        rec = f"{res.get('sensitivity_recall', res.get('recall_sensitivity', 0.0))}%"
        f1 = f"{res.get('f1_score', 0.0)}%"
        print(f"{name:<44} | {acc:<9} | {prec:<9} | {rec:<9} | {f1:<9}")

    print("=" * 85)
    print(f"[*] Benchmark finished in {elapsed_total:.2f} seconds.")

    summary_path = config.OUTPUTS_DIR / "all_methods_summary.json"
    summary_path.write_text(json.dumps(all_results, indent=2))
    print(f"[*] Summary saved to: {summary_path}\n")
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Benchmark for All 4 Methods")
    parser.add_argument("--full", action="store_true", help="Run on full local datasets")
    parser.add_argument("--summary", action="store_true", help="Display all model architectural summaries and exit")
    args = parser.parse_args()

    if args.summary:
        from show_model_summaries import main as show_summaries_main
        show_summaries_main(argv=[])
        sys.exit(0)


    run_all_methods_benchmark(full=args.full)

