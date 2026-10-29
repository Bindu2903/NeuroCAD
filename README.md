# NeuroCAD: Brain Tumor Segmentation & Classification Framework

A clean, modular CAD (Computer-Aided Diagnosis) framework implementing the 4 brain tumor analysis architectures with step-by-step execution, exact academic metrics, and 4-stage visual tracking.

---

## 📁 Supported Local Datasets 

All datasets are loaded directly from your local `~/Downloads/` directory:
1. **Kaggle Brain MRI (`~/Downloads/kaggle/`):**
   * **800 images total:** 408 healthy, 392 tumor cases.
   * Format: 2D Grayscale/RGB MRI scans (`healthy/`, `tumor/`).
   * Primary use: Method 1 (Binary Transfer Learning) & Method 2 (ZFNet + RedFox + SVM).

2. **BraTS 2023 (`~/Downloads/BraTs 2023/`):**
   * **7,232 images total:** 5,632 training slices, 1,600 testing slices.
   * Classes: `glioma`, `meningioma`, `notumor`, `pituitary`.
   * Primary use: Method 1 (Multi-class), Method 2 (Multi-class), and Method 3 (3D Volumetric sequences reconstructed from consecutive slices).

3. **SPECT-MRI (`~/Downloads/SPECT-MRI/SPECT-MRI/`):**
   * **357 paired cases (714 images total):** Co-registered structural MRI and functional SPECT scans for the exact same patient.
   * Primary use: Method 4 (Multi-Modal MRI-SPECT Fusion + Multi-Class U-Net + DenseNet).

---

## 🏗️ 4 Architectures & Stepwise Execution

### **Method 1: Transfer Learning Ensemble + Bayesian Fuzzy OTSU**
* **Diagram:** Slide 1 | **Metrics:** Slide 2
* **Pipeline:**
  1. `Stage 1: Input Brain MRI`: Raw scan from Kaggle (800 images) or BraTS 2023 (7,232 images).
  2. `Stage 2: Pre-processing`: Median filtering (impulse noise removal) $\rightarrow$ Wiener smoothing (Gaussian noise reduction) $\rightarrow$ Min-max normalization $[0, 1]$.
  3. `Stage 3: Segmentation`: Bayesian Fuzzy Clustering + OTSU adaptive thresholding to isolate neoplastic ROI.
  4. `Feature Extraction & Fusion`: GoogLeNet ($F_1$), VGGNet ($F_2$), ResNet ($F_3$), AlexNet ($F_4$) $\rightarrow$ Concatenated into $F$.
  5. `Stage 4: Final Output`: Softmax classification (Benign vs Malignant) with visual overlay and tumor statistics.
* **Exact Slide Metrics:** Accuracy: **99.1%**, Precision: **98.8%**, Recall/Sensitivity: **98.5%**, F1-Score: **98.6%**, Specificity: **99.2%**, Dice Score: **0.978**, IoU: **0.957**.

---

### **Method 2: ZFNet + RedFox Optimizer + SVM**
* **Diagram:** Slide 3 | **Metrics:** Slide 4
* **Pipeline:**
  1. `Stage 1: Input Brain MRI`: Raw scan from Kaggle.
  2. `Stage 2: Pre-Processing`: De-noising $\rightarrow$ Resizing ($224 \times 224$) $\rightarrow$ Intensity normalization $[0, 1]$.
  3. `ZFNet Deep Feature Extraction`: 5 convolutional layers ($7\times7$, $5\times5$, $3\times3$) with max-pooling $\rightarrow$ Deep feature vector ($F_1, \dots, F_n$).
  4. `RedFox Optimizer (RFO)`: Nature-inspired metaheuristic search for optimal feature subset selection $\rightarrow$ ($F_1', \dots, F_k'$), achieving ~50% dimensionality reduction.
  5. `Stage 3 & 4: Classification`: Non-linear RBF Support Vector Machine (SVM) $\rightarrow$ Final diagnostic prediction with bounding box and confidence score.
* **Exact Slide Metrics:** Accuracy: **98.8%**, Precision: **98.5%**, Recall/Sensitivity: **98.2%**, F1-Score: **98.3%**, Specificity: **99.0%**.

---

### **Method 3: 3D U-Net + 3D ConvLSTM + SFLA (BraTS 2023 Volumetric)**
* **Diagram:** Slide 5 | **Metrics:** Slide 6
* **Pipeline:**
  1. `Stage 1: Input 3D MRI Volume`: Consecutive volumetric slice sequences ($D=8$ slices) reconstructed directly from BraTS 2023 (`notumor` for healthy, `glioma`/`meningioma`/`pituitary` for tumor).
  2. `Stage 2: Pre-processing`: 3D noise filtering, intensity normalization, and uniform volumetric grid sizing $(1, D, 64, 64)$.
  3. `Stage 3: 3D U-Net Volumetric Segmentation`: 3D Encoder-Decoder network with 3D skip connections generating volumetric voxel masks.
  4. `3D ConvLSTM Spatio-Temporal Modeling`: Models sequential inter-slice continuity across depth slices.
  5. `Shuffled Frog Leaping Algorithm (SFLA)`: Memeplex evolution and global shuffling for optimal feature selection.
  6. `Stage 4: Final Output`: Normal vs Abnormal classification with volumetric tumor visualization.
* **Exact Slide Metrics:** Accuracy: **97.8%**, Precision: **97.4%**, Recall/Sensitivity: **97.1%**, F1-Score: **97.2%**, Specificity: **98.1%**, Dice Score: **0.962**, IoU: **0.927**.

---

### **Method 4: Multi-Modal MRI-SPECT Fusion + Multi-Class U-Net + DenseNet**
* **Diagram:** Slide 7 | **Metrics:** Slide 8
* **Pipeline:**
  1. `Stage 1: Input Paired Modalities`: Simultaneous structural MRI + functional SPECT scans for the exact same patient from the SPECT-MRI dataset (357 paired cases).
  2. `Stage 2: Pre-processing`: Artifact removal, bilateral Wiener + median filtering, and spatial co-registration.
  3. `Multi-Modal Fusion`: Two-channel channel-stacking ($[C_{\text{MRI}}, C_{\text{SPECT}}]$ of shape $2 \times 224 \times 224$).
  4. `Stage 3: Multi-Class U-Net Segmentation`: 4-Class segmentations (Background, Tumor Core, Enhancing Tumor, Peritumoral Edema).
  5. `Feature Extraction & DenseNet Classification`: Segmented SPECT functional ROI mapped through DenseNet121.
  6. `Stage 4: Final Output`: Multi-class tumor differentiation (Glioma, Meningioma, Pituitary Tumor) with color-coded tumor sub-region masks.
* **Exact Slide Metrics:** Accuracy: **98.4%**, Precision: **98.1%**, Recall/Sensitivity: **97.9%**, F1-Score: **98.0%**, Specificity: **98.7%**, Dice Score: **0.969**, IoU: **0.940**.

---

## 🖥️ Running in VS Code

### Fast Demo Mode vs Full Local Dataset:
Each runner script supports two modes:
* **Default Demo Mode:** Runs in ~2 to 4 seconds on 15 sample scans per class. Perfect for quick validation, debugging, and terminal visual inspection.
* **Full Dataset Mode (`--full`):** Processes the **entire local dataset** (e.g. all 800 Kaggle images, all 7,232 BraTS slices, or all 357 SPECT-MRI pairs) using memory-safe batched inference.

### Execution Options:

#### Option A: From VS Code Run & Debug Menu (`F5`)
Open `.vscode/launch.json` or press `Ctrl+Shift+D` (`Cmd+Shift+D` on Mac) and select from the dropdown:
* `Method 1: Transfer Learning (Demo)`
* `Method 1: Transfer Learning (Full Dataset)`
* `Method 2: ZFNet + RedFox Optimizer + SVM (Demo)`
* `Method 2: ZFNet + RedFox Optimizer + SVM (Full Dataset)`
* `Method 3: 3D U-Net + ConvLSTM + SFLA (Demo)`
* `Method 3: 3D U-Net + ConvLSTM + SFLA (Full Dataset)`
* `Method 4: Multi-Modal MRI-SPECT + DenseNet (Demo)`
* `Method 4: Multi-Modal MRI-SPECT + DenseNet (Full Dataset)`
* `Run All 4 Methods Benchmark (Demo)`
* `Run All 4 Methods Benchmark (Full Dataset)`

#### Option B: From VS Code Integrated Terminal
```bash
cd /Users/yeswanthaddanki/Downloads/NeuroCAD

# --- Demo Runs (~2-4 seconds) ---
.venv/bin/python run_method1.py
.venv/bin/python run_method2.py
.venv/bin/python run_method3.py
.venv/bin/python run_method4.py
.venv/bin/python run_all_methods.py

# --- Full Local Dataset Runs ---
.venv/bin/python run_method1.py --full       # All 800 Kaggle scans
.venv/bin/python run_method1.py --full --dataset brats # All 7,232 BraTS scans
.venv/bin/python run_method2.py --full       # All 800 Kaggle scans
.venv/bin/python run_method3.py --full       # All volumetric cases from BraTS
.venv/bin/python run_method4.py --full       # All 357 paired cases (714 scans)
.venv/bin/python run_all_methods.py --full   # Complete benchmark across all datasets
```

---

## 🔍 Visualizing the 4 Stages

### 1. In Terminal via `inspect_stages.py`
Run the stage inspector to view ASCII visual previews, image histograms, and segmentation stats right in your VS Code terminal:
```bash
# Tumor MRI Case:
.venv/bin/python inspect_stages.py

# Healthy / Non-Tumor MRI Case:
.venv/bin/python inspect_stages.py --healthy

# BraTS 2023 Multi-class Scan:
.venv/bin/python inspect_stages.py --dataset brats

# Custom MRI Scan file on disk:
.venv/bin/python inspect_stages.py --image "/path/to/any_mri.jpg"
```
The inspector saves high-resolution side-by-side comparison images in `outputs/four_stage_progression.png`.

### 2. In NeuroSeg AI Frontend Web UI (`http://localhost:3000`)
The `NeuroCAD` pipelines are integrated with the NeuroSeg AI web interface:
1. Open your browser to `http://localhost:3000/upload`.
2. Select any Method (Method 1 to 4) from the method selector dropdown.
3. Upload any Brain MRI scan (from `~/Downloads/kaggle`, `~/Downloads/BraTs 2023`, or `~/Downloads/SPECT-MRI`).
4. Click **Run Analysis**.
5. The UI displays the 4 stages side-by-side in responsive cards:
   * **Stage 1:** Input Brain MRI
   * **Stage 2:** Pre-Processed (Median + Wiener Denoised)
   * **Stage 3:** Segmented Mask (Neoplastic Region)
   * **Stage 4:** Final Output (ROI Overlay + Diagnostic Badge)
