# DPD-DG: Dual-Physics-Driven Domain Generalization Network for Rotating Machinery Fault Diagnosis under Variable Working Conditions

[English](README.md) | [中文说明](README_ZH.md)

[![Paper](https://img.shields.io/badge/Paper-ScienceDirect-blue.svg)](https://www.sciencedirect.com/science/article/abs/pii/S1568494626016662)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.asoc.2026.116218-brightgreen.svg)](https://doi.org/10.1016/j.asoc.2026.116218)
[![Journal](https://img.shields.io/badge/Journal-Applied%20Soft%20Computing-orange.svg)](https://www.sciencedirect.com/journal/applied-soft-computing)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📢 News & Publication

Official PyTorch implementation of the research paper:

> **DPD-DG: Dual-physics-driven domain generalization network for rotating machinery fault diagnosis under variable working conditions**  
> *Pengfei Xu, Jinyu Zhao, Jinping Liu, Yimei Yang, Haidong Shao*  
> **Applied Soft Computing**, 2026, Article 116218.  
> 🔗 **DOI**: [10.1016/j.asoc.2026.116218](https://doi.org/10.1016/j.asoc.2026.116218)  
> 📄 **ScienceDirect**: [https://www.sciencedirect.com/science/article/abs/pii/S1568494626016662](https://www.sciencedirect.com/science/article/abs/pii/S1568494626016662)

---

## 📖 Abstract

Fault diagnosis of rotating machinery under variable working conditions is often hindered by severe distribution shifts. While existing semi-supervised domain generalization (SSDG) methods primarily rely on pure data-driven statistical alignment, they frequently struggle with the high uncertainty and imprecision induced by fluctuating operations. By largely overlooking deterministic kinematic mechanisms, these approaches tend to yield unreliable decision boundaries in unseen domains and lack physical interpretability. 

To address these critical bottlenecks, this article proposes **DPD-DG**, a novel dual-physics-driven domain generalization network. Serving as a hybrid soft computing framework, DPD-DG seamlessly integrates deterministic physical mechanisms with deep representations through a closed-loop synergy of feature perception, explicit constraints, and implicit verification:

1. **Multi-domain Physical Prior Extraction and Adaptive Perception**: Comprehensively captures kinematic cues; heterogeneous normalization and Global Shared Physics-Aware Attention (SPA) are employed to distill domain-insensitive priors.
2. **Explicit Physics Drive (Dynamic Weight Modulation)**: Leverages low-rank matrix decomposition to map kinematic state variations directly into classifier parameters, preventing decision boundaries from overfitting to environmental noise.
3. **Implicit Physics Drive (Physics-Semantic Dual Verification)**: Formulates a dual verification strategy during self-training to systematically reject high-confidence noisy samples violating physical consistency, effectively mitigating confirmation bias.
4. **Multi-Level Manifold Alignment**: Incorporates point-to-point semantic associations and global second-order statistics (CORAL) to compress inter-domain discrepancies.

Extensive experiments across **six public benchmarks** demonstrate that DPD-DG not only significantly outperforms state-of-the-art methods in terms of accuracy, F1-score, and MCC, but also affords profound physical interpretability, as validated by comprehensive visualization analyses.

---

## 🌟 Key Highlights & Framework Overview

- **Hybrid Soft Computing Paradigm**: Bridges pure statistical deep learning with deterministic kinematics for robust cross-working-condition diagnosis.
- **Physical Prior Engine**: Extracts 48+ domain-informative statistical and spectral prior indicators from raw vibration signals.
- **Dual Physics Drive Mechanism**:
  - *Explicit*: Physics-guided dynamic weight modulation for the classifier.
  - *Implicit*: Physics-consistent self-training pseudo-label filtration.
- **Extensive Benchmarking**: Tested across 6 major rotating machinery datasets covering bearing and gearbox systems under varying speeds, loads, and time-varying conditions.

---

## 📂 Project Structure

```text
DPD_DG/
├── code/
│   ├── data/                               # Dataset loaders and construction scripts
│   │   ├── BJUT_Gear/                      # BJUT Gearbox dataset directory
│   │   ├── CWRU_Bearing/                   # CWRU Bearing dataset directory
│   │   ├── HUST_Gear/                      # HUST Gearbox dataset directory
│   │   ├── Ottawa_Bearing/                 # Ottawa Time-Varying Speed Bearing dataset
│   │   ├── SDUT_Bearing/                   # SDUT Bearing dataset directory
│   │   ├── SDUT_Gear/                      # SDUT Gear dataset directory
│   │   └── construct_loader.py             # Data loader with physical prior integration
│   ├── models/                             # Model architectures
│   │   ├── DPD_DG.py                       # Proposed DPD-DG model (Ours)
│   │   ├── CNN.py                          # 1D-CNN Baseline
│   │   ├── MSDGN.py                        # Multi-Source Domain Generalization Network
│   │   ├── CS.py                           # Crafting Shifts
│   │   ├── CCDG.py, CDDG.py, CEG.py        # Comparison methods
│   │   ├── DGNIS.py, DKGPL.py, mmd.py      # Comparison methods and MMD utilities
│   │   └── __init__.py
│   ├── utils/                              # Utility scripts & visualization tools
│   │   ├── train_test.py                   # Training loop, validation, and evaluation
│   │   ├── prior_features.py               # 48+ physical kinematic prior feature extractor
│   │   ├── save_confusion_matrix.py        # Confusion matrix plotting
│   │   ├── save_t_sne.py                   # t-SNE feature visualization & J-Score calculation
│   │   ├── save_attention_heatmap.py       # Physics attention heatmap generator
│   │   ├── dynamical_pseudo-labels/        # Dynamic pseudo-label evolution analysis
│   │   └── Parameter_Sensitivity/          # 3D parameter sensitivity response surfaces
│   ├── main.py                             # Core single-experiment entry point
│   ├── run_single.py                       # Easy single/custom experiment runner
│   ├── run_6datasets_4tasks_8models.py     # Batch evaluation across 6 datasets & 8 models
│   ├── run_6datasets_4tasks_DPD_ablations.py # Ablation study runner (M0 - M5)
│   └── run_6datasets_4tasks_SeekDPDBestParameters.py # Hyperparameter grid search script
├── requirements.txt                        # Python dependencies
└── README.md                               # Project documentation
```

---

## 📊 Benchmark Datasets

The framework is evaluated on **6 public industrial rotating machinery datasets**:

| Dataset | Machinery Type | Working Condition Variations (Domains) | Task Setting |
| :--- | :--- | :--- | :--- |
| **BJUT_Gear** | Gearbox | Rotation frequencies: `20Hz`, `30Hz`, `40Hz`, `50Hz` | Cross-speed generalization |
| **CWRU_Bearing** | Rolling Element Bearing | Motor loads: `0HP`, `1HP`, `2HP`, `3HP` | Cross-load generalization |
| **HUST_Gear** | Gearbox | Load torques: `0Nm`, `0.113Nm`, `0.226Nm`, `0.339Nm` | Cross-torque generalization |
| **Ottawa_Bearing**| Rolling Element Bearing | Dynamic speed profiles: `up`, `down`, `updown`, `downup` | Time-varying speed generalization |
| **SDUT_Bearing** | Rolling Element Bearing | Rotational speeds: `1500rpm`, `1800rpm`, `2000rpm`, `2500rpm` | Cross-speed generalization |
| **SDUT_Gear** | Gearbox | Rotational speeds: `1500rpm`, `1800rpm`, `2000rpm`, `2500rpm` | Cross-speed generalization |

> Place `.mat` raw data files in their respective folders under `code/data/<DatasetName>/`.

---

## ⚙️ Installation & Environment Setup

### 1. Clone the Repository
```bash
git clone https://github.com/jinyu1034/DPD-DG.git
cd DPD-DG
```

### 2. Create and Activate Virtual Environment
```bash
# Using conda
conda create -n dpd_dg python=3.8 -y
conda activate dpd_dg

# Or using venv
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start & Usage

### 1. Run a Single Experiment
Execute a single cross-domain task (e.g., target domain `40hz` on `BJUT_Gear`):

```bash
python code/main.py \
    --dataset_name BJUT_Gear \
    --target_id 40hz \
    --source_id_list 20hz 30hz 50hz \
    --model_name DPD_DG \
    --epoch 100 \
    --lr 0.001
```

Or configure and run `code/run_single.py` directly:
```bash
python code/run_single.py
```

### 2. Comprehensive Benchmark Evaluation (8 Models on 6 Datasets)
Run all comparison methods across all 6 datasets:
```bash
python code/run_6datasets_4tasks_8models.py
```

### 3. Ablation Experiments
Evaluate ablation variants (M0 baseline to M5 full model) to inspect individual module contributions:
```bash
python code/run_6datasets_4tasks_DPD_ablations.py
```

*Ablation Modes:*
- `M0`: Baseline pure data-driven model
- `M0_`: Physical prior only
- `M1`: Shared Physics-Aware Attention (SPA)
- `M2`: Dynamic Weight Modulation (Explicit Drive)
- `M3`: Physics-Semantic Dual Verification (Implicit Drive)
- `M4`: Multi-level Manifold Alignment (CORAL + SA)
- `M5`: Full **DPD-DG** architecture

### 4. Hyperparameter Sensitivity & Grid Search
Search for optimal sensitivity ranges for hyperparameters:
```bash
python code/run_6datasets_4tasks_SeekDPDBestParameters.py
```

---

## 📈 Visualizations & Interpretability

The repository provides comprehensive visualization and physical interpretability tools, categorized into **online evaluation flags** (controlled via CLI arguments during testing) and **offline post-processing scripts** (analyzing generated logs):

### 1. Online Visualizations (Controlled via CLI Arguments)

Enable these flags in `code/main.py` or `code/run_single.py` during training/testing to automatically generate and save publication-ready figures:

| CLI Argument | Default | Output Directory | Description |
| :--- | :--- | :--- | :--- |
| `--save_tsne True` | `False` | `code/utils/t-SNE/` | Plots inter-domain feature space alignment (Source vs. Target) and calculates the **$J$-Score** ($S_b / S_w$, Fisher criterion) along with scatter matrices. |
| `--save_cm True` | `False` | `code/utils/confusion_matrix/` | Generates normalized confusion matrix heatmaps showing classification percentages and overall test accuracy. |
| `--save_attention True` | `False` | `code/utils/attention_maps/` | Visualizes the class-wise attention weights over 48 physical statistical indicators (SPA module for `DPD_DG`) and exports the attention table to an `.xlsx` spreadsheet. |

**Example Command:**
```bash
python code/main.py \
    --dataset_name BJUT_Gear \
    --target_id 40hz \
    --source_id_list 20hz 30hz 50hz \
    --model_name DPD_DG \
    --save_tsne True \
    --save_cm True \
    --save_attention True
```

### 2. Offline Analysis & Post-Processing (Standalone Python Scripts)

Run these standalone scripts on experimental logs (`.log`) generated in `results/` or utility directories:

1. **Dynamic Pseudo-Label Evolution Curves**:
   Analyzes self-training pseudo-label progression (retention ratio, error rate, and rejected noisy samples violating physical consistency across epochs):
   ```bash
   python "code/utils/dynamical_pseudo-labels/plot_prior_metrics copy.py"
   ```

2. **3D Hyperparameter Sensitivity Response Surfaces**:
   Parses grid search logs for pseudo-label threshold (`threshold`) vs. physical consistency threshold (`physics_thresh`) against target accuracy (`Mean_Acc`), plotting 3D publication surfaces via bicubic interpolation:
   ```bash
   # Example for CWRU Bearing dataset:
   python code/utils/Parameter_Sensitivity/CWRU_Bearing/visualize_3d_sensitivity.py
   ```

---

## 📑 Citation

If you find this code or paper useful in your research, please cite our work as follows:

```bibtex
@article{XU2026116218,
  title = {DPD-DG: Dual-physics-driven domain generalization network for rotating machinery fault diagnosis under variable working conditions},
  journal = {Applied Soft Computing},
  pages = {116218},
  year = {2026},
  issn = {1568-4946},
  doi = {https://doi.org/10.1016/j.asoc.2026.116218},
  url = {https://www.sciencedirect.com/science/article/pii/S1568494626016662},
  author = {Pengfei Xu and Jinyu Zhao and Jinping Liu and Yimei Yang and Haidong Shao},
  keywords = {Semi-supervised domain generalization (SSDG), Hybrid soft computing, Fault diagnosis, Uncertainty calibration, Dual physics-driven, Physics attention mechanism, Manifold alignment, Interpretability}
}
```

---

## 📄 License

This project is licensed under the [MIT License](LICENSE) - see the LICENSE file for details.

## 🤝 Acknowledgements & Contact

For questions, issues, or collaborations regarding this paper and repository, please open an issue in this repository or contact the authors:
- **Pengfei Xu** / **Jinyu Zhao** (`jinyu1034`)
- Repository Link: [https://github.com/jinyu1034/DPD-DG](https://github.com/jinyu1034/DPD-DG)
