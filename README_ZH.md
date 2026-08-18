# DPD-DG: 面向变工况旋转机械故障诊断的双物理驱动域泛化网络

[English](README.md) | [中文说明](README_ZH.md)

[![Paper](https://img.shields.io/badge/Paper-ScienceDirect-blue.svg)](https://www.sciencedirect.com/science/article/abs/pii/S1568494626016662)
[![DOI](https://img.shields.io/badge/DOI-10.1016%2Fj.asoc.2026.116218-brightgreen.svg)](https://doi.org/10.1016/j.asoc.2026.116218)
[![Journal](https://img.shields.io/badge/Journal-Applied%20Soft%20Computing-orange.svg)](https://www.sciencedirect.com/journal/applied-soft-computing)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.8%2B-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📢 论文正式出版与成果介绍

本项目为发表于 Elsevier 权威期刊 **《Applied Soft Computing》** 的学术论文官方开源代码仓库：

> **论文题目**: DPD-DG: Dual-physics-driven domain generalization network for rotating machinery fault diagnosis under variable working conditions  
> **作者**: Pengfei Xu (徐鹏飞), Jinyu Zhao (赵锦宇), Jinping Liu (刘金平), Yimei Yang (杨依枚), Haidong Shao (邵海东)  
> **发表期刊**: *Applied Soft Computing*, Volume 116218, 2026.  
> 🔗 **DOI 直链**: [https://doi.org/10.1016/j.asoc.2026.116218](https://doi.org/10.1016/j.asoc.2026.116218)  
> 📄 **ScienceDirect 官方页面**: [https://www.sciencedirect.com/science/article/abs/pii/S1568494626016662](https://www.sciencedirect.com/science/article/abs/pii/S1568494626016662)

---

## 📖 论文摘要 (Abstract)

在变工况条件下，旋转机械的故障诊断往往受到严重数据分布偏移（Distribution Shifts）的制约。现有的半监督域泛化（Semi-Supervised Domain Generalization, SSDG）方法主要依赖纯数据驱动的统计对齐策略，在工况波动引发的高度不确定性与噪声干扰下极易失效。此外，现有方法很大程度上忽略了确定性的运动学机理（Kinematic Mechanisms），导致模型在未见目标域中生成的决策边界不稳定，且缺乏物理层面的可解释性。

为克服上述核心瓶颈，本文提出了 **DPD-DG（双物理驱动域泛化网络）**。作为一种混合软计算（Hybrid Soft Computing）框架，DPD-DG 通过**“特征感知 - 显式约束 - 隐式校验”**的闭环协同机制，将确定性物理先验与深度特征表征无缝融合：

1. **多域物理先验提取与自适应感知模块**：全面捕获信号中的运动学先验线索，结合异构归一化与全局共享物理感知注意力（Shared Physics-Aware Attention, SPA），蒸馏出对工况不敏感的先验特征；
2. **显式物理驱动（动态权重调制机制）**：基于低秩矩阵分解，将运动学状态的变化直接映射并调制分类器参数，防止决策边界过拟合于环境工况噪声；
3. **隐式物理驱动（物理 - 语义双重校验策略）**：在自训练伪标签生成过程中，系统性剔除违反物理一致性的高置信度噪声样本，有效缓解自训练过程中的确认偏差（Confirmation Bias）；
4. **多级流形对齐目标**：结合点对点语义关联与全局二阶协方差统计量（CORAL），全面压缩域间差异。

在 **6 个主流旋转机械公开数据集**上的大量实验表明，DPD-DG 在准确率（Accuracy）、F1-Score 和马修斯相关系数（MCC）等指标上显著优于现有前沿方法，并通过丰富的可视化分析展现了深厚的物理可解释性。

---

## 🌟 核心创新与方法架构

```
               ┌─────────────────────────────────────────────────────────┐
               │           DPD-DG: 双物理驱动域泛化诊断框架                │
               └─────────────────────────────────────────────────────────┘
                                           │
         ┌─────────────────────────────────┼─────────────────────────────────┐
         ▼                                 ▼                                 ▼
┌──────────────────┐             ┌──────────────────┐             ┌──────────────────┐
│  1. 物理先验感知   │             │  2. 显式物理驱动   │             │  3. 隐式物理校验   │
│  - 48+ 维统计特征 │             │  - 低秩矩阵分解   │             │  - 物理一致性检验 │
│  - 异构归一化     │             │  - 分类器参数调制 │             │  - 拒识高置信噪声 │
│  - SPA 物理注意力 │             │  - 防噪声过拟合   │             │  - 抑制确认偏差   │
└──────────────────┘             └──────────────────┘             └──────────────────┘
                                           │
                                           ▼
                                 ┌──────────────────┐
                                 │  4. 多级流形对齐   │
                                 │  - 点对点语义关联 │
                                 │  - 全局二阶 CORAL│
                                 └──────────────────┘
```

- **混合软计算范式**：结合确定性运动学规律与深度神经网络，打破纯黑盒深度学习的泛化壁垒。
- **物理先验特征引擎**：从原始振动信号中提取 48+ 种时域与频域统计特征指标（RMS、峰度、波形因子、谱熵等）。
- **显隐双物理驱动**：
  - **显式驱动**：通过低秩分解调控分类层权重，使决策边界对工况变化具备适应性。
  - **隐式驱动**：利用物理先验作为安全卫士，过滤置信度高但违背物理机理的伪标签样本。
- **深层可解释性**：能够清晰可视化网络对不同物理指标的注意力分布与特征流形对齐过程。

---

## 📂 项目结构说明

```text
DPD_DG/
├── code/                                   # 核心代码目录
│   ├── data/                               # 数据集加载与构造
│   │   ├── BJUT_Gear/                      # 北京工业大学齿轮数据集
│   │   ├── CWRU_Bearing/                   # 凯斯西储大学轴承数据集
│   │   ├── HUST_Gear/                      # 华中科技大学齿轮数据集
│   │   ├── Ottawa_Bearing/                 # 渥太华大学变速轴承数据集
│   │   ├── SDUT_Bearing/                   # 山东理工大学轴承数据集
│   │   ├── SDUT_Gear/                      # 山东理工大学齿轮数据集
│   │   └── construct_loader.py             # 融入物理先验的数据加载器
│   ├── models/                             # 模型定义
│   │   ├── DPD_DG.py                       # 本文提出的 DPD-DG 核心网络
│   │   ├── CNN.py                          # 1D-CNN 基线模型
│   │   ├── MSDGN.py                        # 多源域泛化网络
│   │   ├── CS.py                           # Crafting Shifts 对比模型
│   │   ├── CCDG.py, CDDG.py, CEG.py        # 域泛化前沿对比方法
│   │   ├── DGNIS.py, DKGPL.py, mmd.py      # 对比方法及 MMD 对齐模块
│   │   └── __init__.py
│   ├── utils/                              # 实验与可视化工具库
│   │   ├── train_test.py                   # 训练与测试流程控制器
│   │   ├── prior_features.py               # 48+ 种物理先验特征提取引擎
│   │   ├── save_confusion_matrix.py        # 混淆矩阵绘制
│   │   ├── save_t_sne.py                   # t-SNE 特征分布降维与 J-Score 计算
│   │   ├── save_attention_heatmap.py       # 物理感知注意力热力图绘制
│   │   ├── dynamical_pseudo-labels/        # 动态伪标签演化分析模块
│   │   └── Parameter_Sensitivity/          # 3D 参数敏感性曲面生成模块
│   ├── main.py                             # 单次实验主程序
│   ├── run_single.py                       # 快捷单组/自定义参数运行脚本
│   ├── run_6datasets_4tasks_8models.py     # 6 数据集 x 8 模型全量基准测试
│   ├── run_6datasets_4tasks_DPD_ablations.py # M0~M5 消融实验自动化脚本
│   └── run_6datasets_4tasks_SeekDPDBestParameters.py # 超参数网格搜索脚本
├── requirements.txt                        # Python 依赖库
├── README.md                               # 英文文档
└── README_ZH.md                            # 中文文档
```

---

## 📊 实验数据集与跨工况任务设置

本方法在 **6 个主流旋转机械公开基准**上进行了全面验证：

| 数据集 | 机械对象 | 工况变化参数 (域划分) | 泛化任务类型 |
| :--- | :--- | :--- | :--- |
| **BJUT_Gear** (北京工业大学齿轮) | 齿轮箱 | 转频：`20Hz`, `30Hz`, `40Hz`, `50Hz` | 跨转速域泛化 |
| **CWRU_Bearing** (凯斯西储大学轴承) | 深沟球轴承 | 电机负载：`0HP`, `1HP`, `2HP`, `3HP` | 跨负载域泛化 |
| **HUST_Gear** (华中科技大学齿轮) | 齿轮箱 | 负载扭矩：`0Nm`, `0.113Nm`, `0.226Nm`, `0.339Nm` | 跨扭矩域泛化 |
| **Ottawa_Bearing** (渥太华大学轴承) | 滚动轴承 | 变速趋势：加速(`up`)、减速(`down`)、加减速(`updown`)、减加速(`downup`) | 时变转速域泛化 |
| **SDUT_Bearing** (山东理工大学轴承) | 滚动轴承 | 旋转转速：`1500rpm`, `1800rpm`, `2000rpm`, `2500rpm` | 跨转速域泛化 |
| **SDUT_Gear** (山东理工大学齿轮) | 齿轮箱 | 旋转转速：`1500rpm`, `1800rpm`, `2000rpm`, `2500rpm` | 跨转速域泛化 |

> 注：原始 `.mat` 数据文件可放置于对应 `code/data/<DatasetName>/` 文件夹中。

---

## ⚙️ 环境配置与安装

### 1. 克隆代码仓库
```bash
git clone https://github.com/jinyu1034/DPD-DG.git
cd DPD-DG
```

### 2. 创建并激活虚拟环境 (推荐 Python 3.8+)
```bash
# 使用 conda
conda create -n dpd_dg python=3.8 -y
conda activate dpd_dg

# 或使用 venv
python -m venv venv
# Linux/macOS:
source venv/bin/activate
# Windows PowerShell:
.\venv\Scripts\activate
```

### 3. 安装依赖库
```bash
pip install -r requirements.txt
```

---

## 🚀 快速上手与运行指南

### 1. 运行单次实验
通过 `main.py` 指定数据集、源域/目标域以及模型：
```bash
python code/main.py \
    --dataset_name BJUT_Gear \
    --target_id 40hz \
    --source_id_list 20hz 30hz 50hz \
    --model_name DPD_DG \
    --epoch 100 \
    --lr 0.001
```

或直接修改并运行 `run_single.py`：
```bash
python code/run_single.py
```

### 2. 批量复现全部 8 种模型对比实验
自动化在 6 个数据集的所有跨域任务上测试 8 种模型：
```bash
python code/run_6datasets_4tasks_8models.py
```

### 3. 运行 DPD-DG 消融实验 (M0 ~ M5)
```bash
python code/run_6datasets_4tasks_DPD_ablations.py
```

*消融模式说明：*
- `M0`：纯数据驱动 Baseline 基线
- `M0_`：仅利用物理先验特征
- `M1`：加入全局共享物理感知注意力 (SPA)
- `M2`：加入动态权重调制 (显式物理驱动)
- `M3`：加入物理 - 语义双重校验 (隐式物理驱动)
- `M4`：加入多级流形对齐 (CORAL + 语义对齐)
- `M5`：**完整 DPD-DG 模型**

### 4. 超参数网格搜索
```bash
python code/run_6datasets_4tasks_SeekDPDBestParameters.py
```

---

## 📈 结果可视化与可解释性分析

项目内置了丰富的可解释性分析与图表绘制工具：

1. **t-SNE 特征空间对齐分布图 & J-Score（类间/类内距离比）**：
   ```bash
   python code/utils/save_t_sne.py
   ```
2. **混淆矩阵 (Confusion Matrix)**：
   ```bash
   python code/utils/save_confusion_matrix.py
   ```
3. **物理特征注意力热力图 (Attention Heatmap)**：
   ```bash
   python code/utils/save_attention_heatmap.py
   ```
4. **动态伪标签演化曲线（硬阈值 vs 物理校验过滤量与错误率）**：
   ```bash
   python code/utils/dynamical_pseudo-labels/plot_prior_metrics\ copy.py
   ```
5. **3D 超参数敏感性响应曲面**：
   ```bash
   python code/utils/Parameter_Sensitivity/CWRU_Bearing/visualize_3d_sensitivity.py
   ```

---

## 📑 论文引用 (Citation)

如果本项目的研究成果或代码对您的科研工作有所启发或帮助，欢迎引用本论文：

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

## 📄 开源许可证 (License)

本项目遵循 [MIT License](LICENSE) 开源许可协议。

## 🤝 交流与反馈 (Contact)

如有任何问题、建议或学术合作意向，欢迎在 GitHub 提交 Issue 或联系作者：
- **徐鹏飞** / **赵锦宇** (`jinyu1034`)
- GitHub 仓库: [https://github.com/jinyu1034/DPD-DG](https://github.com/jinyu1034/DPD-DG)
