# 基于双物理驱动的域泛化故障诊断项目 (DPD-DG)

本项目实现了一种基于双物理驱动（Dual-Physical Driven）的域泛化（Domain Generalization, DG）方法，用于旋转机械（轴承、齿轮）的跨工况故障诊断。项目包含完整的数据预处理、模型定义、训练测试流程以及多种对比方法的实现。

## 1. 环境依赖与安装

请确保已安装 Python 环境（推荐 Python 3.8+），并安装以下依赖库：

```bash
pip install -r requirements.txt
```

核心依赖包括：
*   **PyTorch**: 深度学习框架
*   **Numpy & Scipy**: 数值计算与信号处理
*   **Scikit-learn**: 机器学习工具（t-SNE, 评估指标等）
*   **Matplotlib & Seaborn**: 绘图与可视化
*   **Pandas**: 数据处理
*   **re**: 正则表达式处理（内置库）
*   **scipy.interpolate**: 插值工具（用于3D绘图）

## 2. 文件夹结构说明

工作区主要结构及其作用如下：

```
root/
├── code/                   # 核心代码目录
│   ├── data/               # 数据加载与数据集存放目录
│   ├── models/             # 各种故障诊断模型的定义
│   ├── utils/              # 通用工具函数与绘图脚本
│   │   ├── train_test.py           # 训练/测试流程控制器
│   │   ├── prior_features.py       # 物理先验特征提取工具
│   │   ├── save_*.py               # 各类结果保存与可视化脚本
│   │   ├── dynamical_pseudo-labels/# 动态伪标签过程分析脚本
│   │   └── Parameter_Sensitivity/  # 参数敏感性分析可视化脚本
│   ├── main.py             # 单次实验的主入口脚本
│   ├── run_single.py       # 自定义参数的批量/单次运行脚本
│   └── run_*.py            # 针对特定实验设置的批处理脚本
├── results/                # 实验结果输出目录（日志文件等）
├── 原文/                   # 参考论文或文档
└── requirements.txt        # 项目依赖库列表
```

## 3. 核心文件详细说明

### 3.1 根目录代码 (`code/`)

*   **`code/main.py`**:
    *   **作用**: 项目的主入口文件。
    *   **功能**: 解析命令行参数，设置随机种子，初始化日志，调用 `utils.train_test` 模块进行模型的训练和测试。它定义了所有可调整的超参数（如 `lr`, `batch_size`, `lambda_` 权重等）。
*   **`code/run_single.py`**:
    *   **作用**: 便捷的运行脚本。
    *   **功能**: 用户可以在此文件中配置字典列表 (`dataset_configs`) 和参数网格 (`param_grid`)，脚本会自动构建命令行参数并调用 `main.py`。适合调试或运行特定的一组实验。
*   **`code/run_6datasets_4tasks_*.py`**:
    *   **作用**: 自动化批处理脚本。
    *   **功能**: 针对 6 个数据集和 4 个迁移任务进行大规模实验。常见的有：
        *   `run_6datasets_4tasks_8models.py`: 运行所有对比模型。
        *   `run_6datasets_4tasks_DPD_ablations.py`: 运行 DPD-DG 模型的消融实验。
        *   `run_6datasets_4tasks_SeekDPDBestParameters.py`: 用于网格搜索最佳超参数。

### 3.2 模型定义 (`code/models/`)

该文件夹包含了 DPD-DG 及其对比方法的实现：

*   **`DPD_DG.py`**: **本项目核心模型**。实现了双物理驱动的域泛化网络，包含特征提取器、物理注意力模块、先验特征融合以及物理一致性伪标签生成策略。
*   **`CNN.py`**: 基础的一维卷积神经网络基线模型。
*   **`MSDGN.py`**: 多源域泛化网络 (Multi-Source Domain Generalization Network)。
*   **`CS.py`**: Crafting Shifts，一种基于数据增强的域泛化方法。
*   **`CCDG.py`, `CDDG.py`, `CEG.py`, `DGNIS.py`, `DKGPL.py`, `mmd.py`**: 其他对比算法或辅助模块（如 MMD 距离计算）。
*   **`__init__.py`**: 使 models 文件夹成为一个 Python 包，便于动态调用模型。

### 3.3 工具库详细说明 (`code/utils/`)

此文件夹包含实现项目所有辅助功能的脚本，是复现实验结果和绘制图表的关键。

#### 3.3.1 核心功能脚本
*   **`train_test.py`**:
    *   **核心逻辑**: 封装了 `train` 和 `test` 函数。
    *   **功能**:
        *   管理训练循环（Epoch loop）。
        *   执行模型前向传播、损失计算、反向传播。
        *   执行验证集评估，并根据 Accuracy 或 Loss 保存最佳模型 (`.pth`)。
        *   实现早停机制 (Early Stopping)。
*   **`prior_features.py`**:
    *   **物理引擎**: 提供了从原始振动信号中提取 48+ 种物理统计特征的函数。
    *   **指标**: 包括时域指标（RMS、峰度、波形因子等）和频域指标（谱熵、能量比等）。
    *   被 `code/data/construct_loader.py` 调用，用于为每个样本生成物理先验向量。

#### 3.3.2 可视化脚本
*   **`save_confusion_matrix.py`**: 生成混淆矩阵图，展示模型在各类别上的分类性能。
*   **`save_t_sne.py`**: 使用 t-SNE 算法对高维特征降维并可视化，展示域适应后的特征分布对齐情况，并计算 J-Score（类间类内距离比）。
*   **`save_attention_heatmap.py`**: 绘制物理注意力热力图，展示模型关注哪些物理特征（如 RMS, Kurtosis）来识别特定故障。

#### 3.3.3 分析子模块
*   **`dynamical_pseudo-labels/`**:
    *   **`plot_prior_metrics copy.py`**: 用于分析训练过程中伪标签演变的脚本。它解析日志文件，绘制“硬阈值筛选数量”、“物理一致性通过数量”以及“错误率”随 Epoch 变化的曲线，用于验证 DPD 策略的有效性。
*   **`Parameter_Sensitivity/`**:
    *   **作用**: 用于参数敏感性分析的可视化。
    *   **`visualize_3d_sensitivity.py`**: 该脚本扫描指定的日志文件夹，提取不同超参数（如 `threshold`, `physics_thresh`）组合下的实验准确率，并绘制 3D 表面图，直观展示参数对模型性能的影响。不同数据集子文件夹下可能有针对特定数据集的配置文件。

## 4. 数据集概况

本项目使用了 6 个主流的旋转机械故障诊断数据集，每个数据集通常包含不同工况（载荷/转速）下的数据，用于模拟跨域场景：

1.  **BJUT_Gear (北京工业大学齿轮数据集)**
    *   **包含**: 不同健康状态的齿轮振动数据。
    *   **域划分**: 基于不同的转速频率（20Hz, 30Hz, 40Hz, 50Hz）。
2.  **CWRU_Bearing (凯斯西储大学轴承数据集)**
    *   **包含**: 经典的滚珠轴承故障数据。
    *   **域划分**: 基于不同的电机负载（0HP, 1HP, 2HP, 3HP）。
3.  **HUST_Gear (华中科技大学齿轮数据集)**
    *   **包含**: 齿轮箱在不同负载下的振动信号。
    *   **域划分**: 基于负载扭矩（0Nm, 0.113Nm, 0.226Nm, 0.339Nm）。
4.  **Ottawa_Bearing (渥太华大学轴承数据集)**
    *   **包含**: 轴承在变速条件下的振动数据。
    *   **域划分**: 基于转速变化趋势（加速 up, 减速 down, 加减速 updown, 减加速 downup）。
5.  **SDUT_Bearing (山东理工大学轴承数据集)**
    *   **包含**: 自制实验台的轴承故障数据。
    *   **域划分**: 基于转速（1500, 1800, 2000, 2500 rpm）。
6.  **SDUT_Gear (山东理工大学齿轮数据集)**
    *   **包含**: 自制实验台的齿轮故障数据。
    *   **域划分**: 基于转速（1500, 1800, 2000, 2500 rpm）。

**注意**: 数据文件均为 `.mat` 格式，存放于 `code/data/<DatasetName>/` 目录下。

## 5. 快速开始

### 单次运行
修改 `code/run_single.py` 中的配置，然后直接运行：

```bash
python code/run_single.py
```

### 运行参数敏感性分析
在运行完参数搜索实验后（日志保存在 `results/`），可以使用可视化脚本：

```bash
python code/utils/Parameter_Sensitivity/CWRU_Bearing/visualize_3d_sensitivity.py
```
（需修改脚本中的日志路径指向实际结果文件夹）
