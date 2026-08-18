import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import torch
import logging
from matplotlib import cm

def plot_physical_attention_heatmap(model, dataloader, device, save_path, class_names=None, feature_names=None):
    """
    计算并绘制符合顶刊风格的物理注意力热力图。
    展示不同故障类别下，SPA模块对48个物理特征的平均注意力权重。

    参数:
        model: 训练好的模型 (需要包含 prior_attention 模块)
        dataloader: 测试或验证数据加载器
        device: 运行设备
        save_path: 图片保存路径
        class_names: 类别名称列表 (y轴)
        feature_names: 特征名称列表 (x轴)，默认为 1-48
    """
    # 1. 设置顶刊绘图风格
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['axes.unicode_minus'] = False # 处理负号显示

    # 2. 数据收集
    model.eval()
    
    # 存储每个类别的权重列表字典: {class_idx: [weight_vector1, weight_vector2, ...]}
    class_weights_dict = {} 
    
    logging.info("Collecting attention weights for visualization...")
    
    with torch.no_grad():
        for inputs, labels, domain, _, prior_feats in dataloader:
            prior_feats = prior_feats.to(device)
            labels = labels.to(device)
            
            # 获取 Attention 模块
            # 优先尝试直接从模型获取，或者从分类器获取
            att_module = None
            if hasattr(model, 'prior_attention') and model.prior_attention is not None:
                att_module = model.prior_attention
            elif hasattr(model, 'classifier') and hasattr(model.classifier, 'prior_attention') and model.classifier.prior_attention is not None:
                att_module = model.classifier.prior_attention
            
            if att_module is None:
                logging.warning("No prior_attention module found in model. Cannot plot heatmap.")
                return

            # 前向传播计算权重
            # PriorAttention forward returns (x*w, w)
            _, weights = att_module(prior_feats)
            # weights shape: [batch_size, 48]
            
            weights_np = weights.cpu().numpy()
            labels_np = labels.cpu().numpy()
            
            for i in range(len(labels_np)):
                lbl = int(labels_np[i])
                w = weights_np[i]
                
                if lbl not in class_weights_dict:
                    class_weights_dict[lbl] = []
                class_weights_dict[lbl].append(w)
    
    # 3. 聚合计算 (按类别取平均)
    sorted_classes = sorted(class_weights_dict.keys())
    matrix_rows = []
    
    if class_names is None:
        final_class_names = [f"Class {c}" for c in sorted_classes]
    else:
        # 确保 class_names 覆盖了出现的类别
        final_class_names = [class_names[c] if c < len(class_names) else f"Class {c}" for c in sorted_classes]

    for c in sorted_classes:
        # stack: [num_samples, 48] -> mean -> [48]
        mean_vector = np.mean(np.stack(class_weights_dict[c]), axis=0)
        matrix_rows.append(mean_vector)
    
    #构建最终矩阵: shape [Num_Classes, Num_Features] (e.g., 10 x 48)
    attention_matrix = np.array(matrix_rows)
    
    # === 改进：增强对比度 ===
    # 1. 使用百分位数动态确定颜色范围，避免因为大部分权重都很高导致全红
    # 将颜色范围聚焦在数据的 5% 到 95% 之间，拉伸对比度
    vmin = np.percentile(attention_matrix, 2)
    vmax = np.percentile(attention_matrix, 98)
    
    # 2. (可选) 行归一化：如果整体对比度依然很低，可以考虑取消下面注释
    # 这将展示相对重要性：即对于每一类故障，哪些特征是"相对"更重要的
    # attention_matrix = (attention_matrix - attention_matrix.min(axis=1, keepdims=True)) / (attention_matrix.max(axis=1, keepdims=True) - attention_matrix.min(axis=1, keepdims=True) + 1e-8)
    # vmin, vmax = 0.0, 1.0

    # 4. 绘图
    # 动态调整图片高度，取决于类别数量
    fig_height = max(6, len(sorted_classes) * 0.8)
    fig, ax = plt.subplots(figsize=(16, fig_height), dpi=600) # 增加宽度以适应48个标签
    
    # 处理 X 轴标签
    num_features = attention_matrix.shape[1]
    if feature_names is None:
        if num_features == 48:
            # 使用标准的48个物理特征缩写
            # 顺序对应 build_prior_features 中的拼接顺序
            # 1. Time (15): Max, Min, Peak, Range, AbsMean, SRA, MeanSq, Std, RMS, Kurt, Skew, CF, IF, SF, CLF
            # 2. Envelope (4): EnvMean, EnvStd, EnvMax, EnvKurt
            # 3. Bispectrum (4): BiMean, BiStd, BiMax, BiSkew
            # 4. Freq (15): F-Max, F-Min, F-Peak, F-Range, F-AbsMean, F-SRA, F-MeanSq, F-Std, F-RMS, F-Kurt, F-Skew, F-CF, F-IF, F-SF, F-CLF
            # 5. SpecEnt (1)
            # 6. SpecMoments (4): SpecCent, SpecSprd, SpecSkew, SpecKurt
            # 7. BandEnergy (3): Band1, Band2, Band3
            # 8. Cepstrum (2): CepMax, CepMean
            x_tick_labels = [
                # Time Domain (1-15)
                'Max', 'Min', 'Peak', 'Range', 'AbsMean', 'SRA', 'MeanSq', 'Std', 'RMS', 'Kurt', 'Skew', 'CF', 'IF', 'SF', 'CLF',
                # Envelope (16-19)
                'EnvMean', 'EnvStd', 'EnvMax', 'EnvKurt',
                # Bispectrum (20-23)
                'BiMean', 'BiStd', 'BiMax', 'BiSkew',
                # Freq Domain (24-38)
                'F-Max', 'F-Min', 'F-Peak', 'F-Range', 'F-AbsMean', 'F-SRA', 'F-MeanSq', 'F-Std', 'F-RMS', 'F-Kurt', 'F-Skew', 'F-CF', 'F-IF', 'F-SF', 'F-CLF',
                # Spec Entropy (39)
                'SpecEnt',
                # Spec Moments (40-43)
                'SpecCent', 'SpecSprd', 'SpecSkew', 'SpecKurt',
                # Band Energy (44-46)
                'Band1', 'Band2', 'Band3',
                # Cepstrum (47-48)
                'CepMax', 'CepMean'
            ]
        else:
            # 如果特征数不是48，使用数字索引
            x_tick_labels = [str(i+1) for i in range(num_features)]
    else:
        x_tick_labels = feature_names

    # === Data Export (Added per user request) ===
    # 将注意力权重保存为 Excel 表格，表头为特征名，索引为类别名
    try:
        df_att = pd.DataFrame(attention_matrix, index=final_class_names, columns=x_tick_labels)
        df_att = df_att.round(2) # 保留2位小数
        # 将路径后缀从 .png 改为 .xlsx
        excel_save_path = save_path.rsplit('.', 1)[0] + '.xlsx'
        df_att.to_excel(excel_save_path)
        logging.info(f"Attention data table saved to {excel_save_path}")
    except Exception as e:
        logging.warning(f"Could not save attention table to Excel: {e}")

    # 绘制热力图
    # 选用 'YlOrRd' (黄-橙-红) 代替 'Reds'，因为黄色和红色的亮度差异更大，视觉区分度更强
    heatmap = sns.heatmap(attention_matrix, 
                          annot=False,   # 不在格子里显示数值，因为格子太多
                          cmap="Reds", # 颜色映射：黄色(低)-橙色(中)-红色(高)
                          cbar=True,
                          xticklabels=x_tick_labels,
                          yticklabels=final_class_names,
                          ax=ax,
                          linewidths=0.5,
                          linecolor='white',
                          vmin=vmin, vmax=vmax, # 使用动态范围
                          cbar_kws={"label": "Attention Weight", "shrink": .8})

    # 5. 美化细节
    ax.set_xlabel('Physical Feature Indicator', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Fault Class', fontsize=14, fontweight='bold', labelpad=10)
    
    # 调整刻度字体
    plt.xticks(fontsize=8, rotation=45, ha='right') # 旋转并从右对齐，避免重叠
    plt.yticks(fontsize=11, rotation=0)
    
    # 标题 (可选)
    # plt.title("Class-wise Physical Attention Heatmap", fontsize=16, pad=20)
    
    # 优化 Colorbar 字体
    cbar = heatmap.collections[0].colorbar
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Attention Weight', fontsize=12, fontweight='bold')

    plt.tight_layout()
    
    # 6. 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        logging.info(f"Physical Attention Heatmap saved to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save heatmap: {e}")
    finally:
        plt.close(fig)
