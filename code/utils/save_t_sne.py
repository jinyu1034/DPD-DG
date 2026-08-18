import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
import os
import logging
import seaborn as sns
import pandas as pd

def calculate_j_score(features, labels, domains=None):
    """
    计算基于类间散布矩阵和类内散布矩阵的 J-score (LDA 准则)
    公式参考: J = tr(Sb) / tr(Sw) 或 ||Sb|| / ||Sw||
    其中:
      - sigma_e (Sb): 类间散布矩阵 (Between-class scatter matrix)
      - sigma_a (Sw): 类内散布矩阵 (Within-class scatter matrix)
    注意：为了数值稳定性，通常计算的是迹 (trace) 的比值。
    按照用户提供的图片公式: J = ||sigma_e|| / ||sigma_a||
    这里使用 Frobenius 范数。
    """
    features = np.array(features)
    labels = np.array(labels)
    
    # 确保 features 是 2D 数组
    if len(features.shape) == 1:
        features = features.reshape(-1, 1)

    classes = np.unique(labels)
    n_classes = len(classes)
    n_features = features.shape[1]
    
    # 全局均值 f_hat
    global_mean = np.mean(features, axis=0) # shape (n_features,)
    
    sigma_e = np.zeros((n_features, n_features)) # 类间
    sigma_a = np.zeros((n_features, n_features)) # 类内
    
    for c in classes:
        # 获取属于第 c 类的样本
        class_indices = np.where(labels == c)[0]
        class_samples = features[class_indices] # shape (N_c, n_features)
        N_c = len(class_indices)
        
        if N_c == 0: continue
        
        # 类别均值 f_hat_c
        class_mean = np.mean(class_samples, axis=0) # shape (n_features,)
        
        # 更新类间协方差 sigma_e
        # sigma_e = sum(N_c * (mean_c - mean) * (mean_c - mean)^T)
        diff_mean = (class_mean - global_mean).reshape(-1, 1) # (n_features, 1)
        sigma_e += N_c * np.dot(diff_mean, diff_mean.T)
        
        # 更新类内协方差 sigma_a
        # sigma_a = sum(sum((x_i - mean_c) * (x_i - mean_c)^T))
        # 向量化计算: (X - mean_c)^T * (X - mean_c)
        diff_samples = class_samples - class_mean # (N_c, n_features)
        sigma_a += np.dot(diff_samples.T, diff_samples)
        
    # 计算范数 ||.||
    # 图片公式暗示可能是矩阵范数 (如 Frobenius norm)
    norm_sigma_e = np.linalg.norm(sigma_e, ord='fro')
    norm_sigma_a = np.linalg.norm(sigma_a, ord='fro')
    
    # J = ||sigma_e|| / ||sigma_a||
    # 类间距离越大越好，类内距离越小越好 -> J 越大越好
    if norm_sigma_a == 0:
        j_score = 0.0 # 避免除零
    else:
        j_score = norm_sigma_e / norm_sigma_a
        
    return j_score, norm_sigma_e, norm_sigma_a

def plot_t_sne(features, labels, save_path, title=None, class_names=None, domains=None):
    """
    绘制并保存符合顶刊风格的 t-SNE 可视化图
    参数:
        features: 特征向量 (n_samples, n_features)
        labels: 真实标签 (n_samples,)
        save_path: 保存图片的路径
        title: 图片标题
        class_names: 类别名称列表 (对应的索引)
        domains: 域标签 (n_samples,)，0表示源域，1表示目标域。如果为None，则不区分域。
    """
    # 设置字体
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    # 确保 features 是 numpy 数组
    if hasattr(features, 'cpu'):
        features = features.cpu().numpy()
    elif hasattr(features, 'detach'):
        features = features.detach().cpu().numpy()
        
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()

    if domains is not None:
        if hasattr(domains, 'cpu'):
            domains = domains.cpu().numpy()
    
    # t-SNE 降维
    # 如果样本数量过多，可以考虑先 PCA 降维，或者下采样
    logging.info("Starting t-SNE computation...")
    tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(features)
    logging.info("t-SNE computation completed.")
    
    # 创建 DataFrame 方便绘图
    df = pd.DataFrame()
    df['t-SNE 1'] = tsne_results[:, 0]
    df['t-SNE 2'] = tsne_results[:, 1]
    
    # 将数字标签转换为名称
    if class_names is not None:
        # 确保 labels 是整数类型
        labels = labels.astype(int)
        df['Label'] = [class_names[i] for i in labels]
    else:
        df['Label'] = labels

    if domains is not None:
        domain_map = {0: 'Source', 1: 'Target'}
        df['Domain'] = [domain_map.get(d, 'Unknown') for d in domains]
    else:
        df['Domain'] = 'Target'

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8), dpi=600)
    
    # 获取类别数量
    n_classes = len(np.unique(labels))
    
    # 颜色调色板: 使用 Seaborn 的 bright 或 deep
    palette = sns.color_palette("bright", n_classes) 
    
    # 绘制散点图
    # 如果有域标签，使用 style 区分域，size 区分大小
    if domains is not None:
        sns.scatterplot(
            x='t-SNE 1', y='t-SNE 2',
            hue='Label',
            style='Domain',
            size='Domain',
            sizes={'Source': 40, 'Target': 50}, # 源域小，目标域大
            palette=palette,
            data=df,
            legend="full",
            alpha=0.75,
            edgecolor='w', # 点的边缘颜色
            linewidth=0.3,
            markers={'Source': 'o', 'Target': '^'}, # 源域为圆圈，目标域为三角
            ax=ax
        )
    else:
        sns.scatterplot(
            x='t-SNE 1', y='t-SNE 2',
            hue='Label',
            palette=palette,
            data=df,
            legend="full",
            alpha=0.75,
            s=60, # 点的大小
            edgecolor='w', # 点的边缘颜色
            linewidth=0.5,
            ax=ax
        )

    # 计算J-score
    j_score, sigma_e, sigma_a = calculate_j_score(tsne_results, labels, domains)

    # 显示 J-score，调整位置和格式
    # 如果有标题，将 J-score 放在标题下方；否则作为标题
    # 为了避免重叠，我们手动调整位置
    # 在顶部中间显示两个文本行
    
    if title:
        full_title = f"{title}\nJ-score: {j_score:.4f} (Se: {sigma_e:.2e}, Sa: {sigma_a:.2e})"
        plt.title(full_title, fontsize=16, pad=15)
    else:
        plt.title(f"J-score: {j_score:.4f} (Se: {sigma_e:.2e}, Sa: {sigma_a:.2e})", fontsize=16, pad=15)

    # 标签和标题 (去除默认的 x/ylabel 也许更洁净，或者保留)
    # 顶刊通常保留坐标轴标签，但隐藏刻度，或者都保留
    ax.set_xlabel('Dimension 1', fontsize=14, fontweight='bold', labelpad=10)
    ax.set_ylabel('Dimension 2', fontsize=14, fontweight='bold', labelpad=10)
    
    plt.xticks([])
    plt.yticks([])
            
    # 图例优化
    # 将图例放在图外，防止遮挡
    plt.legend(bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0., fontsize=10, title_fontsize=12, frameon=True)
    
    plt.tight_layout()
    
    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        logging.info(f"t-SNE visualization saved to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save t-SNE plot: {e}")
    finally:
        plt.close(fig)
