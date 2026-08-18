import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import logging
import seaborn as sns

def plot_confusion_matrix(y_true, y_pred, save_path, title=None, class_names=None, accuracy=None):
    """
    绘制并保存符合顶刊风格的混淆矩阵
    参数:
        y_true: 真实标签
        y_pred: 预测标签
        save_path: 保存图片的路径
        title: 图片标题
        class_names: 类别名称列表
        accuracy: 总体准确率
    """
    # 设置字体为 Times New Roman
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.rcParams['mathtext.fontset'] = 'stix'
    
    # 获取混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 归一化
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    if class_names is None:
        n_classes = cm.shape[0]
        class_names = [str(i) for i in range(n_classes)]

    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8), dpi=600)
    
    # 绘制热力图
    sns.heatmap(cm_normalized, annot=False, fmt=".2%", cmap="Blues", cbar=True, 
                xticklabels=class_names, yticklabels=class_names, ax=ax, 
                square=True, linewidths=1.5, linecolor='white', cbar_kws={"shrink": .8})

    # 添加数值标签
    thresh = cm_normalized.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            count = cm[i, j]
            percent = cm_normalized[i, j] * 100
            text = f"{count}\n({percent:.1f}%)"
            ax.text(j + 0.5, i + 0.5, text,
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black",
                    fontsize=14)

    # 标签样式
    ax.set_ylabel('True Label', fontsize=16, fontweight='bold', labelpad=10)
    ax.set_xlabel('Predicted Label', fontsize=16, fontweight='bold', labelpad=10)
    
    plt.xticks(fontsize=14, rotation=0)
    plt.yticks(fontsize=14, rotation=0)
    
    if title:
        plt.title(title, fontsize=18, pad=20)
    elif accuracy is not None:
         plt.title(f"Confusion Matrix (Accuracy: {accuracy:.2%})", fontsize=18, pad=20)
        
    plt.tight_layout()
    
    # 保存
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)
        logging.info(f"Confusion matrix saved to {save_path}")
    except Exception as e:
        logging.error(f"Failed to save confusion matrix: {e}")
    finally:
        plt.close(fig)
