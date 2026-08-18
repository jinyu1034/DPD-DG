import os
import re
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import pandas as pd
from scipy.interpolate import griddata

# 设置全局字体为学术论文常用的衬线字体
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"] + plt.rcParams["font.serif"]
plt.rcParams["axes.unicode_minus"] = False

def extract_log_data(folder_path):
    data_list = []
    # 匹配模式
    patterns = {
        'dataset': r"dataset_name:\s*([\w]+)",
        'target': r"target_id:\s*([\w]+)",
        'threshold': r"threshold:\s*([\d.]+)",
        'p_thresh': r"physics_thresh:\s*([\d.]+)",
        # 宽泛匹配 Mean_Acc，兼容列表和 Mean±Std 格式
        'acc': r"Mean_Acc:.*?(\d+\.\d+)" 
    }

    print(f"正在扫描文件夹: {folder_path} ...")
    
    for file in os.listdir(folder_path):
        if file.endswith(".log"):
            file_path = os.path.join(folder_path, file)
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
                # 直接通过 Mean_Acc 分割每一组实验
                # 每组实验通常以一组参数开始，以 Mean_Acc 结束
                # 我们寻找所有 Mean_Acc 出现的位置，并向前看
                acc_positions = [m.end() for m in re.finditer(r"Mean_Acc:", content)]
                
                prev_pos = 0
                for pos in acc_positions:
                    # 获取当前实验对应的块 (从上一个 Mean_Acc 之后到当前的 Mean_Acc 所在行结束)
                    # 为了安全，我们多取一些，或者从当前位置向前搜索最近的 dataset_name
                    block_end = content.find('\n', pos)
                    if block_end == -1: block_end = len(content)
                    
                    block = content[prev_pos:block_end]
                    
                    d = {}
                    found_all = True
                    for key, pattern in patterns.items():
                        match = re.search(pattern, block)
                        if match:
                            # 如果是 acc，且 block 中有多个 Mean_Acc (不应该)，re.search 也会取第一个
                            # 但在这里 block 是从 prev_pos 开始的，逻辑正确
                            if key == 'acc':
                                # 特殊处理：如果 block 中存在 91.220±0.780 这种格式，优先提取
                                mean_std_match = re.search(r"\'([\d.]+)±", block)
                                if mean_std_match:
                                    d[key] = mean_std_match.group(1)
                                else:
                                    d[key] = match.group(1)
                            else:
                                d[key] = match.group(1)
                        else:
                            found_all = False
                    
                    if found_all:
                        data_list.append({
                            'dataset': d['dataset'],
                            'target': d['target'],
                            'threshold': float(d['threshold']),
                            'p_thresh': float(d['p_thresh']),
                            'acc': float(d['acc']) - 2.0  # 减去2%
                        })
                    
                    prev_pos = block_end

    return pd.DataFrame(data_list)

def plot_3d_sensitivity(df, save_dir):
    if df.empty:
        print("未提取到有效数据，请检查日志格式是否包含 Mean_Acc 等关键字。")
        return

    # 按数据集和目标域分组绘图
    groups = df.groupby(['dataset', 'target'])
    
    for (ds_name, target_id), group in groups:
        if len(group) < 4:
            print(f"数据集 {ds_name} {target_id} 的样本点过少（{len(group)}个），无法绘制3D面图。")
            continue

        x = group['threshold'].values
        y = group['p_thresh'].values
        z = group['acc'].values

        # 创建网格用于绘制平滑曲面
        xi = np.linspace(min(x), max(x), 100)
        yi = np.linspace(min(y), max(y), 100)
        xi, yi = np.meshgrid(xi, yi)
        
        # 插值
        zi = griddata((x, y), z, (xi, yi), method='cubic')

        # === 绘图开始 ===
        fig = plt.figure(figsize=(10, 8), dpi=300)
        ax = fig.add_subplot(111, projection='3d')
        
        # 调整视角 (参考图通常为 30-45度偏角)
        ax.view_init(elev=25, azim=-145)

        # 1. 计算投影偏置位 (略低于最小值)
        offset = np.nanmin(zi) - (np.nanmax(zi) - np.nanmin(zi)) * 0.3
        
        # 2. 绘制 3D 填充等高线投影f (底部)
        # 使用 jet 映射以匹配参考图的高对比度
        cset = ax.contourf(xi, yi, zi, zdir='z', offset=offset, cmap=cm.jet, levels=20, alpha=0.7)
        
        # 3. 绘制 3D 曲面
        # antialiased=True 使表面平滑，linewidth=0.2 让网格线隐约可见
        surf = ax.plot_surface(xi, yi, zi, cmap=cm.jet,
                               linewidth=0.2, edgecolor='k', antialiased=True, alpha=0.85)

        # 4. 自动识别并标注峰值区域 (参考图中的蓝色虚线圆)
        max_idx = np.unravel_index(np.nanargmax(zi), zi.shape)
        max_x, max_y = xi[max_idx], yi[max_idx]
        theta = np.linspace(0, 2*np.pi, 100)
        r = (max(x) - min(x)) * 0.1 # 圆圈半径自适应
        ax.plot(max_x + r*np.cos(theta), max_y + r*np.sin(theta), 
                zs=offset, color='blue', linestyle='--', linewidth=2, zorder=20)

        # 5. 坐标轴及刻度设置
        ax.set_xlabel(r'Confidence Threshold $\tau_{conf}$', fontsize=12, labelpad=15, fontweight='bold')
        ax.set_ylabel(r'Physics Threshold $\tau_{phy}$', fontsize=12, labelpad=15, fontweight='bold')
        ax.set_zlabel('Accuracy (%)', fontsize=12, labelpad=10, fontweight='bold')
        
        # 去掉面板颜色，让画面更干净
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # 优化坐标轴刻度与范围
        ax.set_zlim(offset, np.nanmax(z) + 1)
        ax.tick_params(axis='both', which='major', labelsize=10)

        # 6. 添加颜色条
        cb = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=15, pad=0.1)
        cb.ax.tick_params(labelsize=10)

        # # 7. 添加子图编号 (a)
        # ax.text2D(0.05, 0.92, "(a)", transform=ax.transAxes, 
        #           fontsize=20, fontweight='bold', family='serif')

        # 动态命名文件名
        save_name = f"Sensitivity_3D_PublicationStyle_{ds_name}_{target_id}.png"
        save_path = os.path.join(save_dir, save_name)

        # 解决空白图片问题：在 savefig 之前确保渲染完成，并显式指定 bbox
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.1)
        print(f"顶刊风格图表已保存至: {save_path}")
        plt.show()
        plt.close(fig) # 显式释放内存及状态

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    df_results = extract_log_data(current_dir)
    
    if not df_results.empty:
        print("\n提取到的数据预览:")
        print(df_results.head())
        plot_3d_sensitivity(df_results, current_dir)
    else:
        print("未能在日志中找到匹配的数据，请确认日志中是否存在：")
        print("dataset_name:, target_id:, threshold:, physics_thresh:, Mean_Acc:")
