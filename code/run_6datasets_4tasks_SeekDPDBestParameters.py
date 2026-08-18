import subprocess
import sys
import itertools
import re
import numpy as np
import os
from datetime import datetime

# ================= 配置区域 =================

# 基础命令
python_executable = sys.executable
script_name = os.path.join(os.path.dirname(__file__), "main.py")

# 1. 固定参数
common_args = {
    "--epoch": "80",
    "--operation_num": "2",         
    "--ablation_mode": "M5",
    "--early_stop": "True",
    "--patience": "30",
    "--labeled_source_index": "0", # 明确指定第一个源域为有标签源域
}

# 域定义 (按照从小到大 1-4 排序)
domain_map = {
    'BJUT_Gear': ['20hz','30hz','40hz','50hz'],
    'CWRU_Bearing': ['0HP', '1HP', '2HP', '3HP'],
    'SDUT_Bearing': ['1500', '1800', '2000', '2500'],
    'SDUT_Gear': ['1500', '1800', '2000', '2500'],
    'HUST_Gear': ['0Nm', '0113Nm', '0226Nm', '0339Nm'],
    'Ottawa_Bearing': ['up','down','updown','downup'], 
}

# 数据集特定超参数配置
# 仅在此字典中定义的（未被注释）的数据集会被执行
dataset_hyperparams = {
    "BJUT_Gear": {
        "--lr": "1e-3",
        "--weight_decay": "1e-3",
        "--lambda_sup": "1",
        "--lambda_unsup": "0.1",
        "--lambda_prior": "1",
        "--lambda_sa": "5",
        "--lambda_sim": "10",
        "--lambda_coral": "10",
        "--prior_aug_mode": ["none"],
        "--select_top_k_prior": "48",
    },
    "CWRU_Bearing": {
        "--lr": "1e-3",
        "--weight_decay": "1e-3",
        "--lambda_sup": "10",
        "--lambda_unsup": "1",
        "--lambda_prior": "10",
        "--lambda_sa": "0.05",
        "--lambda_sim": "10",
        "--lambda_coral": "1",
        "--prior_aug_mode": ["none"],
        "--select_top_k_prior": "20",
    },
    "SDUT_Bearing": {
        "--lr": "1e-3",
        "--weight_decay": "1e-3",
        "--lambda_sup": "1",
        "--lambda_unsup": "1",
        "--lambda_prior": "1",
        "--lambda_sa": "5",
        "--lambda_sim": "100",
        "--lambda_coral": "10", 
        "--prior_aug_mode": ["none"],
        "--select_top_k_prior": "20", 
    },
    "SDUT_Gear": {
        "--lr": "1e-3",
        "--weight_decay": "1e-3",
        "--lambda_sup": "10",
        "--lambda_unsup": "1",
        "--lambda_prior": "10",
        "--lambda_sa": "0.05",
        "--lambda_sim": "10",
        "--lambda_coral": "1",
        "--prior_aug_mode": ["none"],
        "--select_top_k_prior": "48", 
    },
    "HUST_Gear": {
        "--lr": "1e-3",
        "--weight_decay": "1e-3",
        "--lambda_sup": "10",
        "--lambda_unsup": "1",
        "--lambda_prior": "10",
        "--lambda_sa": "5",
        "--lambda_sim": "1",
        "--lambda_coral": "1",
        "--prior_aug_mode": ["none"],
        "--select_top_k_prior": "20",      
    },
    "Ottawa_Bearing": {
        "--lr": "1e-3",
        "--weight_decay": "1e-3",
        "--lambda_sup": "1",
        "--lambda_unsup": "0.1",
        "--lambda_prior": "10",
        "--lambda_sa": "0.05",
        "--lambda_sim": "10",
        "--lambda_coral": "1",
        "--prior_aug_mode": ["dropout"],
        "--select_top_k_prior": "20",
    }
}

# 2. 可变参数网格 (Grid Search)
param_grid = {
    "--threshold": ["0.5","0.6","0.7", "0.8", "0.9", "0.95"],
    "--physics_thresh": ["0.3", "0.4","0.5", "0.6", "0.7", "0.8",],
}

# ===========================================

def generate_experiments(grid):
    if not grid:
        return [{}]
    keys = grid.keys()
    values = grid.values()
    combinations = itertools.product(*values)
    return [dict(zip(keys, combo)) for combo in combinations]

def build_command(experiment_params, dataset_specific_params, target, source_list):
    cmd = [python_executable, script_name]
    
    # 固定参数
    for key, value in common_args.items():
        cmd.append(key)
        if value is not None and value != "":
            cmd.append(str(value))
    
    # 添加数据集特定超参数
    
    # 注意：dataset_name 由外部调用者在 cmd 头部添加，或者在这里不处理
    # 为了避免重复，这里只添加 params 中的参数

    for key, value in dataset_specific_params.items():
        cmd.append(key)
        if value is not None and value != "":
            if isinstance(value, list):
                cmd.extend([str(v) for v in value])
            else:
                cmd.append(str(value))

    # 添加实验变化参数
    for key, value in experiment_params.items():
        cmd.append(key)
        if value is not None and value != "":
            if isinstance(value, list):
                cmd.extend([str(v) for v in value])
            else:
                cmd.append(str(value))
    
    # 添加 Target 和 Source List
    cmd.append("--target_id")
    cmd.append(str(target))
    
    cmd.append("--source_id_list")
    cmd.extend([str(s) for s in source_list])
    
    return cmd

def parse_output(output_str):
    """
    从输出日志中提取 Mean_Acc 和 Mean_Accbyloss 的结果列表
    格式示例: Mean_Acc: [96.36, 95.39, 95.64, '95.797±0.411']
    我们只需要提取前 3 个数字 (operation_num=3)
    """
    acc_pattern = re.search(r"Mean_Acc: \[([^\]]+)\]", output_str)
    loss_acc_pattern = re.search(r"Mean_Accbyloss: \[([^\]]+)\]", output_str)
    
    acc_values = []
    loss_acc_values = []
    
    if acc_pattern:
        # 分割字符串，去除最后的字符串项 (带引号和±的)
        parts = acc_pattern.group(1).split(",")
        # 过滤掉带单引号或双引号的项 (即最后的统计项)
        nums = [float(p.strip()) for p in parts if "'" not in p and '"' not in p]
        acc_values = nums
        
    if loss_acc_pattern:
        parts = loss_acc_pattern.group(1).split(",")
        nums = [float(p.strip()) for p in parts if "'" not in p and '"' not in p]
        loss_acc_values = nums
        
    return acc_values, loss_acc_values

def main():
    # 生成网格搜索组合
    experiments = generate_experiments(param_grid)
    
    # 遍历 dataset_hyperparams 中定义的活跃数据集
    # 这样用户可以通过注释放入/移除数据集来控制运行哪些数据集
    for ds_name, ds_params in dataset_hyperparams.items():
        if ds_name not in domain_map:
            print(f"Error: 数据集 {ds_name} 不在 domain_map 中，请在 domain_map 中定义其域列表。")
            continue
            
        domains = domain_map[ds_name]
        print(f"\n====== 开始处理数据集: {ds_name} ======")
        
        # 遍历 Grid Search 参数
        for exp_idx, exp_params in enumerate(experiments):
            print(f"\n--- 实验参数组合 {exp_idx+1}/{len(experiments)}: {exp_params} ---")
            
            all_task_acc = []
            all_task_loss_acc = []
            
            # 生成 4 个任务
            for i in range(4):
                labeled_source = domains[i]
                target_domain = domains[(i+1) % 4]
                
                # 其余为无标签源域
                unlabeled_sources = [d for k, d in enumerate(domains) if k != i and k != (i+1)%4]
                
                # 构建 source_id_list，确保 Labeled source 在第一个
                source_list = [labeled_source] + unlabeled_sources
                
                task_name = f"Task{i+1}: Src(L)={labeled_source} -> Tgt={target_domain}, Src(U)={unlabeled_sources}"
                print(f"Running {task_name}")
                
                # 构建基础命令
                cmd = [python_executable, script_name, "--dataset_name", ds_name]
                
                # 添加其他参数
                cmd_rest = build_command(exp_params, ds_params, target_domain, source_list)
                # build_command 返回的是 [exe, script, args...]，我们只需要 args 部分
                # 修改 build_command 逻辑有些复杂，不如直接构建
                
                final_cmd = cmd + cmd_rest[2:] # 去掉 exe 和 script
                
                # print("Command:", " ".join(final_cmd))
                
                try:
                    # 运行命令，合并 stdout 和 stderr
                    result = subprocess.run(
                        final_cmd, 
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True, 
                        errors='replace',
                        check=True
                    )
                    
                    full_output = result.stdout
                    # print(full_output) # 如果需要完整日志可以取消注释
                    
                    accs, loss_accs = parse_output(full_output)
                    
                    if not accs:
                        print(f"!!! 未能从输出中解析到结果: {task_name}")
                        # 打印最后一部分输出以辅助排查
                        print("Last 1000 chars of output:")
                        print(full_output[-1000:])
                    else:
                        print(f"   {task_name} Acc: {accs}")
                        if loss_accs:
                            print(f"   {task_name} Accbyloss: {loss_accs}")
                        
                        all_task_acc.extend(accs)
                        all_task_loss_acc.extend(loss_accs)
                        
                except subprocess.CalledProcessError as e:
                    print(f"!!! {task_name} 失败 (Code {e.returncode})")
                    print(e.stderr)
                except Exception as e:
                    print(f"!!! {task_name} 发生异常: {e}")

            # 汇总 4 个任务的结果 (共 12 个数据点)
            print(f"\n>>> 数据集 {ds_name} 汇总 (共 {len(all_task_acc)} 次运行) <<<")
            if all_task_acc:
                final_mean_acc = np.mean(all_task_acc)
                final_std_acc = np.std(all_task_acc)
                print(f"Total Mean_Acc: {final_mean_acc:.3f} ± {final_std_acc:.3f}")
                print(f"Detail Accs: {all_task_acc}")
            
            if all_task_loss_acc:
                final_mean_loss = np.mean(all_task_loss_acc)
                final_std_loss = np.std(all_task_loss_acc)
                print(f"Total Mean_Accbyloss: {final_mean_loss:.3f} ± {final_std_loss:.3f}")
                print(f"Detail Accbyloss: {all_task_loss_acc}")

            # --- 写入汇总日志 ---
            # 确定日志路径：项目根目录/results/数据集名/
            results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", ds_name)
            os.makedirs(results_dir, exist_ok=True)
            summary_log_path = os.path.join(results_dir, f"DPD_DG_4task_{ds_name}.log")
            
            with open(summary_log_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*50}\n")
                f.write(f"运行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"数据集名字: {ds_name}\n")
                f.write(f"网格搜索参数 (param_grid): {exp_params}\n")
                f.write(f"数据集超参数 (dataset_hyperparams): {ds_params}\n")
                if all_task_acc:
                    f.write(f"Total Mean_Acc: {final_mean_acc:.3f} ± {final_std_acc:.3f}\n")
                    f.write(f"Detail Accs: {all_task_acc}\n")
                if all_task_loss_acc:
                    f.write(f"Total Mean_Accbyloss: {final_mean_loss:.3f} ± {final_std_loss:.3f}\n")
                    f.write(f"Detail Accbyloss: {all_task_loss_acc}\n")
                f.write(f"{'='*50}\n")
            
            print(f"汇总日志已更新: {summary_log_path}")

if __name__ == "__main__":
    main()
