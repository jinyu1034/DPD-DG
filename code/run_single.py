import subprocess
import sys
import itertools
import os

# ================= 配置区域 =================

# 基础命令 (通常不需要修改)
python_executable = sys.executable # 使用当前运行此脚本的 Python 解释器
import os
script_name = os.path.join(os.path.dirname(__file__), "main.py")

# 1. 固定参数 (每次运行都一样的参数)
# 如果是开关参数(如 --use_mixup)，值设为 None 或 "" 即可，只要键存在就会被添加
# 如果是列表参数(如 --source_id_list)，值设为列表
common_args = {
    "--epoch": "85",
    "--operation_num": "5",         
    "--ablation_mode": "M5",    # 开关参数
    "--early_stop": "True",  # 启用自停
    "--patience": "35",  # 设置耐心值为 10
    #"--select_top_k_prior": "48",
    #"--save_cm": "True",
    # "--save_tsne": "True",
    "--save_attention": "True",
}

# 数据集配置列表
dataset_configs = [
    # {
    #     "--dataset_name": "BJUT_Gear",
    #     "--target_id": "50hz",
    #     "--source_id_list": ['40hz', '30hz', '20hz'],
    #     #"--select_top_k_prior": "20",
    #     "--lambda_sup": "1",
    #     "--lambda_unsup": "1",
    #     "--lambda_prior": "1",
    #     "--lambda_sa": "0.05",
    #     "--lambda_sim": "1",
    #     "--lambda_coral": "10",
    #     "--weight_decay": "1e-3",
    #     "--threshold": "0.8",
    #     "--physics_thresh": "0.7",  
    #     "--prior_aug_mode": ["none"],
    #     "--select_top_k_prior": "48",
    #     # "--lambda_sup": "1",
    #     # "--lambda_unsup": "0.1",
    #     # "--lambda_prior": "1",
    #     # "--lambda_sa": "5",
    #     # "--lambda_sim": "10",
    #     # "--lambda_coral": "10",
    #     # "--weight_decay": "1e-3",
    #     # "--threshold": "0.8",
    #     # "--physics_thresh": "0.6",  
    #     # "--prior_aug_mode": ["none"], 
    # },
    # {
    #     "--dataset_name": "CWRU_Bearing",
    #     "--target_id": "2HP",
    #     "--source_id_list": ['1HP', '0HP', '3HP'],
    #     "--lr": "1e-3",
    #     "--lambda_sup": "10",
    #     "--lambda_unsup": "1",
    #     "--lambda_prior": "10",
    #     "--lambda_sa": "0.05",
    #     "--lambda_sim": "10",
    #     "--lambda_coral": "1",
    #     "--weight_decay": "1e-3",
    #     "--threshold": "0.8",
    #     "--physics_thresh": "0.7",  
    #     "--prior_aug_mode": ["none"],
    #     "--select_top_k_prior": "48",
    # },
    # {
    #     "--dataset_name": "SDUT_Bearing",
    #     # "--target_id": "2500",
    #     # "--source_id_list": ['2000', '1500', '1800'],
    #     "--target_id": "1800",
    #     "--source_id_list": ['1500', '2000', '2500'],
    #     "--lr": "1e-3",
    #     "--lambda_sup": "1",
    #     "--lambda_unsup": "1",
    #     "--lambda_prior": "1",
    #     "--lambda_sa": "0.05",
    #     "--lambda_sim": "10",
    #     "--lambda_coral": "1",
    #     "--weight_decay": "1e-3",
    #     "--threshold": "0.9",
    #     "--physics_thresh": "0.7",  
    #     "--prior_aug_mode": ["dropout"],
    #     "--select_top_k_prior": "20", 
    #     # "--lr": "1e-3",
    #     # "--lambda_sup": "1",
    #     # "--lambda_unsup": "1",
    #     # "--lambda_prior": "1",
    #     # "--lambda_sa": "5",
    #     # "--lambda_sim": "10",
    #     # "--lambda_coral": "1",
    #     # "--weight_decay": "1e-3",
    #     # "--threshold": "0.9",
    #     # "--physics_thresh": "0.7",  
    #     # "--prior_aug_mode": ["dropout"],
    #     # "--select_top_k_prior": "48",
    #     # "--lr": "1e-3",
    #     # "--lambda_sup": "1",
    #     # "--lambda_unsup": "1",
    #     # "--lambda_prior": "1",
    #     # "--lambda_sa": "5",
    #     # "--lambda_sim": "100",
    #     # "--lambda_coral": "10",
    #     # "--weight_decay": "1e-3",
    #     # "--threshold": "0.8",
    #     # "--physics_thresh": "0.7",  
    #     # "--prior_aug_mode": ["none"],
    #     # "--select_top_k_prior": "20",
    # },
    {
        "--dataset_name": "SDUT_Gear",
        "--target_id": "1800",
        "--source_id_list": ['1500', '2000', '2500'],
        "--lr": "1e-3",
        "--lambda_sup": "10",
        "--lambda_unsup": "1",
        "--lambda_prior": "10",
        "--lambda_sa": "0.05",
        "--lambda_sim": "10",
        "--lambda_coral": "1",
        "--weight_decay": "1e-3",
        "--threshold": "0.8",
        "--physics_thresh": "0.7",  
        "--prior_aug_mode": ["none"],
        "--select_top_k_prior": "48",
        # "--lr": "1e-3",
        # "--lambda_sup": "10",
        # "--lambda_unsup": "1",
        # "--lambda_prior": "10",
        # "--lambda_sa": "0.05",
        # "--lambda_sim": "10",
        # "--lambda_coral": "1",
        # "--weight_decay": "1e-2",
        # "--threshold": "0.8",
        # "--physics_thresh": "0.7",  
        # "--prior_aug_mode": ["none"],
        # "--select_top_k_prior": "48", 
    },
    # {
    #     "--dataset_name": "HUST_Gear",
    #     "--target_id": "0339Nm",
    #     "--source_id_list": ['0226Nm', '0Nm', '0113Nm'],
    #     "--lr": "1e-3",
    #     "--lambda_sup": "10",
    #     "--lambda_unsup": "1",
    #     "--lambda_prior": "10",
    #     "--lambda_sa": "5",
    #     "--lambda_sim": "1",
    #     "--lambda_coral": "1",
    #     "--weight_decay": "1e-3",
    #     "--threshold": "0.8",
    #     "--physics_thresh": "0.7",  
    #     "--prior_aug_mode": ["none"],
    #     "--select_top_k_prior": "48",      
    # },
    # {
    #     "--dataset_name": "Ottawa_Bearing",
    #     "--target_id": "downup",
    #     "--source_id_list": ['updown', 'up', 'down'],
    #     "--lr": "1e-3",
    #     "--lambda_sup": "1",
    #     "--lambda_unsup": "0.1",
    #     "--lambda_prior": "10",
    #     "--lambda_sa": "0.05",
    #     "--lambda_sim": "10",
    #     "--lambda_coral": "1",
    #     #"--lambda_align_target": "0.5",
    #     "--weight_decay": "1e-2",
    #     "--threshold": "0.9",
    #     "--physics_thresh": "0.7",  
    #     "--prior_aug_mode": ["dropout"],
    #     "--select_top_k_prior": "48",
    # }
]

# 2. 可变参数网格 (Grid Search)
# 在这里定义你想遍历的参数及其候选值列表
# 脚本会自动生成所有可能的组合 (笛卡尔积)
param_grid = {
    #"--model_name": ["CCDG"],
    "--ablation_mode": ["M5"],
    #"--save_best_after_epoch": ["20"],
    # "--prior_aug_mode": ["dropout"],
    # "--select_top_k_prior": ["48"],
    # "--lambda_k": ["20"],
    # # "--lr": ["0.001"],
    # "--weight_decay": ["1e-2"],
    # #"--prior_stats_source": ["both", "time", "fft"],
    # "--threshold": ["0.5","0.6","0.7", "0.8", "0.9", "0.95"],
    # "--physics_thresh": ["0.3", "0.4","0.5", "0.6", "0.7", "0.8",],
    #"--physics_thresh":
    #"--physics_thresh": ["0.25", "0.3", "0.35", "0.4", "0.45", "0.5"],
    # 你可以在这里添加更多参数，例如:
    # "--lr": ["0.001", "0.01"],
}

# ===========================================

def generate_experiments(grid):
    """生成所有参数组合"""
    if not grid:
        return [{}]
    
    keys = grid.keys()
    values = grid.values()
    # itertools.product 生成笛卡尔积
    combinations = itertools.product(*values)
    
    experiments = []
    for combo in combinations:
        experiments.append(dict(zip(keys, combo)))
    return experiments

def build_command(experiment_params, dataset_params):
    cmd = [python_executable, script_name]
    
    # 添加固定参数
    for key, value in common_args.items():
        cmd.append(key)
        if value is not None and value != "":
            if isinstance(value, list):
                cmd.extend([str(v) for v in value])
            else:
                cmd.append(str(value))
    
    # 添加数据集参数
    for key, value in dataset_params.items():
        cmd.append(key)
        if value is not None and value != "":
            if isinstance(value, list):
                cmd.extend([str(v) for v in value])
            else:
                cmd.append(str(value))

    # 添加实验特定参数
    for key, value in experiment_params.items():
        cmd.append(key)
        if value is not None and value != "":
            if isinstance(value, list):
                cmd.extend([str(v) for v in value])
            else:
                cmd.append(str(value))
                
    return cmd

def main():
    # 生成实验列表
    experiments = generate_experiments(param_grid)
    
    total_per_dataset = len(experiments)
    total_all = total_per_dataset * len(dataset_configs)
    
    print(f"根据参数网格，计划运行 {len(dataset_configs)} 个数据集，每个数据集 {total_per_dataset} 组实验，共 {total_all} 组实验...\n")
    
    # 获取当前环境并显式设置 LOKY_MAX_CPU_COUNT
    current_env = os.environ.copy()
    cpu_count = os.cpu_count()
    if cpu_count is None:
        cpu_count = 4 # Fallback
    current_env['LOKY_MAX_CPU_COUNT'] = str(cpu_count)

    global_count = 0
    
    for ds_idx, ds_config in enumerate(dataset_configs):
        if "--dataset_name" not in ds_config:
            continue
        print(f"====== 正在处理数据集: {ds_config['--dataset_name']} ({ds_idx+1}/{len(dataset_configs)}) ======")
        
        for i, params in enumerate(experiments):
            global_count += 1
            print(f"=== 正在运行总进度 {global_count}/{total_all} (当前数据集 {i+1}/{total_per_dataset}) ===")
            print(f"当前数据集配置: {ds_config}")
            print(f"当前变化参数: {params}")
            
            cmd = build_command(params, ds_config)
            cmd_str = " ".join(cmd)
            print(f"执行命令: {cmd_str}")
            
            try:
                # check=True 会在命令返回非零退出码时抛出异常
                # 显式传递 env
                subprocess.run(cmd, check=True, env=current_env)
                print(f"=== 第 {global_count} 组实验完成 ===\n")
            except subprocess.CalledProcessError as e:
                print(f"!!! 第 {global_count} 组实验失败 (退出码 {e.returncode}) !!!")
                # 如果希望出错后继续运行下一个实验，可以将下面的 break 注释掉
                # break 
            except KeyboardInterrupt:
                print("\n用户中断实验。")
                sys.exit(1)

if __name__ == "__main__":
    main()
