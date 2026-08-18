import argparse
import logging
import os
from datetime import datetime
from pathlib import Path
import torch
import numpy as np
from data.construct_loader import Fault_dataset
from utils.train_test import train_test

def set_random_seed(seed=0):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def setlogger(path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logFormatter = logging.Formatter("%(message)s")

    fileHandler = logging.FileHandler(path)
    fileHandler.setFormatter(logFormatter)
    logger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)

def result_log(Indicators="",target="",source="",results=None):
    results = [round(x, 2) for x in results]
    mean = np.mean(results)
    std = np.std(results)
    results.append(f"{mean:.3f}±{std:.3f}")
    logging.info(f"Indicators:{Indicators} Task:{target}-{source} Mean_{Indicators}: {results}")

# === 主程序概述 ===
# 解析命令行参数 -> 构建数据集与训练器 -> 设置日志 -> 调用训练/测试例程。
# 该脚本是整个项目的入口，其它模块都通过这里被串联。
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def parse_args():
    """
    解析命令行参数的函数
    Returns:
        args: 解析后的命令行参数对象
    """
    # 创建一个解析器
    parser = argparse.ArgumentParser(description='Train')
    # 添加一个参数，表示数据集的名称
    parser.add_argument('--dataset_name', type=str, default="BJUT_Gear", help='name of dataset')
    # 添加一个参数，表示目标域的ID
    parser.add_argument('--target_id', type=str, default='40hz', help='target domain')
    # 添加一个参数，表示源域的ID列表
    parser.add_argument('--source_id_list', type=str, nargs='+', default=['20hz', '30hz', '50hz'], help='List of source domain IDs')
    # 添加一个参数，表示作为有标签源域的索引（0-based），默认为0（即source_id_list中的第一个）
    parser.add_argument('--labeled_source_index', type=int, default=0, help='Index of the source domain to be used as labeled data (0-based)')
    # 添加一个参数，表示数据集划分的比例
    parser.add_argument('--data_ratio', type=float, default=0.8, help='percentage of dataset division (80% train, 20% val)')
    # 添加一个参数，表示要删除的标签
    parser.add_argument('--miss_class', nargs='+', type=int, default=[], help='deleting labels from a class')
    # 添加一个参数，表示是否进行傅里叶变换
    parser.add_argument('--FFT', type=str2bool, default=True, help='whether to Fourier transform the data')
    # 添加一个参数，表示数据归一化的方法
    parser.add_argument('--normalize_type', type=str, default='z-score',choices=['0-1', 'min-max', '-1-1', 'mean-std','z-score'], help='data normalization methods')
    # 添加一个参数，表示方法的名称
    parser.add_argument('--model_name', type=str, default='DPD_DG', help='the name of the method')
    # 消融实验
    parser.add_argument('--ablation_mode', type=str, default='M5', choices=['M0', 'M0_', 'M1', 'M2', 'M3', 'M4', 'M5'], help='Ablation study mode: M0(Baseline), M0_(PriorOnly), M1, M2, M3, M4, M5')
    # 添加一个参数，表示学习率
    parser.add_argument('--lr', type=float, default= 1e-3, help='the learning rate')
    # 添加一个参数，表示最大迭代次数
    parser.add_argument('--epoch', type=int, default=100, help='the max number of epoch')
    # 添加一个参数，表示实验的重复次数
    parser.add_argument('--operation_num', type=int, default=1, help='the repeat operation of experiments')
    # 添加一个参数，表示是否使用物理先验统计信息，可接受数字、True/False 或字符串形式（yes/no）的布尔值
    parser.add_argument('--use_prior_stats', type=str2bool, default=False, help='enable prior-knowledge statistics such as kurtosis/skewness')
    # 添加一个参数，表示先验统计信息的来源,时域、频域或两者
    parser.add_argument('--prior_stats_source', type=str, default='both', choices=['time', 'fft', 'both'], help='which signal space to compute prior stats from')
    # 添加一个参数，表示分类损失项的权重
    parser.add_argument('--lambda_sup', type=float, default=1.0, help='weight for classification loss term')
     # 添加一个参数，表示无监督伪标签损失的权重
    parser.add_argument('--lambda_unsup', type=float, default=1.0, help='weight for unsupervised pseudo-label loss')
    # 添加一个参数，表示先验知识一致性损失的权重
    parser.add_argument('--lambda_prior', type=float, default=10, help='weight for prior knowledge consistency loss')
    # 添加一个参数，表示语义对齐损失(Semantic Alignment)的权重
    parser.add_argument('--lambda_sa', type=float, default=0.5, help='weight for semantic alignment loss')
    # 添加一个参数，表示相似度损失(Similarity)的权重
    parser.add_argument('--lambda_sim', type=float, default=10.0, help='weight for similarity loss')
    # 添加一个参数，表示CORAL损失的权重
    parser.add_argument('--lambda_coral', type=float, default=5.0, help='weight for CORAL loss')
    # 添加一个参数，表示目标域对齐项的整体权重
    parser.add_argument('--lambda_align_target', type=float, default=0.5, help='weight for target domain alignment term')
    # 添加一个参数，表示固定阈值控制
    parser.add_argument('--threshold', type=float, default=0.8, help='fixed threshold for pseudo-label filtering')
    # 添加一个参数，表示物理一致性检查的阈值（余弦相似度）
    parser.add_argument('--physics_thresh', type=float, default=0.7, help='threshold for physics consistency check (cosine similarity)')
    # 添加一个参数，表示是否启用早停机制，若训练无进展则停止训练，可接受数字、True/False 或字符串形式（yes/no）的布尔值
    parser.add_argument('--early_stop', type=str2bool, default=False, help='whether to enable early stopping')
    # 添加一个参数，表示早停的耐心轮数，即在多少轮内验证集性能无提升则停止训练
    parser.add_argument('--patience', type=int, default=100, help='patience rounds for early stopping')
    # 先验增强
    parser.add_argument('--prior_aug_mode', type=str, default='none', choices=['none', 'gaussian', 'dropout'], help='Augmentation mode for prior features: none, gaussian, dropout')
    # 优化：优化器选择和调度器
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adam', 'adamw'], help='optimizer to use for DPD_DG only (others keep baseline)')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay for optimizer')
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['none', 'cosine', 'step'], help='learning rate scheduler to use for DPD_DG')
    # 特征选择
    parser.add_argument('--select_top_k_prior', type=int, default=0, help='If > 0, select top K prior features based on Random Forest importance on labeled source')
    # 模型保存
    parser.add_argument('--save_best_after_epoch', type=int, default=0, help='Start saving the best model only after this epoch')
    # 可视化
    parser.add_argument('--save_cm', type=str2bool, default=False, help='whether to save confusion matrix during testing')    
    parser.add_argument('--save_tsne', type=str2bool, default=False, help='whether to save t-SNE visualization during testing')
    parser.add_argument('--save_attention', type=str2bool, default=False, help='whether to save physical attention heatmap during testing')
    
    # 解析参数
    args = parser.parse_args()
    # 返回参数
    return args

def setup_logging(args, save_dir):
    """
    设置日志记录功能
    参数:
        args: 包含程序运行参数的对象
        save_dir: 日志文件保存的目录路径
    功能:
        1. 创建日志保存目录（如果不存在）
        2. 配置日志记录器
        3. 记录程序运行时间和参数
    """
    # 如果保存目录不存在，则创建
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    # 获取logger对象
    logger = logging.getLogger()
    # 如果logger对象有handler，则清空
    if logger.hasHandlers(): #检查该logger是否已经配置了处理器(handler)
        for handler in list(logger.handlers): #遍历所有的处理器(handler)
            handler.close() #关闭处理器，释放资源
            logger.removeHandler(handler)#移除处理器
    # 设置logger，将日志保存到指定目录
    log_file = save_path / f"{args.model_name}_{args.dataset_name}.log"
    setlogger(str(log_file))
    # 打印空行
    logging.info("\n")
    # 获取当前时间
    time = datetime.strftime(datetime.now(), '%m-%d %H:%M:%S')
    # 打印当前时间
    logging.info('{}'.format(time))
    # 打印args中的所有参数
    # for k, v in args.__dict__.items():
    #     logging.info("{}: {}".format(k, v))
    return str(log_file)

from torch.utils.data import ConcatDataset, DataLoader

def train_and_evaluate(args,operation, dataset):
    """
    训练并评估模型函数

    参数:
        args: 参数配置对象，包含训练所需的各项参数
        operation: 操作对象，包含模型训练和测试的方法
        dataset: 数据集对象，提供数据加载功能
    """
    # 每次 operation 对应一次完整的训练+评估流程，便于统计 different 随机性下的稳定性。
    accuracy=[]
    f1_scores=[]
    mcc_scores=[]
    
    
    accuracy_byloss=[]
    f1_scores_byloss=[]
    mcc_scores_byloss=[]
    

    # 将源数据集ID转换为字符串
    source_list_string = "-".join(map(str, args.source_id_list))
    # 循环执行指定次数的操作，通常用于多次实验取平均
    for i  in range(args.operation_num):
        # 固定随机种子，确保多个 operation 可复现并且日志可对齐。
        set_random_seed(42)
        
        # 1. 加载源域数据 (80% 训练, 20% 验证)
        # 训练集列表 (每个源域的 80%)，包含了3个域数据集，每个信号1600*0.8*1*(512+先验长度)、标签1280*1、域标签1280*1、样本索引1280和先验特征1280*(先验长度)的综合数据集
        # 验证集列表 (每个源域的 20%)，包含了3个域数据集，每个信号1600*0.2*1*(512+先验长度)、标签320*1、域标签320*1、样本索引320和先验特征320*(先验长度)的综合数据集
        source_train_datasets, source_val_datasets = dataset.Loader(args.source_id_list)
        
        # 根据 labeled_source_index 调整 source_train_datasets 的顺序
        # 将指定的有标签源域移动到列表的第一个位置 (索引0)，其余作为无标签域
        if 0 <= args.labeled_source_index < len(source_train_datasets):
            labeled_dataset = source_train_datasets.pop(args.labeled_source_index)
            source_train_datasets.insert(0, labeled_dataset)
            
            # 同时也需要调整验证集顺序，虽然验证集目前是全部合并，但为了逻辑一致性建议保持同步
            labeled_val_dataset = source_val_datasets.pop(args.labeled_source_index)
            source_val_datasets.insert(0, labeled_val_dataset)

            # 打印日志信息
            logging.info("Train_Source:{} prior_dim:{} Test_Target:{}".format(source_list_string, source_train_datasets[0].prior_tensor.shape, args.target_id))
            logging.info(f"已选择 source_id_list[{args.labeled_source_index}] (ID: {args.source_id_list[args.labeled_source_index]}) 作为有标签源域。")
            
            # === 移除无标签源域的真实标签 ===
            # 遍历除了第一个（有标签源域）之外的所有数据集
            # 将其 y_tensor (标签) 全部置为 -1，确保模型无法利用这些标签
            if args.model_name == 'DPD_DG':
                logging.info(f"保留无标签源域的真实标签，仅用于计算DPD_DG伪标签生成的质量（error指标），不参与模型训练。")
            else:
                for idx in range(1, len(source_train_datasets)):
                    unlabeled_ds = source_train_datasets[idx]
                    # 假设 dataset 是 CustomTensorDataset 类型，直接修改其 y_tensor
                    # 创建全 -1 的标签张量，形状与原标签一致
                    fake_labels = torch.full_like(unlabeled_ds.y_tensor, -1)
                    unlabeled_ds.y_tensor = fake_labels
                    logging.info(f"已移除无标签源域 (Index {idx}:(ID: {args.source_id_list[idx]}) ) 的真实标签，替换为 -1。")
        else:
            logging.warning(f"labeled_source_index {args.labeled_source_index} 超出范围，将默认使用第一个源域作为有标签源域。")

        # 将val_dataset中所有3个源域的验证集拼接成一个大的验证集，用于选择最佳模型
        val_dataset = ConcatDataset(source_val_datasets) 
        val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)#DataLoader会自动将数据集分成多个批次，每个批次包含指定数量(128)的样本，shuffle=False：表示在每个批次开始时不打乱数据顺序

        # 2. 加载目标域数据 (100% 测试)
        # 我们需要目标域的全部数据。dataset.Loader会返回分割后的两部分，我们将它们拼接起来。
        target_test_ds1, target_test_ds2 = dataset.Loader([args.target_id])
        full_target_dataset = ConcatDataset(target_test_ds1 + target_test_ds2)
        target_loader = DataLoader(full_target_dataset, batch_size=128, shuffle=False)

        
        # 初始化模型，设置类别数量
        operation.setup(dataset.n_class)
        
        # 训练阶段：
        # 输入: source_train_datasets (源域训练集列表, 用于模型内部构造 batch)
        # 验证: val_loader (源域验证集, 用于选择最佳模型)
        operation.train(i, source_train_datasets, val_loader)
        
        # 测试阶段：
        # 读取最佳权重，对目标域全量数据进行测试
        # 更新以解包相似度分数
        # 传入 val_loader 作为 Source_dataloader 用于 t-SNE 可视化
        acc_best_acc, f1_best_acc, mcc_best_acc, acc_best_loss, f1_best_loss, mcc_best_loss = operation.test(i, target_loader, val_loader)
        
        accuracy.append(acc_best_acc * 100)
        f1_scores.append(f1_best_acc * 100)
        mcc_scores.append(mcc_best_acc * 100)
        
        accuracy_byloss.append(acc_best_loss * 100)
        f1_scores_byloss.append(f1_best_loss * 100)
        mcc_scores_byloss.append(mcc_best_loss * 100)

    result_log(Indicators="Acc", target=args.target_id, source=args.source_id_list , results=accuracy)
    result_log(Indicators="F 1", target=args.target_id, source=args.source_id_list , results=f1_scores)
    result_log(Indicators="MCC", target=args.target_id, source=args.source_id_list , results=mcc_scores)
    
    result_log(Indicators="Accbyloss", target=args.target_id, source=args.source_id_list , results=accuracy_byloss)
    result_log(Indicators="F 1byloss", target=args.target_id, source=args.source_id_list , results=f1_scores_byloss)
    result_log(Indicators="MCCbyloss", target=args.target_id, source=args.source_id_list , results=mcc_scores_byloss)
   
# 如果当前模块是主模块，则执行以下代码
if __name__ == '__main__':
    # 解析命令行并构建数据集，Fault_dataset 会在初始化和 Loader 阶段写入 signal/prior信息。
    args = parse_args()

    # --- Move Logging Setup Here (Moved from bottom) ---
    # 设置项目根目录和结果目录
    project_root = Path(__file__).resolve().parent.parent
    args.project_root = str(project_root)
    args.results_dir = str(project_root / 'results' / args.dataset_name)
    args.trained_models_dir = str(project_root / 'trained_models' / args.dataset_name / args.model_name)
    # 设置日志记录 (Initialize logging BEFORE using logging.info)
    log_file = setup_logging(args, args.results_dir)
    logging.info(f"日志将保存至: {log_file}")
    # ---------------------------------------------------

    # === 消融实验逻辑 ===
    logging.info(f"Ablation Mode Selected: {args.ablation_mode}")
    
    # === 默认值 (M5 - 完整模型) ===
    if args.model_name == 'DPD_DG':
        args.enable_attention = True    # Sec 2.3: SPA
        args.enable_explicit = True     # Sec 2.4: Explicit Drive
        args.enable_papl = True         # Sec 2.5: Implicit Drive (PAPL & Recon)
        args.enable_align = True        # Sec 2.6: Alignment
        args.enable_dfe = True          # Robustness (Keep enabled usually)
        args.enable_rf = True           # Global Static Importance
        args.use_prior_stats = True     # 强制开启统计特征 (M1-M5 需要)，M0 会在下方被关闭
    
    # 确保完整模型的默认 lambda 值为正
    if args.lambda_unsup <= 0: args.lambda_unsup = 1.0
    if args.lambda_prior <= 0: args.lambda_prior = 10.0
    if args.lambda_sa <= 0: args.lambda_sa = 0.5
    if args.lambda_sim <= 0: args.lambda_sim = 10.0
    if args.lambda_coral <= 0: args.lambda_coral = 5.0
    if args.lambda_align_target <= 0: args.lambda_align_target = 0.5

    if args.ablation_mode == 'M0' or args.model_name != 'DPD_DG':
        # M0: Baseline (Source-only CNN) - No Physics, No Adaptation
        args.prior_dim = 0          # Disable physics input entirely
        args.select_top_k_prior = 0 # Ensure Dataset doesn't load prior
        args.use_prior_stats = False # Disable prior stats calculation
        args.lambda_unsup = 0.0     # <--- FIX: Disable pseudo-labeling for true Source-only baseline
        args.lambda_prior = 0.0
        args.lambda_sa = 0.0
        args.lambda_sim = 0.0
        args.lambda_coral = 0.0
        args.lambda_align_target = 0.0
        args.threshold = 0.0
        
        args.enable_attention = False
        args.enable_explicit = False
        args.enable_papl = False
        args.enable_align = False
        args.enable_dfe = False
        args.enable_rf = False
        
        logging.info("Configured M0: Baseline (No Physics, No Adaptation)")

    elif args.ablation_mode == 'M0_':
        # M0_: PriorOnly (Physics Features Only) - No CNN, No Adaptation
        # 强制开启统计特征，但我们只用这个作为输入
        args.use_prior_stats = True
        
        # Disable domain adaptation losses
        args.lambda_unsup = 0.0
        args.lambda_prior = 0.0
        args.lambda_sa = 0.0
        args.lambda_sim = 0.0
        args.lambda_coral = 0.0
        args.lambda_align_target = 0.0
        
        # Disable DPD-DG components that rely on CNN features
        args.enable_attention = False
        args.enable_explicit = False
        args.enable_papl = False
        args.enable_align = False
        args.enable_dfe = False
        args.enable_rf = False
        
        logging.info("Configured M0_: Prior Only Baseline (Physics Features Only)")

    elif args.ablation_mode == 'M1':
        # M1: w/o SPA (Remove Physics Attention)
        # Use raw prior features instead of weighted ones
        args.enable_attention = False
        args.enable_rf = False
        logging.info("Configured M1: w/o SPA (Raw Prior Features)")

    elif args.ablation_mode == 'M2':
        # M2: w/o Explicit (Remove Explicit Modulation)
        # Physics only used for Loss (Implicit)
        args.enable_explicit = False
        logging.info("Configured M2: w/o Explicit Drive")

    elif args.ablation_mode == 'M3':
        # M3: w/o Implicit (Remove PAPL and Reconstruction Loss)
        # Explicit is ON, but Loss is OFF
        args.lambda_prior = 0.0
        args.enable_papl = False
        logging.info("Configured M3: w/o Implicit Drive (No PAPL, No Prior Loss)")

    elif args.ablation_mode == 'M4':
        # M4: w/o Alignment (Remove Proto Align & CORAL)
        args.lambda_sa = 0.0
        args.lambda_sim = 0.0
        args.lambda_coral = 0.0
        args.lambda_align_target = 0.0
        args.enable_align = False
        logging.info("Configured M4: w/o Alignment")

    elif args.ablation_mode == 'M5':
        # M5: Full Model (Ours)
        logging.info("Configured M5: Full DPD-DG Model")

    Dataset = Fault_dataset(args)
    
    # 打印args中的所有参数
    for k, v in args.__dict__.items():
        logging.info("{}: {}".format(k, v))
        
    operation = train_test(args)
    # 设置参数中的类别数量
    setattr(args, 'num_class', Dataset.n_class)
    
    try:
        # 训练和评估模型
        train_and_evaluate(args,operation, Dataset)
    except KeyboardInterrupt:
        logging.warning("检测到手动中断，日志已写入: %s", log_file)
        raise
    except Exception:
        logging.exception("运行过程中出现异常，相关日志保存在: %s", log_file)
        raise
    else:
        logging.info("训练完成，日志保存在: %s", log_file)
    finally:
        logging.shutdown()
