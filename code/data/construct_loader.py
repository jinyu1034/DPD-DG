import os
import torch.utils.data as Data
from torch.utils.data import Dataset
import torch
import scipy.io as scio
from scipy.fftpack import fft
import numpy as np
import logging
from utils.prior_features import build_prior_features
from sklearn.ensemble import RandomForestClassifier

# === 数据模块概览 ===
# 负责将不同工况下的 .mat 数据转换为 PyTorch Dataset/DataLoader，并在需要时拼接先验统计特征。
data_pth={
    "BJUT_Gear": os.path.join(os.path.dirname(__file__), "BJUT_Gear"),
    "CWRU_Bearing": os.path.join(os.path.dirname(__file__), "CWRU_Bearing"),
    "SDUT_Bearing": os.path.join(os.path.dirname(__file__), "SDUT_Bearing"),
    "SDUT_Gear": os.path.join(os.path.dirname(__file__), "SDUT_Gear"),
    "HUST_Gear": os.path.join(os.path.dirname(__file__), "HUST_Gear"),
    "Ottawa_Bearing": os.path.join(os.path.dirname(__file__), "Ottawa_Bearing"),
          }
# 定义类别映射
# .mat 文件中存放一个 1600 × 1025 的矩阵，前 1024 列是时间域采样点，最后一列是类别标签（总 5 类，对应 classes_map['BJUT_Gear']=5）。
classes_map = {
    'BJUT_Gear': 5,
    'CWRU_Bearing': 10,
    'SDUT_Bearing': 4,
    'SDUT_Gear': 7,
    'HUST_Gear': 3,
    'Ottawa_Bearing': 3,
}
# 定义域映射
domain_map = {
    'BJUT_Gear': ['20hz','30hz','40hz','50hz'],
    'CWRU_Bearing': ['0HP', '1HP', '2HP', '3HP'],
    'SDUT_Bearing': ['1500', '1800', '2000', '2500'],
    'SDUT_Gear': ['1500', '1800', '2000', '2500'],
    'HUST_Gear': ['0Nm', '0113Nm', '0226Nm', '0339Nm'],
    'Ottawa_Bearing': ['down', 'downup', 'up', 'updown'],
}
# 获取数据集路径
# 定义一个函数，用于获取指定名称的数据集文件路径
def get_domain_file(name):
    # 如果指定名称的数据集不在data_pth中，则抛出ValueError异常
    if name not in data_pth:
        raise ValueError('Name of datasetpu unknown %s' % name)
    # 返回指定名称的数据集文件路径
    return data_pth[name]
# 获取数据集任务
# 定义一个函数，用于获取域任务
def get_domain_task(name):
    # 如果name不在domain_map的键中，则抛出ValueError异常
    if name not in domain_map:
        raise ValueError('Name of datasetpu unknown %s' % name)
    # 返回domain_map中name对应的值
    return domain_map[name]

# 自定义张量数据集
class CustomTensorDataset(Dataset):
    # 初始化函数，传入数据张量、目标张量和训练集的域标签
    def __init__(self, x_tensor, y_tensor, domain_tensor, prior_tensor=None):
        """
        初始化函数，用于创建数据集对象
        参数:
            x_tensor: 数据张量，包含输入数据
            y_tensor: 目标张量，包含对应的目标标签
            domain_tensor: 训练集的域标签，用于区分不同的域
            prior_tensor: 可选参数，先验知识张量，默认为None
        """
        # 断言数据张量和目标张量的第一维大小相同，这是为了确保每个数据点都有一个对应的目标标签。
        # 如果两个张量的第一维度大小不同，程序会在这里抛出一个 AssertionError 异常，并终止运行。
        assert x_tensor.size(0) == y_tensor.size(0)
        # 将数据张量赋值给self.x_tensor
        self.x_tensor = x_tensor
        # 将目标张量赋值给self.y_tensor
        self.y_tensor = y_tensor
        # 将训练集的域标签赋值给self.domain_tensor
        self.domain_tensor = domain_tensor
        #检查prior_tensor是否为None，如果是则创建一个全零张量
        # if prior_tensor is None:
        # # 创建一个与数据张量行数相同，但列数为0的全零张量
        #     prior_tensor = torch.zeros((self.x_tensor.size(0), 0))
        # 将先验张量赋值给self.prior_tensor
        self.prior_tensor = prior_tensor

    def __len__(self):
        return self.x_tensor.size(0) #返回数据集的样本大小

    def __getitem__(self, index):
        # 输出格式与训练循环一致：(信号, 标签, 域, 样本索引, 先验特征)
        prior_feat = self.prior_tensor[index]
        # # 如果prior特征维度为0，返回一个标量0，这样collate可以正常工作
        if prior_feat.numel() == 0:
            prior_feat = torch.tensor(0.0)  # 返回标量而不是空张量
        return self.x_tensor[index], self.y_tensor[index], self.domain_tensor[index], index, prior_feat

class Fault_dataset(Dataset):
    def __init__(self, args):
        """
        初始化数据集类
        参数:
            args: 配置参数对象，包含数据集相关的各种设置
        """
        self.args = args #self.args 现在指向与 args 相同的对象，因此对 self.args 的任何修改都会反映在 args 中，反之亦然。
        self.n_class = classes_map[args.dataset_name] # 获取当前数据集的类别数
        # 使用基于文件的目录（data_pth 存放目录），后续使用 os.path.join 生成具体文件路径
        # 针对源域 ID 组合出数据文件路径.\data\BJUT_Gear\BJUTGear_20hz_5.mat，并将变量名解析为 BJUTGear_20hz_5，然后交给load_data。

    def apply_heterogeneous_normalization(self, train_prior, val_prior):
        """
        对先验特征应用异构归一化策略。
        严格使用训练集的统计量来归一化验证集。
        """
        if train_prior is None:
            return None, None

        # 特征索引映射 (用户提供的序号 - 1 = Python索引)
        # 1. 双对数变换 + Z-score (3个)
        # 22(21), 20(19), 21(20)
        idx_loglog_z = [21, 19, 20]

        # 2. 对数变换 + Z-score (15个)
        # 7(6), 24(23), 18(17), 4(3), 26(25), 17(16), 27(26), 16(15), 28(27), 47(46), 29(28), 48(47), 30(29), 31(30), 32(31)
        idx_log_z = [6, 23, 17, 3, 25, 16, 26, 15, 27, 46, 28, 47, 29, 30, 31]

        # 3. RobustScaling (7个)
        # 1(0), 2(1), 3(2), 5(4), 6(5), 8(7), 9(8)
        idx_robust = [0, 1, 2, 4, 5, 7, 8]

        # 4. Z-score (10个)
        # 10(9), 33(32), 40(39), 19(18), 11(10), 34(33), 41(40), 23(22), 42(41), 43(42)
        idx_zscore = [9, 32, 39, 18, 10, 33, 40, 22, 41, 42]

        # 5. Min-Max (10个)
        # 12(11), 35(34), 25(24), 13(12), 36(35), 39(38), 14(13), 37(36), 15(14), 38(37)
        idx_minmax = [11, 34, 24, 12, 35, 38, 13, 36, 14, 37]

        # 6. 无需归一化 (3个)
        # 44(43), 45(44), 46(45)
        idx_none = [43, 44, 45]

        epsilon = 1e-8 # 防止除零或log(0)

        # === 策略 1: 双对数变换 (Log-Log) + Z-score ===
        if idx_loglog_z:
            # 变换: log(log(|x| + 1) + 1) 确保正值且平滑
            t_sub = np.log(np.log(np.abs(train_prior[:, idx_loglog_z]) + 1) + 1)
            v_sub = np.log(np.log(np.abs(val_prior[:, idx_loglog_z]) + 1) + 1)
            
            # Z-score
            mu = np.mean(t_sub, axis=0)
            sigma = np.std(t_sub, axis=0)
            sigma[sigma < epsilon] = 1.0
            
            train_prior[:, idx_loglog_z] = (t_sub - mu) / sigma
            val_prior[:, idx_loglog_z] = (v_sub - mu) / sigma

        # === 策略 2: 对数变换 (Log) + Z-score ===
        if idx_log_z:
            # 变换: log(|x| + epsilon)
            t_sub = np.log(np.abs(train_prior[:, idx_log_z]) + epsilon)
            v_sub = np.log(np.abs(val_prior[:, idx_log_z]) + epsilon)
            
            # Z-score
            mu = np.mean(t_sub, axis=0)
            sigma = np.std(t_sub, axis=0)
            sigma[sigma < epsilon] = 1.0
            
            train_prior[:, idx_log_z] = (t_sub - mu) / sigma
            val_prior[:, idx_log_z] = (v_sub - mu) / sigma

        # === 策略 3: RobustScaling (中位数和四分位距) ===
        if idx_robust:
            t_sub = train_prior[:, idx_robust]
            v_sub = val_prior[:, idx_robust]
            
            # 计算统计量
            median = np.median(t_sub, axis=0)
            q75, q25 = np.percentile(t_sub, [75 ,25], axis=0)
            iqr = q75 - q25
            iqr[iqr < epsilon] = 1.0 # 避免除零
            
            train_prior[:, idx_robust] = (t_sub - median) / iqr
            val_prior[:, idx_robust] = (v_sub - median) / iqr

        # === 策略 4: Z-score 标准化 ===
        if idx_zscore:
            t_sub = train_prior[:, idx_zscore]
            v_sub = val_prior[:, idx_zscore]
            
            mu = np.mean(t_sub, axis=0)
            sigma = np.std(t_sub, axis=0)
            sigma[sigma < epsilon] = 1.0
            
            train_prior[:, idx_zscore] = (t_sub - mu) / sigma
            val_prior[:, idx_zscore] = (v_sub - mu) / sigma

        # === 策略 5: Min-Max 归一化 ===
        if idx_minmax:
            t_sub = train_prior[:, idx_minmax]
            v_sub = val_prior[:, idx_minmax]
            
            min_val = np.min(t_sub, axis=0)
            max_val = np.max(t_sub, axis=0)
            range_val = max_val - min_val
            range_val[range_val < epsilon] = 1.0
            
            train_prior[:, idx_minmax] = (t_sub - min_val) / range_val
            val_prior[:, idx_minmax] = (v_sub - min_val) / range_val

        # === 策略 6: 无需归一化 ===
        # 直接保留原值，不做操作
        
        return train_prior, val_prior

    # 加载数据
    def load_data(self, path, temp, sum_class, data_ratio, mis_class, FFT=True, normalize_type="mean-std"):
        # 该函数完成 .mat -> (train/val) Tensor 的全部转换流程。
        """
        加载并预处理数据函数
        参数:
            path: 数据文件路径
            temp: 数据文件中的变量名
            sum_class: 总类别数
            data_ratio: 训练集数据比例
            mis_class: 需要排除的类别列表
            FFT: 是否应用快速傅里叶变换，默认为True
            normalize_type: 数据归一化类型，默认为"0-1"
        返回:
            train_x: 训练数据特征
            train_y: 训练数据标签
            val_x: 验证数据特征
            val_y: 验证数据标签
            train_prior: 训练数据的先验特征
            val_prior: 验证数据的先验特征
        """
    # 从.mat文件加载数据
        try:
            data_temp = scio.loadmat(path)  # 使用scipy.io的loadmat函数加载.mat文件
            data = data_temp.get(temp)  # 从加载的.mat文件中获取指定变量名的数据
        except NotImplementedError: #如果文件是mat最新格式，则使用h5py库加载
            import h5py
            with h5py.File(path, 'r') as f:
                data = np.array(f[temp]).T

    # 计算有效类别数量（总类别数减去需要排除的类别数）
        self.n_class = int(sum_class) - len(mis_class)  # 计算有效类别数量，排除不需要的类别
    # 默认不删除，删除需要排除的类别数据,后续可做缺失标签的分类诊断研究
        for i_c in mis_class:  # 遍历需要排除的类别列表
            mis_class_id = np.argwhere(data[:, -1] == i_c) #在最后一行的数据标签找到==i_c,并将其标记为为True的位置（即找到所有属于当前类别的样本的索引）
            data = np.delete(data, mis_class_id, axis=0) #从数据中删除这些索引对应的行，axis=0表示按行删除。
    # 获取样本总数，这是一种元组解包的写法，将.shape返回的元组的第一个元素赋值给class_sample，第二个元素赋值给_（被忽略）。
        class_sample, _ = data.shape # 1600*1024,在Python中，下划线通常用作一个"不在乎"的变量名，表示我们想忽略这个值。在这里，它接收了.shape返回的第二个维度（列数），但我们不关心这个值。
    # 根据比例计算训练集样本数
        sample_ratio=int(class_sample * data_ratio) #1600*0.8=1280

    # 根据 args.data_ratio=0.8 将前 1280 行作为训练集，其余 320 行作为验证集；train_x/train_y 与 val_x/val_y 完成初始划分。
    # 分割训练集和验证集
        train_x= data[:sample_ratio, :-1]  # 训练集特征，选取从开始到 sample_ratio 的行，以及除最后一列外的所有列作为训练集特征
        train_y = data[:sample_ratio, -1]  # 训练集标签，选取从开始到 sample_ratio 的行，以及最后一列作为训练集标签
        val_x= data[sample_ratio:, :-1]   # 验证集特征
        val_y = data[sample_ratio:, -1]   # 验证集标签
        time_train = train_x.copy()        #train_x 变为 1280 × 512，val_x 变为 1280 × 512。
        time_val = val_x.copy()            #time_train/time_val 仍保存原始时间域副本，供可能的先验特征使用。
    # 如果需要，应用快速傅里叶变换并保留前512个特征，FFT的结果是复数形式，包含实部和虚部，np.abs() 函数计算复数的模（幅度），即sqrt(real² + imag²)
        if FFT: #其他函数axis=0就是纵向操作，1横向操作，drop函数这个例外，axis=0表示行就是删除行，1表示列就是删除列
            train_x = np.abs(fft(train_x, axis=1))[:, :512] #axis=1 参数表示沿着第二个轴（即每行）进行FFT变换
            val_x =np.abs(fft(val_x , axis=1))[:, :512]   # 这是对变换后的数据进行切片操作，保留所有样本（第一个冒号表示行），只取前512个频率分量（第二个冒号和512表示列），这样做是因为对于实数信号的FFT，结果是对称的，我们只需要前半部分就能包含所有频率信息
        
        # 保存未归一化的频域数据用于先验特征计算
        freq_train_raw = train_x.copy()
        freq_val_raw = val_x.copy()

        # 数据归一化处理
        train_x = self.Normalize(train_x, normalize_type) #采用 args.normalize_type='mean-std'，对每个样本的 512 维频域向量逐行减均值除以标准差，避免零方差时的除零问题。
        val_x = self.Normalize(val_x, normalize_type)   #调用Normalize函数对验证数据进行归一化处理
        if getattr(self.args, 'use_prior_stats', False):  #检查是否使用了先验统计信息,计算诸如偏度、峰度等统计量，并根据 args.prior_stats_source 选择时间/频域/两者。默认不开启，因此 prior_dim=0，不与主特征拼接。
            # 两者形状1280*先验长度, 320*先验长度；可选：为每个样本计算时域/频域统计量，作为附加先验知识特征;
            prior_train = build_prior_features(time_data=time_train, freq_data=freq_train_raw, source=self.args.prior_stats_source)
            prior_val = build_prior_features(time_data=time_val, freq_data=freq_val_raw, source=self.args.prior_stats_source)
            
            # === 先验特征归一化 ===
            # 应用异构归一化策略
            if prior_train is not None:
                prior_train, prior_val = self.apply_heterogeneous_normalization(prior_train, prior_val)
        else:
            prior_train = None
            prior_val = None
        if prior_train is None: #检查prior_train是否为None，如果是则创建一个全零张量,用全 0、列数为 0 的张量，是为了接口统一、代码简洁、类型与形状安全，同时通过 prior_dim = 0 明确表示“当前没有使用先验特征”。
            prior_train = np.zeros((train_x.shape[0], 0), dtype=np.float32) #之后无论是否使用先验特征，都可以直接：转成张量：torch.FloatTensor(prior_train)，
            prior_val = np.zeros((val_x.shape[0], 0), dtype=np.float32)   #在 CustomTensorDataset.__getitem__ 里统一返回 self.prior_tensor[index]
            prior_dim = 0
        else: #如果prior_train不为None，则计算先验特征的维度
            prior_dim = prior_train.shape[1]
        self.args.prior_dim = prior_dim # 动态为args添加先验特征维度赋值给prior_dim（即使args没有prior_dim，也会为其创建一个）
        # if prior_dim > 0: #如果先验特征维度大于0，则将先验特征与原始特征进行拼接
        #     # 将先验特征与频域特征拼接，供模型共享提取。
        #     train_x = np.concatenate([train_x, prior_train], axis=1) # 1280*(512+先验长度)，在列向拼接
        #     val_x = np.concatenate([val_x, prior_val], axis=1)       # 320*(512+先验长度)
        setattr(self.args, 'signal_length', train_x.shape[1]) #=512+先验长度，将训练数据的特征长度赋值给self.args.signal_length
    # 将数据转换为PyTorch张量格式,频域特征转换为 torch.FloatTensor 后 unsqueeze(1)，形成 (样本数, 1, 512)：训练集 800 × 1 × 512，验证集 800 × 1 × 512。
        train_x = torch.FloatTensor(train_x).unsqueeze(1) #将输入的训练特征数据(train_x)转换为PyTorch的浮点型张量(FloatTensor)。FloatTensor是32位浮点数类型的张量，通常用于存储神经网络的输入特征。
        train_y = torch.LongTensor(train_y) #将训练标签数据(train_y)转换为PyTorch的长整型张量(LongTensor)。LongTensor是64位整数类型的张量，通常用于分类任务中的类别标签，因为PyTorch的损失函数(如CrossEntropyLoss)期望目标值是长整型。
        val_x = torch.FloatTensor(val_x).unsqueeze(1) #在张量的第1维(从0开始计数)上增加一个维度。这通常用于将数据适配到某些网络层期望的输入格式。例如，如果原始train_x的形状是[N, L]，经过unsqueeze(1)后会变成[N, 1, L]，这在处理序列数据或图像数据时很常见，增加了"通道"维度。
        val_y = torch.LongTensor(val_y) #将验证标签数据(val_y)转换为PyTorch的长整型张量(LongTensor)。
        train_prior = torch.FloatTensor(prior_train) #将训练先验特征(prior_train)转换为PyTorch的浮点型张量(FloatTensor)。
        val_prior = torch.FloatTensor(prior_val) #将验证先验特征(prior_val)转换为PyTorch的浮点型张量(FloatTensor)。
    # 返回处理后的数据,训练集 1280 × 1 × 512+先验长度，验证集 320 × 1 × 512+先验长度,标签为 1280或320 × 1；先验特征则为 1280*先验长度，320*先验长度；
        return train_x, train_y, val_x, val_y, train_prior, val_prior

    def Normalize(self, data, type):
        # 针对每个样本行进行归一化，避免不同幅值的域间分布差异干扰训练。
        """
        对输入数据进行归一化处理

        参数:
            data: 输入数据，需要进行归一化的序列
            type: 归一化类型，支持以下几种:
                "0-1" / "min-max": 将数据归一化到[0,1]区间
                "-1-1": 将数据归一化到[-1,1]区间
                "mean-std" / "z-score": 使用均值和标准差进行标准化处理

        返回:
            归一化处理后的数据
        """
        seq = data  # 将输入数据赋值给seq变量
        
        if type == "0-1" or type == "min-max":
            # 0-1归一化：将数据线性变换到[0,1]区间
            Zmax = seq.max(axis=1, keepdims=True)
            Zmin = seq.min(axis=1, keepdims=True)
            # 避免除以0
            denom = Zmax - Zmin
            denom[denom == 0] = 1e-8
            seq = (seq - Zmin) / denom
            
        elif type == "-1-1":
            # -1到1归一化：将数据线性变换到[-1,1]区间
            Zmax = seq.max(axis=1, keepdims=True)
            Zmin = seq.min(axis=1, keepdims=True)
            denom = Zmax - Zmin
            denom[denom == 0] = 1e-8
            seq = 2 * (seq - Zmin) / denom - 1
            
        elif type == "mean-std" or type == "z-score":
            # 均值-标准差标准化：使数据均值为0，标准差为1
            mean = np.mean(seq, axis=1, keepdims=True)  # 计算每行的均值
            std = np.std(seq, axis=1, keepdims=True)    # 计算每行的标准差
            std[std == 0] = 1.0  # 避免除以0
            seq = (seq - mean) / std  # 进行标准化处理
            
        else:
            # 如果type不是以上三种，则不做处理
            pass

        return seq  # 返回归一化后的数据


    # 加载器
    def Loader(self,data_list_name=[]):
        # data_list_name：希望加载的工况编号
        """
        数据加载器函数，用于加载和预处理训练及验证数据

        参数:
            data_list_name (list): 数据集名称列表

        返回:
            train_loader: 训练数据加载器
            val_loader: 验证数据加载器
        """
        train_loader = []  # 用于存储训练数据集
        val_loader = []   # 用于存储验证数据集
        
        # 临时存储所有域的数据，以便后续统一处理先验特征选择
        all_train_data = [] 
        all_val_data = []
        
        # 遍历每个mat文件所有域的数据，将BJUTGear_20hz_5、BJUTGear_30hz_5、BJUTGear_50hz_5等依次处理，
        for domain_id,domain in enumerate(data_list_name): #domain_id 是当前域的索引(0,1,2,3)，domain 是当前域的名称(20hz,30hz,40hz,50hz)
            sum_class = classes_map[self.args.dataset_name]  # 获取数据集类别总数,5
        # 构建数据文件路径
            if self.args.dataset_name == "BJUT_Gear":
                file_name = f"BJUTGear_{domain}_{sum_class}.mat" 
            elif self.args.dataset_name == "CWRU_Bearing":
                file_name = f"CwruBearing_{domain}_{sum_class}.mat"
            elif self.args.dataset_name == "SDUT_Bearing":
                file_name = f"SdutBearing_{domain}_{sum_class}.mat"
            elif self.args.dataset_name == "SDUT_Gear":
                file_name = f"SdutGear_{domain}_{sum_class}.mat"
            elif self.args.dataset_name == "HUST_Gear":
                file_name = f"HUSTGear_{domain}_{sum_class}.mat"
            elif self.args.dataset_name == "Ottawa_Bearing":
                file_name = f"OttawaBearing_{domain}_{sum_class}.mat"
            else:
                raise ValueError(f"Unknown dataset name: {self.args.dataset_name}")

            root = os.path.join(data_pth[self.args.dataset_name], file_name)#例如 "/home/user/data/BJUT_Gear/BJUTGear_20hz_5.mat"
            temp = os.path.splitext(os.path.basename(root))[0]  # BJUTGear_20hz_5,basename(root)会从给定的路径中提取最后的文件名部分，root 是 "/home/user/document.txt"，它会返回 "document.txt"；splitext(...)会将文件名分割成两部分：(文件名, 扩展名)，[0]就是提取文件名（不含扩展名）
            # 训练集 1280 × 1 × 512+先验长度，验证集 320 × 1 × 512+先验长度,标签为 1280或320 × 1；先验特征则为 1280*先验长度，320*先验长度；
            train_x, train_y, val_x, val_y, train_prior, val_prior= self.load_data(root,temp, sum_class,self.args.data_ratio, self.args.miss_class, self.args.FFT, self.args.normalize_type)
            
            # 临时存储原始 numpy/tensor 数据
            all_train_data.append({
                'x': train_x, 'y': train_y, 'prior': train_prior, 'domain_id': domain_id
            })
            all_val_data.append({
                'x': val_x, 'y': val_y, 'prior': val_prior, 'domain_id': domain_id
            })

        # === 随机森林进行人工选择最相关的先验特征选择 ===
        selected_indices = None
        
        # 检查是否已经在 args 中缓存了先验特征索引（以避免在目标域上重复选择）
        if hasattr(self.args, 'selected_prior_indices') and self.args.selected_prior_indices is not None:
            selected_indices = self.args.selected_prior_indices
            self.args.prior_dim = len(selected_indices)
            print(f"使用缓存的先验特征索引: {selected_indices}")
            
        elif getattr(self.args, 'select_top_k_prior', 0) > 0 and self.args.prior_dim > 0:
            k = self.args.select_top_k_prior
            # 使用带标签的源域数据来选择先验特征
            # labeled_source_index 默认为 0
            labeled_idx = self.args.labeled_source_index
            if 0 <= labeled_idx < len(all_train_data):
                # 从带标签的源域中获取数据
                source_data = all_train_data[labeled_idx]
                # 如果需要，将张量转换回 numpy 供 sklearn 使用（当前为张量）
                X_prior = source_data['prior'].numpy()
                y_labels = source_data['y'].numpy()
                
                if X_prior.shape[1] >= k:
                    print(f"Selecting top {k} prior features using Random Forest on domain index {labeled_idx}...")
                    rf = RandomForestClassifier(n_estimators=400, random_state=42, n_jobs=-1)
                    rf.fit(X_prior, y_labels)
                    importances = rf.feature_importances_
                    selected_indices = np.argsort(importances)[::-1][:k]
                    logging.info(f"After {rf.n_estimators} estimators, the top {k} selected feature indices: {selected_indices}")
                    
                    # === 改进：保存 RF 重要性向量 ===
                    selected_importances = importances[selected_indices]
                    # 归一化到 [0, 1] 以便与 Attention 权重融合
                    if selected_importances.max() > 0:
                        selected_importances = selected_importances / selected_importances.max()
                    self.args.rf_importance_vec = selected_importances
                    logging.info(f"RF Importance Vector (Top 5): {selected_importances[:5]}")

                    # 更新 args 中的 prior_dim
                    self.args.prior_dim = k
                    # 将 selected_indices 缓存到 args 中
                    self.args.selected_prior_indices = selected_indices
                else:
                    print(f"Requested top {k} features but only {X_prior.shape[1]} available. Skipping selection.")
            else:
                print(f"Warning: labeled_source_index {labeled_idx} out of range. Skipping feature selection.")

        # === 构建数据集 ===
        for i in range(len(all_train_data)):
            train_d = all_train_data[i]
            val_d = all_val_data[i]
            domain_id = train_d['domain_id']
            
            # 如果确定了 selected_indices，则应用特征选择
            t_prior = train_d['prior']
            v_prior = val_d['prior']
            
            if selected_indices is not None:
                # 将 selected_indices 转换为列表或正步幅数组以避免负步幅问题
                # 在 numpy 中使用 [::-1] 会生成负步幅（negative strides），Torch 无法直接处理
                indices = selected_indices.copy()
                #print(f"Selected prior indices: {indices}, before_shape: {t_prior.shape}")
                t_prior = t_prior[:, indices] #将原始先验特征矩阵按选定的索引进行列选择，得到新的先验特征矩阵
                v_prior = v_prior[:, indices] #40个先验特征变成选定的影响大的k个先验特征
                #print(f"Selected prior indices: {indices}, after_shape: {t_prior.shape}")   
            
            train_domain = torch.full_like(train_d['y'], domain_id)# 1280*1，全为当前域的索引(0,1,2,3)等，使用torch.full_like函数创建一个与train_y形状相同的张量，其中所有元素都填充为domain_id
            train_dataset = CustomTensorDataset(train_d['x'], train_d['y'], train_domain, t_prior) #一个同时包含信号1280*1*(512+先验长度)、标签1280*1、域标签1280*1、域索引1280*1和先验特征1280*(先验长度)的综合数据集，方便在训练过程中使用。
            #print(f"train_dataset_x_tensor: {train_dataset.x_tensor.shape}; train_dataset_y_tensor: {train_dataset.y_tensor.shape}; train_dataset_domain_tensor: {train_dataset.domain_tensor.shape}; train_dataset_prior_tensor: {train_dataset.prior_tensor.shape};")
            
            val_domain = torch.full_like(val_d['y'], domain_id)
            val_dataset = CustomTensorDataset(val_d['x'], val_d['y'], val_domain, v_prior)
            #print(f"val_dataset_x_tensor: {val_dataset.x_tensor.shape};     val_dataset_y_tensor: {val_dataset.y_tensor.shape};     val_dataset_domain_tensor: {val_dataset.domain_tensor.shape};     val_dataset_prior_tensor: {val_dataset.prior_tensor.shape};")
            
            train_loader.append(train_dataset)# 将训练数据集添加汇总的数据集列表中
            val_loader.append(val_dataset)# 将验证数据集添加汇总的数据集列表中
            
        return train_loader, val_loader
        # 最终train_loader包含了3个域数据集，每个信号1600*0.8*1*(512+0或10或20)、标签1280*1、域标签1280*1、样本索引1280和先验特征1280*(0或10或20)的综合数据集
        # val_loader同理，但样本数为1600*0.2
