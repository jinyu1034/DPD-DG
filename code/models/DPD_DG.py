# -*- coding: utf-8 -*-  # 指定文件编码为 UTF-8，支持中文注释
""" 
DPD-DG: 双物理驱动的域泛化故障诊断模型
"""
import math  # 导入数学模块，用于数学函数
from collections import Counter  # 导入 Counter 用于计数伪标签分布等
import numpy as np  # 导入 numpy，作为数值工具
import logging
import torch  # 导入 PyTorch 主库
import torch.nn as nn  # 导入神经网络模块的别名 nn
from torch import optim  # 导入优化器模块
from torch.nn import functional as F  # 导入函数式接口 F（如 softmax、relu）
import torch.utils.data as Data  # 导入数据工具模块, 并别名为 Data
import os
from datetime import datetime

# === 模型概览 ===  # 概览说明：包含 DFE、卷积特征提取器、可适配权重分类器与多类损失
# 包含分布特征扩展(DFE)、卷积特征提取器、带权重自适应的分类器与多项损失（监督、伪标签、原型对齐、MMD、先验约束）。

class DFE(nn.Module):
    """
    Distribution Feature Expansion (DFE)
    分布特征扩展模块：用于特征扰动以提升鲁棒性
    """
    def __init__(self, eps=1e-6, alpha=0.1):
        super().__init__()
        self.eps = eps
        self.beta = torch.distributions.Beta(alpha, alpha)

    def forward(self, x):
        # 通过随机采样均值/标准差进行归一化扰动，提升特征分布多样性，缓解域间偏移。
        N, C, L = x.shape
        mu = x.mean(dim=2, keepdim=True)
        var = x.var(dim=2, keepdim=True)
        sig = (var + self.eps).sqrt()
        x_perturbed = x
        
        if self.training:
            mu, sig = mu.detach(), sig.detach()
            x_normed = (x - mu) / sig
            
            mu_random = torch.empty((N, C, 1), dtype=torch.float32).uniform_(0.5, 1.0).to(x.device)
            var_random = torch.empty((N, C, 1), dtype=torch.float32).uniform_(0.5, 1.0).to(x.device)
            
            lam = self.beta.sample((N, C, 1)).to(x.device)
            bernoulli = torch.bernoulli(lam).to(x.device)
            
            mu_mix = mu_random * bernoulli + mu * (1. - bernoulli)
            sig_mix = var_random * bernoulli + sig * (1. - bernoulli)
            
            x_perturbed = x_normed * sig_mix + mu_mix

        return x_perturbed

class MSBlock(nn.Module):
    """
    物理驱动的多尺度卷积模块
    模拟多分辨率信号分析：
    - 小核 (3-5): 捕捉高频冲击 (Bearing Faults)
    - 中核 (11-19): 捕捉中频结构振动
    - 大核 (50+): 捕捉低频趋势与周期性 (Unbalance/Misalignment)
    """
    def __init__(self, in_channels, out_channels):
        super(MSBlock, self).__init__()
        # 确保每个分支输出通道数之和等于 out_channels
        # 这里我们将 out_channels 分配给三个分支
        branch_channels = out_channels // 4
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=65, stride=1, padding=32), # 大感受野
            nn.InstanceNorm1d(branch_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels, kernel_size=17, stride=1, padding=8),  # 中感受野
            nn.InstanceNorm1d(branch_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, branch_channels * 2, kernel_size=3, stride=1, padding=1),   # 小感受野 (分配更多通道以捕捉丰富的高频细节)
            nn.InstanceNorm1d(branch_channels * 2),
            nn.ReLU(inplace=True)
        )
        
        # 1x1 卷积用于融合不同尺度的特征
        self.fusion = nn.Conv1d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        # 确保输入长度适配 padding (简单的处理方式，也可以用 same padding)
        # 注意：上面的 padding 是手动计算的 (k-1)/2，假设 stride=1
        
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        # 截取以匹配最小长度 (防止 padding 导致的微小尺寸差异)
        
        out = torch.cat([b1, b2, b3], dim=1)
        out = self.fusion(out)
        return out

class PriorAttention(nn.Module):
    def __init__(self, in_dim, reduction=4):
        super(PriorAttention, self).__init__()
        mid_dim = max(in_dim // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(in_dim, mid_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mid_dim, in_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        w = self.fc(x)
        return x * w, w

class PriorMLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(PriorMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, out_dim)
        )
    def forward(self, x):
        return self.net(x)

class PriorBaselineClassifier(nn.Module):
    """
    仅使用物理先验特征进行分类的基准模型 (for M0_)
    """
    def __init__(self, in_dim, num_classes):
        super(PriorBaselineClassifier, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.net(x)


class DPD_DG_Classifier(nn.Module):  # 分类器模块，支持基于先验引导的权重自适应
    def __init__(self, num_features, num_classes, prior_dim=0, domain_knowledge=False, method="DPD_DG", prior_aug_mode='none'):
        super().__init__()  # 调用父类构造函数
        self.method = method  # 存储模式/方法标识
        self.prior_dim = prior_dim # 记录先验维度
        self.prior_aug_mode = prior_aug_mode # Augmentation mode
        
        if domain_knowledge:  # 若启用 DK 分支（领域知识引导的权重调整），初始化额外的全连接层
            self.head_1 = nn.Linear(num_features, num_features)  # 线性变换 h1
            self.head_2 = nn.Linear(num_features, num_classes)  # 线性变换 h2

        # === 改进部分：先验特征融合层 ===
        if self.prior_dim > 0:
            # 将先验特征映射到与深度特征相同的维度，以便融合
            self.prior_fusion = nn.Sequential(
                nn.Linear(prior_dim, num_features),
                nn.LayerNorm(num_features), # 归一化以匹配特征分布尺度，它的原理是计算单个样本内部所有特征的均值和方差
                nn.ReLU(inplace=True)
            )
            # === 新增：Sigmoid 门控参数 ===
            # 初始化为 0，经过 Sigmoid 后为 0.5，表示初始状态下融合一半的先验信息，让模型自己学习最佳比例
            self.prior_gate = nn.Parameter(torch.tensor(0.0))
            # === 新增：先验注意力机制 (初始化为 None，由外部 DPD_DG 注入以保持共享) ===
            self.prior_attention = None
        # ==============================

        self.proj_1 = nn.Linear(num_features, num_features // 2)  # 用于生成掩码的网络 p1..p5
        self.proj_2 = nn.Linear(num_features // 2, num_features// 4 )  # p2 层
        self.proj_3 = nn.Linear(num_features // 4, num_features // 8)  # p3 层
        self.proj_4 = nn.Linear(num_features // 8, num_features // 2)  # p4 层（反向扩展）- Modified input dim due to noise removal
        self.proj_5 = nn.Linear(num_features // 2, num_features)  # p5 层
        self.weight = nn.Parameter(torch.Tensor(num_classes, num_features))  # 分类器权重参数，纯粹的矩阵张量，可以随心所欲地对它进行切片、加权、掩码操作，然后再手动执行 torch.matmul。这在实现这种“非标准”的动态计算图时，比封装好的 nn.Linear 更灵活、更直观。

        stdv = 1. / math.sqrt(self.weight.size(1))  # 初始化权重的标准范围
        self.weight.data.uniform_(-stdv, stdv)  # 随机初始化权重


    def forward(self, x, prior=None, domain_knowledge=False):  # 前向传播：支持可选权重调整
        # domain_knowledge=True 时触发“先验引导的权重调整”：
        #   1) 统计同域样本的压缩向量
        #   2) 生成掩码调制共享分类器参数 w，实现不同工况下的自适应。
        if domain_knowledge:  # 当启用 DK 时，使用先验来动态调整分类器权重
            # 改为样本级融合，不再取均值
            x_instance = x 
            
            # === 改进部分：融合物理先验知识 ===
            if self.prior_dim > 0 and prior is not None:
                # Augmentation for Prior Features
                if self.training:
                    if self.prior_aug_mode == 'gaussian':
                        noise = torch.randn_like(prior) * 0.05
                        prior = prior + noise
                    elif self.prior_aug_mode == 'dropout':
                        # Dropout with p=0.2 (keep probability 0.8)
                        mask = (torch.rand_like(prior) > 0.2).float()
                        prior = prior * mask / 0.8

                # 计算物理先验特征的领域质心
                if hasattr(self, 'prior_attention') and self.prior_attention is not None:
                    prior, _ = self.prior_attention(prior) # 计算并应用注意力权重
                    
                # 不再取均值，保留样本级先验
                # prior_mean = prior.mean(dim=0, keepdim=True) 
                
                # 映射并融合 (这里采用相加融合，也可以尝试拼接)
                # 融合后的 x_instance 既包含深度特征的分布信息，也包含物理指标的统计信息
                # === 应用 Sigmoid 门控 ===
                gate = torch.sigmoid(self.prior_gate)
                x_instance = x_instance + gate * self.prior_fusion(prior) 
            # ====================================

            x_instance = torch.relu(self.proj_1(x_instance))  # 通过 p1 -> relu
            x_instance = torch.relu(self.proj_2(x_instance))  # 通过 p2 -> relu
            x_instance = torch.relu(self.proj_3(x_instance))  # 通过 p3 -> relu
            
            x_instance = torch.relu(self.proj_4(x_instance))  # 通过 p4 -> relu
            x_mask = torch.relu(self.proj_5(x_instance))  # 通过 p5 -> 得到 x_mask
            
            x1 = self.head_1(x_mask)  # [Batch, Dim]，# 生成特征向量 V: [1, 1536]表示在当前工况下，哪些特征通道是重要的（与类别无关的全局重要性）。
            x2 = self.head_2(x_mask)  # [Batch, Num_Classes] # 生成类别向量 Q: [1, num_classes]表示在当前工况下，哪些故障类别是概率较高的（或者需要重点关注的）。
            
            # 样本级权重生成: [Batch, Num_Classes, Dim]
            # x2.unsqueeze(2): [Batch, Num_Classes, 1]
            # x1.unsqueeze(1): [Batch, 1, Dim]
            attention_scores = torch.matmul(x2.unsqueeze(2), x1.unsqueeze(1)) #两者的结合。它强制模型学习一种规律——“特征的重要性”和“类别的关注度”是解耦的，然后通过乘法重新组合。这极大地减少了过拟合的风险。
            # 维度计算: [5, 1] * [1, 1536] = [5, 1536]
            weight_mask = torch.sigmoid(attention_scores)  # [Batch, Num_Classes, Dim]
            
            # adapted_weight: [Batch, Num_Classes, Dim]
            adapted_weight = self.weight.unsqueeze(0) * weight_mask 
            
            # 应用权重进行分类
            # x: [Batch, Dim] -> [Batch, 1, Dim]
            # adapted_weight: [Batch, Num_Classes, Dim]
            # element-wise mul -> [Batch, Num_Classes, Dim]
            # sum(dim=2) -> [Batch, Num_Classes]
            logits = (x.unsqueeze(1) * adapted_weight).sum(dim=2)
            
            return logits
        else:
            return torch.matmul(x, self.weight.t())  # 默认直接使用权重 w 进行线性变换

class DPD_DG_Fea_Extraction(nn.Module):  # 特征提取器：一系列 1D 卷积层和池化
    def __init__(self, in_channel=1, method="DPD_DG"):
        super().__init__()  # 调用父类构造
        self.method = method  # 存储方法名称（用于条件逻辑）
        self.dfe = DFE() # 初始化分布特征扩展模块（DFE）
        
        self.layers = nn.ModuleList([
            # Layer 1: 改进点：使用多尺度模块替换原始 Layer1
            nn.Sequential(
                MSBlock(in_channel, 32), # 输出 32 通道
                nn.MaxPool1d(kernel_size=2, stride=2)
            ),
            # Layer 2: 调整 Layer2 的输入通道数 (从 16 变为 32)
            nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=16, stride=1), # 输入 32, 输出 64 (原代码输出32，这里适当加宽)
                nn.InstanceNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ),
            # Layer 3: 调整 Layer3 的输入通道数 (从 32 变为 64)
            nn.Sequential(
                nn.Conv1d(64, 64, kernel_size=5, stride=1), # 输入 64
                nn.BatchNorm1d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ),
            # Layer 4
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=5, stride=1),
                nn.BatchNorm1d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ),
            # Layer 5
            nn.Sequential(
                nn.Conv1d(128, 256, kernel_size=5, stride=1),
                nn.BatchNorm1d(256),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2)
            ),
            # Layer 6
            nn.Sequential(
                nn.Conv1d(256, 512, kernel_size=5, stride=1),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2)
            )
        ])

    def forward(self, x, enable_dfe=False, return_layer=None):  # 前向传播：支持插入 DFE 以及返回中间层
        for i, layer in enumerate(self.layers):
            current_layer_idx = i + 1
            
            # 当 enable_dfe=True 时，在层间插入 DFE
            # 将 DFE 插入第一层与第二层之间，只在训练时生效。
            # Ablation: Check enable_dfe from config/args
            use_dfe = enable_dfe and getattr(self, 'enable_dfe', True)
            if current_layer_idx == 2 and use_dfe:
                x = self.dfe(x)
            
            # MixStyle removed

            x = layer(x)
            
            if return_layer == current_layer_idx:
                return x
        
        out = x.view(x.size(0), -1)  # 展平特征为 (batch, dim)
        return out  # 返回最终特征向量

    def forward_from(self, x, start_layer, enable_dfe=False):  # 从指定层开始的前向传播（用于流形混合）
        # 从指定层开始前向传播
        # 注意：start_layer 是已经完成的层，所以从 start_layer (索引 start_layer) 开始执行
        # 例如 start_layer=1，说明 layer1 已执行，应从 layer2 (索引1) 开始
        
        # 特殊处理 DFE: 如果 start_layer=1，说明刚过 layer1，需要检查是否插入 DFE
        use_dfe = enable_dfe and getattr(self, 'enable_dfe', True)
        if start_layer == 1 and use_dfe:
            x = self.dfe(x)
            
        for i in range(start_layer, len(self.layers)):
            x = self.layers[i](x)
            
        out = x.view(x.size(0), -1)  # 展平输出特征
        return out  # 返回

class DPD_DG(nn.Module):  # 主模型类，集成特征提取、分类器、先验分支与训练流程
    def __init__(self, in_channel=1, num_classes=5, lr=0.01, set="DPD_DG", args=None):
        super(DPD_DG, self).__init__()  # 调用父类构造
        
        # === 内部配置参数管理 ===
        self.args = args
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.lr = lr
        self.method = set
        
        # 从 args 获取参数，若无则使用默认值
        self.lambda_sup = getattr(args, 'lambda_sup', 1.0)
        self.lambda_prior = getattr(args, 'lambda_prior', 10.0)
        self.lambda_unsup = getattr(args, 'lambda_unsup', 1.0)
        self.lambda_sa = getattr(args, 'lambda_sa', 0.5)
        self.lambda_sim = getattr(args, 'lambda_sim', 10.0)
        self.lambda_coral = getattr(args, 'lambda_coral', 5.0)
        self.lambda_align_target = getattr(args, 'lambda_align_target', 0.5)
        self.physics_thresh = getattr(args, 'physics_thresh', 0.3)
        self.threshold = getattr(args, 'threshold', 0.5)
        self.weight_decay = getattr(args, 'weight_decay', 1e-3)
        self.epoch = getattr(args, 'epoch', 100)
        self.signal_length = getattr(args, 'signal_length', 512)
        self.prior_dim = getattr(args, 'prior_dim', 0)
        
        # 检查消融模式
        self.ablation_mode = getattr(args, 'ablation_mode', 'M5')

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备判断
        
        if self.ablation_mode == 'M0_':
            # M0_: 仅先验 - 初始化特定分类器
            self.prior_baseline_net = PriorBaselineClassifier(self.prior_dim, self.num_classes).to(self.device)
            # 必须初始化 criterion，否则 train_epoch 会报错
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(self.device)
            self.optimizer = optim.AdamW(self.prior_baseline_net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epoch)
            
            # 禁用其他组件
            self.feature_extractor = None
            self.classifier = None
            self.prior_head = None
            self.prior_attention = None
            logging.info(f"DPD_DG Initialized in M0_ (Prior Only) mode. Input Dim: {self.prior_dim}")
            return # 提前退出初始化
            
        self.feature_extractor = DPD_DG_Fea_Extraction(in_channel=self.in_channel, method=self.method).to(self.device) # 初始化特征提取器
        
        # 消融：将标志传递给特征提取器
        self.feature_extractor.enable_dfe = getattr(args, 'enable_dfe', True)

        input_sample = torch.zeros(1, self.in_channel, self.signal_length).to(self.device) # 生成示例输入张量
        # 移除 MixStyle 和 Noise 标志
        try:
            with torch.no_grad():
                feat_dim = self.feature_extractor(input_sample, enable_dfe=True).shape[1] # 获取特征维度
                logging.info(f"Determined feature dimension: {feat_dim}") # 打印特征维度
        except Exception:
            feat_dim = 1024  # 容错默认特征维度
            
        prior_dim = self.prior_dim  # 先验特征维度
        
        # 从参数获取增强模式
        prior_aug_mode = getattr(args, 'prior_aug_mode', 'none')
        
        enable_explicit = getattr(args, 'enable_explicit', True)
        self.enable_explicit = enable_explicit
        self.classifier = DPD_DG_Classifier(feat_dim, num_classes, prior_dim=prior_dim, domain_knowledge=enable_explicit, method=self.method, prior_aug_mode=prior_aug_mode).to(self.device)  # 初始化分类器
        
        self.enable_papl = getattr(args, 'enable_papl', True)
        
        # 确保 prior_head 的输出维度与当前的 prior_dim 一致
        self.prior_head = PriorMLP(feat_dim, prior_dim).to(self.device) if prior_dim > 0 else None  # 先验分支
        
        # 消融：检查 enable_attention
        enable_attn = getattr(args, 'enable_attention', True)
        self.prior_attention = PriorAttention(prior_dim).to(self.device) if (prior_dim > 0 and enable_attn) else None # 先验注意力层
        
        # === 共享 Attention 模块 ===
        self.classifier.prior_attention = self.prior_attention

        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1).to(self.device)  # 交叉熵损失监督交叉熵损失
        self.unsup_loss = nn.CrossEntropyLoss(reduction="none").to(self.device)  # 无监督伪标签时的逐样本交叉熵
        self.prior_loss = nn.MSELoss(reduction="mean").to(self.device) if prior_dim > 0 else None  # 先验回归损失
        
        # === 物理一致性原型 ===
        if prior_dim > 0:
            self.prior_prototypes = torch.zeros(self.num_classes, prior_dim).to(self.device) 
            self.register_buffer('prior_prototypes_buffer', self.prior_prototypes)
            
            # === 新增：对齐融合门控 ===
            # 初始化为 0，经过 Sigmoid 后为 0.5，表示初始状态下各占一半 (或根据公式调整)
            # 这是一个可学习的参数，用于自适应调整深度特征相似度与物理先验相似度的融合比例
            self.align_gate = nn.Parameter(torch.tensor(0.0))
            
            # === 新增：RF融合门控 ===
            # 用于控制 Attention 权重与 RF 全局重要性的融合比例
            self.rf_gate = nn.Parameter(torch.tensor(0.0))
        
        # 使用 AdamW 优化器
        self.optimizer = optim.AdamW(
            [{'params': self.feature_extractor.parameters(), 'lr': self.lr},
             {'params': self.classifier.parameters(), 'lr': self.lr},],
            weight_decay=self.weight_decay
        )
        # 引入余弦退火学习率调度器
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epoch)
        
        # 确保 prior_head 的参数被加入到优化器中
        if self.prior_head is not None:
            self.optimizer.add_param_group({'params': self.prior_head.parameters(), 'lr': self.lr})
            # 将 align_gate 和 rf_gate 加入优化器
            self.optimizer.add_param_group({'params': [self.align_gate, self.rf_gate], 'lr': self.lr})
        # if self.prior_attention is not None:
        #     self.optimizer.add_param_group({'params': self.prior_attention.parameters(), 'lr': self.lr})

        # === 改进：加载 RF 重要性 ===
        enable_rf = getattr(args, 'enable_rf', True)
        if args is not None and hasattr(args, 'rf_importance_vec') and args.rf_importance_vec is not None and enable_rf:
            # 注册为 buffer，不参与梯度更新
            self.register_buffer('rf_importance', torch.tensor(args.rf_importance_vec, dtype=torch.float32).to(self.device))
            print(f"DPD_DG: 已加载 RF 特征重要性作为 Attention 基准。")
        else:
            self.rf_importance = None

    def forward(self, TR_dataloader, epoch_it):
        """
        前向传播：作为训练循环的入口。
        委托给 train_epoch 以保持代码清晰。
        """
        return self.train_epoch(TR_dataloader, epoch_it)

    def train_epoch(self, TR_dataloader, epoch_it):  # 主训练循环：接收各源域 dataloader 的列表（或迭代器）
        # M0_: Prior Only Mode - Simplified Training Loop
        if getattr(self, 'ablation_mode', 'M5') == 'M0_':
            self.prior_baseline_net.train()
            total_loss = 0.0
            batch_size = 128
            
            # Unpack dataloader list
            # TR_dataloader contains [train_loader_src_0, train_loader_src_1, train_loader_src_2]
            # src_0 is labeled, src_1/2 are unlabeled (but for M0_ we might only use src_0 or use pseudo labels?
            # User requirement: "M0_ comparable to M0". M0 is Source Only. 
            # So M0_ should also be Source Only (Supervised on Source).
            
            # Just retrieve the first dataset (Source Labeled)
            dataset_src_0 = TR_dataloader[0]
            train_loader = Data.DataLoader(dataset_src_0, batch_size=batch_size, shuffle=True)
            
            for batch_idx, (batch_x, batch_y, batch_domain, x_index, prior) in enumerate(train_loader):
                batch_x, batch_y, prior = batch_x.to(self.device), batch_y.to(self.device), prior.to(self.device)
                
                # Forward Prior Only
                logits = self.prior_baseline_net(prior)
                loss = self.criterion(logits, batch_y)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_loader)
            if epoch_it % 10 == 0:
                 logging.info(f"Epoch {epoch_it} [M0_ PriorOnly] Loss: {avg_loss:.4f}")
            return avg_loss

        #训练集列表 (每个源域的 80%)，包含了3个域数据集，每个信号1600*0.8*1*512、标签1280*1、域标签1280*1、样本索引1280和先验特征1280*维度的综合数据集
        # 主训练循环：TR_dataloader 是不同源域 Dataset 列表，本函数内部自行构建 batch 并计算各类损失。
        self.current_epoch= epoch_it
        hard_pseudo_count = 0
        hard_pseudo_count_error = 0
        physics_pass_count = 0
        physics_pass_count_error = 0
        soft_pseudo_count = 0.0
        soft_pseudo_count_error = 0.0
        
        # === 优化：分阶段课程学习策略 (Staggered Curriculum Learning) ===
        # 1. 先验损失 (Prior): 最早介入，引导特征符合物理规律 (第 1 -> 10 轮)
        prior_weight = self._get_rampup_weight(1, 10)
        
        # 2. 对齐损失 (Align): 特征初步稳定后，开始对齐域分布 (第 1 -> 30 轮)
        align_weight = self._get_rampup_weight(3, 20)
        
        # 3. 无监督损失 (Unsup): 对齐较好后，才开始信任伪标签 (第 10 -> 60 轮)
        # 修正：让伪标签更早介入，但权重增长更缓慢，给物理约束更多发挥空间
        unsup_warmup = self._get_rampup_weight(5, 50)

        self.feature_extractor.train()  # 将 feature_extractor 设置为训练模式
        self.classifier.train()  # 将 classifier 设置为训练模式
        total_loss = 0.0  # 累积总损失
        
        # === 新增：详细损失记录累积变量 ===
        epoch_loss_unsup_1_masked = 0.0
        epoch_loss_unsup_2_masked = 0.0
        epoch_prior_penalty = 0.0
        epoch_loss_align_target = 0.0
        epoch_coral_loss = 0.0
        # === 新增：更详细的损失记录 ===
        epoch_loss_sa_1 = 0.0
        epoch_loss_sim_1 = 0.0
        epoch_loss_sa_2 = 0.0
        epoch_loss_sim_2 = 0.0
        
        # === New: Accumulate FINAL weighted losses for consistency ===
        epoch_loss_final_sup = 0.0
        epoch_loss_final_unsup = 0.0
        epoch_loss_final_prior = 0.0
        epoch_loss_final_align_src = 0.0
        epoch_loss_final_align_tgt = 0.0
        
        batch_size = 128  # 每个批次大小
        
        epoch_att_weights_sum = None # 初始化 Attention 权重累积变量

        for domain, dataset in enumerate(TR_dataloader):  # 遍历不同源域的数据集
            if domain == 0:
                # 约定 domain 0 为有标签的监督源域，其余域进入伪标签流程。
                train_loader_src_0 = Data.DataLoader(dataset, batch_size)  # 创建数据加载器，一个可迭代对象，可以用for循环遍历，每次迭代返回一个批次的数据（128个样本）

            elif domain == 1:
                train_loader_src_1 = Data.DataLoader(dataset, batch_size)  # 域1加载器
                selected_label_1 = torch.ones((len(dataset),), dtype=torch.long) * -1 
                selected_label_1 = selected_label_1.to(self.device)  # 移动 selected_label_1 到设备
            elif domain == 2:
                train_loader_src_2 = Data.DataLoader(dataset, batch_size)  # 域2加载器
                selected_label_2 = torch.ones((len(dataset),), dtype=torch.long) * -1  # 初始化选择标签
                selected_label_2 = selected_label_2.to(self.device)  # 移动到设备
        
        threshold_1_list=[]  # 用于记录域1 的阈值列表（便于监控）
        threshold_2_list=[]  # 用于记录域2 的阈值列表
        
        for batch_idx, (batch_src_0, batch_src_1, batch_src_2) in enumerate(zip(train_loader_src_0, train_loader_src_1, train_loader_src_2)): 
            # 统一移动数据到设备
            batch_x_0, batch_y_0, batch_domain_0, x_index_0, prior_0 = [x.to(self.device) if torch.is_tensor(x) else x for x in batch_src_0]
            batch_x_1, batch_y_1, batch_domain_1, x_index_1, prior_1 = [x.to(self.device) if torch.is_tensor(x) else x for x in batch_src_1]
            batch_x_2, batch_y_2, batch_domain_2, x_index_2, prior_2 = [x.to(self.device) if torch.is_tensor(x) else x for x in batch_src_2]

            # === 更新物理原型(源域) ===  # 使用带动量的均值更新每类的先验原型以保证稳定
            # 不断更新每个类别的“物理特征原型”（即该类故障通常具有什么样的物理指标），用于下一轮的物理一致性检查。
            if self.prior_head is not None:
                with torch.no_grad():  # 更新原型时不计算梯度
                    for c in range(self.num_classes):  # 遍历每个类别
                        mask_c = (batch_y_0 == c)  # 用于筛选源域类别的样本，如有标签源域中标签为0的所有样本
                        if mask_c.sum() > 0:
                            current_mean = prior_0[mask_c].mean(dim=0)  # 1*先验特征长度，每一个值均为该批次所有样本的该类先验均值
                            
                            # 确保维度匹配，如果 self.prior_prototypes 维度不对，则重新初始化  
                            if self.prior_prototypes.shape[1] != current_mean.shape[0]:
                                logging.info(f"self.prior_prototypes与current_mean维度不匹配，在这里将先验原型重新初始化为0")
                                self.prior_prototypes = torch.zeros(self.num_classes, current_mean.shape[0]).to(self.device)
                                self.register_buffer('prior_prototypes_buffer', self.prior_prototypes)
                            
                            # 动量更新以获得稳定原型
                            # 如果原型仍为初始全0状态，直接赋值，避免前几个Epoch数值过小
                            if self.prior_prototypes[c].abs().sum() == 0:
                                self.prior_prototypes[c] = current_mean  #防止初试阶段原型数值过小
                            else:
                                self.prior_prototypes[c] = 0.8 * self.prior_prototypes[c] + 0.2 * current_mean  # 动量式更新
            
            #这段代码的作用是生成伪标签 (Pseudo-labels)。在后续的 loss_unsup 计算中，这些伪标签被当作“真实标签（Ground Truth）”来监督模型训练。
            #如果不加 no_grad()，计算图会一直连接到生成伪标签的过程。模型在反向传播时，不仅会更新参数以接近伪标签，还会试图更新参数来修改伪标签本身以降低 Loss。
            #这就像考试时，学生（模型）不仅可以修改自己的答案，还可以修改标准答案来让自己得分更高。这会导致模型坍塌（Model Collapse），学不到任何有用的特征。
            with torch.no_grad():  # 切断梯度流（防止“作弊”）。在伪标签与物理一致性计算中暂时关闭梯度
                # 模块: PCPL (物理一致性感知伪标签模块)
                # 1) 推理伪标签并基于类别自适应阈值筛选高置信样本
                
                # 处理域 1
                features_1, logits_1, pseudo_labels_1, mask_1, threshold_1, hard_pseudo_count_1, physics_pass_count_1, soft_pseudo_count_1, hard_pseudo_count_error_1, physics_pass_count_error_1, soft_pseudo_count_error_1 = self._process_unlabeled_domain(batch_x_1, batch_y_1, prior_1, 1)
                threshold_1_list.append(threshold_1.cpu())

                # 处理域 2
                features_2, logits_2, pseudo_labels_2, mask_2, threshold_2, hard_pseudo_count_2, physics_pass_count_2, soft_pseudo_count_2, hard_pseudo_count_error_2, physics_pass_count_error_2, soft_pseudo_count_error_2 = self._process_unlabeled_domain(batch_x_2, batch_y_2, prior_2, 2)
                threshold_2_list.append(threshold_2.cpu())
                
                hard_pseudo_count += hard_pseudo_count_1 + hard_pseudo_count_2
                physics_pass_count += physics_pass_count_1 + physics_pass_count_2
                soft_pseudo_count += soft_pseudo_count_1 + soft_pseudo_count_2
                hard_pseudo_count_error += hard_pseudo_count_error_1 + hard_pseudo_count_error_2
                physics_pass_count_error += physics_pass_count_error_1 + physics_pass_count_error_2
                soft_pseudo_count_error += soft_pseudo_count_error_1 + soft_pseudo_count_error_2
           
            #===干净样本的监督损失===#
            features_clean_0 = self.feature_extractor(batch_x_0, enable_dfe=True)
            logits_clean_0 = self.classifier(features_clean_0, prior=prior_0, domain_knowledge=self.enable_explicit)
            loss_clean_0 = self.criterion(logits_clean_0, batch_y_0)
            loss_supervised = loss_clean_0

            #第一次计算是为了生成“标准答案”（目标），它们在 Loss 函数中充当 y（真实标签）的角色。
            #第二次计算是为了生成“学生作业”（预测），它们在 Loss 函数中充当 y_pre（预测值）的角色。只有“学生作业”才需要反向传播来修改模型。)

            #第一次计算是为了生成“标准答案”（目标），它们在 Loss 函数中充当 y（真实标签）的角色。
            #第二次计算是为了生成“学生作业”（预测），它们在 Loss 函数中充当 y_pre（预测值）的角色。只有“学生作业”才需要反向传播来修改模型。
            features_1 = self.feature_extractor(batch_x_1.clone(),enable_dfe=True)  # 对域1 的样本进行扰动前向
            logits_1 = self.classifier(features_1, prior=prior_1, domain_knowledge=self.enable_explicit)  # 获取 logits
            loss_unsup_1 = self.unsup_loss(logits_1, pseudo_labels_1)  # 计算伪标签损失（每样本）
            loss_unsup_1_masked = (loss_unsup_1 * mask_1).mean()  # 应用掩码，仅计算高置信样本的平均损失
            
            features_2 = self.feature_extractor(batch_x_2.clone(),enable_dfe=True)  # 同理域2
            logits_2 = self.classifier(features_2, prior=prior_2, domain_knowledge=self.enable_explicit)  # 获取 logits
            loss_unsup_2 = self.unsup_loss(logits_2, pseudo_labels_2)  # 计算伪标签损失（每样本）
            loss_unsup_2_masked = (loss_unsup_2 * mask_2).mean()  # 应用掩码，仅计算高置信样本的平均损失
           
           
                # features_clean_0 在上文已计算
            loss_prior_0, w_clean = self._compute_prior_loss_term(features_clean_0, prior_0)               
                
                # 记录 Attention 权重 (仅使用源域干净数据作为代表)
            if w_clean is not None:
                batch_mean_w = w_clean.mean(dim=0).detach()
                if epoch_att_weights_sum is None:
                    epoch_att_weights_sum = batch_mean_w
                else:
                    epoch_att_weights_sum += batch_mean_w
            
            # 2. 目标域 1 
            loss_prior_1, _ = self._compute_prior_loss_term(features_1, prior_1)
            
            # 3. 目标域 2
            loss_prior_2, _ = self._compute_prior_loss_term(features_2, prior_2)
            
            # 平均化所有损失项
            prior_penalty = (loss_prior_0 + loss_prior_1 + loss_prior_2) / 3.0
            
            epoch_prior_penalty += prior_penalty.item()


            # === 改进：使用渐进式权重 (Ramp-up) 替代硬阈值 ===
            # 相比于 >50 的硬截断，渐进式策略能避免 Loss 突变，且能更早利用对齐信息
            
            loss_sa_source = torch.tensor(0.0, device=self.device)
            loss_sim_source = torch.tensor(0.0, device=self.device)
            loss_sa_target = torch.tensor(0.0, device=self.device)
            loss_sim_target = torch.tensor(0.0, device=self.device)
            loss_semantic_align_0 = torch.tensor(0.0, device=self.device)
            loss_similarity_0 = torch.tensor(0.0, device=self.device)
            loss_semantic_align_1 = torch.tensor(0.0, device=self.device)
            loss_similarity_1 = torch.tensor(0.0, device=self.device)
            loss_semantic_align_2 = torch.tensor(0.0, device=self.device)
            loss_similarity_2 = torch.tensor(0.0, device=self.device)
            coral_loss_1 = torch.tensor(0.0, device=self.device)
            coral_loss_2 = torch.tensor(0.0, device=self.device)
            coral_loss = torch.tensor(0.0, device=self.device)
            enable_align = getattr(self.args, 'enable_align', True)
            if align_weight > 0 and enable_align:
                # > epoch 后启用原型对齐：源域原型与伪标签原型对齐以提升跨域一致性。
                class_prototypes = {}  # 保存各域类别原型
                
                # 计算0有标签源域类别原型0 (使用干净特征 features_clean_0)，源域0有5个类别原型
                class_prototypes[0] = self._compute_domain_prototypes(features_clean_0.detach(), batch_y_0, self.num_classes)

                # === 修正：原型对齐逻辑 ===
                # 核心思想：所有域的样本都应该向“源域原型”看齐，而不是各自为政。
                # 源域原型 (prototypes_0) 是最可靠的锚点。
                
                prototypes_0_norm = F.normalize(class_prototypes[0], p=2, dim=1)  # 归一化源域原型 (Anchor)
                
                # 1. 源域样本 vs 源域原型 (类内紧凑性)
                features_clean_0_norm = F.normalize(features_clean_0, p=2, dim=1)
                similarity_0 = torch.mm(features_clean_0_norm, prototypes_0_norm.t())
                mask_0 = torch.ones(features_clean_0_norm.size(0)).to(self.device) #

                # 2. 域1样本 vs 源域原型 (域间对齐)
                features_1_norm = F.normalize(features_1, p=2, dim=1)
                similarity_1 = torch.mm(features_1_norm, prototypes_0_norm.t())
                
                # 3. 域2样本 vs 源域原型 (域间对齐)
                features_2_norm = F.normalize(features_2, p=2, dim=1)
                similarity_2 = torch.mm(features_2_norm, prototypes_0_norm.t())

                # === 融入先验知识到原型对齐 ===
                if self.prior_head is not None:
                    # === 新增：应用注意力机制 ===
                    # 如果存在 prior_attention，则先对先验特征和原型进行加权，使对齐更加关注重要物理特征
                    if hasattr(self, 'prior_attention') and self.prior_attention is not None:
                        # 对原型应用注意力
                        prior_prototypes_weighted, _ = self.prior_attention(self.prior_prototypes)
                        prior_proto_norm = F.normalize(prior_prototypes_weighted, p=2, dim=1)
                        
                        # 对各域先验应用注意力
                        prior_0_weighted, _ = self.prior_attention(prior_0)
                        prior_0_norm = F.normalize(prior_0_weighted, p=2, dim=1)
                        
                        prior_1_weighted, _ = self.prior_attention(prior_1)
                        prior_1_norm = F.normalize(prior_1_weighted, p=2, dim=1)
                        
                        prior_2_weighted, _ = self.prior_attention(prior_2)
                        prior_2_norm = F.normalize(prior_2_weighted, p=2, dim=1)
                    else:
                        # 原有逻辑：直接使用原始先验特征
                        prior_proto_norm = F.normalize(self.prior_prototypes, p=2, dim=1)
                        prior_0_norm = F.normalize(prior_0, p=2, dim=1)
                        prior_1_norm = F.normalize(prior_1, p=2, dim=1)
                        prior_2_norm = F.normalize(prior_2, p=2, dim=1)
                    
                    # 域0: 融合物理相似度
                    prior_sim_0 = torch.mm(prior_0_norm, prior_proto_norm.t())
                    # 使用门控机制进行融合: (1-gate) * Deep + gate * Prior
                    align_gate_val = torch.sigmoid(self.align_gate)
                    similarity_0 = (1 - align_gate_val) * similarity_0 + align_gate_val * prior_sim_0 
                    
                    # 域1: 融合物理相似度
                    prior_sim_1 = torch.mm(prior_1_norm, prior_proto_norm.t())
                    similarity_1 = (1 - align_gate_val) * similarity_1 + align_gate_val * prior_sim_1
                    
                    # 域2: 融合物理相似度
                    prior_sim_2 = torch.mm(prior_2_norm, prior_proto_norm.t())
                    similarity_2 = (1 - align_gate_val) * similarity_2 + align_gate_val * prior_sim_2

                loss_semantic_align_0 = self.compute_sa_loss_multi(similarity_0,  mask_0, batch_y_0)  # 基于原型的语义对齐损失，通常会通过交叉熵或类似的逻辑，强制要求样本与其真实标签对应的原型相似度最大化。
                sim_top1_0, _ = similarity_0.topk(k=1, dim=1, sorted=True)  # 找出每个样本与所有原型中最相似的那一个的值（即最大相似度）。
                loss_similarity_0 = ((1 - sim_top1_0.squeeze(1)) * mask_0).mean()  # 计算最大相似度与理想值 1.0 的差距，无论样本属于哪一类，它至少应该非常接近某一个原型。这是一种类内紧凑性 (Intra-class Compactness) 的约束，让特征分布更收敛，不发散。

                loss_semantic_align_1 = self.compute_sa_loss_multi(similarity_1,  mask_1, pseudo_labels_1)  
                sim_top1_1, _ = similarity_1.topk(k=1, dim=1, sorted=True)
                loss_similarity_1 = ((1 - sim_top1_1.squeeze(1)) * mask_1).mean()  

                loss_semantic_align_2 = self.compute_sa_loss_multi(similarity_2,  mask_2, pseudo_labels_2)  
                sim_top1_2, _ = similarity_2.topk(k=1, dim=1, sorted=True)
                loss_similarity_2 = ((1 - sim_top1_2.squeeze(1))* mask_2).mean()  
                # === 分离源域和目标域损失 ===
                loss_sa_source = loss_semantic_align_0
                loss_sim_source = loss_similarity_0
                loss_sa_target = loss_semantic_align_1 + loss_semantic_align_2
                loss_sim_target = loss_similarity_1 + loss_similarity_2
                 # === 使用 CORAL ===
                # CORAL 对齐二阶统计量 (协方差)，计算更高效且对旋转机械信号往往更敏感
                coral_loss_1 = self.coral(features_clean_0, features_1)
                coral_loss_2 = self.coral(features_clean_0, features_2)
                coral_loss = coral_loss_1 + coral_loss_2       
            else:
                loss_sa_source = torch.tensor(0.0, device=self.device)
                loss_sim_source = torch.tensor(0.0, device=self.device)
                loss_sa_target = torch.tensor(0.0, device=self.device)
                loss_sim_target = torch.tensor(0.0, device=self.device)
                coral_loss=0.0
            
            epoch_loss_sa_1 += loss_semantic_align_1.item() if torch.is_tensor(loss_semantic_align_1) else loss_semantic_align_1
            epoch_loss_sim_1 += loss_similarity_1.item() if torch.is_tensor(loss_similarity_1) else loss_similarity_1
            epoch_loss_sa_2 += loss_semantic_align_2.item() if torch.is_tensor(loss_semantic_align_2) else loss_semantic_align_2
            epoch_loss_sim_2 += loss_similarity_2.item() if torch.is_tensor(loss_similarity_2) else loss_similarity_2
            epoch_coral_loss += coral_loss.item() if torch.is_tensor(coral_loss) else coral_loss


            # 总损失 = 监督损失 + 伪标签损失 * warmup + 先验一致性 + 源域对齐(全权重) + 目标域对齐(热身权重)
            # 增加 prior_penalty 的权重系数 (x10) 以平衡量级
            
            # === 计算各部分最终损失 (Weighted) ===
            
            # 1. 监督损失
            loss_supervised_final = loss_supervised * self.lambda_sup
            
            # 2. 无监督伪标签损失 (带 Warmup)
            loss_unsup_final = self.lambda_unsup * (loss_unsup_1_masked + loss_unsup_2_masked) * unsup_warmup
            
            # 3. 先验一致性损失 (带 Warmup)
            loss_prior_final = self.lambda_prior * prior_penalty * prior_weight
            
            # 4. 源域对齐损失 (始终生效)
            # 包含语义对齐 (SA) 和 相似度 (Sim)
            loss_align_source_raw = loss_sa_source * self.lambda_sa + loss_sim_source * self.lambda_sim
            loss_align_source_final = loss_align_source_raw
            
            # 5. 目标域对齐损失 (带 Warmup)
            # 包含语义对齐 (SA), 相似度 (Sim) 和 CORAL
            # 修正：明确对两个目标域取平均，使量级与源域对齐一致
            loss_align_target_raw = (loss_sa_target * self.lambda_sa + loss_sim_target * self.lambda_sim) / 2.0
            loss_coral_weighted = (coral_loss * self.lambda_coral) / 2.0
            
            # align_weight 控制整个目标域对齐项的介入
            loss_align_target_final = (loss_align_target_raw + loss_coral_weighted) * align_weight * self.lambda_align_target

            # === 总损失 ===
            loss = loss_supervised_final + loss_unsup_final + loss_prior_final + loss_align_source_final + loss_align_target_final

            self.optimizer.zero_grad()  # 优化器梯度清零
            loss.backward()  # 反向传播
            self.optimizer.step()  # 优化
            
            # 修正：使用当前 batch 计算出的最终 loss 值进行累加，确保日志一致性
            total_loss += loss.item() 
            
            # === 累积最终组件 ===
            epoch_loss_final_sup += loss_supervised_final.item()
            epoch_loss_final_unsup += loss_unsup_final.item()
            epoch_loss_final_prior += loss_prior_final.item()
            epoch_loss_final_align_src += loss_align_source_final.item() if torch.is_tensor(loss_align_source_final) else loss_align_source_final
            epoch_loss_final_align_tgt += loss_align_target_final.item() if torch.is_tensor(loss_align_target_final) else loss_align_target_final


            # === 新增：累积详细损失 ===
            epoch_loss_unsup_1_masked += loss_unsup_1_masked.item()
            epoch_loss_unsup_2_masked += loss_unsup_2_masked.item()
            epoch_prior_penalty += prior_penalty.item()
            epoch_loss_align_target += loss_align_target_raw.item() if torch.is_tensor(loss_align_target_raw) else loss_align_target_raw
            epoch_coral_loss += coral_loss.item() if torch.is_tensor(coral_loss) else coral_loss

        logging.info(f"Epoch [{epoch_it}] - Hard_pseudo_count: {hard_pseudo_count} (erro: {hard_pseudo_count_error/(hard_pseudo_count+0.1)*100:.1f}%), Physics_pass_count: {physics_pass_count} (error: {physics_pass_count_error/(physics_pass_count+0.1)*100:.1f}%), Soft_pseudo_count: {soft_pseudo_count:.1f} (erro: {soft_pseudo_count_error/(soft_pseudo_count+0.1)*100:.1f}%)")
        # 更新学习率，每个 epoch 结束后，上面是每个epoch里面多个批次的循环操作
        if self.scheduler is not None:
            self.scheduler.step()

        # === 记录 Attention 权重与详细损失 ===
        target_epochs = [0, 1, 5, 10, 15, 25, 50, 60]
        num_batches = batch_idx + 1
        
        # 计算当前 Epoch 的平均总损失
        avg_total_loss = total_loss / num_batches

        if epoch_it in target_epochs:
            # 记录 Attention Weights
            if epoch_att_weights_sum is not None:
                epoch_avg_w = epoch_att_weights_sum / num_batches
                w_str = ", ".join([f"{x:.3f}" for x in epoch_avg_w.cpu().numpy()])
                logging.info(f"Epoch {epoch_it} Attention Weights: [{w_str}]")
            
            # 记录详细损失
            avg_unsup1_raw = epoch_loss_unsup_1_masked / num_batches
            avg_unsup2_raw = epoch_loss_unsup_2_masked / num_batches
            avg_prior_raw = epoch_prior_penalty / num_batches
            avg_coral_raw = epoch_coral_loss / num_batches
            
            # === 新增：详细的平均值 ===
            avg_sa_1 = epoch_loss_sa_1 / num_batches
            avg_sim_1 = epoch_loss_sim_1 / num_batches
            avg_sa_2 = epoch_loss_sa_2 / num_batches
            avg_sim_2 = epoch_loss_sim_2 / num_batches
            avg_align_target = epoch_loss_align_target / num_batches

            # === 1. 计算总损失的 5 个部分 (最终加权组件) ===
            # 使用累积的最终损失，确保 Sum == AvgTotal
            final_sup = epoch_loss_final_sup / num_batches
            final_unsup = epoch_loss_final_unsup / num_batches
            final_prior = epoch_loss_final_prior / num_batches
            final_align_src = epoch_loss_final_align_src / num_batches
            final_align_tgt = epoch_loss_final_align_tgt / num_batches
            
            # avg_coral_weighted 仅用于详细日志
            avg_coral_weighted = avg_coral_raw * self.lambda_coral
            
            # 验证总和是否等于平均总损失
            sum_components = final_sup + final_unsup + final_prior + final_align_src + final_align_tgt
            
            logging.info(f"Epoch {epoch_it} [Total Loss Components]: "
                         f"Sum={sum_components:.2f} (AvgTotal={avg_total_loss:.2f}), "
                         f"Sup={final_sup:.2f}, "
                         f"Unsup={final_unsup:.2f}, "
                         f"Prior={final_prior:.2f}, "
                         f"AlignSrc={final_align_src:.2f}, "
                         f"AlignTgt={final_align_tgt:.2f}")

            # === 2. 输出详细的 Raw Loss 和 参数 ===
            avg_sa_target = (avg_sa_1 + avg_sa_2) / 2.0
            avg_sim_target = (avg_sim_1 + avg_sim_2) / 2.0
            avg_coral_raw_mean = avg_coral_raw / 2.0
            avg_coral_weighted_mean = avg_coral_weighted / 2.0
            
            logging.info(f"Epoch {epoch_it} [Detailed Raw Losses & Params]: "
                         f"Unsup1={avg_unsup1_raw:.4f}, Unsup2={avg_unsup2_raw:.4f} (L_unsup={self.lambda_unsup}, w={unsup_warmup:.2f}) | "
                         f"Prior={avg_prior_raw:.4f} (L_prior={self.lambda_prior}, w={prior_weight:.2f}) | "
                         f"SA_Tgt_Mean={avg_sa_target:.4f} (L_sa={self.lambda_sa}), Sim_Tgt_Mean={avg_sim_target:.4f} (L_sim={self.lambda_sim}) | "
                         f"AlignTgtRaw_Mean={avg_align_target:.4f} | "
                         f"Coral_Mean={avg_coral_raw_mean:.4f} (L_coral={self.lambda_coral}) => CoralW_Mean={avg_coral_weighted_mean:.4f} | "
                         f"Global_AlignTgt (L_align={self.lambda_align_target}, w={align_weight:.2f})")

        return avg_total_loss  # 返回 epoch 的平均损失

    def _get_rampup_weight(self, start, end):
        """计算线性预热权重"""
        if self.current_epoch < start:
            return 0.0
        if self.current_epoch >= end:
            return 1.0
        return (self.current_epoch - start) / (end - start) 

    def _process_unlabeled_domain(self, x, y, prior, dataset_idx):
        """
        处理无标签域的通用逻辑：特征提取、伪标签生成、物理一致性检查、更新类精度
        """
        current_batch_hard_pseudo_count = 0
        current_batch_physics_pass_count = 0 # 新增：物理检查通过的绝对数量
        current_batch_soft_pseudo_count = 0.0
        current_batch_hard_pseudo_count_error = 0
        current_batch_physics_pass_count_error = 0
        current_batch_soft_pseudo_count_error = 0.0
        
        features = self.feature_extractor(x, enable_dfe=True)  # 特征提取
        logits = self.classifier(features, prior=prior, domain_knowledge=self.enable_explicit)  # 通过分类器获得 logits
        probs = F.softmax(logits, dim=1)  # 计算概率
        max_probs, pseudo_labels = probs.max(dim=1)  # 置信度与预测类别
        
        # 使用固定阈值
        threshold = torch.tensor(self.threshold).to(self.device)
        mask = (max_probs >= threshold).float()  # 生成高置信样本掩码
        current_batch_hard_pseudo_count = mask.sum().item() #  计算当前 batch 超过阈值的硬标签数量
        current_batch_hard_pseudo_count_error = ((mask == 1) & (pseudo_labels != y)).sum().item()  # 计算当前 batch 错误的硬标签数量

        # 物理一致性检查 (添加 Warmup 机制：前 10 个 Epoch 不进行物理过滤)
        # Ablation: enable_papl controls whether we perform this check
        if self.prior_head is not None and self.current_epoch >= 0 and self.enable_papl:
            pred_prototypes = self.prior_prototypes[pseudo_labels] # 获取对应预测类别的原型
            
            if self.prior_attention is not None:
                _, w = self.prior_attention(pred_prototypes) #计算原型自身的注意力权重
                prior_w = prior * w
                pred_prototypes_w = pred_prototypes * w
            else:
                prior_w = prior
                pred_prototypes_w = pred_prototypes
            
            # 计算余弦相似度
            physics_sim = F.cosine_similarity(prior_w, pred_prototypes_w, dim=1)
            
            # === 改进：不再硬性过滤，而是作为置信度的修正系数 ===
            # 如果物理相似度高，保持或增加置信度；如果低，降低权重，而不是直接置0
            # 映射相似度 [-1, 1] -> [0, 1]
            sim_score = (physics_sim + 1) / 2
            
            # 只有当物理相似度极低时（确实完全不符合物理规律），才进行硬截断
            mask_physics = (physics_sim > self.physics_thresh).float()
            current_batch_physics_pass_count = mask_physics.sum().item()  # 计算当前 batch 物理检查通过的样本数量
            current_batch_physics_pass_count_error = ((mask_physics == 1) & (pseudo_labels != y)).sum().item()  # 计算当前 batch 物理检查通过但错误的样本数量
            
            # 策略 A: 硬截断 (原策略，风险大)
            # mask = mask * mask_physics
            
            # 策略 B: 软加权 (推荐，更鲁棒)
            # 将物理相似度作为样本权重的系数，保留样本但降低其Loss贡献
            # mask = mask * sim_score
            
            # 策略 C + 兜底: 软加权，但如果低于阈值则直接置0，物理相似度低于 physics_thresh 的样本实在太离谱，连软加权都不配，直接丢弃
            mask = mask * sim_score * mask_physics 
            current_batch_soft_pseudo_count = mask.sum().item() # 2. 计算当前 batch 超过阈值的软标签数量
            current_batch_soft_pseudo_count_error = mask[(mask > 0) & (pseudo_labels != y)].sum().item()  # 计算当前 batch 错误的软标签数量
           

        return features, logits, pseudo_labels, mask, threshold, current_batch_hard_pseudo_count, current_batch_physics_pass_count, current_batch_soft_pseudo_count,current_batch_hard_pseudo_count_error, current_batch_physics_pass_count_error, current_batch_soft_pseudo_count_error

    def _compute_prior_loss_term(self, features, target_prior):
        """
        计算单个特征-先验对的先验损失
        """
        if self.prior_head is None:
            return torch.tensor(0.0, device=self.device), None
            
        pred_prior = self.prior_head(features)
        
        # 注意：target_prior 已经在 DataLoader 中做过全局 Z-Score 归一化，此处不再重复归一化
        # 避免双重归一化导致 Batch 间差异信息丢失

        if self.prior_attention is not None:
            # === Attention 加权 ===
            # 修正：使用真实先验 (target_norm) 而非预测先验 (pred_prior) 来计算权重
            # 原因 1: 保持一致性。在物理一致性检查中，Attention 是作用于真实先验的。
            # 原因 2: 避免作弊。防止 PriorMLP 通过输出特定值来诱导 Attention 生成小权重以降低 Loss，而非真正去拟合数据。
            _, w = self.prior_attention(target_prior)
            
            # === 改进：融合 RF 重要性 (Scaling Gating) ===
            if getattr(self, 'rf_importance', None) is not None:
                # 融合策略：动态权重与静态全局重要性的加权平均
                # 这保留了 Attention 的动态性，同时利用 RF 的全局经验防止训练初期的盲目性
                # 使用门控机制: (1-gate) * Attention + gate * RF
                rf_gate_val = torch.sigmoid(self.rf_gate)
                w = (1 - rf_gate_val) * w + rf_gate_val * self.rf_importance

            # 2. 计算原始误差 (不带权)
            squared_diff = (pred_prior - target_prior).pow(2)

            # 3. 应用 Attention 权重
            weighted_squared_diff = w * squared_diff

            # 4. 计算最终损失 (MSE + 正则化)
            mse_loss = weighted_squared_diff.mean()
            reg_loss = (1.0 - w).pow(2).mean()
            
            return mse_loss + 0.1 * reg_loss, w
        else:
            # 无 Attention，直接 MSE
            return self.prior_loss(pred_prior, target_prior), None

    def _compute_domain_prototypes(self, features, labels, num_classes):
        """计算给定特征和标签的类原型"""
        prototypes = []
        for c in range(num_classes):
            mask = (labels == c)
            if mask.sum() > 0:
                prototype = features[mask].mean(dim=0)
            else:
                prototype = torch.zeros(features.size(1), device=features.device)
            prototypes.append(prototype)
        return torch.stack(prototypes)

    def compute_sa_loss_multi(self, similarity, mask_x, num_classes):  # 计算多域的 prototype SA 损失（cross-entropy）
        # 根据 similarity 与伪标签/真实标签计算交叉熵，mask 控制仅使用高置信样本。
        loss_similarity = F.cross_entropy(similarity,num_classes.to(self.device), reduction="none")*3  # 按样本计算交叉熵并乘系数
        loss_similarity = (loss_similarity * mask_x).mean()  # 根据 mask 进行加权平均
        return loss_similarity  # 返回 SA 损失

    def coral(self, source, target):
        d = source.data.shape[1]
        # source covariance
        xm = source - torch.mean(source, 0, keepdim=True)
        xc = torch.matmul(xm.t(), xm)
        # target covariance
        ym = target - torch.mean(target, 0, keepdim=True)
        yc = torch.matmul(ym.t(), ym)
        # frobenius norm
        loss = torch.sum(torch.pow(xc - yc, 2))
        loss = loss / (4*d*d)
        return loss

    def model_inference(self, input, prior=None):  # 推理：禁用噪声、DFE 等并返回 logits
        # M0_: Prior Only Mode
        if getattr(self, 'ablation_mode', 'M5') == 'M0_':
            self.prior_baseline_net.eval()
            with torch.no_grad():
                if prior is not None:
                    prior = prior.to(self.device)
                    prediction = self.prior_baseline_net(prior)
                else:
                    # Fallback or error if no prior provided (should not happen in M0_)
                    return torch.zeros(input.size(0), self.num_classes).to(self.device)
            return prediction

        # 推理阶段关闭DFE 并返回分类 logits。
        self.feature_extractor.eval()  # 设为推理模式
        self.classifier.eval()  # 设为推理模式
        with torch.no_grad():  # 禁用梯度计算
            features = self.feature_extractor(input.to(self.device), enable_dfe=False)
            # 如果提供了 prior，则将其移动到设备并开启 domain_knowledge 引导
            if prior is not None:
                prior = prior.to(self.device)
                dk_flag = self.enable_explicit
            else:
                dk_flag = False
            prediction = self.classifier(features, prior=prior, domain_knowledge=dk_flag)
        return prediction  # 返回预测 logits

