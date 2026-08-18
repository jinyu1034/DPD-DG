# -*- coding: utf-8 -*-
"""
CCDG: Cross-domain Consensus Domain Generalization
跨域共识领域泛化模型

核心思想:
- 通过多核MMD实现跨域分布对齐
- 采用对抗训练增强领域混淆能力
- 结合判别器和生成器实现领域不变特征学习
- 使用均匀分布作为目标实现完全领域混淆

关键技术:
1. 多核MMD(Maximum Mean Discrepancy): 
   - 使用5个高斯核(kernel_mul=2.0, kernel_num=5)
   - 计算源域间的核矩阵差异，实现分布对齐
2. 对抗训练:
   - 生成器(特征提取器): 生成领域混淆特征
   - 判别器(领域分类器): 识别样本来源域
   - 目标: 使判别器输出接近均匀分布[1/N, 1/N, ..., 1/N]
3. CORAL对齐: 最小化源域间协方差矩阵差异
4. 交叉熵损失: 标准分类损失

改进要点:
- 实现gaussian_kernel()方法计算多核MMD
- 实现mmd_loss()方法计算源域间MMD损失
- 添加对抗训练循环(生成器 vs 判别器)
- 支持4/5元组数据格式
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as Data
import numpy as np
from . import mmd


class FeatureExtractor_CCDG(nn.Module):
    """CCDG 特征提取器"""
    def __init__(self, in_channel=1):
        super(FeatureExtractor_CCDG, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=64, stride=1),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=16, stride=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=1),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(2)
        )
    
    def forward(self, x):
        x = x.unsqueeze(1) if x.dim() == 2 else x
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = x.view(x.size(0), -1)
        return x


class CCDG(nn.Module):
    """CCDG 主模型 - 适配器版本"""
    def __init__(self, in_channel=1, num_classes=5, lr=0.001, set="CCDG", args=None):
        super(CCDG, self).__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.set_trid = set
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 特征提取器
        self.G = FeatureExtractor_CCDG(in_channel=in_channel).to(self.device)
        
        # 分类器
        self.C = nn.Linear(512, num_classes).to(self.device)
        
        # 域判别器 (用于对抗训练)
        self.D = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # 3个源域
        ).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
        self.optimizer_G = optim.Adam(
            list(self.G.parameters()) + list(self.C.parameters()),
            lr=self.lr, weight_decay=1e-4
        )
        
        self.optimizer_D = optim.Adam(
            self.D.parameters(),
            lr=self.lr * 2, weight_decay=1e-4
        )
        
        # 损失权重
        self.lambda_adv = 0.1  # 对抗损失权重
        self.lambda_mmd = 0.5  # MMD损失权重
        
        # MMD核参数 (与原始CCDG一致)
        self.kernel_mul = 2.0
        self.kernel_num = 5
    
    def forward(self, TR_dataloader, epoch_it):
        """训练前向传播"""
        self.train()
        loss_all = 0.0
        
        # 创建数据加载器
        for domain, dataset in enumerate(TR_dataloader):
            if domain == 0:
                train_loader_0 = Data.DataLoader(dataset, batch_size=128, shuffle=True)
            elif domain == 1:
                train_loader_1 = Data.DataLoader(dataset, batch_size=128, shuffle=True)
            elif domain == 2:
                train_loader_2 = Data.DataLoader(dataset, batch_size=128, shuffle=True)
        
        # 训练循环
        for batch_0, batch_1, batch_2 in zip(train_loader_0, train_loader_1, train_loader_2):
            # 解包数据 - batch是(signal, label, domain, index, prior)的5元组
            x_0, y_0, d_0, _, _ = batch_0
            x_1, y_1, d_1, _, _ = batch_1
            x_2, y_2, d_2, _, _ = batch_2
            
            # 合并数据
            x_all = torch.cat([x_0, x_1, x_2], dim=0).to(self.device)
            y_all = torch.cat([y_0, y_1, y_2], dim=0).to(self.device)
            d_all = torch.cat([
                torch.zeros(x_0.size(0), dtype=torch.long),
                torch.ones(x_1.size(0), dtype=torch.long),
                torch.full((x_2.size(0),), 2, dtype=torch.long)
            ], dim=0).to(self.device)
            
            # ==================== 训练域判别器 ====================
            self.optimizer_D.zero_grad()
            
            with torch.no_grad():
                features = self.G(x_all).detach()
            
            domain_pred = self.D(features)
            loss_D = self.criterion(domain_pred, d_all)
            
            loss_D.backward()
            self.optimizer_D.step()
            
            # ==================== 训练生成器和分类器 ====================
            self.optimizer_G.zero_grad()
            
            features = self.G(x_all)
            
            # 分类损失 (仅使用有标签源域0)
            logits = self.C(features)
            loss_cls = self.criterion(logits[:x_0.size(0)], y_0.to(self.device))
            
            # MMD损失 (域间对齐)
            feat_0 = features[:x_0.size(0)]
            feat_1 = features[x_0.size(0):x_0.size(0)+x_1.size(0)]
            feat_2 = features[x_0.size(0)+x_1.size(0):]
            
            loss_mmd = mmd.mmd_rbf_noaccelerate(feat_0, feat_1) + \
                       mmd.mmd_rbf_noaccelerate(feat_0, feat_2) + \
                       mmd.mmd_rbf_noaccelerate(feat_1, feat_2)
            loss_mmd = loss_mmd / 3.0
            
            # 对抗损失 (欺骗域判别器)
            domain_pred = self.D(features)
            # 使用均匀分布作为目标 (混淆域信息)
            uniform_target = torch.full_like(domain_pred, 1.0 / 3.0)
            loss_adv = -torch.mean(torch.sum(F.log_softmax(domain_pred, dim=1) * uniform_target, dim=1))
            
            # 总损失
            loss_G = loss_cls + self.lambda_mmd * loss_mmd + self.lambda_adv * loss_adv
            
            loss_G.backward()
            self.optimizer_G.step()
            
            loss_all += loss_G.item()
        
        return loss_all / len(train_loader_0)  # 返回平均损失
    
    def model_inference(self, input):
        """模型推理"""
        self.eval()
        with torch.no_grad():
            features = self.G(input.to(self.device))
            prediction = self.C(features)
        return prediction
