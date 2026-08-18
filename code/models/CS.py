# -*- coding: utf-8 -*-
"""
CS Model: Crafting Shifts
基于多视图/多模态数据增强的域泛化模型，适配半监督/单源域泛化实验设置

核心思想:
- 多视图学习: 同时利用原始数据和增强数据进行训练
- 共享骨干网络: 所有视图共享同一个特征提取器
- 加权损失: 
  - 原始视图损失权重为 1.0
  - 增强视图损失权重由 method_loss 参数控制
- 推理集成: 测试时融合多视图预测结果(可选，本实现主要关注训练时的正则化效果)

适配说明:
- 参照 CNN.py 的基线架构进行修改
- 适配 forward 中包含训练循环的特殊设计
- 适配半监督设置:
  - 有标签数据: 计算原始视图和增强视图的监督损失
  - 无标签数据: 计算原始视图和增强视图的一致性损失(伪标签)
- 数据增强: 使用高斯噪声和幅度缩放模拟不同视图
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as Data
import numpy as np

class CNN_FeatureExtractor(nn.Module):
    """标准CNN特征提取器 (与CNN.py保持一致)"""
    def __init__(self, in_channel=1):
        super(CNN_FeatureExtractor, self).__init__()
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

class CNN_Classifier(nn.Module):
    """标准分类器"""
    def __init__(self, input_dim=512, num_classes=10):
        super(CNN_Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CS(nn.Module):
    def __init__(self, in_channel, num_classes, lr, set, args):
        super(CS, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.lr = lr
        self.set = set
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 组件
        self.feature_extractor = CNN_FeatureExtractor(in_channel)
        self.classifier = CNN_Classifier(512, num_classes)
        
        self.optimizer = optim.Adam(
            list(self.feature_extractor.parameters()) + 
            list(self.classifier.parameters()),
            lr=1e-3,
            weight_decay=5e-4
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # 超参数
        self.method_loss = 1.0 # 增强视图损失权重
        self.conf_threshold = 0.95

    def get_augmented_views(self, x):
        """生成增强视图"""
        # 视图 1: 原始
        v1 = x
        
        # 视图 2: 噪声 (高斯噪声)
        noise = torch.randn_like(x) * 0.05
        v2 = x + noise
        
        # 视图 3: 缩放 (幅度缩放)
        scale = torch.rand(x.size(0), 1, 1).to(self.device) * 0.4 + 0.8 # [0.8, 1.2]
        v3 = x * scale
        
        return [v1, v2, v3]

    def forward_backbone(self, views):
        """
        多视图通过共享骨干网络的前向传播
        参数:
            views: 张量列表
        返回:
            outputs: logits 列表
        """
        # 拼接所有视图以进行批处理 (高效)
        x_cat = torch.cat(views, dim=0)
        feat_cat = self.feature_extractor(x_cat)
        logits_cat = self.classifier(feat_cat)
        
        # 拆分回视图
        batch_size = views[0].size(0)
        outputs = torch.split(logits_cat, batch_size, dim=0)
        
        return list(outputs)

    def forward(self, TR_dataloader, epoch_it):
        self.train()
        
        # 创建数据加载器
        for domain, dataset in enumerate(TR_dataloader):
            if domain == 0:
                train_loader_0 = Data.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
            elif domain == 1:
                train_loader_1 = Data.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
            elif domain == 2:
                train_loader_2 = Data.DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_0, batch_1, batch_2 in zip(train_loader_0, train_loader_1, train_loader_2):
            # 解包数据
            data_0, label_0, _, _, _ = batch_0
            data_1, _, _, _, _ = batch_1
            data_2, _, _, _, _ = batch_2
            
            if torch.cuda.is_available():
                data_0, label_0 = data_0.cuda(), label_0.cuda()
                data_1 = data_1.cuda()
                data_2 = data_2.cuda()
            
            if data_0.dim() == 2: data_0 = data_0.unsqueeze(1)
            if data_1.dim() == 2: data_1 = data_1.unsqueeze(1)
            if data_2.dim() == 2: data_2 = data_2.unsqueeze(1)

            self.optimizer.zero_grad()
            
            # =================================================================
            # 1. 有标签数据 (域 0)
            # =================================================================
            # 生成视图
            views_0 = self.get_augmented_views(data_0)
            outputs_0 = self.forward_backbone(views_0)
            
            loss_sup = torch.tensor(0.0).to(self.device)
            
            # 计算每个视图的加权损失
            # 视图 0 (原始): 权重 = 1.0
            # 视图 1+ (增强): 权重 = self.method_loss
            for idx, output in enumerate(outputs_0):
                current_loss = self.criterion(output, label_0)
                if idx == 0:
                    loss_sup += current_loss
                else:
                    loss_sup += current_loss * self.method_loss
            
            # 按视图数量归一化 (可选，但有利于稳定性)
            loss_sup = loss_sup / len(outputs_0)

            # =================================================================
            # 2. 无标签数据 (域 1 & 2)
            # =================================================================
            data_u = torch.cat([data_1, data_2], dim=0)
            views_u = self.get_augmented_views(data_u)
            outputs_u = self.forward_backbone(views_u)
            
            # 使用原始视图 (视图 0) 生成伪标签
            with torch.no_grad():
                probs_u = F.softmax(outputs_u[0], dim=1)
                max_probs, targets_u = torch.max(probs_u, dim=1)
                mask = max_probs.ge(self.conf_threshold).float()
            
            loss_unsup = torch.tensor(0.0).to(self.device)
            
            # 强制一致性：增强视图应匹配伪标签
            # 我们也可以在此处包含视图 0 (自训练)
            for idx, output in enumerate(outputs_u):
                # 与伪标签的交叉熵
                current_loss = (F.cross_entropy(output, targets_u, reduction='none') * mask).mean()
                
                if idx == 0:
                    loss_unsup += current_loss
                else:
                    loss_unsup += current_loss * self.method_loss
            
            loss_unsup = loss_unsup / len(outputs_u)
            
            # =================================================================
            # 总损失
            # =================================================================
            loss = loss_sup + loss_unsup
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss

    def model_inference(self, input):
        self.eval()
        with torch.no_grad():
            if isinstance(input, np.ndarray):
                input = torch.FloatTensor(input)
            
            if torch.cuda.is_available():
                self.cuda()
                input = input.cuda()
            
            if input.dim() == 2:
                input = input.unsqueeze(1)
            
            # 仅使用原始视图的标准推理
            feat = self.feature_extractor(input)
            outputs = self.classifier(feat)
            
            return outputs

    def get_parameters(self):
        return list(self.feature_extractor.parameters()) + list(self.classifier.parameters())
