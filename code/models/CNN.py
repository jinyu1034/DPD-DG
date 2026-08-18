# -*- coding: utf-8 -*-
"""
CNN Baseline: 基础卷积神经网络基线模型
不包含任何领域泛化策略的简单CNN模型

核心思想:
- 使用标准CNN架构进行特征提取
- 直接使用模型生成的伪标签作为无标签源域的真实标签
- 不使用任何领域对齐、分布匹配、对抗训练等策略
- 作为领域泛化方法的基线对比模型

关键特点:
1. 标准CNN架构: Conv1d + BatchNorm1d + ReLU + MaxPool
2. 简单伪标签策略: 
   - 对有标签数据使用真实标签训练
   - 对无标签数据使用模型预测作为伪标签
   - 不使用置信度筛选或加权
3. 标准交叉熵损失: 所有样本同等对待
4. 无领域适应: 不考虑源域差异

模型用途:
- 作为领域泛化方法的性能基线
- 验证领域泛化策略的有效性
- 对比分析不同方法的改进幅度
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as Data
import numpy as np


class CNN_FeatureExtractor(nn.Module):
    """标准CNN特征提取器"""
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


class CNN(nn.Module):
    """CNN基线模型 - 无领域泛化策略"""
    def __init__(self, in_channel, num_classes, lr, set, args):
        super(CNN, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.lr = lr
        self.set = set
        self.args = args
        
        # 特征提取器
        self.feature_extractor = CNN_FeatureExtractor(in_channel)
        
        # 分类器
        self.classifier = CNN_Classifier(input_dim=512, num_classes=num_classes)
        
        # 优化器
        self.optimizer = optim.Adam(
            list(self.feature_extractor.parameters()) + 
            list(self.classifier.parameters()),
            lr=1e-3,
            weight_decay=5e-4
        )
        
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, TR_dataloader, epoch_it):
        """
        训练一个epoch
        
        Args:
            TR_dataloader: 训练数据加载器列表，包含不同源域的Dataset
            epoch_it: 当前epoch编号
            
        Returns:
            float: 训练损失
        """
        self.train()
        
        # 创建数据加载器
        for domain, dataset in enumerate(TR_dataloader):
            if domain == 0:
                train_loader_0 = Data.DataLoader(dataset, batch_size=128, shuffle=True)
            elif domain == 1:
                train_loader_1 = Data.DataLoader(dataset, batch_size=128, shuffle=True)
            elif domain == 2:
                train_loader_2 = Data.DataLoader(dataset, batch_size=128, shuffle=True)
        
        total_loss = 0.0
        num_batches = 0
        
        # 训练循环
        for batch_0, batch_1, batch_2 in zip(train_loader_0, train_loader_1, train_loader_2):
            # 解包数据 - batch是(signal, label, domain, index, prior)的5元组
            data_0, label_0, domain_0, index_0, _ = batch_0
            data_1, label_1, domain_1, index_1, _ = batch_1
            data_2, label_2, domain_2, index_2, _ = batch_2
            
            # 强制将域1和域2的标签设为-1 (无标签)
            label_1[:] = -1
            label_2[:] = -1
            
            # 合并所有源域数据
            data = torch.cat([data_0, data_1, data_2], dim=0).float()
            label = torch.cat([label_0, label_1, label_2], dim=0).long()
            
            if torch.cuda.is_available():
                data = data.cuda()
                label = label.cuda()
            
            self.optimizer.zero_grad()
            
            # 前向传播
            features = self.feature_extractor(data)
            outputs = self.classifier(features)
            
            # 对于有标签数据，使用真实标签
            # 对于无标签数据，使用伪标签（模型预测）
            # 注意：这里假设label=-1表示无标签数据
            labeled_mask = (label != -1)
            
            if labeled_mask.any():
                # 有标签数据的损失
                loss_labeled = self.criterion(outputs[labeled_mask], label[labeled_mask])
            else:
                loss_labeled = 0.0
            
            # 无标签数据使用伪标签
            unlabeled_mask = (label == -1)
            if unlabeled_mask.any():
                # 获取无标签数据的输出
                unlabeled_outputs = outputs[unlabeled_mask]
                
                # 直接取最大值作为伪标签，不设阈值
                with torch.no_grad():
                    pseudo_labels = torch.argmax(unlabeled_outputs, dim=1)
                
                # 计算无标签损失
                loss_unlabeled = self.criterion(unlabeled_outputs, pseudo_labels)
                    
                loss = loss_labeled + loss_unlabeled
            else:
                loss = loss_labeled
            
            # 如果所有数据都有标签（实际场景），直接使用真实标签
            if not unlabeled_mask.any() and labeled_mask.all():
                loss = self.criterion(outputs, label)
            
            # 反向传播
            if isinstance(loss, torch.Tensor):
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            else:
                total_loss += float(loss)
            
            num_batches += 1
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def model_inference(self, input):
        """
        模型推理
        
        Args:
            input: 输入数据 [batch_size, seq_length] 或 [batch_size, channels, seq_length]
            
        Returns:
            torch.Tensor: 预测结果 [batch_size, num_classes]
        """
        self.eval()
        with torch.no_grad():
            if isinstance(input, np.ndarray):
                input = torch.FloatTensor(input)
            
            # 确保模型和输入在同一设备上
            if torch.cuda.is_available():
                self.cuda()
                input = input.cuda()
            
            # 特征提取
            features = self.feature_extractor(input)
            
            # 分类
            outputs = self.classifier(features)
            
            return outputs
    
    def get_parameters(self):
        """获取模型参数"""
        return list(self.feature_extractor.parameters()) + list(self.classifier.parameters())
