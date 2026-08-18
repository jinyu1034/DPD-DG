# -*- coding: utf-8 -*-
"""
CEG Model: Cluster-based Entropy-guided Generalization
基于聚类熵引导的泛化模型，适配半监督/单源域泛化实验设置

核心思想:
- 结合主动学习(Active Learning)和半监督学习(Semi-supervised Learning)
- 伪标签生成: 利用高置信度预测生成伪标签
- 一致性正则化: 强弱数据增强的一致性约束
- 混合增强(Mixup): 
  - 利用伪标签进行类内(Intra)和类间(Inter)混合
  - 增强决策边界的平滑性和类内紧凑性

适配说明:
- 参照 CNN.py 的基线架构进行修改
- 适配 forward 中包含训练循环的特殊设计
- 简化了原版 CEG 中的 K-Means 聚类过程，直接使用伪标签作为聚类/类别标识进行 Mixup
- 使用高斯噪声模拟强数据增强
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

class CEG(nn.Module):
    def __init__(self, in_channel, num_classes, lr, set, args):
        super(CEG, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.lr = lr
        self.set = set
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Components
        self.feature_extractor = CNN_FeatureExtractor(in_channel)
        self.classifier = CNN_Classifier(512, num_classes)
        
        self.optimizer = optim.Adam(
            list(self.feature_extractor.parameters()) + 
            list(self.classifier.parameters()),
            lr=1e-3,
            weight_decay=5e-4
        )
        
        self.criterion = nn.CrossEntropyLoss()
        
        # Hyperparameters
        self.conf_threshold = 0.95
        self.alpha_mix = 0.2
        self.w_cons = 1.0
        self.w_mix = 1.0

    def mixup_data(self, x, y, alpha=0.2):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(self.device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, pred, y_a, y_b, lam):
        return lam * self.criterion(pred, y_a) + (1 - lam) * self.criterion(pred, y_b)

    def forward(self, TR_dataloader, epoch_it):
        self.train()
        
        # Create DataLoaders
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
            # Unpack data
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
            # 1. Supervised Loss (Labeled Data)
            # =================================================================
            feat_0 = self.feature_extractor(data_0)
            logits_0 = self.classifier(feat_0)
            loss_sup = self.criterion(logits_0, label_0)
            
            # =================================================================
            # 2. Unlabeled Data Processing (Pseudo-labeling & Consistency)
            # =================================================================
            data_u = torch.cat([data_1, data_2], dim=0)
            
            # Weak Augmentation (Original) -> Pseudo Labels
            with torch.no_grad():
                feat_u = self.feature_extractor(data_u)
                logits_u = self.classifier(feat_u)
                probs_u = F.softmax(logits_u, dim=1)
                max_probs, targets_u = torch.max(probs_u, dim=1)
                mask = max_probs.ge(self.conf_threshold).float()
            
            # Strong Augmentation (Simulated with Noise) -> Consistency Loss
            noise = torch.randn_like(data_u) * 0.05 # Add 5% noise
            data_u_aug = data_u + noise
            
            feat_u_aug = self.feature_extractor(data_u_aug)
            logits_u_aug = self.classifier(feat_u_aug)
            
            loss_cons = (F.cross_entropy(logits_u_aug, targets_u, reduction='none') * mask).mean()
            
            # =================================================================
            # 3. Mixup Loss (Cluster-based / Pseudo-label based)
            # =================================================================
            loss_mix = torch.tensor(0.0).to(self.device)
            
            # Only perform mixup if we have enough high-confidence samples
            if mask.sum() > 2:
                idx_select = torch.nonzero(mask).squeeze()
                data_sel = data_u[idx_select]
                targets_sel = targets_u[idx_select]
                
                # Intra-Mixup & Inter-Mixup simplified:
                # We perform standard mixup on the selected high-confidence subset.
                # This implicitly covers both cases (mixing same class and different class)
                # and encourages linearity between samples.
                
                mixed_x, y_a, y_b, lam = self.mixup_data(data_sel, targets_sel, self.alpha_mix)
                
                feat_mix = self.feature_extractor(mixed_x)
                logits_mix = self.classifier(feat_mix)
                
                loss_mix = self.mixup_criterion(logits_mix, y_a, y_b, lam)
            
            # =================================================================
            # Total Loss
            # =================================================================
            loss = loss_sup + self.w_cons * loss_cons + self.w_mix * loss_mix
            
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
                
            feat = self.feature_extractor(input)
            outputs = self.classifier(feat)
            
            return outputs

    def get_parameters(self):
        return list(self.feature_extractor.parameters()) + list(self.classifier.parameters())
