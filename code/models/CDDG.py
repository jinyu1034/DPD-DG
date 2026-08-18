# -*- coding: utf-8 -*-
"""
CDDG Model: Causal Disentanglement Domain Generalization
基于因果解耦的域泛化模型，适配半监督/单源域泛化实验设置

核心思想:
- 双编码器架构: 
  - encoder_h: 提取因果特征(Causal Features)，与类别相关
  - encoder_m: 提取边缘特征(Marginal/Spurious Features)，与域相关
- 解码器重构: 保证特征的完整性
- 损失函数:
  - Classification Loss (CL): 保证分类准确性
  - Reconstruction Loss (RC): 保证信息完整性
  - Reduce Redundancy Loss (RR): 保证因果特征与边缘特征解耦
  - Causal Aggregation Loss (CA): 促进类内紧凑和域内紧凑

适配说明:
- 参照 CNN.py 的基线架构进行修改
- 适配 forward 中包含训练循环的特殊设计
- 适配半监督设置：对无标签数据使用伪标签计算 CL 和 CA 损失
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as Data
import numpy as np
import math

class CDDG_Encoder(nn.Module):
    """
    CDDG 专用编码器
    基于 CNN_FeatureExtractor 修改，返回特征图和特征向量
    """
    def __init__(self, in_channel=1):
        super(CDDG_Encoder, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=64, stride=1, padding=32), # padding to keep length roughly same before pool
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=16, stride=1, padding=8),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
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
        map_out = self.layer5(x) # (B, 256, 2)
        vec_out = map_out.view(map_out.size(0), -1) # (B, 512)
        return map_out, vec_out

class CDDG_Decoder(nn.Module):
    """
    CDDG 专用解码器
    尝试从 (fm_map, fh_map) 重构原始信号
    """
    def __init__(self, out_channel=1):
        super(CDDG_Decoder, self).__init__()
        # Input: (B, 512, 2) -> (B, 256 + 256, 2)
        
        self.layer5_inv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(512, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True)
        )
        self.layer4_inv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(128, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        self.layer3_inv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True)
        )
        self.layer2_inv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(32, 16, kernel_size=16, stride=1, padding=8),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True)
        )
        self.layer1_inv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose1d(16, out_channel, kernel_size=64, stride=1, padding=32),
            # No ReLU at the end for signal reconstruction usually, but depends on data normalization
        )

    def forward(self, x):
        x = self.layer5_inv(x)
        x = self.layer4_inv(x)
        x = self.layer3_inv(x)
        x = self.layer2_inv(x)
        x = self.layer1_inv(x)
        # Output shape might not be exactly 1024 due to padding/stride math, 
        # usually we interpolate to match input size if needed.
        return x

class CDDG_Classifier(nn.Module):
    """标准分类器"""
    def __init__(self, input_dim=512, num_classes=10):
        super(CDDG_Classifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CDDG(nn.Module):
    def __init__(self, in_channel, num_classes, lr, set, args):
        super(CDDG, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.lr = lr
        self.set = set
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Hyperparameters (can be moved to args)
        self.w_rc = 1.0
        self.w_rr = 1.0
        self.w_ca = 1.0
        
        # Modules
        self.encoder_m = CDDG_Encoder(in_channel) # Marginal/Spurious
        self.encoder_h = CDDG_Encoder(in_channel) # Causal/Semantic
        self.decoder = CDDG_Decoder(in_channel)
        self.classifier = CDDG_Classifier(input_dim=512, num_classes=num_classes)
        
        # Optimizer
        self.optimizer = optim.Adam(
            list(self.encoder_m.parameters()) + 
            list(self.encoder_h.parameters()) + 
            list(self.decoder.parameters()) + 
            list(self.classifier.parameters()),
            lr=1e-3,
            weight_decay=5e-4
        )

    def cal_reconstruction_loss(self, x, x_rec):
        # Resize x_rec to match x if necessary (due to conv arithmetic)
        if x_rec.shape[-1] != x.shape[-1]:
            x_rec = F.interpolate(x_rec, size=x.shape[-1], mode='linear', align_corners=False)
        return (x_rec-x).pow(2).mean()

    def cal_reduce_redundancy_loss(self, fm_vec, fh_vec):
        B = fm_vec.shape[0]
        D = fm_vec.shape[1]
        
        fm_vec = F.normalize(fm_vec, p=2, dim=0) #(B,D)
        fh_vec = F.normalize(fh_vec, p=2, dim=0) #(B,D)
        sim_fm_vec = torch.matmul(fm_vec.T, fm_vec) #(D,D)
        sim_fh_vec = torch.matmul(fh_vec.T, fh_vec) #(D,D)
        
        E = torch.eye(D).to(self.device)

        loss_fm =  ((1-E)*sim_fm_vec).pow(2).sum()/torch.sum(1-E)
        loss_fh =  ((1-E)*sim_fh_vec).pow(2).sum()/torch.sum(1-E)

        loss_fmh = torch.matmul(fh_vec.T, fm_vec).div(B).pow(2).mean()
        
        loss = loss_fm + loss_fh + loss_fmh
        return loss

    def cal_causal_aggregation_loss(self, fm_vec, fh_vec, labels, domain_labels):
        B = fm_vec.shape[0]
        D = fm_vec.shape[1]

        fm_vec = F.normalize(fm_vec, p=2, dim=1) # (B,D)
        fh_vec = F.normalize(fh_vec, p=2, dim=1) # (B,D)

        labels= labels.contiguous().view(-1, 1)
        mask_fh = torch.eq(labels, labels.T).float().to(self.device) # (B,B)
        sim_fh_vec = torch.matmul(fh_vec, fh_vec.T)/D # (B,B)
        
        # Avoid division by zero if mask sum is 0
        sum_mask_fh = torch.sum(mask_fh)
        sum_inv_mask_fh = torch.sum(1-mask_fh)
        
        term1_fh = -(mask_fh*sim_fh_vec).sum()/sum_mask_fh if sum_mask_fh > 0 else torch.tensor(0.0).to(self.device)
        term2_fh = ((1-mask_fh)*sim_fh_vec).sum()/sum_inv_mask_fh if sum_inv_mask_fh > 0 else torch.tensor(0.0).to(self.device)
        loss_fh = term1_fh + term2_fh

        domain_labels= domain_labels.contiguous().view(-1, 1)
        mask_fm = torch.eq(domain_labels, domain_labels.T).float().to(self.device) # (B,B)
        sim_fm_vec = torch.matmul(fm_vec, fm_vec.T)/D # (B,B)
        
        sum_mask_fm = torch.sum(mask_fm)
        sum_inv_mask_fm = torch.sum(1-mask_fm)
        
        term1_fm = -(mask_fm*sim_fm_vec).sum()/sum_mask_fm if sum_mask_fm > 0 else torch.tensor(0.0).to(self.device)
        term2_fm = ((1-mask_fm)*sim_fm_vec).sum()/sum_inv_mask_fm if sum_inv_mask_fm > 0 else torch.tensor(0.0).to(self.device)
        loss_fm = term1_fm + term2_fm

        loss = loss_fm + loss_fh
        return loss

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
        
        # 训练循环
        # 注意：CDDG计算相关性矩阵需要较大的Batch Size，这里如果Batch太小可能不稳定
        # CNN.py 使用 128，这里为了计算 loss_rr 和 loss_ca，建议保持较大 batch
        
        for batch_0, batch_1, batch_2 in zip(train_loader_0, train_loader_1, train_loader_2):
            # 解包数据
            data_0, label_0, domain_0, index_0, _ = batch_0
            data_1, label_1, domain_1, index_1, _ = batch_1
            data_2, label_2, domain_2, index_2, _ = batch_2
            
            # 标记无标签数据
            label_1[:] = -1
            label_2[:] = -1
            
            # 合并数据
            data = torch.cat([data_0, data_1, data_2], dim=0).float()
            labels_all = torch.cat([label_0, label_1, label_2], dim=0).long()
            
            # 构造域标签 (0, 1, 2)
            domain_labels = torch.cat([
                torch.zeros(len(label_0)),
                torch.ones(len(label_1)),
                torch.ones(len(label_2)) * 2
            ], dim=0).long()

            if torch.cuda.is_available():
                data = data.cuda()
                labels_all = labels_all.cuda()
                domain_labels = domain_labels.cuda()
            
            # 确保输入维度正确 (B, 1, L)
            if data.dim() == 2:
                data = data.unsqueeze(1)

            self.optimizer.zero_grad()
            
            # 1. Forward Pass
            fm_map, fm_vec = self.encoder_m(data)
            fh_map, fh_vec = self.encoder_h(data)
            
            # Concatenate maps for decoder: (B, 256, 2) + (B, 256, 2) -> (B, 512, 2)
            fmh_map = torch.cat([fm_map, fh_map], dim=1)
            x_rec = self.decoder(fmh_map)
            
            logits = self.classifier(fh_vec)
            
            # 2. Pseudo-labeling for Unlabeled Data
            labeled_mask = (labels_all != -1)
            unlabeled_mask = (labels_all == -1)
            
            # 填充用于 loss_ca 的完整标签向量
            # 对于无标签数据，使用预测的伪标签
            current_labels = labels_all.clone()
            
            if unlabeled_mask.any():
                with torch.no_grad():
                    pseudo_logits = logits[unlabeled_mask]
                    pseudo_labels = torch.argmax(pseudo_logits, dim=1)
                    current_labels[unlabeled_mask] = pseudo_labels
            
            # 3. Calculate Losses
            
            # Loss RC: Reconstruction (All data)
            loss_rc = self.cal_reconstruction_loss(data, x_rec)
            
            # Loss RR: Reduce Redundancy (All data)
            loss_rr = self.cal_reduce_redundancy_loss(fm_vec, fh_vec)
            
            # Loss CA: Causal Aggregation (All data, using pseudo labels)
            loss_ca = self.cal_causal_aggregation_loss(fm_vec, fh_vec, current_labels, domain_labels)
            
            # Loss CL: Classification
            # Labeled data: Cross Entropy with true labels
            if labeled_mask.any():
                loss_cl_labeled = F.cross_entropy(logits[labeled_mask], labels_all[labeled_mask])
            else:
                loss_cl_labeled = 0.0
                
            # Unlabeled data: Cross Entropy with pseudo labels
            if unlabeled_mask.any():
                loss_cl_unlabeled = F.cross_entropy(logits[unlabeled_mask], current_labels[unlabeled_mask])
            else:
                loss_cl_unlabeled = 0.0
                
            loss_cl = loss_cl_labeled + loss_cl_unlabeled
            
            # Total Loss
            loss = self.w_rc * loss_rc + self.w_rr * loss_rr + self.w_ca * loss_ca + loss_cl
            
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
                
            _, fh_vec = self.encoder_h(input)
            outputs = self.classifier(fh_vec)
            
            return outputs

    def get_parameters(self):
        return list(self.encoder_m.parameters()) + \
               list(self.encoder_h.parameters()) + \
               list(self.decoder.parameters()) + \
               list(self.classifier.parameters())
