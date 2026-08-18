# -*- coding: utf-8 -*-
"""
MSDGN Model: Multi-Source Domain Generalization Network
基于多源域泛化网络的故障诊断模型，适配半监督/单源域泛化实验设置

核心思想:
- 三分支架构:
  - Net1 & Net2: 辅助网络，用于协同训练和生成伪标签
  - Net3 (Main): 主网络，采用CSD (Common-Specific Decomposition) 结构
- 训练策略:
  - 利用 Net1 和 Net2 对无标签数据进行预测，基于熵选择高置信度样本
  - 使用 MMD 对齐不同分支的特征分布
  - Net3 利用 CSD 结构同时学习领域无关(Common)和领域特定(Specific)特征
  - 结合有标签数据和筛选后的伪标签数据训练主网络

适配说明:
- 参照 CNN.py 的基线架构进行修改，使用相同的特征提取器
- 适配 forward 中包含训练循环的特殊设计
- 适配半监督设置：将单源域作为有标签域(Domain 0)，其他作为无标签域(Domain 1, 2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as Data
import numpy as np
import math
from . import mmd

# =============================================================================
# Utils
# =============================================================================

def EntropyLoss(input_):
    mask = input_.ge(0.000001)
    mask_out = torch.masked_select(input_, mask)
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))
    return entropy / float(input_.size(0))

def one_hot(ids, depth, device):
    z = torch.zeros(len(ids), depth).to(device)
    z.scatter_(1, ids.unsqueeze(1), 1)
    return z

# =============================================================================
# Components
# =============================================================================

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

class CSD_Classifier(nn.Module):
    """
    Common-Specific Decomposition (CSD) 分类器
    用于 Net3，实现领域无关和领域特定特征的解耦
    """
    def __init__(self, input_dim=512, num_classes=10, num_domains=3, K=2):
        super(CSD_Classifier, self).__init__()
        self.num_classes = num_classes
        self.K = K
        self.input_dim = input_dim
        
        # 特定模块 (sms): K 组权重 [K, input_dim, num_classes]
        self.sms = nn.Parameter(torch.normal(0, 1e-1, size=[K, input_dim, num_classes]), requires_grad=True)
        self.sm_biases = nn.Parameter(torch.normal(0, 1e-1, size=[K, num_classes]), requires_grad=True)
        
        # 域嵌入，用于选择特定模块 [num_domains, K-1]
        self.embs = nn.Parameter(torch.normal(mean=0., std=1e-4, size=[num_domains, K - 1]), requires_grad=True)
        self.cs_wt = nn.Parameter(torch.normal(mean=.1, std=1e-4, size=[]), requires_grad=True)

    def forward(self, x, uids):
        """
        参数:
            x: 特征 (B, input_dim)
            uids: 独热编码域 ID (B, num_domains)
        """
        # 通用 logits (使用第一个分量作为通用分量)
        w_c, b_c = self.sms[0, :, :], self.sm_biases[0, :]
        logits_common = torch.matmul(x, w_c) + b_c
        
        # 特定 logits
        # 基于域嵌入计算每个 K 分量的权重
        c_wts = torch.matmul(uids, self.embs) # (B, K-1)
        
        # 为第一个分量添加通用权重 (cs_wt)
        batch_size = uids.shape[0]
        c_wts = torch.cat((torch.ones((batch_size, 1), device=x.device) * self.cs_wt, c_wts), 1)
        c_wts = torch.tanh(c_wts) # (B, K)
        
        # 基于 c_wts 组合 sms
        # w_d: (B, input_dim, num_classes)
        w_d = torch.einsum("bk,krl->brl", c_wts, self.sms)
        b_d = torch.einsum("bk,kl->bl", c_wts, self.sm_biases)
        
        logits_specialized = torch.einsum("brl,br->bl", w_d, x) + b_d
        
        return logits_specialized, logits_common

# =============================================================================
# MSDGN Model
# =============================================================================

class MSDGN(nn.Module):
    def __init__(self, in_channel, num_classes, lr, set, args):
        super(MSDGN, self).__init__()
        self.in_channel = in_channel
        self.num_classes = num_classes
        self.lr = lr
        self.set = set
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 网络 1 (辅助)
        self.net1_feat = CNN_FeatureExtractor(in_channel)
        self.net1_cls = CNN_Classifier(512, num_classes)
        
        # 网络 2 (辅助)
        self.net2_feat = CNN_FeatureExtractor(in_channel)
        self.net2_cls = CNN_Classifier(512, num_classes)
        
        # 网络 3 (主网络 - CSD)
        self.net3_feat = CNN_FeatureExtractor(in_channel)
        # 假设 3 个域：0 (有标签), 1 (无标签), 2 (无标签)
        self.net3_cls = CSD_Classifier(512, num_classes, num_domains=3) 
        
        self.optimizer = optim.Adam(
            list(self.net1_feat.parameters()) + list(self.net1_cls.parameters()) +
            list(self.net2_feat.parameters()) + list(self.net2_cls.parameters()) +
            list(self.net3_feat.parameters()) + list(self.net3_cls.parameters()),
            lr=1e-3,
            weight_decay=5e-4
        )
        
        self.criterion = nn.CrossEntropyLoss()

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
        
        # Hyperparameters
        ratio = 0.5 # Pseudo-label selection ratio
        lambd = 0.5 # Weight for final classification loss
        
        for batch_0, batch_1, batch_2 in zip(train_loader_0, train_loader_1, train_loader_2):
            # Unpack data
            data_0, label_0, domain_0, index_0, _ = batch_0
            data_1, label_1, domain_1, index_1, _ = batch_1
            data_2, label_2, domain_2, index_2, _ = batch_2
            
            # Mark unlabeled data
            label_1[:] = -1
            label_2[:] = -1
            
            if torch.cuda.is_available():
                data_0, label_0 = data_0.cuda(), label_0.cuda()
                data_1 = data_1.cuda()
                data_2 = data_2.cuda()
            
            # 确保输入维度
            if data_0.dim() == 2: data_0 = data_0.unsqueeze(1)
            if data_1.dim() == 2: data_1 = data_1.unsqueeze(1)
            if data_2.dim() == 2: data_2 = data_2.unsqueeze(1)

            self.optimizer.zero_grad()
            
            # =================================================================
            # 1. 训练辅助网络 (Net1, Net2) & MMD
            # =================================================================
            
            # 前向传播有标签数据 (域 0)
            feat1_0 = self.net1_feat(data_0)
            pred1_0 = self.net1_cls(feat1_0)
            
            feat2_0 = self.net2_feat(data_0)
            pred2_0 = self.net2_cls(feat2_0)
            
            # 前向传播无标签数据 (域 1 & 2)
            feat1_1 = self.net1_feat(data_1)
            pred1_1 = self.net1_cls(feat1_1)
            
            feat2_2 = self.net2_feat(data_2)
            pred2_2 = self.net2_cls(feat2_2)
            
            # 分类损失 (有标签)
            loss_cls_aux = self.criterion(pred1_0, label_0) + self.criterion(pred2_0, label_0)
            
            # MMD 损失 (对齐有标签与无标签特征)
            # 对齐 Net1(域0) <-> Net1(域1) 和 Net2(域0) <-> Net2(域2)
            loss_mmd = mmd.mmd_rbf_noaccelerate(feat1_0, feat1_1) + mmd.mmd_rbf_noaccelerate(feat2_0, feat2_2)
            
            # 熵损失 (无标签)
            loss_H = EntropyLoss(F.softmax(pred1_1, dim=1)) + EntropyLoss(F.softmax(pred2_2, dim=1))
            
            # =================================================================
            # 2. 伪标签选择
            # =================================================================
            
            # 使用 Net 1 从域 1 选择样本
            prob1_1 = F.softmax(pred1_1, dim=1)
            entropy1_1 = -torch.sum(prob1_1 * torch.log(prob1_1 + 1e-6), dim=1)
            # 选择熵最低的前 % 样本
            num_select = int(len(data_1) * ratio)
            _, idx1 = torch.sort(entropy1_1)
            idx1_select = idx1[:num_select]
            
            data_1_sel = data_1[idx1_select]
            pseudo_label_1 = torch.argmax(prob1_1[idx1_select], dim=1)
            
            # 使用 Net 2 从域 2 选择样本
            prob2_2 = F.softmax(pred2_2, dim=1)
            entropy2_2 = -torch.sum(prob2_2 * torch.log(prob2_2 + 1e-6), dim=1)
            _, idx2 = torch.sort(entropy2_2)
            idx2_select = idx2[:num_select]
            
            data_2_sel = data_2[idx2_select]
            pseudo_label_2 = torch.argmax(prob2_2[idx2_select], dim=1)
            
            # =================================================================
            # 3. 训练主网络 (Net3 - CSD)
            # =================================================================
            
            # 为 Net3 准备数据
            # 结合有标签数据 + 选定的伪标签数据
            data_all = torch.cat([data_0, data_1_sel, data_2_sel], dim=0)
            label_all = torch.cat([label_0, pseudo_label_1, pseudo_label_2], dim=0)
            
            # CSD 的域 ID
            # 0 代表有标签, 1 代表无标签1, 2 代表无标签2
            uids_0 = torch.zeros(len(data_0), dtype=torch.long).to(self.device)
            uids_1 = torch.ones(len(data_1_sel), dtype=torch.long).to(self.device)
            uids_2 = torch.ones(len(data_2_sel), dtype=torch.long).to(self.device) * 2
            uids_all = torch.cat([uids_0, uids_1, uids_2], dim=0)
            uids_onehot = one_hot(uids_all, 3, self.device)
            
            feat3 = self.net3_feat(data_all)
            logits_spec, logits_comm = self.net3_cls(feat3, uids_onehot)
            
            # 分类损失 (主网络)
            loss_cls_final = self.criterion(logits_spec, label_all)
            
            # CSD 的正交损失
            sms = self.net3_cls.sms
            K = self.net3_cls.K
            diag_tensor = torch.stack([torch.eye(K) for _ in range(self.num_classes)], dim=0).to(self.device)
            cps = torch.stack(
                [torch.matmul(sms[:, :, i], torch.transpose(sms[:, :, i], 0, 1)) for i in range(self.num_classes)], dim=0)
            loss_orth = torch.mean((cps - diag_tensor) ** 2)
            
            # =================================================================
            # 总损失
            # =================================================================
            
            loss = loss_cls_aux + loss_mmd + loss_H + lambd * loss_cls_final + loss_orth
            
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
            
            # 使用 Net3 (主网络) 进行推理
            # 对于推理，我们可能不知道域 ID，或者我们想要“通用”预测？
            # 或者我们需要一个虚拟 ID？
            # 参考代码在目标测试中使用 dummy_ids = one_hot(zeros, 3)。
            # 这意味着使用域 0 (源域) 的特定预测器，或者仅仅依赖于通用性？
            # 实际上，CSD 设计上包含一个通用部分。
            # 让我们使用虚拟 ID 0 (类似源域) 或者平均值？
            # 参考: dummy_ids = one_hot(np.zeros(...), 3)
            
            dummy_uids = torch.zeros(input.size(0), dtype=torch.long).to(self.device)
            uids_onehot = one_hot(dummy_uids, 3, self.device)
            
            feat3 = self.net3_feat(input)
            logits_spec, logits_comm = self.net3_cls(feat3, uids_onehot)
            
            # 返回特定 logits (根据参考 test_target)
            return logits_spec

    def get_parameters(self):
        return list(self.net1_feat.parameters()) + list(self.net1_cls.parameters()) + \
               list(self.net2_feat.parameters()) + list(self.net2_cls.parameters()) + \
               list(self.net3_feat.parameters()) + list(self.net3_cls.parameters())
