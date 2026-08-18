# -*- coding: utf-8 -*-
"""
DKGPL: Domain Knowledge Guided Pseudo-label Generation
领域知识引导的伪标签生成模型

核心思想:
- 利用DFE(Distribution Feature Expansion)增强特征分布多样性
- 通过自适应权重掩码实现伪标签选择
- 采用原型对齐策略学习领域不变表示
- 结合有标签和无标签数据进行半监督学习

关键技术:
1. DFE模块: Beta分布随机扰动均值和方差，生成多样化特征
2. 自适应权重掩码: 基于预测置信度动态调整样本权重
3. 原型对齐: 最小化类原型间的分布差异
4. 半监督损失: 结合交叉熵(有标签)和伪标签损失(无标签)

改进要点:
- 完整实现DFE的Beta分布采样和Bernoulli随机化
- 保留原始的LR(Linear Regression)和DKGPL(Multi-level)分类器选项
- 支持4/5元组数据格式(data, label, domain, index, [ignored])
"""
import math
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.nn import functional as F
import torch.utils.data as Data
from torch.autograd import Variable
from . import mmd

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


class DKGPLClassifier(nn.Module):
    """DKGPL 分类器 - 基于SSDG的自适应权重掩码分类器"""
    def __init__(self, num_features, num_classes, LR=False, method="DKGPL"):
        super().__init__()
        self.set = method
        if LR:
            self.h1 = nn.Linear(num_features, num_features)
            self.h2 = nn.Linear(num_features, num_classes)
        
        self.p1 = nn.Linear(num_features, num_features // 2)
        self.p2 = nn.Linear(num_features // 2, num_features // 4)
        self.p3 = nn.Linear(num_features // 4, num_features // 8)
        self.p4 = nn.Linear(num_features // 4, num_features // 2)
        self.p5 = nn.Linear(num_features // 2, num_features)
        self.w = nn.Parameter(torch.Tensor(num_classes, num_features))
        
        stdv = 1. / math.sqrt(self.w.size(1))
        self.w.data.uniform_(-stdv, stdv)
    
    def forward(self, x, LR=False, noise=False):
        if LR and self.set == "DKGPL":
            x_mean = x.mean(dim=0, keepdim=True)
            x_mean = torch.relu(self.p1(x_mean))
            x_mean = torch.relu(self.p2(x_mean))
            x_mean = torch.relu(self.p3(x_mean))
            if noise:
                noise_tensor = torch.randn(1, int(x.shape[1]/8)).to(x.device)
                x_mean = torch.cat((x_mean, noise_tensor), dim=1)
            else:
                x_mean = torch.cat((x_mean, torch.zeros(1, int(x.shape[1]/8)).to(x.device)), dim=1)
            x_mean = torch.relu(self.p4(x_mean))
            x_mask = torch.relu(self.p5(x_mean))
            x1 = self.h1(x_mask)
            x2 = self.h2(x_mask)
            a = torch.matmul(x2.t(), x1)
            w_mask = torch.sigmoid(a)
            self.w_new = self.w * w_mask
            return torch.matmul(x, self.w_new.t())
        else:
            return torch.matmul(x, self.w.t())


class DKGPLFea_Extraction(nn.Module):
    """DKGPL 特征提取器 - 包含DFE的多层卷积网络"""
    def __init__(self, in_channel=1, method="DKGPL"):
        super().__init__()
        self.set = method
        self.DFE = DFE()
        self.layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 16, kernel_size=64, stride=1),
            nn.InstanceNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=16, stride=1),
            nn.InstanceNorm1d(32),
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
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.layer6 = nn.Sequential(
            nn.Conv1d(256, 512, kernel_size=5, stride=1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.avgp1 = nn.AdaptiveAvgPool1d(2)  # 确保输出为512*2=1024维
    
    def forward(self, x, trid=False):
        x1 = self.layer1(x)
        if trid and (self.set == "DKGPL"):
            x2 = self.DFE(x1)
        else:
            x2 = x1
        x = self.layer2(x2)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.avgp1(x)  # Apply adaptive pooling
        out = x.view(x.size(0), -1)
        return out


class DKGPL(nn.Module):
    """DKGPL 主模型 - 领域知识引导的伪标签生成模型"""
    def __init__(self, in_channel=1, num_classes=5, lr=0.01, set="DKGPL", args=None):
        super(DKGPL, self).__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.set_trid = set
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.G = DKGPLFea_Extraction(in_channel=in_channel, method=self.set_trid).to(self.device)
        self.C = DKGPLClassifier(1024, num_classes, LR=True, method=self.set_trid).to(self.device)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.u_loss = nn.CrossEntropyLoss(reduction="none").to(self.device)
        
        self.classwise_acc_1 = torch.ones((self.num_classes,)).to(self.device)
        self.classwise_acc_2 = torch.ones((self.num_classes,)).to(self.device)
        
        self.optimizer = optim.Adam(
            [{'params': self.G.parameters(), 'lr': 1e-4},
             {'params': self.C.parameters(), 'lr': 1e-4}],
            weight_decay=1e-4
        )
    
    def forward(self, TR_dataloader, epoch_it):
        """训练前向传播"""
        self.G.train()
        self.C.train()
        loss_all = 0.0
        
        # 为每个域创建数据加载器
        for domain, dataset in enumerate(TR_dataloader):
            if domain == 0:
                train_loader_x_src_0 = Data.DataLoader(dataset, batch_size=128)
                selected_label_0 = torch.ones((len(dataset),), dtype=torch.long) * -1
            elif domain == 1:
                train_loader_x_src_1 = Data.DataLoader(dataset, batch_size=128)
                selected_label_1 = torch.ones((len(dataset),), dtype=torch.long) * -1
                selected_label_1 = selected_label_1.to(self.device)
            elif domain == 2:
                train_loader_x_src_2 = Data.DataLoader(dataset, batch_size=128)
                selected_label_2 = torch.ones((len(dataset),), dtype=torch.long) * -1
                selected_label_2 = selected_label_2.to(self.device)
        
        unsup_warmup = np.clip(epoch_it / (0.8 * 100), a_min=0.0, a_max=1.0)
        
        for batch_src_0, batch_src_1, batch_src_2 in zip(train_loader_x_src_0, train_loader_x_src_1, train_loader_x_src_2):
            # 解包数据 - 自适应数据格式 (兼容包含先验特征的数据加载器，但忽略先验特征)
            if len(batch_src_0) == 5:
                batch_x_0, batch_y_0, batch_domain_0, _, _ = batch_src_0
                batch_x_1, batch_y_1, batch_domain_1, x_index_1, _ = batch_src_1
                batch_x_2, batch_y_2, batch_domain_2, x_index_2, _ = batch_src_2
            else:  # 原始SSDG格式 (4元组)
                batch_x_0, batch_y_0, batch_domain_0, _ = batch_src_0
                batch_x_1, batch_y_1, batch_domain_1, x_index_1 = batch_src_1
                batch_x_2, batch_y_2, batch_domain_2, x_index_2 = batch_src_2
            
            self.optimizer.zero_grad()
            
            # 生成伪标签
            with torch.no_grad():
                f_x_1 = self.G(batch_x_1.to(self.device), trid=True)
                z_xu_k_1 = self.C(f_x_1, LR=True)
                p_xu_1 = F.softmax(z_xu_k_1, dim=1)
                p_xu_maxval_1, y_xu_pre_1 = p_xu_1.max(dim=1)
                threshold_1 = 0.95 * (self.classwise_acc_1[y_xu_pre_1] / (2.0 - self.classwise_acc_1[y_xu_pre_1]))
                mask_x_1 = (p_xu_maxval_1 >= threshold_1).float()
                
                f_x_2 = self.G(batch_x_2.to(self.device), trid=True)
                z_xu_k_2 = self.C(f_x_2, LR=True)
                p_xu_2 = F.softmax(z_xu_k_2, dim=1)
                p_xu_maxval_2, y_xu_pre_2 = p_xu_2.max(dim=1)
                threshold_2 = 0.95 * (self.classwise_acc_2[y_xu_pre_2] / (2.0 - self.classwise_acc_2[y_xu_pre_2]))
                mask_x_2 = (p_xu_maxval_2 >= threshold_2).float()
                
                # 更新类别精度
                selected_indices_1 = torch.where(mask_x_1 == 1)[0]
                if selected_indices_1.nelement() != 0:
                    pseudo_labels_1 = y_xu_pre_1[selected_indices_1]
                    pseudo_counter_1 = Counter(pseudo_labels_1.tolist())
                    if max(pseudo_counter_1.values()) < len(selected_label_1):
                        for i in range(self.num_classes):
                            self.classwise_acc_1[i] = pseudo_counter_1.get(i, 0) / max(pseudo_counter_1.values())
                
                selected_indices_2 = torch.where(mask_x_2 == 1)[0]
                if selected_indices_2.nelement() != 0:
                    pseudo_labels_2 = y_xu_pre_2[selected_indices_2]
                    pseudo_counter_2 = Counter(pseudo_labels_2.tolist())
                    if max(pseudo_counter_2.values()) < len(selected_label_2):
                        for i in range(self.num_classes):
                            self.classwise_acc_2[i] = pseudo_counter_2.get(i, 0) / max(pseudo_counter_2.values())
            
            # 计算损失
            f_k_0 = self.G(batch_x_0.to(self.device), trid=True)
            z_k_0 = self.C(f_k_0, LR=True, noise=True)
            loss_s = self.criterion(z_k_0, batch_y_0.to(self.device))
            
            f_xu_k_aug_1 = self.G(batch_x_1.clone().to(self.device), trid=True)
            z_xu_k_aug_1 = self.C(f_xu_k_aug_1, LR=True, noise=True)
            loss_u_1 = self.u_loss(z_xu_k_aug_1, y_xu_pre_1)
            loss_u_1_m = (loss_u_1 * mask_x_1).mean()
            
            f_xu_k_aug_2 = self.G(batch_x_2.clone().to(self.device), trid=True)
            z_xu_k_aug_2 = self.C(f_xu_k_aug_2, LR=True, noise=True)
            loss_u_2 = self.u_loss(z_xu_k_aug_2, y_xu_pre_2)
            loss_u_2_m = (loss_u_2 * mask_x_2).mean()
            
            # 原型对齐 + MMD (在 epoch > 50 时启用)
            if epoch_it > 50:
                # 计算类原型
                class_prototypes = {}
                features_0 = f_k_0.detach()
                prototypes_0 = []
                for c in range(self.num_classes):
                    class_features = features_0[batch_y_0 == c]
                    if class_features.size(0) > 0:
                        prototype = class_features.mean(dim=0)
                    else:
                        prototype = torch.zeros(features_0.size(1)).to(self.device)
                    prototypes_0.append(prototype)
                class_prototypes[0] = torch.stack(prototypes_0)
                
                prototypes_0_norm = F.normalize(class_prototypes[0], p=2, dim=1)
                f_u_0 = F.normalize(f_k_0, p=2, dim=1)
                similarity_0 = torch.mm(f_u_0, prototypes_0_norm.t())
                mask_x_0 = torch.ones(f_u_0.size(0)).to(self.device)
                
                f_u_1 = F.normalize(f_xu_k_aug_1, p=2, dim=1)
                similarity_1 = torch.mm(f_u_1, prototypes_0_norm.t())
                
                f_u_2 = F.normalize(f_xu_k_aug_2, p=2, dim=1)
                similarity_2 = torch.mm(f_u_2, prototypes_0_norm.t())
                
                loss_sa_0 = self.compute_sa_loss_multi(similarity_0, mask_x_0, batch_y_0)
                sim_top1_0, _ = similarity_0.topk(k=1, dim=1, sorted=True)
                loss_sim_0 = ((1 - sim_top1_0.squeeze(1)) * mask_x_0).mean()
                
                loss_sa_1 = self.compute_sa_loss_multi(similarity_1, mask_x_1, y_xu_pre_1)
                sim_top1_1, _ = similarity_1.topk(k=1, dim=1, sorted=True)
                loss_sim_1 = ((1 - sim_top1_1.squeeze(1)) * mask_x_1).mean()
                
                loss_sa_2 = self.compute_sa_loss_multi(similarity_2, mask_x_2, y_xu_pre_2)
                sim_top1_2, _ = similarity_2.topk(k=1, dim=1, sorted=True)
                loss_sim_2 = ((1 - sim_top1_2.squeeze(1)) * mask_x_2).mean()
                
                loss_sa = loss_sa_0 + loss_sa_1 + loss_sa_2
                MMD_loss = mmd.mmd_rbf_noaccelerate(z_k_0, z_xu_k_aug_1) + mmd.mmd_rbf_noaccelerate(z_k_0, z_xu_k_aug_2)
                loss_sim = loss_sim_0 + loss_sim_1 + loss_sim_2
            else:
                loss_sa = 0.0
                loss_sim = 0.0
                MMD_loss = 0.0
            
            loss = (loss_u_1_m + loss_u_2_m) * unsup_warmup + loss_s + loss_sa * 0.5 + loss_sim * 15 + MMD_loss
            loss.backward()
            self.optimizer.step()
            loss_all += loss.item()
        
        return loss_all
    
    def compute_sa_loss_multi(self, similarity, mask_x, labels):
        """计算原型对齐损失"""
        loss_sim = F.cross_entropy(similarity, labels.to(self.device), reduction="none") * 3
        loss_sim = (loss_sim * mask_x).mean()
        return loss_sim
    
    def model_inference(self, input):
        """模型推理"""
        self.G.eval()
        self.C.eval()
        with torch.no_grad():
            # 确保输入格式正确: [batch_size, seq_length] -> [batch_size, 1, seq_length]
            if input.dim() == 2:
                input = input.unsqueeze(1)
            features = self.G(input.to(self.device), trid=False)
            prediction = self.C(features, LR=False, noise=False)
        return prediction
