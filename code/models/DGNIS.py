# -*- coding: utf-8 -*-
"""
DGNIS: Domain Generalization with Nuisance Invariant Subspace
领域泛化与不相关不变子空间学习模型 (单源域泛化改进版)

核心思想:
- 学习领域不变特征(feature1)用于分类
- 学习领域特定特征(feature2)用于领域识别
- 通过CORAL对齐领域分布
- 在单源域设置下，利用伪标签技术挖掘无标签源域信息

改进要点 (针对单源域泛化):
1. 仅使用源域0的真实标签进行有监督训练
2. 对源域1和源域2使用伪标签策略 (基于置信度筛选)
3. 域判别器仍然使用所有源域的域标签 (0, 1, 2)
4. CORAL损失用于对齐有标签源域和无标签源域
5. 三元组损失结合真实标签(源域0)和高置信度伪标签(源域1,2)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as Data
import numpy as np
import math


def pdist_torch(emb1, emb2):
    '''
    compute the eucilidean distance matrix between embeddings1 and embeddings2
    using gpu
    '''
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim = 1, keepdim = True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim = 1, keepdim = True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm(emb1, emb2.t(), beta=1, alpha=-2)
    dist_mtx = dist_mtx.clamp(min = 1e-12).sqrt()
    return dist_mtx


class BatchHardTripletSelector(object):
    '''
    a selector to generate hard batch embeddings from the embedded batch
    '''
    def __init__(self, *args, **kwargs):
        super(BatchHardTripletSelector, self).__init__()

    def __call__(self, embeds, labels):
        dist_mtx = pdist_torch(embeds, embeds).detach().cpu().numpy()# 计算距离
        labels = labels.contiguous().cpu().numpy().reshape((-1, 1))
        num = labels.shape[0]
        dia_inds = np.diag_indices(num)#返回对角线索引
        lb_eqs = labels == labels.T
        lb_eqs[dia_inds] = False
        dist_same = dist_mtx.copy()
        dist_same[lb_eqs == False] = -np.inf
        pos_idxs = np.argmax(dist_same, axis = 1)
        dist_diff = dist_mtx.copy()
        lb_eqs[dia_inds] = True
        dist_diff[lb_eqs == True] = np.inf
        neg_idxs = np.argmin(dist_diff, axis = 1)
        pos = embeds[pos_idxs].contiguous().view(num, -1)
        neg = embeds[neg_idxs].contiguous().view(num, -1)
        return embeds, pos, neg


class TripletLoss(nn.Module):
    '''
    Compute normal triplet loss or soft margin triplet loss given triplets
    '''
    def __init__(self, margin = None):
        super(TripletLoss, self).__init__()
        self.margin = margin
        if self.margin is None:  # use soft-margin
            self.Loss = nn.SoftMarginLoss()
        else:
            self.Loss = nn.TripletMarginLoss(margin = margin, p = 2)

    def forward(self, anchor, pos, neg):
        if self.margin is None:
            num_samples = anchor.shape[0]
            y = torch.ones((num_samples, 1)).view(-1)
            if anchor.is_cuda: y = y.cuda()
            ap_dist = torch.norm(anchor - pos, 2, dim = 1).view(-1)
            an_dist = torch.norm(anchor - neg, 2, dim = 1).view(-1)
            loss = self.Loss(an_dist - ap_dist, y)
        else:
            loss = self.Loss(anchor, pos, neg)

        return loss


class CNN1D_Feature(nn.Module):
    """DGNIS 1D CNN 特征提取器 (升级版 - 对齐CNN基线)"""
    def __init__(self, in_channel=1):
        super(CNN1D_Feature, self).__init__()
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


class DGNIS(nn.Module):
    """DGNIS 主模型 - 单源域泛化适配版"""
    def __init__(self, in_channel=1, num_classes=5, lr=0.001, set="DGNIS", args=None):
        super(DGNIS, self).__init__()
        self.lr = lr
        self.num_classes = num_classes
        self.set_trid = set
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.l2_decay = 5e-4
        
        # 两个共享特征提取器
        self.sharedNet1 = CNN1D_Feature(in_channel=in_channel).to(self.device) # 用于分类 (不变特征)
        self.sharedNet2 = CNN1D_Feature(in_channel=in_channel).to(self.device) # 用于域预测 (特定特征)
        
        # 每个域一个分类器 (输入512维 - 对应升级后的Backbone)
        self.cls_fc_1 = nn.Linear(512, num_classes).to(self.device)
        self.cls_fc_2 = nn.Linear(512, num_classes).to(self.device)
        self.cls_fc_3 = nn.Linear(512, num_classes).to(self.device)
        
        # 域分类器 (用于预测样本来自哪个源域)
        self.test_domain_fc = nn.Linear(512, 3).to(self.device)
        
        self.criterion = nn.CrossEntropyLoss().to(self.device)
        
        # 三元组损失参数
        self.triplet_margin = 5.0

        # 初始化优化器 (修复: 移至__init__以保持动量状态)
        self.optimizer = torch.optim.Adam([
            {'params': self.sharedNet1.parameters()},
            {'params': self.sharedNet2.parameters()},
            {'params': self.cls_fc_1.parameters()},
            {'params': self.cls_fc_2.parameters()},
            {'params': self.cls_fc_3.parameters()},
            {'params': self.test_domain_fc.parameters()},
        ], lr=self.lr, weight_decay=self.l2_decay)
    
    def forward(self, TR_dataloader, epoch_it):
        """训练前向传播"""
        self.train()
        loss_all = 0.0
        iteration = 100  # 内部迭代次数限制
        
        # 创建数据加载器
        for domain, dataset in enumerate(TR_dataloader):
            if domain == 0:
                train_loader_src_0 = Data.DataLoader(dataset, batch_size=128, shuffle=True)
            elif domain == 1:
                train_loader_src_1 = Data.DataLoader(dataset, batch_size=128, shuffle=True)
            elif domain == 2:
                train_loader_src_2 = Data.DataLoader(dataset, batch_size=128, shuffle=True)
        
        # 训练循环
        for i, (batch_0, batch_1, batch_2) in enumerate(zip(train_loader_src_0, train_loader_src_1, train_loader_src_2)):
            if i >= iteration:
                break
                
            # 学习率调度 (更新优化器中的学习率)
            # GLEARNING_RATE = self.lr / math.pow((1 + 10 * i / iteration), 0.75) # 原调度策略可能过于激进
            # 使用简单的余弦退火或保持恒定，这里为了稳定先保持恒定或微调
            # 实际上，train_test.py 中可能有 scheduler，但这里我们手动控制
            # 为了修复bug，我们暂时不修改LR，或者仅在epoch级别修改
            
            # 解包数据
            x_0, y_0, d_0, _, _ = batch_0
            x_1, y_1, d_1, _, _ = batch_1
            x_2, y_2, d_2, _, _ = batch_2
            
            # 移动到设备
            x_0, y_0 = x_0.to(self.device), y_0.to(self.device)
            x_1 = x_1.to(self.device)
            x_2 = x_2.to(self.device)
            
            # 构造域标签 (我们总是知道数据来自哪个域)
            d_0 = torch.zeros(x_0.size(0), dtype=torch.long).to(self.device)
            d_1 = torch.ones(x_1.size(0), dtype=torch.long).to(self.device)
            d_2 = torch.full((x_2.size(0),), 2, dtype=torch.long).to(self.device)
            
            # 合并数据
            x_combined = torch.cat([x_0, x_1, x_2], dim=0)
            domain_labels = torch.cat([d_0, d_1, d_2], dim=0)
            
            # ==================== 前向传播 ====================
            feature1 = self.sharedNet1(x_combined)  # 不变特征
            feature2 = self.sharedNet2(x_combined)  # 特定特征
            
            # 1. 域分类损失 (Supervised)
            test_domain_pre = self.test_domain_fc(feature2)
            test_domain_loss = self.criterion(test_domain_pre, domain_labels)
            
            # 2. 分类损失 (Supervised on Source 0)
            src_pred1 = self.cls_fc_1(feature1)
            src_pred2 = self.cls_fc_2(feature1)
            src_pred3 = self.cls_fc_3(feature1)
            
            # 提取各域预测
            pred1_d0 = src_pred1[:x_0.size(0)]
            pred2_d0 = src_pred2[:x_0.size(0)] # 也可以让其他分类器在源域0上训练
            pred3_d0 = src_pred3[:x_0.size(0)]
            
            # 3. 伪标签训练 (Pseudo-Labeling for Source 1 & 2)
            # 使用 cls_fc_1 (在源域0上训练得最好) 来生成伪标签
            # 改进: 增加置信度阈值筛选，避免噪声标签
            threshold = 0.95
            with torch.no_grad():
                # 简单起见，使用 cls_fc_1 生成伪标签
                pred1_d1 = src_pred1[x_0.size(0):x_0.size(0)+x_1.size(0)]
                pred1_d2 = src_pred1[x_0.size(0)+x_1.size(0):]
                
                # 获取伪标签和置信度
                probs_1 = F.softmax(pred1_d1, dim=1)
                max_probs_1, pseudo_y_1 = torch.max(probs_1, dim=1)
                mask_1 = max_probs_1.ge(threshold).float()
                
                probs_2 = F.softmax(pred1_d2, dim=1)
                max_probs_2, pseudo_y_2 = torch.max(probs_2, dim=1)
                mask_2 = max_probs_2.ge(threshold).float()
            
            # 源域0的分类损失 (主要训练 cls_fc_1)
            cls_loss_src = F.nll_loss(F.log_softmax(pred1_d0, dim=1), y_0)
            
            # 改进: 让 cls_fc_1 也学习高置信度的伪标签 (使其成为全局分类器，对齐CNN基线能力)
            if mask_1.sum() > 0:
                cls_loss_pseudo_1_global = (F.nll_loss(F.log_softmax(pred1_d1, dim=1), pseudo_y_1, reduction='none') * mask_1).mean()
            else:
                cls_loss_pseudo_1_global = torch.tensor(0.0).to(self.device)
                
            if mask_2.sum() > 0:
                cls_loss_pseudo_2_global = (F.nll_loss(F.log_softmax(pred1_d2, dim=1), pseudo_y_2, reduction='none') * mask_2).mean()
            else:
                cls_loss_pseudo_2_global = torch.tensor(0.0).to(self.device)
            
            # cls_fc_1 的总损失
            cls_loss = cls_loss_src + 0.5 * (cls_loss_pseudo_1_global + cls_loss_pseudo_2_global)

            # 使用伪标签训练对应的分类器 (cls_fc_2 负责域1, cls_fc_3 负责域2)
            pred2_d1 = src_pred2[x_0.size(0):x_0.size(0)+x_1.size(0)]
            pred3_d2 = src_pred3[x_0.size(0)+x_1.size(0):]
            
            # 仅对高置信度样本计算损失
            if mask_1.sum() > 0:
                loss_pseudo_1 = (F.nll_loss(F.log_softmax(pred2_d1, dim=1), pseudo_y_1, reduction='none') * mask_1).mean()
            else:
                loss_pseudo_1 = torch.tensor(0.0).to(self.device)
                
            if mask_2.sum() > 0:
                loss_pseudo_2 = (F.nll_loss(F.log_softmax(pred3_d2, dim=1), pseudo_y_2, reduction='none') * mask_2).mean()
            else:
                loss_pseudo_2 = torch.tensor(0.0).to(self.device)
            
            # 4. CORAL 域对齐损失
            feat_d0 = feature1[:x_0.size(0)]
            feat_d1 = feature1[x_0.size(0):x_0.size(0)+x_1.size(0)]
            feat_d2 = feature1[x_0.size(0)+x_1.size(0):]
            
            # 确保维度匹配 (取最小batch size)
            minlen = min([feat_d0.size(0), feat_d1.size(0), feat_d2.size(0)])
            if minlen > 1: # CORAL需要至少2个样本
                f0 = feat_d0[:minlen]
                f1 = feat_d1[:minlen]
                f2 = feat_d2[:minlen]
                
                MMD_loss = self._coral(f0, f1) + \
                           self._coral(f0, f2) + \
                           self._coral(f1, f2)
            else:
                MMD_loss = torch.tensor(0.0).to(self.device)
            
            # 5. 三元组损失
            # 改进: 仅使用源域0(真实标签)和高置信度伪标签样本进行三元组挖掘
            # 构造掩码: 源域0全选(mask=1), 源域1/2根据置信度选
            mask_0 = torch.ones(x_0.size(0)).to(self.device)
            combined_mask = torch.cat([mask_0, mask_1, mask_2], dim=0).bool()
            
            if combined_mask.sum() > 1: # 至少要有2个样本才能计算Triplet
                combined_features = feature1[combined_mask]
                combined_labels = torch.cat([y_0, pseudo_y_1, pseudo_y_2], dim=0)[combined_mask]
                
                # 使用 Batch Hard Triplet Selector
                selector = BatchHardTripletSelector()
                anchor, pos, neg = selector(combined_features, combined_labels)
                
                triplet_loss_func = TripletLoss(margin=self.triplet_margin).to(self.device)
                triplet_loss = triplet_loss_func(anchor, pos, neg)
            else:
                triplet_loss = torch.tensor(0.0).to(self.device)
            
            # 动态权重
            lambd = 2 / (1 + math.exp(-10 * i / iteration)) - 1
            
            # 预热策略 (Warm-up): 前20个Epoch仅使用有监督损失，让模型先学好基础特征
            if epoch_it < 20:
                loss = cls_loss_src
            else:
                # 总损失
                loss = cls_loss + \
                       0.5 * (loss_pseudo_1 + loss_pseudo_2) + \
                       test_domain_loss + \
                       MMD_loss * lambd + \
                       triplet_loss * lambd
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            loss_all += loss.item()
        
        return loss_all / (i + 1)
    
    def _coral(self, source, target):
        """CORAL 域对齐损失"""
        d = source.size(1)
        ns, nt = source.size(0), target.size(0)
        
        # 计算协方差矩阵
        source_mean = source.mean(dim=0, keepdim=True)
        target_mean = target.mean(dim=0, keepdim=True)
        
        source_centered = source - source_mean
        target_centered = target - target_mean
        
        cov_source = torch.mm(source_centered.t(), source_centered) / (ns - 1)
        cov_target = torch.mm(target_centered.t(), target_centered) / (nt - 1)
        
        loss = torch.mean((cov_source - cov_target) ** 2)
        return loss
    
    def model_inference(self, input):
        """模型推理 - 使用加权集成"""
        self.eval()
        with torch.no_grad():
            input = input.to(self.device)
            feature1 = self.sharedNet1(input)
            feature2 = self.sharedNet2(input)
            
            pred1 = self.cls_fc_1(feature1)
            pred2 = self.cls_fc_2(feature1)
            pred3 = self.cls_fc_3(feature1)
            test_domain_pre = self.test_domain_fc(feature2)
            
            # 使用域预测权重进行加权集成
            m = nn.Softmax(dim=1)
            tgt_domain_pre = m(test_domain_pre)
            
            # 加权融合
            pred = m(pred1) * tgt_domain_pre[:, 0].reshape(-1, 1) + \
                   m(pred2) * tgt_domain_pre[:, 1].reshape(-1, 1) + \
                   m(pred3) * tgt_domain_pre[:, 2].reshape(-1, 1)
            
            # 返回 logits (取 log)
            prediction = torch.log(pred + 1e-10)
        
        return prediction
