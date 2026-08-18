import logging
import os
import time
import torch
from torch import nn
from torch import optim
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.metrics import f1_score, matthews_corrcoef
import models 
from utils.save_confusion_matrix import plot_confusion_matrix
from utils.save_t_sne import plot_t_sne
from utils.save_attention_heatmap import plot_physical_attention_heatmap

# 1.将 models 文件夹组织为 Python 包（通过 __init__.py）而不是单个 .py 文件的原因如下：1.便于扩展多个模型文件现在只有 DPD_DG.py，以后可能还有 ResNet.py、Transformer.py 等;
# 2.用文件夹时，可以一个模型一个文件，结构清晰：models/DPD_DG.py、models/ResNet.py 等。如果用单个 models.py，所有模型代码都堆在一起，文件会越来越长，难以维护。
# === 训练/测试调度器 ===
# 负责根据 args 配置模型、优化器与损失，并组织训练/验证流程。
class train_test(object):
    def __init__(self, args):
        """
        初始化函数
        参数:
            args: 包含模型配置和其他参数的对象
        """
        self.args = args
    def setup(self,n_class):
        """
        设置模型、优化器和其他必要组件
        参数:
            n_class: 分类任务的类别数量
        """
        args = self.args
        self.method=args.model_name
        # 设置运行设备(CPU或GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # 根据方法名称设置模型名称
        if self.method in  ['DPD_DG']:
            self.model_name='DPD_DG'
        elif self.method == 'DKGPL':
            self.model_name = 'DKGPL'
        elif self.method == 'CNN':
            self.model_name = 'CNN'
        elif self.method == 'CCDG':
            self.model_name = 'CCDG'
        elif self.method == 'DGNIS':
            self.model_name = 'DGNIS'
        elif self.method == 'CDDG':
            self.model_name = 'CDDG'
        elif self.method == 'MSDGN':
            self.model_name = 'MSDGN'
        elif self.method == 'CEG':
            self.model_name = 'CEG'
        elif self.method == 'CS':
            self.model_name = 'CS'
        # 初始化模型
        self.model =getattr(models,self.model_name)(in_channel=1,num_classes=n_class,lr=args.lr,set=self.method,args=args) #动态从models包中获取名为self.model_name（例如 'DPD_DG'）的类，然后调用参数实例化它。
        # 设置优化器
        # 如果模型类内部已经定义了优化器（例如模型自己负责训练流程），优先使用模型内部的 optimizer，避免重复定义。
        if hasattr(self.model, 'optimizer') and getattr(self.model, 'optimizer') is not None:
            # 若模型内部定义了多组优化器（如 optimizer_G/optimizer_CC 等），train_test 只做引用而不创建新优化器。
            self.optimizer = getattr(self.model, 'optimizer')
            logging.info('train_test: using optimizer defined inside model (%s).', type(self.optimizer).__name__)
        else:
            # 若模型没有内部 optimizer，则由 train_test 创建一个默认优化器
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=0.0001)
            logging.info('train_test: created default Adam optimizer for model at lr=%s', self.args.lr)
        # 将模型移动到指定设备
        self.model.to(self.device)
        # 设置损失函数
        self.criterion = nn.CrossEntropyLoss()

        return
    def train(self, op_num, TR_dataloader, Val_dataloader):
        # op_num 用于区分第几次重复实验；TR_dataloader 是源域训练集列表；Val_dataloader 是源域验证集 DataLoader。
        args = self.args
        best_epoch_acc = 0
        best_epoch_loss = 0
        best_acc = 0.0
        best_val_loss = float('inf')
        
        # 用于耐心度跟踪
        patience_best_acc = 0.0
        patience_best_loss = float('inf')
        time_all=0.0
        
        # 注意：CCDG、DKGPL、CNN这些模型需要分域处理，不能合并数据集
        # if self.model_name in ['...'] and isinstance(TR_dataloader, list):
        #     concat_dataset = ConcatDataset(TR_dataloader)
        #     TR_dataloader = DataLoader(concat_dataset, batch_size=128, shuffle=True, drop_last=False)

        if self.model_name in ['DPD_DG','DKGPL', 'CNN', 'CCDG',  'DGNIS', 'CDDG', 'MSDGN', 'CEG', 'CS']:
            patience_counter = 0
            for epoch in range(args.epoch):
                epoch_start = time.time()
                # 模型内部 forward 会遍历 TR_dataloader，完成伪标签、原型对齐、MMD 等全部损失。
                loss=self.model(TR_dataloader,epoch_it=epoch)
                logging.info('Opration-{}, Epoch: {}, Loss: {:.2f},Time {:.4f} sec'.format(op_num, epoch, loss,time.time() - epoch_start))
                time_all+= time.time() - epoch_start

                # 每个 epoch 后都使用源域验证集评估，不接触目标域数据，选出泛化能力最强的模型
                self.model.eval()
                total_correct = 0
                total_samples = 0
                val_loss = 0.0
                all_preds = []
                all_labels = []
                with torch.no_grad():
                    for batch_idx, (inputs, labels,domain, _, prior_feats) in enumerate(Val_dataloader):
                        #print("Val_dataloader_batch_Num: {}, Oparation-{} batch_idx: {}".format(len(Val_dataloader),op_num, batch_idx))
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        # 改进：在验证阶段传入先验特征以启用物理引导推理
                        if self.model_name == 'DPD_DG':
                            logits = self.model.model_inference(inputs, prior=prior_feats)
                        else:
                            logits = self.model.model_inference(inputs)
                        
                        loss_v = self.criterion(logits, labels)
                        val_loss += loss_v.item()

                        pred = logits.argmax(dim=1)
                        total_correct += (pred == labels).sum().item()
                        total_samples += labels.size(0)
                        all_preds.extend(pred.cpu().numpy())
                        all_labels.extend(labels.cpu().numpy())

                    epoch_acc = (total_correct / total_samples)*100
                    avg_val_loss = val_loss / len(Val_dataloader)
                    logging.info("Source Validation Accuracy: {:.2f} (Correct: {}/{}), Val Loss: {:.4f}".format(epoch_acc, total_correct, total_samples, avg_val_loss))
                    
                    if hasattr(self.model, 'writer') and self.model.writer is not None:
                        self.model.writer.add_scalar('Val/Accuracy', epoch_acc, epoch)
                        self.model.writer.add_scalar('Val/Loss', avg_val_loss, epoch)

                    # 仅当验证集表现创下新高且当前轮数大于等于设定的起始轮数时才覆盖权重文件
                    if epoch >= args.save_best_after_epoch:
                        save_dir = os.path.join('./trained_models/{}/{}'.format(args.dataset_name, args.model_name))
                        if not os.path.exists(save_dir):
                            os.makedirs(save_dir)

                        # 保存最佳准确率模型
                        if epoch_acc > best_acc or (epoch_acc == best_acc and avg_val_loss < best_val_loss):
                            best_acc = epoch_acc
                            best_epoch_acc = epoch
                            torch.save(self.model.state_dict(), os.path.join('{}/{}.pth'.format(save_dir, 'operation_' + str(op_num) + '_best_acc')))

                        # 保存最佳损失模型
                        if avg_val_loss < best_val_loss:
                            best_val_loss = avg_val_loss
                            best_epoch_loss = epoch
                            torch.save(self.model.state_dict(), os.path.join('{}/{}.pth'.format(save_dir, 'operation_' + str(op_num) + '_best_loss')))

                        # 耐心度逻辑
                       
                        if epoch_acc > patience_best_acc:
                            patience_best_acc = epoch_acc
                            patience_counter = 0
                        else:
                            patience_counter += 1
                                
                        if args.early_stop and patience_counter >= args.patience:
                            logging.info("Early stopping triggered at epoch {}".format(epoch))
                            break
                    else:
                        logging.info("Skipping model save check (Current Epoch: {} < {})".format(epoch, args.save_best_after_epoch))


            logging.info(f"Operation_{op_num}, Training time {time_all:.2f}, Best_epoch_acc_{op_num}: {best_epoch_acc}, Best_Acc_{op_num}: {best_acc:.2f}, Best_epoch_loss_{op_num}: {best_epoch_loss}, Best_Val_Loss_{op_num}: {best_val_loss:.4f}")

        return

    def test(self, op_num, Test_dataloader, Source_dataloader=None):
        # 测试阶段仅需要目标域 DataLoader
        args = self.args
        save_dir = os.path.join(f'./trained_models/{args.dataset_name}/{args.model_name}')
        
        # 评估辅助函数
        def evaluate_model(model_filename):
            model_path = os.path.join(save_dir, model_filename)
            if not os.path.exists(model_path):
                logging.warning(f"Model file {model_path} not found.")
                return 0.0, 0.0, 0.0

            self.model.load_state_dict(torch.load(model_path, weights_only=True), strict=False)
            self.model.eval()
            
            total_acc, total_samples = 0.0, 0.0
            all_preds = []
            all_labels = []
            
            def get_feats(inputs):
                if args.model_name == 'DPD_DG':
                    return self.model.feature_extractor(inputs, enable_dfe=False)
                elif args.model_name == 'DKGPL':
                    return self.model.G(inputs, trid=False)
                elif args.model_name == 'CCDG':
                    return self.model.G(inputs)
                elif args.model_name == 'DGNIS':
                    return self.model.sharedNet1(inputs)
                elif args.model_name == 'MSDGN':
                    return self.model.net3_feat(inputs)
                elif args.model_name == 'CDDG':
                    return self.model.encoder_h(inputs)
                elif hasattr(self.model, 'feature_extractor'):
                    return self.model.feature_extractor(inputs)
                return None

            with torch.no_grad():
                # === 重构: 分离 指标计算 和 可视化数据收集 ===
                
                # A. 计算 Target 指标 (必须全量)
                for inputs, labels, domain, _, prior_feats in Test_dataloader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)

                    if args.model_name in ['DPD_DG', 'DKGPL', 'CNN', 'CCDG',  'DGNIS', 'CDDG', 'MSDGN', 'CEG', 'CS']:
                        if args.model_name == 'DPD_DG':
                            logits = self.model.model_inference(inputs, prior=prior_feats)
                        else:
                            logits = self.model.model_inference(inputs)
                    else:
                        logits = self.model(inputs)
                    
                    pred = logits.argmax(dim=1)
                    total_acc += (pred == labels).sum().item()
                    total_samples += labels.size(0)
                    all_preds.extend(pred.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())

                # B. 如果需要保存 t-SNE，收集数据
                tsne_features = []
                tsne_labels = []
                tsne_domains = []
                
                if args.save_tsne:
                    # 1. 收集 Source 数据
                    if Source_dataloader is not None:
                        count = 0
                        for inputs, labels, domain, _, prior_feats in Source_dataloader:
                            if count > 500: break # 限制源域样本数，防止覆盖目标域
                            inputs = inputs.to(self.device)
                            feats = get_feats(inputs)
                            
                            if feats is not None:
                                if len(feats.shape) > 2:
                                    feats = torch.nn.functional.adaptive_avg_pool1d(feats, 1).squeeze(-1)
                                tsne_features.extend(feats.cpu().numpy())
                                tsne_labels.extend(labels.numpy())
                                tsne_domains.extend([0] * len(labels)) # 0 for Source
                                count += len(labels)

                    # 2. 收集 Target 数据 (我们可能不需要全量 Target，取一部分或者全量)
                    target_count = 0
                    for inputs, labels, domain, _, prior_feats in Test_dataloader:
                        if target_count > 500: break # 同样限制数量保持平衡
                        inputs = inputs.to(self.device)
                        feats = get_feats(inputs)

                        if feats is not None:
                            if len(feats.shape) > 2:
                                feats = torch.nn.functional.adaptive_avg_pool1d(feats, 1).squeeze(-1)
                            tsne_features.extend(feats.cpu().numpy())
                            tsne_labels.extend(labels.numpy())
                            tsne_domains.extend([1] * len(labels)) # 1 for Target
                            target_count += len(labels)
            
            final_acc = total_acc / total_samples
            final_f1 = f1_score(all_labels, all_preds, average='macro')
            final_mcc = matthews_corrcoef(all_labels, all_preds)

            # 获取类别名称
            dataset_classes = {
                'CWRU_Bearing': ['Normal', 'Ball 007', 'Ball 014', 'Ball 021', 
                                'Inner Race 007', 'Inner Race 014', 'Inner Race 021', 
                                'Outer Race 007', 'Outer Race 014', 'Outer Race 021'],
                'BJUT_Gear': ['Normal', 'Crack in root', 'Gear wear', 'Missing tooth', 'Broken tooth'],
                'SDUT_Bearing': ['Normal', 'Inner Race', 'Outer Race', 'Ball'],
                'SDUT_Gear': ['Normal', 'Sun fracture', 'Sun pitting', 'Sun wear', 'Plant fracture', 'Plant pitting', 'Plant wear'],
                'HUST_Gear': ['Normal', 'Broken tooth', 'Missing tooth'],
                'Ottawa_Bearing': ['Normal', 'Inner Race', 'Outer Race']
            }
            class_names = dataset_classes.get(args.dataset_name, None)

            if args.save_cm:
                # 构造文件名: 数据集_目标域_模型_消融参数_准确率.png
                cm_filename = "{}_{}_{}_{}_{}_ACC{:.1f}_CM.png".format(
                    args.dataset_name,
                    args.target_id, 
                    args.model_name,
                    model_filename.replace('.pth', ''), 
                    args.ablation_mode,
                    final_acc * 100
                )
                
                cm_save_dir = os.path.join(os.path.dirname(__file__), "confusion_matrix")
                cm_save_path = os.path.join(cm_save_dir, cm_filename)
                
                plot_confusion_matrix(all_labels, all_preds, cm_save_path, 
                                      title="{} - {} {} (Acc: {:.2f}%)".format(args.dataset_name, args.model_name, args.ablation_mode, final_acc*100),
                                      class_names=class_names,
                                      accuracy=final_acc)
                                      
            if args.save_tsne and len(tsne_features) > 0:
                tsne_filename = "{}_{}_{}_{}_{}_ACC{:.1f}_tSNE.png".format(
                    args.dataset_name,
                    args.target_id, 
                    args.model_name,
                    model_filename.replace('.pth', ''), 
                    args.ablation_mode,
                    final_acc * 100
                )
                tsne_save_dir = os.path.join(os.path.dirname(__file__), "t-SNE")
                tsne_save_path = os.path.join(tsne_save_dir, tsne_filename)
                
                # 构建更详细的标题
                # 格式: 数据集 - 模型(M5) (Acc: 98.5%)
                tsne_title = "{} - {} {} (Acc: {:.2f}%)".format( args.dataset_name,  args.model_name,args.ablation_mode,  final_acc*100)
                
                import numpy as np
                plot_t_sne(np.array(tsne_features), np.array(tsne_labels), tsne_save_path,
                           title=tsne_title,
                           class_names=class_names,
                           domains=np.array(tsne_domains))

            # === Save Physical Attention Heatmap (New) ===
            if args.model_name == 'DPD_DG' and getattr(args, 'save_attention', False):
                 att_filename = "{}_{}_{}_{}_{}_ACC{:.1f}_Attention.png".format(
                    args.dataset_name,
                    args.target_id, 
                    args.model_name,
                    model_filename.replace('.pth', ''), 
                    args.ablation_mode,
                    final_acc * 100
                )
                 att_save_dir = os.path.join(os.path.dirname(__file__), "attention_maps")
                 att_save_path = os.path.join(att_save_dir, att_filename)
                 
                 # Plot using Test_dataloader to show attention on Target Domain
                 plot_physical_attention_heatmap(self.model, Test_dataloader, self.device, att_save_path, class_names=class_names)

            return final_acc, final_f1, final_mcc

        # Test Best Accuracy Model
        acc_best_acc, f1_best_acc, mcc_best_acc = evaluate_model(f'operation_{op_num}_best_acc.pth')
        logging.info(f"Operation_{op_num} [Best Ac c Model], Final Target Test Acc: {acc_best_acc*100:.2f}%, F1: {f1_best_acc*100:.2f}%, MCC: {mcc_best_acc*100:.2f}%")

        # Test Best Loss Model
        acc_best_loss, f1_best_loss, mcc_best_loss = evaluate_model(f'operation_{op_num}_best_loss.pth')
        logging.info(f"Operation_{op_num} [Best Loss Model], Final Target Test Acc: {acc_best_loss*100:.2f}%, F1: {f1_best_loss*100:.2f}%, MCC: {mcc_best_loss*100:.2f}%")

        return acc_best_acc, f1_best_acc, mcc_best_acc, acc_best_loss, f1_best_loss, mcc_best_loss
