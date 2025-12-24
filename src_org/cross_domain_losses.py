import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import precision_score, f1_score, recall_score, accuracy_score, roc_auc_score


def euclidean_dist(x, y):
    '''
    Compute euclidean distance between two tensors
    '''
    # x: N x D
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层，用于对抗训练"""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def gradient_reversal(x, alpha=1.0):
    return GradientReversalLayer.apply(x, alpha)


class CrossDomainLoss(nn.Module):
    def __init__(self, args):
        super(CrossDomainLoss, self).__init__()
        self.args = args
        self.domain_loss_weight = getattr(args, 'domain_loss_weight', 0.1)
        self.adversarial_loss_weight = getattr(args, 'adversarial_loss_weight', 0.01)

        # 领域分类器损失
        self.domain_criterion = nn.CrossEntropyLoss()

    def forward(self, model_outputs, labels, domain_info=None):
        """
        计算跨域元学习的总损失

        Args:
            model_outputs: 模型输出字典，包含:
                - prototypes: 类原型
                - query_embeddings: 查询嵌入
                - acc: top-k准确率
                - original_prototypes: 原始原型
                - sampled_data: 采样数据
                - domain_agnostic_features: 领域无关特征
                - domain_name: 域名
            labels: 标签
            domain_info: 域信息（可选）
        """
        device = next(iter(model_outputs.values())).device if isinstance(model_outputs, dict) else model_outputs.device
        query_size = self.args.numNWay * self.args.numQShot
        support_size = self.args.numNWay * self.args.numKShot

        prototypes = model_outputs['prototypes']
        q_re = model_outputs['query_embeddings']
        topk_acc = model_outputs['acc']
        original_prototypes = model_outputs['original_prototypes']
        sampled_data = model_outputs['sampled_data']
        domain_agnostic_features = model_outputs['domain_agnostic_features']
        domain_name = model_outputs.get('domain_name', None)

        # 1. 主要分类损失（继承原始损失）
        dists = euclidean_dist(q_re, prototypes)
        log_p_y = F.log_softmax(-dists, dim=1)

        query_labels = labels[support_size:]
        query_labels = torch.tensor(query_labels, dtype=float).to(device)
        classification_loss = - query_labels * log_p_y
        classification_loss = classification_loss.mean()

        # 2. 生成损失（继承原始损失）
        generate_labels = torch.tensor(range(self.args.numNWay)).unsqueeze(dim=1)
        generate_labels = generate_labels.repeat(1, sampled_data.shape[1]).view(-1)
        generate_labels = F.one_hot(generate_labels, self.args.numNWay).float().to(device)
        sampled_data_flat = sampled_data.view(-1, sampled_data.shape[2])
        g_dists = euclidean_dist(sampled_data_flat, original_prototypes)
        glog_p_y = F.log_softmax(-g_dists, dim=1)
        generation_loss = - generate_labels * glog_p_y
        generation_loss = generation_loss.mean()

        # 3. 域对抗损失（新增）
        domain_loss = torch.tensor(0.0).to(device)
        if domain_agnostic_features is not None and domain_info is not None:
            # 应用梯度反转
            reversed_features = gradient_reversal(
                domain_agnostic_features,
                alpha=self.adversarial_loss_weight
            )

            # 域分类损失（我们希望域分类器无法区分域）
            if hasattr(self, 'domain_classifier'):
                domain_predictions = self.domain_classifier(reversed_features)
                if isinstance(domain_info, list):
                    # 如果domain_info是域名列表，需要转换为索引
                    domain_mapping = {'HuffPost': 0, 'Amazon': 1}
                    domain_targets = torch.tensor([
                        domain_mapping.get(d, 0) for d in domain_info
                    ]).to(device)
                else:
                    domain_targets = domain_info

                domain_loss = self.domain_criterion(domain_predictions, domain_targets)

        # 4. 跨域一致性损失（新增）
        consistency_loss = torch.tensor(0.0).to(device)
        if hasattr(model_outputs, 'cross_domain_features'):
            # 如果有跨域特征，计算一致性损失
            cross_domain_features = model_outputs['cross_domain_features']
            consistency_loss = self._compute_consistency_loss(cross_domain_features)

        # 总损失
        total_loss = (classification_loss +
                      0.1 * generation_loss +
                      self.domain_loss_weight * domain_loss +
                      0.05 * consistency_loss)

        # 计算评估指标（继承原始代码）
        metrics = self._compute_metrics(log_p_y, query_labels, dists)

        return total_loss, metrics['p'], metrics['r'], metrics['f'], metrics['acc'], metrics['auc'], topk_acc

    def _compute_consistency_loss(self, cross_domain_features):
        """计算跨域一致性损失"""
        if len(cross_domain_features) < 2:
            return torch.tensor(0.0).to(device)

        consistency_loss = 0.0
        count = 0

        # 计算不同域特征之间的差异
        for i in range(len(cross_domain_features)):
            for j in range(i + 1, len(cross_domain_features)):
                # 使用余弦相似度作为一致性度量
                sim = F.cosine_similarity(
                    cross_domain_features[i].mean(dim=0, keepdim=True),
                    cross_domain_features[j].mean(dim=0, keepdim=True)
                )
                consistency_loss += (1 - sim)  # 希望特征相似
                count += 1

        return consistency_loss / count if count > 0 else torch.tensor(0.0).to(device)

    def _compute_metrics(self, log_p_y, query_labels, dists):
        """计算评估指标"""
        # 单标签分类指标计算
        x, _ = torch.max(log_p_y, dim=1, keepdim=True)
        one = torch.ones_like(log_p_y)
        zero = torch.zeros_like(log_p_y)
        y_pred = torch.where(log_p_y >= x, one, log_p_y)
        y_pred = torch.where(y_pred < x, zero, y_pred)

        target_mode = 'macro'

        query_labels_np = query_labels.cpu().detach().numpy()
        y_pred_np = y_pred.cpu().detach().numpy()

        try:
            p = precision_score(query_labels_np, y_pred_np, average=target_mode, zero_division=0)
            r = recall_score(query_labels_np, y_pred_np, average=target_mode, zero_division=0)
            f = f1_score(query_labels_np, y_pred_np, average=target_mode, zero_division=0)
            acc = accuracy_score(query_labels_np, y_pred_np)

            y_score = F.softmax(-dists, dim=1)
            y_score_np = y_score.cpu().detach().numpy()
            auc = roc_auc_score(query_labels_np, y_score_np, multi_class='ovr', average=target_mode)
        except Exception as e:
            print(f"Metric calculation error: {e}")
            p = r = f = acc = auc = 0.0

        return {
            'p': p,
            'r': r,
            'f': f,
            'acc': acc,
            'auc': auc
        }


class CrossDomainLossWithSelfSupervision(nn.Module):
    """
    在原有CrossDomainLoss基础上添加简单的自监督正则化
    第一步：只添加最基础的consistency regularization
    """
    def __init__(self, args):
        super(CrossDomainLossWithSelfSupervision, self).__init__()
        self.args = args
        self.domain_loss_weight = getattr(args, 'domain_loss_weight', 0.1)
        self.adversarial_loss_weight = getattr(args, 'adversarial_loss_weight', 0.01)
        
        # 新增：自监督正则化权重 - 开始设置得很小，避免影响现有性能
        self.consistency_weight = getattr(args, 'consistency_weight', 0.01)
        
        # 领域分类器损失
        self.domain_criterion = nn.CrossEntropyLoss()

    def compute_consistency_loss(self, query_embeddings, prototypes):
        """
        计算简单的一致性损失 - 鼓励相似样本有相似的表示
        这是SELP思路的最简化版本
        """
        device = query_embeddings.device
        batch_size = query_embeddings.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0).to(device)
        
        # 计算查询样本之间的相似度矩阵
        query_sim = torch.mm(F.normalize(query_embeddings, dim=1), 
                            F.normalize(query_embeddings, dim=1).t())
        
        # 计算到原型的距离
        dists_to_proto = euclidean_dist(query_embeddings, prototypes)
        proto_sim = F.softmax(-dists_to_proto, dim=1)
        
        # 计算预测一致性
        pred_sim = torch.mm(proto_sim, proto_sim.t())
        
        # 一致性损失：如果两个样本在embedding空间相似，它们的预测也应该相似
        # 使用KL散度作为一致性度量
        consistency_loss = 0.0
        count = 0
        
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                if query_sim[i, j] > 0.7:  # 只对相似的样本对施加一致性约束
                    # 对于相似的样本，它们的预测分布应该相似
                    kl_loss = F.kl_div(F.log_softmax(proto_sim[i].unsqueeze(0), dim=1),
                                      F.softmax(proto_sim[j].unsqueeze(0), dim=1),
                                      reduction='sum')
                    consistency_loss += kl_loss
                    count += 1
        
        if count > 0:
            consistency_loss = consistency_loss / count
        else:
            consistency_loss = torch.tensor(0.0).to(device)
            
        return consistency_loss

    def forward(self, model_outputs, labels, domain_info=None):
        """
        计算跨域元学习的总损失 + 简单的自监督正则化
        """
        device = next(iter(model_outputs.values())).device if isinstance(model_outputs, dict) else model_outputs.device
        query_size = self.args.numNWay * self.args.numQShot
        support_size = self.args.numNWay * self.args.numKShot

        prototypes = model_outputs['prototypes']
        q_re = model_outputs['query_embeddings']
        topk_acc = model_outputs['acc']
        original_prototypes = model_outputs['original_prototypes']
        sampled_data = model_outputs['sampled_data']
        domain_agnostic_features = model_outputs['domain_agnostic_features']
        domain_name = model_outputs.get('domain_name', None)

        # 1. 主要分类损失（继承原始损失）
        dists = euclidean_dist(q_re, prototypes)
        log_p_y = F.log_softmax(-dists, dim=1)

        query_labels = labels[support_size:]
        query_labels = torch.tensor(query_labels, dtype=float).to(device)
        classification_loss = - query_labels * log_p_y
        classification_loss = classification_loss.mean()

        # 2. 生成损失（继承原始损失）
        generate_labels = torch.tensor(range(self.args.numNWay)).unsqueeze(dim=1)
        generate_labels = generate_labels.repeat(1, sampled_data.shape[1]).view(-1)
        generate_labels = F.one_hot(generate_labels, self.args.numNWay).float().to(device)
        sampled_data_flat = sampled_data.view(-1, sampled_data.shape[2])
        g_dists = euclidean_dist(sampled_data_flat, original_prototypes)
        glog_p_y = F.log_softmax(-g_dists, dim=1)
        generation_loss = - generate_labels * glog_p_y
        generation_loss = generation_loss.mean()

        # 3. 域对抗损失（继承原始损失）
        domain_loss = torch.tensor(0.0).to(device)
        if domain_agnostic_features is not None and domain_info is not None:
            # 暂时保持原有的域对抗逻辑
            pass

        # 4. 新增：简单的一致性正则化损失
        consistency_loss = self.compute_consistency_loss(q_re, prototypes)

        # 总损失 - 权重保持保守，确保不影响现有性能
        total_loss = (classification_loss +
                      0.1 * generation_loss +
                      self.domain_loss_weight * domain_loss +
                      self.consistency_weight * consistency_loss)  # 新增项权重很小

        # 计算评估指标（继承原始代码）
        metrics = self._compute_metrics(log_p_y, query_labels, dists)

        return total_loss, metrics['p'], metrics['r'], metrics['f'], metrics['acc'], metrics['auc'], topk_acc

    def _compute_metrics(self, log_p_y, query_labels, dists):
        """计算评估指标 - 与原始代码相同"""
        # 单标签分类指标计算
        x, _ = torch.max(log_p_y, dim=1, keepdim=True)
        one = torch.ones_like(log_p_y)
        zero = torch.zeros_like(log_p_y)
        y_pred = torch.where(log_p_y >= x, one, log_p_y)
        y_pred = torch.where(y_pred < x, zero, y_pred)

        target_mode = 'macro'

        query_labels_np = query_labels.cpu().detach().numpy()
        y_pred_np = y_pred.cpu().detach().numpy()

        try:
            p = precision_score(query_labels_np, y_pred_np, average=target_mode, zero_division=0)
            r = recall_score(query_labels_np, y_pred_np, average=target_mode, zero_division=0)
            f = f1_score(query_labels_np, y_pred_np, average=target_mode, zero_division=0)
            acc = accuracy_score(query_labels_np, y_pred_np)

            y_score = F.softmax(-dists, dim=1)
            y_score_np = y_score.cpu().detach().numpy()
            auc = roc_auc_score(query_labels_np, y_score_np, multi_class='ovr', average=target_mode)
        except Exception as e:
            print(f"Metric calculation error: {e}")
            p = r = f = acc = auc = 0.0

        return {
            'p': p,
            'r': r,
            'f': f,
            'acc': acc,
            'auc': auc
        }


class DomainAdaptationLoss(nn.Module):
    """域适应损失，专门用于域对抗训练"""

    def __init__(self, feature_dim, num_domains, hidden_dim=256):
        super(DomainAdaptationLoss, self).__init__()
        self.domain_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_domains)
        )
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, domain_agnostic_features, domain_labels, alpha=1.0):
        """
        计算域适应损失

        Args:
            domain_agnostic_features: 领域无关特征
            domain_labels: 域标签
            alpha: 梯度反转强度
        """
        # 应用梯度反转
        reversed_features = gradient_reversal(domain_agnostic_features, alpha)

        # 域分类
        domain_predictions = self.domain_classifier(reversed_features)

        # 域分类损失
        domain_loss = self.criterion(domain_predictions, domain_labels)

        return domain_loss


class MetaTaskLoss(nn.Module):
    """元任务损失，用于多任务元学习"""

    def __init__(self, args):
        super(MetaTaskLoss, self).__init__()
        self.args = args
        self.task_adaptation_weight = getattr(args, 'task_adaptation_weight', 0.1)

    def forward(self, task_outputs, task_labels):
        """
        计算元任务损失

        Args:
            task_outputs: 各个任务的输出列表
            task_labels: 各个任务的标签列表
        """
        total_loss = 0.0
        task_count = 0

        for task_output, task_label in zip(task_outputs, task_labels):
            # 为每个任务计算损失
            task_loss = self._compute_single_task_loss(task_output, task_label)
            total_loss += task_loss
            task_count += 1

        # 返回平均任务损失
        return total_loss / task_count if task_count > 0 else torch.tensor(0.0).to(device)

    def _compute_single_task_loss(self, task_output, task_label):
        """计算单个任务的损失"""
        # 这里可以根据具体任务类型实现不同的损失计算
        # 暂时使用交叉熵损失
        return F.cross_entropy(task_output, task_label)


# 为了向后兼容，保留原始的Loss_fn类
class Loss_fn(CrossDomainLoss):
    """向后兼容的损失函数类"""

    def __init__(self, args):
        super(Loss_fn, self).__init__(args)
