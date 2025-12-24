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
        self.domain_criterion = nn.CrossEntropyLoss()

    def forward(self, model_outputs, labels, domain_info=None):
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
        dists = euclidean_dist(q_re, prototypes)
        log_p_y = F.log_softmax(-dists, dim=1)

        query_labels = labels[support_size:]
        query_labels = torch.tensor(query_labels, dtype=float).to(device)
        classification_loss = - query_labels * log_p_y
        classification_loss = classification_loss.mean()
        generate_labels = torch.tensor(range(self.args.numNWay)).unsqueeze(dim=1)
        generate_labels = generate_labels.repeat(1, sampled_data.shape[1]).view(-1)
        generate_labels = F.one_hot(generate_labels, self.args.numNWay).float().to(device)
        sampled_data_flat = sampled_data.view(-1, sampled_data.shape[2])
        g_dists = euclidean_dist(sampled_data_flat, original_prototypes)
        glog_p_y = F.log_softmax(-g_dists, dim=1)
        generation_loss = - generate_labels * glog_p_y
        generation_loss = generation_loss.mean()

        domain_loss = torch.tensor(0.0).to(device)
        if domain_agnostic_features is not None and domain_info is not None:
            reversed_features = gradient_reversal(
                domain_agnostic_features,
                alpha=self.adversarial_loss_weight
            )
            if hasattr(self, 'domain_classifier'):
                domain_predictions = self.domain_classifier(reversed_features)
                if isinstance(domain_info, list):
                    domain_mapping = {'HuffPost': 0, 'Amazon': 1}
                    domain_targets = torch.tensor([
                        domain_mapping.get(d, 0) for d in domain_info
                    ]).to(device)
                else:
                    domain_targets = domain_info

                domain_loss = self.domain_criterion(domain_predictions, domain_targets)

        consistency_loss = torch.tensor(0.0).to(device)
        if hasattr(model_outputs, 'cross_domain_features'):
            cross_domain_features = model_outputs['cross_domain_features']
            consistency_loss = self._compute_consistency_loss(cross_domain_features)

        total_loss = (classification_loss +
                      0.1 * generation_loss +
                      self.domain_loss_weight * domain_loss +
                      0.05 * consistency_loss)

        metrics = self._compute_metrics(log_p_y, query_labels, dists)

        return total_loss, metrics['p'], metrics['r'], metrics['f'], metrics['acc'], metrics['auc'], topk_acc

    def _compute_consistency_loss(self, cross_domain_features):
        if len(cross_domain_features) < 2:
            return torch.tensor(0.0).to(device)

        consistency_loss = 0.0
        count = 0

        for i in range(len(cross_domain_features)):
            for j in range(i + 1, len(cross_domain_features)):
                sim = F.cosine_similarity(
                    cross_domain_features[i].mean(dim=0, keepdim=True),
                    cross_domain_features[j].mean(dim=0, keepdim=True)
                )
                consistency_loss += (1 - sim) 
                count += 1

        return consistency_loss / count if count > 0 else torch.tensor(0.0).to(device)

    def _compute_metrics(self, log_p_y, query_labels, dists):
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
    def __init__(self, args):
        super(CrossDomainLossWithSelfSupervision, self).__init__()
        self.args = args
        self.domain_loss_weight = getattr(args, 'domain_loss_weight', 0.1)
        self.adversarial_loss_weight = getattr(args, 'adversarial_loss_weight', 0.01)        
        self.consistency_weight = getattr(args, 'consistency_weight', 0.01)
        self.domain_criterion = nn.CrossEntropyLoss()

    def compute_consistency_loss(self, query_embeddings, prototypes):
        device = query_embeddings.device
        batch_size = query_embeddings.size(0)
        
        if batch_size < 2:
            return torch.tensor(0.0).to(device)
        query_sim = torch.mm(F.normalize(query_embeddings, dim=1), 
                            F.normalize(query_embeddings, dim=1).t())
        dists_to_proto = euclidean_dist(query_embeddings, prototypes)
        proto_sim = F.softmax(-dists_to_proto, dim=1)
        pred_sim = torch.mm(proto_sim, proto_sim.t())
        consistency_loss = 0.0
        count = 0
        
        for i in range(batch_size):
            for j in range(i+1, batch_size):
                if query_sim[i, j] > 0.7:  
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
        dists = euclidean_dist(q_re, prototypes)
        log_p_y = F.log_softmax(-dists, dim=1)
        query_labels = labels[support_size:]
        query_labels = torch.tensor(query_labels, dtype=float).to(device)
        classification_loss = - query_labels * log_p_y
        classification_loss = classification_loss.mean()
        generate_labels = torch.tensor(range(self.args.numNWay)).unsqueeze(dim=1)
        generate_labels = generate_labels.repeat(1, sampled_data.shape[1]).view(-1)
        generate_labels = F.one_hot(generate_labels, self.args.numNWay).float().to(device)
        sampled_data_flat = sampled_data.view(-1, sampled_data.shape[2])
        g_dists = euclidean_dist(sampled_data_flat, original_prototypes)
        glog_p_y = F.log_softmax(-g_dists, dim=1)
        generation_loss = - generate_labels * glog_p_y
        generation_loss = generation_loss.mean()
        domain_loss = torch.tensor(0.0).to(device)
        if domain_agnostic_features is not None and domain_info is not None:      
            pass

        consistency_loss = self.compute_consistency_loss(q_re, prototypes)
        total_loss = (classification_loss +
                      0.1 * generation_loss +
                      self.domain_loss_weight * domain_loss +
                      self.consistency_weight * consistency_loss)  
        metrics = self._compute_metrics(log_p_y, query_labels, dists)

        return total_loss, metrics['p'], metrics['r'], metrics['f'], metrics['acc'], metrics['auc'], topk_acc

    def _compute_metrics(self, log_p_y, query_labels, dists):
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
        reversed_features = gradient_reversal(domain_agnostic_features, alpha)
        domain_predictions = self.domain_classifier(reversed_features)
        domain_loss = self.criterion(domain_predictions, domain_labels)
        return domain_loss


class MetaTaskLoss(nn.Module):
    def __init__(self, args):
        super(MetaTaskLoss, self).__init__()
        self.args = args
        self.task_adaptation_weight = getattr(args, 'task_adaptation_weight', 0.1)

    def forward(self, task_outputs, task_labels):
        total_loss = 0.0
        task_count = 0

        for task_output, task_label in zip(task_outputs, task_labels):
            task_loss = self._compute_single_task_loss(task_output, task_label)
            total_loss += task_loss
            task_count += 1

        return total_loss / task_count if task_count > 0 else torch.tensor(0.0).to(device)

    def _compute_single_task_loss(self, task_output, task_label):
        return F.cross_entropy(task_output, task_label)


class Loss_fn(CrossDomainLoss):
    def __init__(self, args):
        super(Loss_fn, self).__init__(args)
