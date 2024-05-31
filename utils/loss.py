import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from utils.helper import *
from utils.logging_utils import AverageMeter
import math

__all__ = ['MultiLabelCrossEntropyLoss', 'MetricLoss', 'KL_u_p_loss']


class MultiLabelCrossEntropyLoss(nn.Module):
    """NLL loss with label smoothing."""

    def __init__(self, eps: float=0, alpha: float=0.2, topk_pos: int=-1, temp: float=1., **kwargs):
        """Construct LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(MultiLabelCrossEntropyLoss, self).__init__()
        self.eps = eps
        self.alpha = alpha
        self.topk_pos = topk_pos
        self.temp = temp


    def __repr__(self):
        return "MultiLabelCrossEntropyLoss(eps={}, alpha={})".format(self.eps, self.alpha)


    def forward(self, input: torch.Tensor, targets: torch.Tensor, reduction: bool = True, beta: float = None, 
    uncertainty: torch.Tensor = None, class_ratio: torch.Tensor = None, level: float = None, progress: float = None, data_label: torch.Tensor = None) -> torch.Tensor:

        N, C = input.size()
        E = self.eps

        input[input==np.inf] = -np.inf # to ignore np.inf

        if beta is not None:
            weights = torch.ones_like(targets)
            weights[targets==0] = beta
            input += torch.log(weights)/self.temp

        log_probs = F.log_softmax(input, dim=1)
        loss_ = (-targets * log_probs)
        loss_[loss_==np.inf] = 0.
        loss_[loss_==-np.inf] = 0.
        loss_[loss_.isnan()] = 0.
        loss = loss_.sum(dim=1)


        with torch.no_grad():
            non_zero_cnt = max(loss.nonzero(as_tuple=False).size(0), 1)

        if reduction:
            loss = loss.sum() / non_zero_cnt
        else:
            loss = loss


        return loss

    
class MetricLoss(nn.Module):

    def __init__(self, topk_pos=-1, topk_neg=-1, temp=1, eps=0., pair=None, loss_type=None, beta=None,
                 pos_sample_type=None, neg_sample_type=None, pos_loss_type=None, neg_loss_type=None,
                 threshold=1, **kwargs):
        super(MetricLoss, self).__init__()
        self.pair = pair
        
        self.topk_pos = self.pair.get('topk_pos') or topk_pos
        self.topk_neg = self.pair.get('topk_neg') or topk_neg
        self.pos_sample_type = self.pair.get('pos_sample_type') or pos_sample_type
        self.neg_sample_type = self.pair.get('neg_sample_type') or neg_sample_type
        
        self.temp = self.pair.get('temp') or temp
        
        self.loss_type = self.pair.get('loss_type') or loss_type
        self.pos_loss_type = self.pair.get('pos_loss_type') or pos_loss_type 
        self.neg_loss_type = self.pair.get('neg_loss_type') or neg_loss_type 
        self.beta = self.pair.get('beta') or beta # weight for negative samples. 

        self.threshold = self.pair.get('threshold') or threshold

        self.criterion = MultiLabelCrossEntropyLoss(topk_pos=topk_pos, temp=temp, eps=eps,) 


    def __set_num_classes__(self, num_classes):
        self.num_classes = num_classes

    def __repr__(self):
        return "{}(topk_pos={}, topk_neg={}, temp={}, crit={}), pair={})".format(
            type(self).__name__, self.topk_pos, self.topk_neg, self.temp, self.criterion, self.pair)

    def get_classwise_mask(self, target):
        B = target.size(0)
        classwise_mask = target.expand(B, B).eq(target.expand(B, B).T)
        return classwise_mask
    

    def get_topk_neg(self, sim, pos_mask=None, topk_neg=None, topk_pos=None, labels=None,):

        sim_neg = sim.clone()
        B = sim_neg.size(0)

        neg_loss_type = self.neg_loss_type if self.neg_loss_type else self.loss_type

        if 'unsupervised' in neg_loss_type:
            pos_mask = self.get_classwise_mask(torch.arange(B)).to(labels.device) # 1 only if the same sample
        else:
            pos_mask = self.get_classwise_mask(labels)


        if self.neg_sample_type:
            if self.neg_sample_type == 'debug':
                breakpoint()
                
            if self.neg_sample_type == 'all':
                
                sim_neg[torch.eye(B)==1] = -np.inf
                sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)[0]

            elif self.neg_sample_type == 'intra_class':
                classwise_mask = self.get_classwise_mask(labels)
                sim_neg[~classwise_mask] = -np.inf
                sim_neg[torch.eye(B)==1] = -np.inf
                
                sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)[0]

            elif self.neg_sample_type == 'intra_class_thresholding':
                classwise_mask = self.get_classwise_mask(labels)
                sim_neg[~classwise_mask] = np.inf
                sim_neg[torch.eye(B)==1] = np.inf
                
                sim_neg = torch.topk(sim_neg, min(topk_pos, B), dim=1, largest=False)[0]
                idx = sim_neg < self.threshold
                sim_neg[idx] = -1
                sim_neg[sim_neg == np.inf] = -1

            elif self.pos_sample_type == 'center':
                sim_neg[pos_mask==1] = np.nan
                sim_neg = sim_neg.nanmean(1, keepdim=True).repeat(1, topk_neg)

            elif self.neg_sample_type == 'inter_class':
                
                classwise_mask = self.get_classwise_mask(labels)
                sim_neg[classwise_mask] = -np.inf
                
                sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)[0]

            elif self.neg_sample_type == 'easy':
                sim_neg[pos_mask==1] = np.inf
                sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=False)[0]

            elif self.neg_sample_type == 'random':
                sim_neg[pos_mask==1] = np.nan
                random_neg = torch.rand_like(sim_neg)
                random_neg[pos_mask==1] = -np.inf
                random_neg_inds = torch.topk(random_neg, min(topk_neg, B), dim=1, largest=True)[1]
                sim_neg = sim_neg.gather(1, random_neg_inds)
            else:
                raise ValueError

        else:
            sim_neg[pos_mask==1] = -np.inf
            sim_neg = torch.topk(sim_neg, min(topk_neg, B), dim=1, largest=True)[0]


        return sim_neg
    

    def get_topk_pos(self, sim, topk_pos=None, labels=None, uncertainty=None):

        sim_pos = sim.clone()
        B = sim.size(0)

        pos_loss_type = self.pos_loss_type if self.pos_loss_type else self.loss_type
        if pos_loss_type == 'unsupervised':
            pos_mask = self.get_classwise_mask(torch.arange(B)).to(labels.device)
        else:
            pos_mask = self.get_classwise_mask(labels)

        if self.pos_sample_type:
            if self.pos_sample_type == 'easy': # high sim
                sim_pos[pos_mask==0] = -np.inf
                sim_pos, inds = torch.topk(sim_pos, topk_pos, dim=1, largest=True)

            elif self.pos_sample_type == 'no_grad':
                sim_pos[pos_mask==0] = np.inf
                sim_pos, inds = torch.topk(sim_pos, topk_pos, dim=1, largest=False)
                sim_pos.fill_(1)

            elif self.pos_sample_type == 'center':
                sim_pos[pos_mask==0] = np.nan
                sim_pos = sim_pos.nanmean(1, keepdim=True).repeat(1, self.topk_pos)

            elif self.pos_sample_type == 'random':
                sim_pos[pos_mask==0] = np.nan
                random_pos = torch.rand_like(sim_pos)
                random_pos[pos_mask==0] = -np.inf
                random_pos_inds = torch.topk(random_pos, topk_pos, dim=1, largest=True)[1]
                sim_pos = sim_pos.gather(1, random_pos_inds)

        else:
            sim_pos[pos_mask==0] = np.inf
            sim_pos, inds = torch.topk(sim_pos, topk_pos, dim=1, largest=False)


        return sim_pos





    def forward(self, old_feat, new_feat, target, reduction=True, pair=None, topk_pos=None, topk_neg=None, 
    uncertainty=None, class_ratio=None, level=None, progress=None, name="loss1"):

        if old_feat.dim() > 2:
            old_feat = old_feat.squeeze(-1).squeeze(-1)
        if new_feat.dim() > 2:
            new_feat = new_feat.squeeze(-1).squeeze(-1)

        B, C = new_feat.size()    

        old_feat_ = F.normalize(old_feat, p=2, dim=1)
        new_feat_ = F.normalize(new_feat, p=2, dim=1)

        sims = {}
        all_pair_types = set(self.pair.pos.split(' ') + self.pair.neg.split(' '))

        sims['oo'] = torch.mm(old_feat_, old_feat_.t()) 
        sims['no'] = torch.mm(new_feat_, old_feat_.t()) if 'no' in all_pair_types else None
        sims['on'] = torch.mm(old_feat_, new_feat_.t()) if 'on' in all_pair_types else None
        sims['nn'] = torch.mm(new_feat_, new_feat_.t()) if 'nn' in all_pair_types else None
        sims['nd'] = torch.mm(new_feat_, (new_feat_.detach().clone().t())) if 'nd' in all_pair_types else None
        sims['dn'] = torch.mm(new_feat_.detach().clone(), new_feat_.t()) if 'dn' in all_pair_types else None
        sims['dd'] = torch.mm(new_feat_.detach().clone(), (new_feat_.detach().clone().t())) if 'dd' in all_pair_types else None

        loss = 0.

        sim_poss, sim_negs = {}, {}
        ind_poss = {}

        if topk_pos is None:
            topk_pos = self.topk_pos
        if topk_neg is None:
            topk_neg = self.topk_neg
            

        neg_loss_type = self.neg_loss_type if self.neg_loss_type else self.loss_type

        for pair_type in all_pair_types:
            sim_poss[pair_type] = self.get_topk_pos(sims[pair_type], topk_pos=topk_pos, labels=target)
            sim_negs[pair_type] = self.get_topk_neg(sims[pair_type], topk_neg=topk_neg, topk_pos=topk_pos, labels=target)
            

        if pair is None:
            pair = self.pair

        pair_poss, pair_negs = [], []

        for pos_name in pair['pos'].split(' '):
            pair_poss.append(sim_poss[pos_name])

        for neg_name in pair['neg'].split(' '):
            pair_negs.append(sim_negs[neg_name])

        pair_poss = torch.cat(pair_poss, 1) # B*P
        pair_negs = torch.cat(pair_negs, 1) # B*N


        pair_poss_ = pair_poss.unsqueeze(2).repeat(1, 1, 1) # B*P*1
        pair_negs_ = pair_negs.unsqueeze(1).repeat(1, pair_poss_.shape[1] , 1)
        pair_all_ = torch.cat((pair_poss_, pair_negs_), 2) # B*P*(N+1)

        binary_zero_labels_ = torch.zeros_like(pair_all_)
        binary_zero_labels_[:, :, 0] = 1

        loss = self.criterion(
            input=pair_all_.reshape(-1, pair_all_.size(2))/self.temp,
            targets=binary_zero_labels_.reshape(-1, pair_all_.size(2)),
            reduction=reduction, beta=self.beta, class_ratio=class_ratio, level=level, progress=progress,)
        loss = loss.reshape(B, -1).mean(1)
            
        return loss





def KL_u_p_loss(outputs):
    # KL(u||p)
    num_classes = outputs.size(1)
    uniform_tensors = torch.ones(outputs.size())
    uniform_dists = torch.autograd.Variable(uniform_tensors / num_classes).cuda()
    instance_losses = F.kl_div(F.log_softmax(outputs, dim=1), uniform_dists, reduction='none').sum(dim=1)
    return instance_losses

