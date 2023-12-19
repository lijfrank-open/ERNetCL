import torch
import torch.nn as nn
import numpy as np

from loss import MaskedNLLLoss

class CurrLearnLoss(nn.Module):
    def __init__(self, weight=None, sigma=0.1, delt_epoch=6, dataset=''):
        super(CurrLearnLoss, self).__init__()
        if weight is not None:
            self.weight = weight.cuda()
        if dataset=='IEMOCAP':
            self.loss = MaskedNLLLoss(self.weight)
        elif dataset=='MELD':
            self.loss = MaskedNLLLoss()
        elif dataset=='EmoryNLP':
            self.loss = MaskedNLLLoss()
        elif dataset=='DailyDialog':
            self.loss = MaskedNLLLoss()

        self.sigma = sigma
        self.delt_epoch = delt_epoch
        self.mapping_func = lambda weights_dialogue,mu,sigma:1/(1+torch.exp(-(mu-weights_dialogue)/sigma))
        
    def _make_class_utterance_weights(self, label, umask, qmask):

        _, bts = label.shape
        weights_list = []
        for b in range(bts):
            mask  = umask[:, b]==1
            label = label[:, b]
            weights_list.append(1 - (torch.bincount(label[mask])[label] + 1e-5)/mask.sum())
        weights = torch.stack(weights_list)

        return weights.transpose(0,1)
        
    def _make_class_dialogue_weights(self, label, umask, qmask):

        qmask_ = torch.argmax(qmask, -1)

        max_p, min_p = torch.max(qmask_).item(), torch.min(qmask_).item()
        assert min_p >= 0

        seq_len, batch_size = qmask_.size()
        mem_near = (-1) * torch.ones(batch_size, max_p+1, dtype=torch.int8).cuda()
        conv_id  = torch.zeros_like(qmask_)
        batch_id = torch.arange(batch_size).cuda()

        label_shift= torch.zeros_like(qmask_)

        person_emo_shift  = torch.zeros(batch_size, max_p+1, dtype=torch.float).to(qmask_.device)

        person_emo_length = torch.ones_like(person_emo_shift) * 1e-5

        for i in range(seq_len):
            conv_id[i] = mem_near[batch_id, qmask_[i]]
            mem_near[batch_id, qmask_[i]] = i

            cur_conv_id = torch.where(conv_id[i]==-1, 0, conv_id[i])

            label_shift[i] = torch.where((label[i] != label[cur_conv_id, batch_id]) | (conv_id[i]==-1), 1 ,0)

            person_emo_shift[batch_id, qmask_[i]] += label_shift[i] * (umask[i]==1).type(torch.float)
            person_emo_length[batch_id, qmask_[i]] += (umask[i]==1).type(torch.float)

        weight_class_dialogue = (person_emo_shift / person_emo_length / (person_emo_length>=1.0).type(torch.float).sum(-1, keepdim=True)).sum(dim=-1)

        return weight_class_dialogue[None, :]
    
    def curriculum_weights(self, label, umask, qmask, iterations):

        weights_dialogue  = self._make_class_dialogue_weights(label, umask, qmask) * torch.ones_like(umask)

        mu = iterations / self.delt_epoch
       
        return self.mapping_func(weights_dialogue, mu=mu, sigma=self.sigma) 

    def forward(self, prob, label, umask, qmask, mi=None, iterations=0):

        curr_weight = self.curriculum_weights(label, umask, qmask, iterations)

        prob_ = prob.view(-1, prob.size()[-1])
        label_ = label.view(-1)
        umask_ = umask.view(-1,1)
        curr_weight_  = curr_weight.view(-1,1)

        loss = self.loss(prob_*umask_*curr_weight_, label_)
        return loss