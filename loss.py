import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class MaskedNLLLoss(nn.Module):

    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight,
                               reduction='sum')

    def forward(self, pred, target, mask=None):
        """
        pred -> batch*seq_len, n_classes
        target -> batch*seq_len
        mask -> batch, seq_len
        """
        if mask!=None:
            mask_ = mask.view(-1, 1)
            if type(self.weight) == type(None):
                loss = self.loss(pred * mask_, target) / torch.sum(mask)
            else:
                loss = self.loss(pred * mask_, target) \
                    / torch.sum(self.weight[target] * mask_.squeeze())
        else:
            if type(self.weight) == type(None):
                loss = self.loss(pred, target)
            else:
                loss = self.loss(pred, target) \
                    / torch.sum(self.weight[target])
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target, mask=None):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)
            input = input.transpose(1, 2)
            input = input.contiguous().view(-1, input.size(2))
        target = target.view(-1, 1)

        logpt = input.gather(1, target).view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


class Focal_Loss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2, num_classes=5, size_average=True):

        super(Focal_Loss,self).__init__()
        self.size_average = size_average
        if isinstance(alpha,list):
            assert len(alpha)==num_classes 
            self.alpha = torch.Tensor(alpha)
        elif torch.is_tensor(alpha):
            self.alpha = alpha
        else:
            assert alpha<1 
            self.alpha = torch.zeros(num_classes)
            self.alpha[0] += alpha
            self.alpha[1:] += (1-alpha) 

        self.gamma = gamma

    def forward(self, preds, labels, mask=None):
       
        
        preds = preds.view(-1,preds.size(-1))
        self.alpha = self.alpha.to(preds.device)
        
        preds_logsoft = F.log_softmax(preds, dim=1) 
        
        preds_softmax = torch.exp(preds_logsoft)    
   
        preds_softmax = preds_softmax.gather(1,labels.view(-1,1)) 
        preds_logsoft = preds_logsoft.gather(1,labels.view(-1,1))

        self.alpha = self.alpha.gather(0,labels.view(-1))

        loss = -torch.mul(torch.pow((1-preds_softmax), self.gamma), preds_logsoft) 
    
        loss = torch.mul(self.alpha, loss.t())
        
        if self.size_average:
            loss = loss.mean()
        else:
            loss = loss.sum()
       
        return loss

class AutomaticWeightedLoss(nn.Module):

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

def make_positions(tensor, padding_idx):

    mask = tensor.ne(padding_idx).int()
    return (
        torch.cumsum(mask, dim=1).type_as(mask) * mask
    ).long() + padding_idx