import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn
from torchmetrics.functional.regression import pearson_corrcoef

class MSECCC(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_function = torch.nn.MSELoss()

    def forward(self, features, labels):
        return (self.loss_function(features, labels) + (2 * torch.cov(torch.cat([features, labels], dim=1)) / (
                    features.var() + labels.var() + (features.mean() - labels.mean()) ** 2)) / 2).mean()


class CCC(nn.Module):

    def __init__(self):
        super().__init__()
        self.loss_function = torch.nn.MSELoss()

    def forward(self, features, labels):
        return 2 * torch.cov(torch.cat([features, labels], dim=1)) / (
                    features.var() + labels.var() + (features.mean() - labels.mean()) ** 2)


class MSE(nn.Module):

    def __init__(self, reduction='mean'):
        super().__init__()
        self.loss_function = torch.nn.MSELoss(reduction=reduction)

    def forward(self, features, labels):
        return self.loss_function(features, labels)

class CORR(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_function = pearson_corrcoef

    def forward(self, predictions, labels):
        return (1 - torch.nan_to_num(self.loss_function(predictions, labels), nan=-1.0)).mean()

class MTLoss(nn.Module):
    # cf. https://arxiv.org/pdf/1705.07115.pdf
    # the loss module learns the weigths in the linear combination of each task's loss
    # store the weights as log(std_i) -> 1/std_i**2 == exp(-2*weight_i)
    def __init__(self, num_task, loss_fn, label_smoothing=0.0):
        super(MTLoss, self).__init__()
        self.requires_grad_(True)
        self.log_std = nn.Parameter(torch.zeros(num_task))
        self.loss_fn = loss_fn.__class__(reduction='none')
        self.label_smoothing = label_smoothing
        self.weight = None
        # weights have to be set, else load_state_dict throws error
        #self.ce_country = nn.CrossEntropyLoss(reduction='none', weight=torch.tensor(tuple([1/4]*4), device=device))
        #self.ce_type = nn.CrossEntropyLoss(reduction='none', weight=torch.tensor(tuple([1/8]*8), device=device))

    def forward(self, predict: dict, target: dict):
        #loss_c = self.ce_country(predict['country'], target['country'])
        #loss_t = self.ce_type(predict['type'], target['type'])
        #loss_hi = 1.0 - self.ccc(predict['high'], target['high'])
        #loss_cu = 1.0 - self.ccc(predict['culture'], target['culture'])
        #loss_va = 1.0 - self.ccc(predict['valaro'], target['valaro'])
        #loss = torch.stack((loss_c, loss_t, loss_hi, loss_cu, loss_va))
        #loss = loss.transpose(0, 1)
        l1 = self.loss_fn(predict[0], target).mean(axis=1)
        l2 = self.loss_fn(predict[1], target).mean(axis=1)
        loss = torch.stack((l1, l2)).transpose(0, 1)
        self.weight = torch.exp(-2.0*self.log_std)
        ret = torch.matmul(loss, self.weight)
        ret += self.log_std.sum()
        ret = ret.mean()
        return {'loss': ret, 'task_loss': loss.mean(dim=0), 'task_weight': self.weight,
                'task_uncertainty': torch.exp(self.log_std),}
                #'ccc_hi': (1-loss_hi).mean(),
                #'ccc_cu': (1-loss_cu).mean(),
                #'ccc_va': (1-loss_va).mean()}

    def train(self: nn.Module, mode: bool = True) -> nn.Module:
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)

        # only during train: inversely proportional weights to counter class imbalance combined with label smoothing
        '''
        if mode:
            self.ce_country.weight = torch.tensor((0.21442458972180697, 0.21568838887537362, 0.15371799207163983,
                                                   0.41616902933117955), device=device)
            self.ce_country.label_smoothing = self.label_smoothing
            self.ce_type.weight = torch.tensor((0.09220052044246965, 0.023945658814239372, 0.12453144964594179,
                                                0.12628801797799294, 0.03443521461869565, 0.12619433250471548,
                                                0.36426115678020665, 0.1081436492157384), device=device)
            self.ce_type.label_smoothing = self.label_smoothing
        else:
            self.ce_country.weight = torch.tensor(tuple([1/4]*4), device=device)
            self.ce_country.label_smoothing = 0.0
            self.ce_type.weight = torch.tensor(tuple([1/8]*8), device=device)
            self.ce_type.label_smoothing = 0.0
        '''
        return self