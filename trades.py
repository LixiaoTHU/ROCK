import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import utils
import random
import torchattacks
class TradesLoss(nn.Module):
    def __init__(self, alpha = 6.0, num_steps = 10, eps = None, step_size = None, 
                            mtype = "cls", attacker = None, base_criterion = None, 
                            categ2pids = None, return_x = False):
        super(TradesLoss, self).__init__()
        self.alpha = alpha
        self.mtype = mtype
        self.num_steps = num_steps
        self.attacker = attacker
        self.criterion_kl = nn.KLDivLoss(reduction='sum')
        self.base_criterion = base_criterion
        self.eps = eps
        self.step_size = step_size
        self.categ2pids = categ2pids
        self.return_x = return_x
    
    def forward(self, model, x_natural, labels, cids):
        if self.mtype == "cls":
            model.eval()
            x_adv = x_natural.detach() + torch.empty_like(x_natural).uniform_(-self.eps / 255, self.eps / 255)
            logits = model(x_natural)
            for _ in range(self.num_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    loss_kl = self.criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                        F.softmax(logits, dim=1))
                grad = torch.autograd.grad(loss_kl, [x_adv])[0]
                x_adv = x_adv.detach() + (self.step_size / 255) * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.eps / 255), x_natural + self.eps / 255)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
            
            if self.return_x:
                return x_adv
            
            model.train()
            logits = model(x_natural)
            adv_logits = model(x_adv)
            loss_natural = self.base_criterion(logits, cids)
            loss_robust = (1.0 / (logits.shape[0])) * self.criterion_kl(F.log_softmax(adv_logits, dim=1),
                                                    F.softmax(logits, dim=1))
            loss = loss_natural + self.alpha * loss_robust
            return logits, loss

        elif self.mtype == "segmentation":
            x_adv = self.attacker.DAG_attack(model, x_natural, labels, cids, 
                                num_steps = self.num_steps, attacktype = "random", random_start = True)
            
            if self.return_x:
                return x_adv
            model.train()
            ori_pred = model(x_natural)
            loss_natural = self.base_criterion(ori_pred, labels)
            pred = model(x_adv)
            N, C, H, W = pred.shape
            logits = ori_pred.permute(0,2,3,1)
            logits = logits.reshape(N*H*W, C)
            adv_logits = pred.permute(0,2,3,1)
            adv_logits = adv_logits.reshape(N*H*W, C)
            loss_robust = (1.0 / (N*H*W)) * self.criterion_kl(F.log_softmax(adv_logits, dim=1),
                                                            F.softmax(logits, dim=1))
            loss = loss_natural + self.alpha * loss_robust
            return ori_pred, loss