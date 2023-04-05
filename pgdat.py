import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import utils
import random
import torchattacks
class PGDATLoss(nn.Module):
    def __init__(self, eps, step_size, num_steps = 10, mtype = "cls",
                        categ2pids = None, attacker = None, base_criterion = None):
        super(PGDATLoss, self).__init__()
        self.mtype = mtype
        self.num_steps = num_steps
        self.base_criterion = base_criterion
        self.eps = eps
        self.step_size = step_size
        self.attacker = attacker

        self.m = nn.Softmax(dim=1)
        self.categ2pids = categ2pids
        if self.categ2pids is not None:
            num_class = 1
            for i in range(len(self.categ2pids)):
                num_class += len(self.categ2pids[i])
            masklist = []
            for i in range(len(self.categ2pids)):
                part_ids = self.categ2pids[i]
                part_ids = torch.tensor(part_ids)
                mask = torch.zeros(num_class)
                mask[part_ids] += 1
                masklist.append(mask.unsqueeze(1))
            self.mask = torch.cat(masklist, dim = 1) # [num_classes, category]
            self.l = torch.nn.CrossEntropyLoss()
    
    def forward(self, model, x_natural, labels, cids):
        if self.mtype == "cls":
            model.eval()
            x_adv = x_natural.detach() + torch.empty_like(x_natural).uniform_(-self.eps / 255, self.eps / 255)
            for _ in range(self.num_steps):
                x_adv.requires_grad_()
                with torch.enable_grad():
                    logits = model(x_adv)
                    loss = self.base_criterion(logits, cids)
                grad = torch.autograd.grad(loss, [x_adv])[0]
                x_adv = x_adv.detach() + (self.step_size / 255) * torch.sign(grad.detach())
                x_adv = torch.min(torch.max(x_adv, x_natural - self.eps / 255), x_natural + self.eps / 255)
                x_adv = torch.clamp(x_adv, 0.0, 1.0)
            
            model.train()
            adv_logits = model(x_adv)
            loss = self.base_criterion(adv_logits, cids)
            return adv_logits, loss

        elif self.mtype == "segmentation":
            model.eval()
            x_adv = self.attacker.DAG_attack(model, x_natural, labels, cids, 
                                num_steps = self.num_steps, attacktype = "random", random_start = True)
            model.train()
            pred = model(x_adv)
            loss = self.base_criterion(pred, labels)
            return pred, loss