import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, gamma=0, ignore_label = None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.ce = torch.nn.CrossEntropyLoss(reduction="none")
    def forward(self, pred, gold):
        B, C, H, W = pred.shape
        pred = pred.permute(0,2,3,1)
        pred = pred.reshape(B*H*W, C)
        gold = gold.reshape(B*H*W)
        logp = self.ce(pred, gold)
        p = torch.exp(-logp)
        loss = ((1 - p) ** self.gamma) * logp
        return loss.mean()

class ConcentrationLoss(nn.Module):
    def __init__(self,device):
        super(ConcentrationLoss,self).__init__()
        self.device = device
    
    def forward(self,pred):
        device = self.device
        num_class = pred.size(1)
        score = F.softmax(pred,dim=1)
        z = torch.sum(score,dim=(2,3))+1e-8

        c_x = torch.tensor(range(pred.size(3)),dtype=torch.float32).to(device)
        c_x_ = c_x.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        c_x = torch.sum(c_x_*score,dim=(2,3))/z

        c_y = torch.tensor(range(pred.size(2)),dtype=torch.float32).to(device)
        c_y_ = c_y.unsqueeze(0).unsqueeze(0).unsqueeze(3)
        c_y = torch.sum(c_y_*score,dim=(2,3))/z

        weight = (c_x_-c_x.unsqueeze(2).unsqueeze(3))**2 + (c_y_-c_y.unsqueeze(2).unsqueeze(3))**2
        loss = torch.sum(weight*score,dim=(2,3))/z
        loss = torch.sum(loss[:,1:])
        loss = loss/pred.size(0)/(num_class-1)/pred.size(2)#/pred.size(3)
        return loss
