import torch
import torch.nn as nn
import random
import torch.nn.functional as F
import time
class PartAttack():
    def __init__(self, judger, categ2pids = None, num_class = None, c_classes = None, eps = 1 / 255, alpha = 1 / (255 * 8)):
        self.judger = judger
        self.eps = eps
        self.alpha = alpha
        self.num_class = num_class
        self.c_classes = c_classes
        self.categ2pids = categ2pids
        self.indiceslist = []
        masklist = []
        for i in range(len(self.categ2pids)):
            part_ids = self.categ2pids[i]
            part_ids = torch.tensor(part_ids)
            mask = torch.zeros(num_class)
            mask[part_ids] += 1
            masklist.append(mask.unsqueeze(1))
        self.mask = torch.cat(masklist, dim = 1) # [num_classes, category]
        self.l = torch.nn.CrossEntropyLoss()

    def opt_attack(self, model, imgs, cids, random_start = False, 
                        return_single_grad = False, num_steps = 40):
        model.eval()
        adv_imgs = imgs.clone().detach()
        if random_start:
            adv_imgs = adv_imgs + torch.empty_like(adv_imgs).uniform_(-self.eps, self.eps)
            adv_imgs = torch.clamp(adv_imgs, min=0, max=1).detach()
        for j in range(num_steps):
            with torch.enable_grad():
                adv_imgs.requires_grad = True
                pred = model(adv_imgs)
                z = self.judger.eval(pred, return_grad = True)
                loss = self.l(z, cids)
                grad = torch.autograd.grad(loss, adv_imgs, retain_graph=False, create_graph=False)[0]
                if return_single_grad:
                    return grad.detach()

                adv_imgs = adv_imgs.detach() + self.alpha * grad.sign()
                delta = torch.clamp(adv_imgs - imgs, min=-self.eps, max=self.eps)
                adv_imgs = torch.clamp(imgs + delta, min=0, max=1).detach()
        return adv_imgs
    

    def DAG_attack(self, model, imgs, gt_label, cids, random_start = False, num_steps = 40, 
                        attacktype = "targeted", return_single_grad = False):    
        model.eval()
        wrong_label = self.__get_wrong_label(gt_label, cids, attacktype)
        adv_imgs = imgs.clone().detach()


        if random_start:
            adv_imgs = adv_imgs + torch.empty_like(adv_imgs).uniform_(-self.eps, self.eps)
            adv_imgs = torch.clamp(adv_imgs, min=0, max=1).detach()

        one_hot = F.one_hot(cids, num_classes=self.c_classes)
        for j in range(num_steps):
            with torch.enable_grad():
                adv_imgs.requires_grad = True
                pred = model(adv_imgs)
            
                fgt_mask = F.one_hot(gt_label, self.num_class).transpose(2,3).transpose(1,2)
                fgt = torch.mean(fgt_mask * pred,dim=1)

                fwr_mask = F.one_hot(wrong_label, self.num_class).transpose(2,3).transpose(1,2)
                fwr = torch.mean(fwr_mask * pred,dim=1)
                if attacktype == "untargeted":
                    loss = torch.sum((-fgt))
                else:
                    loss = torch.sum((fwr-fgt))
                grad = torch.autograd.grad(loss, adv_imgs, retain_graph=False, create_graph=False)[0]

                
                if return_single_grad:
                    return grad.detach()
                
                adv_imgs = adv_imgs.detach() + self.alpha * grad.sign()
                delta = torch.clamp(adv_imgs - imgs, min=-self.eps, max=self.eps)
                adv_imgs = torch.clamp(imgs + delta, min=0, max=1).detach()

        return adv_imgs
    

    def __get_wrong_label(self, gt_label, ids, attacktype):
        x = gt_label
        if attacktype == "targeted":
            x = x.clone()
            xd = x.clone()
            ss = [i for i in range(x.shape[0])]
            for i in range(x.shape[0]):
                random.shuffle(ss)
                flag = False
                for j in ss:
                    if ids[i] != ids[j]:
                        x[i] = xd[j]
                        flag = True
                        break
                if not flag:
                    x[i] = 0
                    print("zero attack")
            return x
        elif attacktype == "background" or attacktype == "untargeted":
            return x - x
        elif attacktype == "random":
            target = torch.rand(x.shape).to(x.device) * self.num_class
            return (1-(x==0).to(torch.int64)) * target.long()

    
