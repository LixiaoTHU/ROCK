from skimage.measure import label as sklabel
import torch
import argparse
import os
import sys
from torch.utils import data
from collections import defaultdict
import torch.nn.functional as F
import time
import torch.nn as nn
import numpy as np
import math
import time
import cv2


class Judgement():
    def __init__(self, categ2pids, pid2name, clabels, plabels, connection, stat, finegrained = False, withknowledge = True, withat = False):
        self.categ2pids = categ2pids
        self.clabels = clabels
        self.connection = connection
        self.plabels = plabels
        self.m = nn.Softmax(dim=1)
        self.stat = stat
        self.withknowledge = withknowledge
        self.withat = withat
        self.pid2name = pid2name

        self.translated_connection = {}
        self.part2c = {}
        self.weight_connection = {}

        if finegrained:
            for label in clabels:
                for part in plabels:
                    if label in part:
                        supercategory = part.split("_")[1]
                        supercategory = supercategory.split(" ")[0]
                        break
                cid = self.clabels.index(label)
                self.translated_connection[cid] = []
                pairs = self.connection[supercategory]
                for pair in pairs:
                    pid0 = self.plabels.index(label + "_" + pair[0])+1
                    pid1 = self.plabels.index(label + "_" + pair[1])+1
                    self.translated_connection[cid].append((pid0,pid1))
                    if pid0 in self.weight_connection:
                        self.weight_connection[pid0] += 1
                    else:
                        self.weight_connection[pid0] = 1
                    if pid1 in self.weight_connection:
                        self.weight_connection[pid1] += 1
                    else:
                        self.weight_connection[pid1] = 1
                    self.part2c[pid0] = cid
                    self.part2c[pid1] = cid
        else:
            for key,pairs in self.connection.items():
                cid = self.clabels.index(key)
                self.translated_connection[cid] = []
                for pair in pairs:
                    pid0 = self.plabels.index(pair[0])+1
                    pid1 = self.plabels.index(pair[1])+1
                    self.translated_connection[cid].append((pid0,pid1))
                    if pid0 in self.weight_connection:
                        self.weight_connection[pid0] += 1
                    else:
                        self.weight_connection[pid0] = 1
                    if pid1 in self.weight_connection:
                        self.weight_connection[pid1] += 1
                    else:
                        self.weight_connection[pid1] = 1
                    self.part2c[pid0] = cid
                    self.part2c[pid1] = cid
        
        num_class = len(plabels) + 1
        masklist = []
        for i in range(len(self.categ2pids)):
            part_ids = self.categ2pids[i]
            part_ids = torch.tensor(part_ids)
            mask = torch.zeros(num_class)
            mask[part_ids] += 1
            masklist.append(mask.unsqueeze(1))
        self.mask = torch.cat(masklist, dim = 1) # [num_classes, category]


    def componentscore(self, indice, cid, part_ids, n, pred): #H W
        maxi = len(part_ids)
        allblocks = []
        for i in range(1, 1 + maxi):
            imap = indice.clone().numpy()
            imap[imap != i] = 0
            label_img, num = sklabel(imap, connectivity=1, background=0, return_num=True)
            max_num = 0
            max_label = 0
            if num > 0:
                for k in range(1, num+1):
                    t = np.sum(label_img == k)
                    if t > max_num:
                        max_label = k
                        max_num = t
                lcc = (label_img == max_label)
                allblocks.append(lcc)
            else:
                allblocks.append(None)

        total = 0.0
        score = 0.0
        for group in self.translated_connection[cid]:
            bid1 = part_ids.index(group[0])
            bid2 = part_ids.index(group[1])
            t1 = allblocks[bid1]
            t2 = allblocks[bid2]

            index = (self.pid2name[group[0]], self.pid2name[group[1]])
            if index in self.stat:
                total += self.stat[index]
            imap = np.zeros_like(imap)
            if t1 is not None and t2 is not None and np.sum(t1) > 1 and np.sum(t2) > 1:
                imap[t1] = 1
                imap[t2] = 1
                label_img, num = sklabel(imap, connectivity=1, background=0, return_num=True)
                if num == 1:
                    mask1 = torch.tensor(t1).to(pred.device)
                    mask2 = torch.tensor(t2).to(pred.device)
                    d1 = self.weight_connection[group[0]]
                    d2 = self.weight_connection[group[1]]
                    if index in self.stat:
                        score +=  self.stat[index] * ( (torch.sum(pred[n, group[0]] * mask1) / d1) + (torch.sum(pred[n, group[1]] * mask2) / d2) )
        return score / total

    def showpart(self, bg, cpred, part_ids, raw_imgs, batchid = 9):
        up = nn.Upsample(size=(224, 224), mode='bilinear')
        cpred = up(cpred)
        bg = up(bg.unsqueeze(1))[:,0]
        values, indices = cpred.max(1)
        allc = torch.sum(cpred, dim = 1)
        mask = allc > bg
        indices = (indices + 1) * mask

        color = { #BGR
            'red':(0,0,255),
            'blue':(255,0,0),
            'green':(0,255,0),
            'yellow':(0,255,255),
            'pink':(255,0,255),
            'violet': (226,43,138)
        }
        color_list = list(color.keys())

        raw_imgs = raw_imgs * 255
        raw_img = raw_imgs[batchid]
        raw_img = raw_img.permute(1,2,0)
        raw_img = raw_img.cpu().numpy().astype(np.uint8)
        indices = indices[batchid]
        indices = indices.cpu().numpy().astype(np.uint8)
        cv2.imwrite('imgs/raw.png',raw_img)
        
        raw_black = (np.ones(raw_img.shape)*0).astype(np.uint8)
        ctoname = []
        ctoname.append(self.pid2name[part_ids[0]].split("_")[0])
        for idx in range(len(part_ids)):
            mask = (indices==idx+1).astype(np.int32)
            mask = np.expand_dims(mask,axis=-1)
            mask_color = color_list[idx]
            color_BGR = color[mask_color]
            ch0 = (mask*color_BGR[0]).astype(np.uint8)
            ch1 = (mask*color_BGR[1]).astype(np.uint8)
            ch2 = (mask*color_BGR[2]).astype(np.uint8)
            mask = np.concatenate((ch0,ch1,ch2),axis=2)
            raw_img = cv2.addWeighted(raw_img, 1, mask, 0.5, 0)
            raw_black = cv2.addWeighted(raw_black, 1, mask, 0.5, 0)
            ctoname.append((color_list[idx], self.pid2name[part_ids[idx]].split("_")[1]))
        
        cv2.imwrite('imgs/masked_black_{}.png'.format(self.pid2name[part_ids[0]].split("_")[0]),raw_black)
        print(ctoname)
            
    
    def eval(self, pred, k = 10, raw_imgs = None, return_grad = False):
        k = min(k, len(self.categ2pids))
        pred = self.m(pred) # N, C, H, W
        bg = pred[:,0]
        indiceslist = []
        
        P_c = []
        num_pixels = []
        for i in range(len(self.categ2pids)):
            part_ids = self.categ2pids[i]
            cpred = pred[:,part_ids]
            values, indices = cpred.max(1)
            allc = torch.sum(cpred, dim = 1)
            if self.withat:
                mask = allc > (bg / math.sqrt(len(self.categ2pids)))
            else:
                mask = allc > (bg)

            P_c.append(torch.sum(allc, dim=[1,2]))

            indices = (indices + 1) * mask
            num_pixels.append(torch.sum(mask, dim=[1,2]))
            indices = indices.cpu()
            if raw_imgs != None:
                self.showpart(bg, cpred, part_ids, raw_imgs)
            indiceslist.append(indices)
        num_pixels = torch.stack(num_pixels, 1)

        N = pred.shape[0]
        score = 0 * pred[:,:len(self.categ2pids),0,0]  # init
        for n in range(N):
            np = num_pixels[n] # C
            _, maxk = torch.topk(np, k)
            for i in range(k):
                index = maxk[i].item()
                indice = indiceslist[index][n]
                part_ids = self.categ2pids[index]
                Q_c = self.componentscore(indice, index, part_ids, n, pred)
                score[n, index] += Q_c
        score = score / (torch.sum(score, dim = 1).unsqueeze(1).detach() + 1e-20)
        return score