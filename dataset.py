import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
import cv2
import pickle
from torch.utils import data
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from rle.np_impl import dense_to_rle, rle_length, rle_to_dense
from PIL import Image
from pycocotools.coco import COCO

class RandomResizedCrop(transforms.transforms.RandomResizedCrop):
    def __init__(self,size, scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.), interpolation=Image.BILINEAR, interpolation_label = Image.NEAREST):
        super().__init__(size,scale,ratio,interpolation)
        self.interpolation_label = interpolation_label

    def crop(self,img,label,train):
        img = torch.tensor(img)
        # H, W = img.shape[0], img.shape[1]
        out_img = img.transpose(1,2).transpose(0,1)
        i, j, h, w = self.get_params(out_img, self.scale, self.ratio)
        if not train:
            i,j = 0,0
            h,w = out_img.size()[1],out_img.size()[2]
        out_img = F.resized_crop(out_img, i, j, h, w, self.size, self.interpolation)
        if label is not None:
            # print("before: ", label.shape)
            # label = cv2.resize(label ,(W, H),interpolation=cv2.INTER_NEAREST)
            # print("after: ", label.shape)
            label = torch.tensor(label)
            out_label = label.unsqueeze(dim=0)
            out_label = F.resized_crop(out_label, i, j, h, w, self.size, self.interpolation_label)
            return out_img.transpose(0,1).transpose(1,2).numpy().astype(np.float32),out_label.squeeze(dim=0).numpy().astype(np.float32)
        else:
            return out_img.transpose(0,1).transpose(1,2).numpy().astype(np.float32)



class VOCDataset(data.Dataset):
    def __init__(self, root, list_path, Part_Seg_folder, mode = "train", 
                        crop_size=(350, 450), output_size=(224,224), scale=True, 
                        mirror=True, ignore_label=255, other_data = None, corruption = None):
        self.root = root 
        self.list_path = os.path.join(root, list_path)
        self.mode = mode
        self.corruption = corruption

        self.img_ids = [i_id.strip() for i_id in open(self.list_path)]
        
        
        self.crop_h, self.crop_w = crop_size
        self.output_size = output_size
        self.scale = scale
        self.is_mirror = mirror
        self.ignore_label = ignore_label

        self.files = []
        for name in self.img_ids:
            if other_data == None:
                img_file = osp.join(self.root, "JPEGImages_Aug/%s.jpg" % name)
                if not osp.exists(img_file):
                    img_file = osp.join(self.root, "JPEGImages_Aug/%s.JPEG" % name)
            else:
                img_file = osp.join(other_data, "%s.jpg" % name)
                if not osp.exists(img_file):
                    img_file = osp.join(other_data, "%s.JPEG" % name)
                    
            label_file = osp.join(self.root, "singleAnnos_Aug/%s.pkl" % name)
            
            with open(label_file,'rb') as f:
                label_data = pickle.load(f)
                assert len(label_data)==1
            if label_data[0]['label'] not in ["dog",'person','cat','aeroplane','car','sheep','train','horse','bird','bus']:
                continue
            
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name,
                "has_part": not img_file.split('/')[-1].startswith('IN')
            })

        # load part labels
        f = open(osp.join(self.root, "ImageSets/{}/partlabel.txt".format(Part_Seg_folder)), "r")
        self.plabels = list(eval(f.read()))
        f.close()

        # load category labels
        f = open(osp.join(self.root, "ImageSets/{}/categorylabel.txt".format(Part_Seg_folder)), "r")
        self.clabels = list(eval(f.read()))
        f.close()
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)
        if self.corruption != None:
            from imagecorruptions import corrupt
            image = corrupt(image, corruption_name=self.corruption[0], severity=self.corruption[1])


        cid, name, label = self.loadlabel(datafiles["label"],datafiles['has_part'])

    
        Resized_Crop = RandomResizedCrop(self.output_size)
        image,label = Resized_Crop.crop(image, label, self.mode=='train')

        image = image.transpose((2, 0, 1)) # change to BGR

        # for mirror
        if self.is_mirror and self.mode == "train":
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        image = image / 255 # [0,255]->[0,1]
        
        return image.copy(), label.copy(), cid, name, datafiles['has_part']


    def loadlabel(self, name, has_part):
        from collections import defaultdict
        with open(name,'rb') as f:
            data = pickle.load(f)
            shape = data[0]['obj_mask_shape']
            c = data[0]['label']
            label = np.zeros(shape)
            label -= 1 # -1
            for i in range(len(data)):
                cid = self.clabels.index(data[i]['label'])
                parts = data[i]['parts']
                for key in parts:
                    pid = self.plabels.index(c + "_" + key)+1
                    for a in parts[key]:
                        pmask = rle_to_dense(a)
                        pmask = pmask.reshape(shape)
                        label[pmask > 0] = pid
            if not has_part:
                label[label < -1e-4] = self.ignore_label # IGNORE_LABEL
            else:
                label[label < -1e-4] = 0 # background  
        return cid, name, label



class FaceVOCDataset(data.Dataset):
    def __init__(self, root, list_path, Part_Seg_folder, mode = "train", crop_size=(350, 450), output_size=(224,224), scale=True, mirror=True, ignore_label=255, other_data = None):
        self.root = root 
        self.list_path = os.path.join(root, list_path)
        self.mode = mode

        self.img_ids = [i_id.strip() for i_id in open(self.list_path)]
        
        
        self.crop_h, self.crop_w = crop_size
        self.output_size = output_size
        self.scale = scale
        self.is_mirror = mirror
        self.ignore_label = ignore_label

        self.files = []
        for name in self.img_ids:
            if other_data == None:
                img_file = osp.join(self.root, "Images/%s.jpg" % name)
                if not osp.exists(img_file):
                    img_file = osp.join(self.root, "Images/%s.JPEG" % name)
            else:
                img_file = osp.join(other_data, "%s.jpg" % name)
                if not osp.exists(img_file):
                    img_file = osp.join(other_data, "%s.JPEG" % name)
                    
            label_file = osp.join(self.root, "Anno/%s.pkl" % name)
            
            with open(label_file,'rb') as f:
                label_data = pickle.load(f)
                assert len(label_data)==1
            
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name,
                "has_part": not img_file.split('/')[-1].startswith('IN')
            })

        # load part labels
        f = open(osp.join(self.root, "partlabel.txt".format(Part_Seg_folder)), "r")
        self.plabels = list(eval(f.read()))
        f.close()

        # load category labels
        f = open(osp.join(self.root, "categorylabel.txt".format(Part_Seg_folder)), "r")
        self.clabels = list(eval(f.read()))
        f.close()
    def __len__(self):
        return len(self.files)
    def __getitem__(self, index):
        datafiles = self.files[index]
        image = cv2.imread(datafiles["img"], cv2.IMREAD_COLOR)


        cid, name, label = self.loadlabel(datafiles["label"],datafiles['has_part'])

    
        Resized_Crop = RandomResizedCrop(self.output_size)
        image,label = Resized_Crop.crop(image, label, self.mode=='train')

        image = image.transpose((2, 0, 1)) # change to BGR

        # for mirror
        if self.is_mirror and self.mode == "train":
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        image = image / 255 # [0,255]->[0,1]
        
        return image.copy(), label.copy(), cid, name, datafiles['has_part']


    def loadlabel(self, name, has_part):
        from collections import defaultdict
        with open(name,'rb') as f:
            data = pickle.load(f)
            shape = data[0]['obj_mask_shape']
            c = data[0]['label']
            label = np.zeros(shape)
            label -= 1 # -1
            for i in range(len(data)):
                cid = self.clabels.index(data[i]['label'])
                parts = data[i]['parts']
                for key in parts:
                    pid = self.plabels.index(c + "_" + key)+1
                    for a in parts[key]:
                        pmask = rle_to_dense(a)
                        pmask = pmask.reshape(shape)
                        label[pmask > 0] = pid
            if not has_part:
                label[label < -1e-4] = self.ignore_label # IGNORE_LABEL
            else:
                label[label < -1e-4] = 0 # background  
        return cid, name, label
                    


class CocoDataset(data.Dataset):
    def __init__(self, root, output_size=(224,224), mode = 'train', anns = "allannotations", supercategory = False, other_data = None, corruption = None, fewshot = False):
        self.root = root
        assert(mode in ['train', 'test', 'test_imagenet'])
        self.mode = mode
        self.output_size = output_size
        self.alldata = ["train", "val", "test"]
        self.cocolist = []
        self.split = []
        self.imgscounts = 0
        self.supercategory = supercategory
        self.other_data = other_data
        self.corruption = corruption
        self.fewshot = fewshot
        if self.fewshot:
            f = open(os.path.join(self.root, anns, "fewshotc.txt"), "r")
            self.fewshotc = eval(f.read())
            f.close()

            f = open(os.path.join(self.root, anns, "namedict.txt"), "r")
            self.namedict = eval(f.read())
            f.close()


        if self.mode == "train":
            self.Resized_Crop = RandomResizedCrop(self.output_size)
        else:
            if self.other_data == None:
                self.Resized_Crop = RandomResizedCrop((256,256))
            else:
                self.Resized_Crop = RandomResizedCrop((224,224))

        if self.mode == "train":
            sp = "9"
        else:
            sp = "1"
        for c in self.alldata:
            annFile = os.path.join(self.root, anns, "%s_0_%s.json" % (c, sp))
            coco = COCO(annFile)
            self.cocolist.append(coco)
            self.imgscounts += len(coco.imgs)
            self.split.append(self.imgscounts)
        
        if not self.supercategory:
            f = open(os.path.join(self.root, anns, "categorylabel.txt")) # 125
            self.categorylabel = eval(f.read())
            f.close()

            f = open(os.path.join(self.root, anns, "partlabel.txt")) # 503
            self.partlabel = eval(f.read())
            f.close()
        else:
            f = open(os.path.join(self.root, anns, "supercategory.txt")) # 8
            self.categorylabel = eval(f.read())
            f.close()

            f = open(os.path.join(self.root, anns, "categorytosuper.txt")) # 8
            self.catetosuper = eval(f.read())
            f.close()

            f = open(os.path.join(self.root, anns, "superpartlabel.txt")) # 33
            self.partlabel = eval(f.read())
            f.close()
        if self.mode == "test_imagenet":
            f = open(os.path.join(self.root, anns, "imagenet_a_plus.txt"))
            self.testlist = eval(f.read())
            f.close()
            self.imgscounts = len(self.testlist)
    def __len__(self):
        if self.mode == "train":
            return self.imgscounts
        else:
            # return len(self.testlist)
            return self.imgscounts
    
    def __getitem__(self, index):
        # coco index get
        if self.mode == "train" or self.mode == "test":
            if index < self.split[0]:
                coco = self.cocolist[0]
                foldname = self.alldata[0]
            elif index < self.split[1]:
                coco = self.cocolist[1]
                index = index - self.split[0]
                foldname = self.alldata[1]
            else:
                coco = self.cocolist[2]
                index = index - self.split[1]
                foldname = self.alldata[2]
            if self.fewshot:
                cid, imgpath, image, label, is_few = self.load(coco, index, foldname)
            else:
                cid, imgpath, image, label = self.load(coco, index, foldname)
        else:
            cid, imgpath, image, label = self.loadtest(index)

        if self.mode == "train":
            image, label = self.Resized_Crop.crop(image, label, self.mode=='train')
        elif self.mode == "test":
            image, label = self.Resized_Crop.crop(image, label, self.mode=='train')
            if self.other_data == None:
                image = image[16:240,16:240,:]
                label = label[16:240,16:240]
        else:
            image = cv2.resize(image, (256, 256)).astype(np.float32)
            image = image[16:240,16:240,:]
            label = cv2.resize(label, (256, 256)).astype(np.float32)
            label = label[16:240,16:240]

        image = image.transpose((2, 0, 1)) # change to BGR

        # for mirror
        if self.mode == "train":
            flip = np.random.choice(2) * 2 - 1
            image = image[:, :, ::flip]
            label = label[:, ::flip]

        image = image / 255 # [0,255]->[0,1]
        
        if self.mode == "train":
            return image.copy(), label.copy(), cid, imgpath, is_few # has part token
        else:
            return image.copy(), label.copy(), cid, imgpath, 1
        
    def load(self, coco, index, foldname):
        infos = coco.loadImgs(index)[0]
        
        imgname = infos['file_name'] # e.g., n04252225_8354.JPEG
        category = imgname.split("_")[0]

        # if category in self.fewshotc:
        #     if len(self.namedict[category]) < 10:
        #         self.namedict[category].append(imgname)

        if self.other_data == None:
            imgpath = os.path.join(self.root, foldname, category, imgname)
        else:
            imgpath = os.path.join(self.other_data, imgname.split(".")[0] + ".jpg")
        image = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        if self.corruption != None:
            from imagecorruptions import corrupt
            image = corrupt(image, corruption_name=self.corruption[0], severity=self.corruption[1])

        if not self.supercategory:
            cid = self.categorylabel.index(category)
        else:
            cid = self.categorylabel.index(self.catetosuper[category])

        annos = coco.loadAnns(coco.getAnnIds(imgIds=index))
        label = np.zeros((infos['height'], infos['width']))
        for anno in annos:
            partname = coco.cats[anno['category_id']]['name'] # e.g., Biped Hand
            if len(anno['segmentation'][0]) == 4:
                h = anno['segmentation'][0]
                anno['segmentation'][0].append(h[0])
                anno['segmentation'][0].append(h[3])
                anno['segmentation'][0].append(h[2])
                anno['segmentation'][0].append(h[1])
            elif len(anno['segmentation'][0]) < 4:
                continue
            seglabel = coco.annToMask(anno)
            if not self.supercategory:
                fullname = category + "_" + partname
                segid = self.partlabel.index(fullname)
            else:
                segid = self.partlabel.index(partname)
            label[seglabel > 0] = segid + 1
        if self.fewshot:
            if category in self.fewshotc and imgname not in self.namedict[category]:
                return cid, imgpath, image, label, False
            else:
                return cid, imgpath, image, label, True
        else:
            return cid, imgpath, image, label
    
    def loadtest(self, index):
        imgpath = os.path.join(self.root, "imagenettest", self.testlist[index])
        image = cv2.imread(imgpath, cv2.IMREAD_COLOR)
        category = self.testlist[index].split("/")[0]
        if not self.supercategory:
            cid = self.categorylabel.index(category)
        else:
            cid = self.categorylabel.index(self.catetosuper[category])
        label = np.zeros((image.shape[0], image.shape[1]))
        label += 255 # ignore label
        return cid, imgpath, image, label

