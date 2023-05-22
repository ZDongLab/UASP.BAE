# -*- coding: utf-8 -*-
"""
Created by @author: Carlson Zhang.
Faculty of Electronic and Information Engineering, Xi'an Jiaotong University.
The Code compliance with LR license.
"""

from email import message
import os
import sys
from matplotlib import patches
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
import numpy as np
import albumentations as A
import random
import torch
import json
from patchify import patchify

age_names_label_dict = {
                        "a5-13":{str(n):n for n in range(5,14)},
                        "a14-24":{str(n):n for n in range(14,25)},
                        "b5-18":{str(n):n for n in range(5,19)},
                        "b19-24":{str(n):n for n in range(19,25)},
                        "5-24":{str(n):n for n in range(5,25) }}

def normalize(pixels):
    mean, std = pixels.mean(), pixels.std()
    pixels = (pixels - mean) / std
    pixels = np.clip(pixels, -1.0, 1.0)
    pixels = (pixels + 1.0) / 2.0   

    return pixels

def gaussifun(x, mean=0, sigma=0.45): ## sig^2=0.2 
    prob = 1/(sigma*np.sqrt(2*np.pi))*np.exp(-(x-mean)**2 / (2*sigma**2))
    return prob

def GussiCut(x, agerange, std=1.5, mapnum=30):                                                                        
    (start, end) = agerange                                   
    dataidx = np.arange(start, end, (end-start)/mapnum)
    data = gaussifun(dataidx, mean=x, sigma=std)
    data[data>=0.1]=1
    return (data/(data.max()),dataidx)

class toothopg(Dataset):
    def __init__(self, imgs_dir,label_csv, imagesize=(224,224), tempmemary=False,agerange=(5,25),mapnum=10,sex=None,istest=False):
        self.imgs_dir = imgs_dir
        
        self.imgsize = imagesize
        self.imgchl = 1
        self.input_transform = True
        self.tempmemary = tempmemary
        self.agerange = agerange
        self.mapnum = mapnum
        self.sex = sex
        if self.sex == "male":
            pddata = pd.read_csv(label_csv)
            self.pd_data = pddata[pddata["gender"]==1].reset_index(drop = True)
        elif self.sex == "female":
            pddata = pd.read_csv(label_csv)
            self.pd_data = pddata[pddata["gender"]==2].reset_index(drop = True)
        else:
            self.pd_data = pd.read_csv(label_csv)


        self.istest = istest
        self.istest_transformations = A.Compose([
            A.Resize(self.imgsize[0],self.imgsize[1]),
        ])
        self.transformations = A.Compose([
                A.Resize(self.imgsize[0],self.imgsize[1]),
                
                A.OneOf([
                    #A.RandomRotate90(p=0.5),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    #A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
                    #A.RandomBrightnessContrast(p=0.2),
                    #A.GridDistortion(p=0.2),
                    #A.ElasticTransform(p=0.2)
                ]), 
        ])

    def readimg(self, imgpath): 
        if os.path.exists(os.path.join(self.imgs_dir, imgpath)):
            if self.imgchl == 1:
                img = cv2.imread(os.path.join(self.imgs_dir, imgpath), cv2.IMREAD_GRAYSCALE).astype(np.float32) # [H, W, 1]
                img /=255
                img = np.expand_dims(img, -1)
            else:
                img = cv2.imread(os.path.join(self.imgs_dir, imgpath), cv2.IMREAD_COLOR).astype(np.float32) # [H, W, 3]
                img /=255
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            if self.input_transform:
                random.seed(123456)
                if self.istest:
                    transformed = self.istest_transformations(image=img)
                    img = transformed['image'] 
                else: 
                    transformed = self.transformations(image=img)
                    img = transformed['image'] 
            
            img = normalize(img)

            img = np.transpose(img, (2, 0, 1))    
            return torch.from_numpy(img)
        else:
            messagestr = "This image no exist: "+os.path.join(self.imgs_dir, imgpath)
            print(messagestr)
            raise AssertionError(messagestr)

    def __getitem__(self, index):
        labelinfo = self.pd_data.loc[index]
        img_id = "%s_OPGs.jpg"%(labelinfo["check_id"])
        gender = torch.from_numpy(np.array(labelinfo["gender"]-1))
        age = torch.from_numpy(np.array(labelinfo["age"]))
        data, dataidx = GussiCut(np.array(labelinfo["age"]), self.agerange, std=0.15, mapnum=self.mapnum)
        gsage = (torch.from_numpy(data), torch.from_numpy(dataidx))
        imgpath = os.path.join("%02d"%(int(labelinfo["age"])), "jpg/%s"%img_id)
        img = self.readimg(imgpath) 
        return img_id, gender, age, gsage, img
    
    def __len__(self):
        return len(self.pd_data)
'''
if __name__ == "__main__":
    
    imgs_dir, label_csv = "./datasets/train","./datasets/train22.csv"
    ThyroidCls = toothopg(imgs_dir, label_csv)      
    dataloaderCls = torch.utils.data.DataLoader(ThyroidCls,batch_size=12,shuffle=True,num_workers=0)   
    for index,(img_id, gender, age, img) in enumerate(dataloaderCls):
        print(img_id, gender, age, img.shape)
    
'''