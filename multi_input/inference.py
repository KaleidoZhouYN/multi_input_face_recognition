import numpy as np
import torch
import cv2
import os

import sys
pwd = os.path.abspath(os.path.dirname(__file__))

insightface_path = os.path.abspath(os.path.join(pwd,'..','..','face_recognition','insightface'))
sys.path.insert(0,os.path.join(insightface_path,'recognition','arcface_torch'))

from backbones_multi import get_model
from utils_multi.utils_config import get_config

config = 'configs/ms1mv3_r100'
cfg = get_config(config)
backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size)
dict_checkpoint = torch.load(os.path.join(cfg.output, f"model.pt"))
backbone.load_state_dict(dict_checkpoint)
backbone.eval()

p1 = os.path.abspath('./demo/p1')
p2 = os.path.abspath('./demo/p2')

fp1 ,fp2 = [],[]
input3_1,input3_2 = [],[]

for i in range(3):
    path1 = os.path.join(p1,'{}.jpg'.format(i))
    img1 = cv2.imread(path1)
    input1 = torch.from_numpy(img1/255 - 0.5)[None,:].permute(0,3,1,2)
    input3_1.append(input1)
    input1 = torch.cat((input1.clone(),input1.clone(),input1.clone()),dim=1).float()
    f1 = backbone(input1)[0].detach().numpy().reshape(-1)
    #f1 = f1 / np.linalg.norm(f1)
    fp1.append(f1)

for j in range(3):
    path2 = os.path.join(p2,'{}.jpg'.format(j))
    img2 = cv2.imread(path2)
    input2 = torch.from_numpy(img2/255 - 0.5)[None,:].permute(0,3,1,2)
    input3_2.append(input2)
    input2 = torch.cat((input2.clone(),input2.clone(),input2.clone()),dim=1).float()
    f2 = backbone(input2)[0].detach().numpy().reshape(-1)  
    #f2 = f2 / np.linalg.norm(f2)
    fp2.append(f2)

for i in range(3):
    for j in range(3):
        sim = np.dot(fp1[i],fp2[j]) / (np.linalg.norm(fp1[i]) * np.linalg.norm(fp2[j]))
        print('sim for each pair p1_{} vs p2_{} : {}'.format(i,j,sim))

fc_1 = np.zeros_like(fp1[0])
for i in range(3):
    fc_1 += fp1[i]
fc_2 = np.zeros_like(fp2[0])
for j in range(3):
    fc_2 += fp2[i]
sim = np.dot(fc_1,fc_2) / (np.linalg.norm(fc_1) * np.linalg.norm(fc_2))
print('sim use add combine : {}'.format(sim))

input3_1 = torch.cat(input3_1,dim=1).float()
input3_2 = torch.cat(input3_2,dim=1).float()
f3_1 = backbone(input3_1)[0].detach().numpy().reshape(-1)
f3_2 = backbone(input3_2)[0].detach().numpy().reshape(-1)
f3_1 = f3_1 / np.linalg.norm(f3_1)
f3_2 = f3_2 / np.linalg.norm(f3_2)
sim = np.dot(f3_1,f3_2)
print('sim use multi-input : {}'.format(sim))

