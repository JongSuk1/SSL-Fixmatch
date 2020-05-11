from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import os.path
import argparse

from PIL import Image

import numpy as np
import shutil
import random
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.utils.data

import torchvision
from torchvision import datasets, models, transforms
#from tensorboardX import SummaryWriter

import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

from ImageDataLoader import SimpleImageLoader, FixMatchImageLoader
from models import Res18, Res50, Dense121, Res18_basic
from randaugment import RandAugmentMC

import nsml

DATASET_PATH = '/workspace/cs492h-ssl/meta/'

def split_ids(path):
    with open(path) as f:
        ids_l = []
        ids_u = []
        for i, line in enumerate(f.readlines()):
            if i == 0 or line == '' or line == '\n':
                continue
            line = line.replace('\n', '').split('\t')
            if int(line[1]) >= 0:
                ids_l.append(int(line[0]))
            else:
                ids_u.append(int(line[0]))

    ids_u = np.array(ids_u)
    return ids_u


os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Currently using GPU {}".format("0,1,2,3,4,5,6,7"))
    cudnn.benchmark = True
    torch.cuda.manual_seed_all(123)

unlabeled = split_ids(os.path.join(DATASET_PATH, 'train/train_label'))

imnames = []
imclasses = []

validation_loader = torch.utils.data.DataLoader(
    SimpleImageLoader(DATASET_PATH, 'unlabel', unlabeled,
                       transform=transforms.Compose([
                           transforms.Resize(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])),
                       batch_size=64, num_workers=4, pin_memory=True, drop_last=False,
                       shuffle=True)

model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=265).cuda()
model = torch.nn.DataParallel(model)  
m = torch.load('./runs/gb_tuning_f_e32.pth.tar')
model.load_state_dict(m)
model.eval()

def validation(validation_loader, model,  use_gpu):
    countlabel = [0] * 265 
            
    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader):
            inputs, labels = data
            if use_gpu :
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            preds = model(inputs)
            
            _, outputs = torch.max(preds.data.cpu(), 1)
            for label in outputs:
                #print(label)
                countlabel[label] += 1

    print(countlabel)

validation(validation_loader, model, 1)
