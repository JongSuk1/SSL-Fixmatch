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

from ImageDataLoader import SimpleImageLoader, FixMatchImageLoader
from models import Res18, Res50, Dense121, Res18_basic
from randaugment import RandAugmentMC

import nsml
from nsml import DATASET_PATH


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

    ids_l = np.array(ids_l)

    return ids_l

labeled = split_ids(os.path.join(DATASET_PATH, 'train/train_label'))
impath = os.path.join(DATASET_PATH, 'train/train_data')
meta_file = os.path.join(DATASET_PATH, 'train/train_label')

imnames = []
imclasses = []

with open(meta_file, 'r') as rf:
    for i, line in enumerate(rf):
        if i == 0:
            continue
        instance_id, label, file_name = line.strip().split()
        if os.path.exists(os.path.join(impath, file_name)):
            imnames.append(file_name)
            imclasses.append(int(label))
countlabel = [0] * 265
for label in imclasses:
    if label != -1:
        countlabel[label] += 1
print(countlabel)