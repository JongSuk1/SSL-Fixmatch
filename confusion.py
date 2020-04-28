from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import argparse

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

import seaborn as sns
import matplotlib.pyplot as plt


import torchvision
from torchvision import datasets, models, transforms
#from tensorboardX import SummaryWriter

import torch.nn.functional as F

from ImageDataLoader import SimpleImageLoader, FixMatchImageLoader
from models import Res18, Res50, Dense121, Res18_basic, weights_init_classifier
from randaugment import RandAugmentMC


NUM_CLASSES = 265
DATASET_PATH = '/workspace/cs492h-ssl/meta/'
    


def top_n_accuracy_score(y_true, y_prob, n=5, normalize=True):
    num_obs, num_labels = y_prob.shape
    idx = num_labels - n - 1
    counter = 0
    argsorted = np.argsort(y_prob, axis=1)
    for i in range(num_obs):
        if y_true[i] in argsorted[i, idx+1:]:
            counter += 1
    if normalize:
        return counter * 1.0 / num_obs
    else:
        return counter
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def split_ids(path):
    with open(path) as f:
        ids_l = []
        for i, line in enumerate(f.readlines()):
            if i == 0 or line == '' or line == '\n':
                continue
            line = line.replace('\n', '').split('\t')
            if int(line[1]) >= 0:
                ids_l.append(int(line[0]))

    ids_l = np.array(ids_l)

    perm = np.random.permutation(np.arange(len(ids_l)))
    val_ids = ids_l[perm][:]

    return val_ids


def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        no_progress = float(current_step) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))
    return LambdaLR(optimizer, _lr_lambda)

######################################################################
# Options
######################################################################
parser = argparse.ArgumentParser(description='Sample Product200K Training')
parser.add_argument('--start_epoch', type=int, default=1, metavar='N', help='number of start epoch (default: 1)')
parser.add_argument('--epochs', type=int, default=300, metavar='N', help='number of epochs to train (default: 200)')

# basic settings
parser.add_argument('--name',default='HA_trial3', type=str, help='output model name')
parser.add_argument('--gpu_ids',default='0,1,2,3,4,5,6,7', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
parser.add_argument('--seed', type=int, default=123, help='random seed')

# basic hyper-parameters
parser.add_argument('--momentum', type=float, default=0.9, metavar='LR', help=' ')
parser.add_argument('--lr', type=float, default=0.03, metavar='LR', help='learning rate (default: 5e-5)')
parser.add_argument('--imResize', default=256, type=int, help='')
parser.add_argument('--imsize', default=224, type=int, help='')
parser.add_argument('--lossXent', type=float, default=1, help='lossWeight for Xent')

# arguments for logging and backup
parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='logging training status')
parser.add_argument('--save_epoch', type=int, default=50, help='saving epoch interval')

# hyper-parameters for fixmatch
parser.add_argument('--lambda-u', default=1, type=float)
parser.add_argument('--mu', default=1 , type=int, help="Batch ratio between labeled/unlabeled")
parser.add_argument('--threshold', type=float, default=0.95, help='Threshold setting for Fixmatch')


parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--val-iteration', type=int, default=100, help='Number of labeled data')

### DO NOT MODIFY THIS BLOCK ###
# arguments for nsml 
parser.add_argument('--pause', type=int, default=0)
parser.add_argument('--mode', type=str, default='train')
################################

def main():
    global opts
    opts = parser.parse_args()
    opts.cuda = 0

#    global writer
#    writer = SummaryWriter("runs/"+opts.name)
    # Set GPU
    seed = opts.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_ids
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        opts.cuda = 1
        print("Currently using GPU {}".format(opts.gpu_ids))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(seed)
    else:
        print("Currently using CPU (GPU is highly recommended)")


    # Set model
    model = Res50(NUM_CLASSES)
    
    checkpoint = torch.load("./runs/model.pt")
    model.load_state_dict(checkpoint)
    #model.classifier.apply(weights_init_classifier)
        
    if use_gpu:
        model.cuda()


    model.eval()
    # Set dataloader
    val_ids = split_ids(os.path.join(DATASET_PATH, 'train/train_label'))
    print('found {} validation images'.format(len(val_ids)))


    validation_loader = torch.utils.data.DataLoader(
        SimpleImageLoader(DATASET_PATH, 'val', val_ids,
                           transform=transforms.Compose([
                               transforms.Resize(opts.imResize),
                               transforms.CenterCrop(opts.imsize),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])),
                           batch_size=opts.batchsize, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    print('validation_loader done')



    # Validation 


    print('start validation')
    acc_top1, acc_top5 = validation(opts, validation_loader, model, use_gpu)



def validation(opts,validation_loader, model, use_gpu):
    model.eval()
    avg_top1= 0.0
    avg_top5 = 0.0
    nCnt =0
    epoch=10
    cm = np.zeros((NUM_CLASSES,NUM_CLASSES))
    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader):
            inputs, labels = data
            if use_gpu :
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            nCnt +=1
            embed_fea, preds = model(inputs)

            acc_top1 = top_n_accuracy_score(labels.numpy(), preds.data.cpu().numpy(), n=1)*100
            acc_top5 = top_n_accuracy_score(labels.numpy(), preds.data.cpu().numpy(), n=5)*100
            avg_top1 += acc_top1
            avg_top5 += acc_top5
            
            _, outputs = torch.max(preds.data.cpu(), 1)
            #print(labels.numpy())
            #print(outputs.numpy())
            for t, p in zip(labels.numpy(), outputs.numpy()):
                cm[t,p] +=1

        avg_top1 = float(avg_top1/nCnt)   
        avg_top5= float(avg_top5/nCnt)   
        print('Top1_acc_val:{:.2f}% Top5_acc_val:{:.2f}% '.format(avg_top1, avg_top5))
        #print(confusion_matrix)
        
        cm = cm.astype(float)/cm.sum(axis=1)[:,np.newaxis]

        plt.figure()
        sns.heatmap(cm,xticklabels = False, yticklabels=False)
        plt.xlabel("prediction")
        plt.ylabel("True")
        plt.savefig("./confusion_matrix/"+opts.name+"_epoch"+str(epoch)+".png")
        
    return avg_top1, avg_top5



if __name__ == '__main__':
    main()


