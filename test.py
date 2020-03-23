# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import numpy as np
import shutil
import torchvision
from torchvision import datasets, models, transforms

import time
import os

import tensorflow as tf

from ImageDataLoader import TripletImageLoader, SimpleImageLoader
from models import Res18, Res50, Dense121, Res18_basic
from tqdm import tqdm

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
       
        
######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Sample Product200K Training')
parser.add_argument('--datadir',default='/home1/irteam/users/shine0624/kaist-naver/kaist_naver_product200k/', type=str, help='training dir path')

parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--batchsize', default=400, type=int, help='batchsize')

parser.add_argument('--imResize', default=256, type=int, help='')
parser.add_argument('--imsize', default=224, type=int, help='')

parser.add_argument('--reuse', action='store_true', default=True, help='using latest checkpoint (default: Ture)')
parser.add_argument('--name', default='your trained model name', type=str, help='output model name')
parser.add_argument('--testfile', default='kaist_naver_prod200k_class265_test.txt', type=str, help='path to latest checkpoint (default: none)')


def main():
    global opts
    opts = parser.parse_args()
    opts.cuda = 0

    ##########################
    # Set GPU
    ##########################
    torch.manual_seed(1)
    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_ids
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        opts.cuda = 1
        print("Currently using GPU {}".format(opts.gpu_ids))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(1)
    else:
        print("Currently using CPU (GPU is highly recommended)")

    ##########################
    # Set dataloader
    ##########################    
    test_loader = torch.utils.data.DataLoader(
        SimpleImageLoader(opts, 'test', 
                           transform=transforms.Compose([
                               transforms.Resize(opts.imResize),
                               transforms.CenterCrop(opts.imsize),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                           ])), batch_size=opts.batchsize, shuffle=False, num_workers=4, pin_memory=True)
    print('test_loader done')

    ##########################
    # Set model
    ##########################
    class_numbers =test_loader.dataset.classnumber
    model = Res18_basic(class_numbers)  

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))            
    
    if use_gpu:
        model.cuda()

    if opts.reuse:
        print("=> model loading from checkpoint '{}'".format(opts.name))
        checkpoint = torch.load('runs/%s/' % (opts.name) + 'model_best.pth.tar', encoding='latin1')
        opts.start_epoch = checkpoint['epoch']
        best_acc = checkpoint['best_prec1']
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' done from (epoch {}), best_acc:{}" .format(opts.name, checkpoint['epoch'], best_acc ))
            
    print('start test')
    acc_top1_test, acc_top5_test  = test(test_loader, model, use_gpu)
    print('Test evaluation Top1_acc_test:{:.2f}% Top5_acc_test:{:.2f}% '.format(acc_top1_test, acc_top5_test))
  
        
def test(test_loader, model, use_gpu):
    model.eval()
    avg_top1= 0.0
    avg_top5 = 0.0
    nCnt =0 

    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(test_loader)):
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
        
        avg_top1 = float(avg_top1/nCnt)    
        avg_top5= float(avg_top5/nCnt)    
        print('Test Top1_acc_test:{:.2f}% Top5_acc_test:{:.2f}% '.format(avg_top1, avg_top5))
    return avg_top1, avg_top5
         
            
if __name__ == '__main__':
    main()
            
        
