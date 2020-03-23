# encoding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import random
import time
import numpy as np
import shutil

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms

import tensorflow as tf
from tqdm import tqdm

from ImageDataLoader import SimpleImageLoader, TripletImageLoader
from models import Res18, Res50, Dense121, Res18_basic

from pytorch_metric_learning import miners
from pytorch_metric_learning import losses as lossfunc

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
        
class SummaryWriter(object):
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.summary_writer = tf.compat.v1.summary.FileWriter(log_dir) 
        
    def add_scalar(self, tag, value, step):
        summary = tf.compat.v1.Summary()
        summary.value.add(tag=tag, simple_value=value)
        self.summary_writer.add_summary(summary, step)
        self.summary_writer.flush()
        
def adjust_learning_rate(opts, optimizer, epoch):
    lr = opts.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr        

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    directory = "runs/%s/" % (opts.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (opts.name) + 'model_best.pth.tar')
        
        
######################################################################
# Options
######################################################################
parser = argparse.ArgumentParser(description='Sample Product200K Training')
parser.add_argument('--datadir',default='/home1/irteam/users/shine0624/kaist-naver/kaist_naver_product200k/', type=str, help='training dir path')

parser.add_argument('--start_epoch', type=int, default=1, metavar='N', help='number of start epoch (default: 1)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 10)')

parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--batchsize', default=200, type=int, help='batchsize')

parser.add_argument('--momentum', type=float, default=0.9, metavar='LR', help=' ')
parser.add_argument('--seed', type=int, default=123, help='random seed')

parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 5e-5)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='logging training status')
parser.add_argument('--imResize', default=256, type=int, help='')
parser.add_argument('--imsize', default=224, type=int, help='')

parser.add_argument('--name',default='ResNet50_kaist_naver_prod200k_train_0.1', type=str, help='output model name')
parser.add_argument('--reuse', action='store_true', help='using pretrained net' )
parser.add_argument('--tripletmode', action='store_true', help='using pretrained net' )

parser.add_argument('--trainfile', default='kaist_naver_prod200k_class265_train01.txt', type=str, help='file name of train')
parser.add_argument('--validfile', default='kaist_naver_prod200k_class265_val.txt', type=str, help='file name of validation')
parser.add_argument('--testfile', default='kaist_naver_prod200k_class265_test.txt', type=str, help='file name of test')
parser.add_argument('--lossXent', type=float, default=1, help='lossWeight for Xent')
parser.add_argument('--lossTri', type=float, default=1, help='lossWeight for metric learning')

def main():
    global opts
    opts = parser.parse_args()
    opts.cuda = 0

    ##########################
    # Set GPU
    ##########################
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

    ##########################
    # Set dataloader
    ##########################    
    if opts.tripletmode:
        train_loader = torch.utils.data.DataLoader(
            TripletImageLoader(opts, 'train', 
                               transform=transforms.Compose([
                                   transforms.Resize(opts.imResize),
                                   transforms.RandomSizedCrop(opts.imsize),
                                   transforms.RandomHorizontalFlip(),
                                   transforms.RandomVerticalFlip(),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                               ])), batch_size=opts.batchsize, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        print('train_loader done')   

        validation_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(opts, 'validation', 
                               transform=transforms.Compose([
                                   transforms.Resize(opts.imResize),
                                   transforms.CenterCrop(opts.imsize),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                               ])), batch_size=opts.batchsize, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        print('validation_loader done')    
    else:            
        train_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(opts, 'train',
                              transform=transforms.Compose([
                                  transforms.Resize(opts.imResize),
                                  transforms.RandomResizedCrop(opts.imsize),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),
                                  transforms.ToTensor(),
                                  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])), 
            batch_size=opts.batchsize, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        print('train_loader done')            
        
        validation_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(opts, 'validation', 
                               transform=transforms.Compose([
                                   transforms.Resize(opts.imResize),
                                   transforms.CenterCrop(opts.imsize),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                               ])), batch_size=opts.batchsize, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        print('validation_loader done')       
   
    
    ##########################
    # Set model
    ##########################
    class_numbers =train_loader.dataset.classnumber
    model = Res18_basic(class_numbers)    

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))            
    
    if use_gpu: 
        model.cuda()
        
    ##########################
    # Set optimizer
    ##########################    
    writer = SummaryWriter( 'runs/%s/' % (opts.name) )    
    # optimizer = optim.Adam(model.parameters(), lr=opts.lr)
    optimizer = optim.SGD(model.parameters(), lr=opts.lr,  momentum=opts.momentum )
    
    # INSTANTIATE LOSS CLASS
    criterion = nn.CrossEntropyLoss().cuda()
   
    # INSTANTIATE STEP LEARNING SCHEDULER CLASS
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones=[50, 150], gamma=0.1)   
    
    best_acc = 0.0
    for epoch in range(opts.start_epoch, opts.epochs + 1):    
        scheduler.step()
        
        print('start training')
        loss, acc_top1, acc_top5 = train(opts, train_loader, model, criterion, optimizer, epoch, writer, use_gpu)
        is_best = acc_top1 > best_acc    
        best_acc = max(acc_top1, best_acc)    
        save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'best_prec1': best_acc,}, is_best)   
        writer.add_scalar('train_epoch/loss', loss, epoch)
        writer.add_scalar('train_epoch/acc_train_top1', acc_top1, epoch)       
        writer.add_scalar('train_epoch/acc_train_top5', acc_top5, epoch)  
        
        print('start validation')
        acc_top1_test, acc_top5_test = validation(opts, validation_loader, model, epoch, writer, use_gpu)
        writer.add_scalar('test_epoch/acc_val_top1', acc_top1_test, epoch)  
        writer.add_scalar('test_epoch/acc_val_top5', acc_top5_test, epoch)  
            
    checkpoint = torch.load('runs/%s/' % (opts.name) + 'model_best.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
                
def train(opts, train_loader, model, criterion, optimizer, epoch, writer, use_gpu):
    avg_loss = 0.0
    avg_top1 = 0.0
    avg_top5 = 0.0
    
    losses = AverageMeter()
    losses_xent = AverageMeter()
    losses_triplet = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()
    
    model.train()
    nCnt =0 
    steps = (epoch - 1) * len(train_loader)    
    
    for batch_idx, data in enumerate(tqdm(train_loader)):
        
        if opts.tripletmode:
            anc, label_anc, pos, label_pos, neg,  label_neg = data
            if use_gpu :
                anc, label_anc, pos, label_pos, neg,  label_neg = anc.cuda(), label_anc.cuda(), pos.cuda(), label_pos.cuda(), neg.cuda(),  label_neg.cuda()
            anc, label_anc, pos, label_pos, neg,  label_neg = Variable(anc),Variable(label_anc),Variable(pos),Variable(label_pos),Variable(neg),Variable(label_neg)
        else:
            anc, labels = data
            if use_gpu :
                anc, labels = anc.cuda(), labels.cuda()
            anc, labels = Variable(anc), Variable(labels)         
        
        steps += 1
        nCnt += 1 
        optimizer.zero_grad()
        
        if opts.tripletmode:        
            ##########################   
            embed_anc, preds_anc = model(anc)
            embed_pos, preds_pos = model(pos)
            embed_neg, preds_neg = model(neg)
            ##########################           
            preds_all = torch.cat((torch.cat((preds_anc, preds_pos)), preds_neg))
            label_all = torch.cat((torch.cat((label_anc, label_pos)), label_neg))
            loss_xent = criterion(preds_all, label_all)
            ##########################   
            fea_all = torch.cat((torch.cat((embed_anc, embed_pos)), embed_neg))
            loss_func_triplet = lossfunc.TripletMarginLoss(margin=0.3)       
            miner_ms = miners.MultiSimilarityMiner(epsilon=0.1)
            hard_pairs_ms = miner_ms(fea_all, label_all)                    
            loss_triplet = loss_func_triplet(fea_all, label_all, hard_pairs_ms)
            loss = opts.lossXent*loss_xent + opts.lossTri*loss_triplet
        else:      
            ##########################   
            embed, preds = model(anc)            
            loss_xent = criterion(preds, labels)        
            loss = opts.lossXent*loss_xent
        
        ##########################             
        loss.backward()
        optimizer.step()        
        if opts.tripletmode: 
            acc_top1b = top_n_accuracy_score(label_all.data.cpu().numpy(), preds_all.data.cpu().numpy(), n=1)*100
            acc_top5b = top_n_accuracy_score(label_all.data.cpu().numpy(), preds_all.data.cpu().numpy(), n=5)*100   
        else: 
            acc_top1b = top_n_accuracy_score(labels.data.cpu().numpy(), preds.data.cpu().numpy(), n=1)*100
            acc_top5b = top_n_accuracy_score(labels.data.cpu().numpy(), preds.data.cpu().numpy(), n=5)*100               
            
        ##########################           
        losses.update(loss.data.cpu(), len(anc))
        losses_xent.update(loss_xent.data.cpu(), len(anc))
        if opts.tripletmode:           
            losses_triplet.update(loss_triplet.data.cpu(), len(anc))
        acc_top1.update(torch.as_tensor(acc_top1b), len(anc))        
        acc_top5.update(torch.as_tensor(acc_top5b), len(anc))        
        avg_loss += loss.data.cpu()
        avg_top1 += acc_top1b
        avg_top5 += acc_top5b   

        if batch_idx % opts.log_interval == 0:
            print('Train Epoch:{} [{}/{}] Loss:{:.4f}({:.4f}) Top-1:{:.2f}%({:.2f}%) Top-5:{:.2f}%({:.2f}%) '.format( 
                epoch, batch_idx * len(anc), len(train_loader.dataset), 
                losses.val, losses.avg, acc_top1.val, acc_top1.avg, acc_top5.val, acc_top5.avg))            
            
        writer.add_scalar('train_step/loss', loss.data.cpu(), steps)
        writer.add_scalar('train_step/loss_Xent', loss_xent.data.cpu(), steps)
        writer.add_scalar('train_step/acc_top1', acc_top1b, steps)     
        writer.add_scalar('train_step/acc_top5', acc_top5b, steps)     
        if opts.tripletmode:           
            writer.add_scalar('train_step/loss_Trip', loss_triplet.data.cpu(), steps)   
      
    avg_loss =  float(avg_loss/nCnt)
    avg_top1 = float(avg_top1/nCnt)
    avg_top5 = float(avg_top5/nCnt)
        
    return  avg_loss, avg_top1, avg_top5


def validation(opts, validation_loader, model, epoch, writer, use_gpu):
    model.eval()
    avg_top1= 0.0
    avg_top5 = 0.0
    nCnt =0 
    steps = (epoch - 1) * len(validation_loader)
    with torch.no_grad():
        for batch_idx, data in enumerate(tqdm(validation_loader)):
            inputs, labels = data
            if use_gpu :
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            steps += 1        
            nCnt +=1
            embed_fea, preds = model(inputs)
            acc_top1 = top_n_accuracy_score(labels.numpy(), preds.data.cpu().numpy(), n=1)*100
            acc_top5 = top_n_accuracy_score(labels.numpy(), preds.data.cpu().numpy(), n=5)*100
            avg_top1 += acc_top1
            avg_top5 += acc_top5
            writer.add_scalar('test_step/acc_val_top1', acc_top1, steps)    
            writer.add_scalar('test_step/acc_val_top5', acc_top5, steps)   
        avg_top1 = float(avg_top1/nCnt)   
        avg_top5= float(avg_top5/nCnt)   
        print('Test Epoch:{} Top1_acc_val:{:.2f}% Top5_acc_val:{:.2f}% '.format(epoch, avg_top1, avg_top5))
    return avg_top1, avg_top5

            
if __name__ == '__main__':
    main()
            