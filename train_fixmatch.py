# encoding: utf-8
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
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import torchvision
from torchvision import datasets, models, transforms
from tensorboardX import SummaryWriter

import torch.nn.functional as F

from ImageDataLoader import SimpleImageLoader, TripletImageLoader, FixMatchImageLoader
from models import Res18, Res50, Dense121, Res18_basic
from randaugment import RandAugmentMC

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
        
def adjust_learning_rate(opts, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = opts.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr        

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    """Saves checkpoint to disk"""
    directory = "runs/%s/" % (opts.name)
    if not os.path.exists(directory):
        os.makedirs(directory)
    filename = directory + filename
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'runs/%s/' % (opts.name) + 'model_best.pth.tar')
        
def linear_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current / rampup_length, 0.0, 1.0)
        return float(current)
        
class SemiLoss(object):
    def __call__(self, outputs_x, targets_x, outputs_u, targets_u, epoch, final_epoch):
        probs_u = torch.softmax(outputs_u, dim=1)
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        Lu = torch.mean((probs_u - targets_u)**2)
        return Lx, Lu, opts.lambda_u * linear_rampup(epoch, final_epoch)

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def get_cosine_schedule_with_warmup(optimizer,
                                    num_warmup_steps,
                                    num_training_steps,
                                    num_cycles=7./16.,
                                    last_epoch=-1):
    def _lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        no_progress = float(current_step - num_warmup_steps) / \
            float(max(1, num_training_steps - num_warmup_steps))
        return max(0., math.cos(math.pi * num_cycles * no_progress))

    return LambdaLR(optimizer, _lr_lambda, last_epoch)

######################################################################
# Options
# --------
parser = argparse.ArgumentParser(description='Sample Product200K Training')
parser.add_argument('--datadir',default='/home1/irteam/users/shine0624/kaist-naver/kaist_naver_product200k/', type=str, help='training dir path')

parser.add_argument('--start_epoch', type=int, default=1, metavar='N', help='number of start epoch (default: 1)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 10)')

parser.add_argument('--gpu_ids',default='0,1', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--batchsize', default=200, type=int, help='batchsize')

parser.add_argument('--momentum', type=float, default=0.9, metavar='LR', help=' ')
parser.add_argument('--seed', type=int, default=123, help='random seed')

parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 5e-5)')
parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='logging training status')
parser.add_argument('--imResize', default=256, type=int, help='')
parser.add_argument('--imsize', default=224, type=int, help='')

parser.add_argument('--name',default='ResNet50_kaist_naver_prod200k_train_0.1', type=str, help='output model name')
parser.add_argument('--resume', default='', type=str, help='path to latest checkpoint (default: none)')

parser.add_argument('--trainfile', default='kaist_naver_prod200k_class265_train01.txt', type=str, help='file name of train')
parser.add_argument('--validfile', default='kaist_naver_prod200k_class265_val.txt', type=str, help='file name of validation')
parser.add_argument('--testfile', default='kaist_naver_prod200k_class265_test.txt', type=str, help='file name of test')
parser.add_argument('--unlabelfile', default='kaist_naver_prod200k_class265_unlabel.txt', type=str, help='file name of test')

parser.add_argument('--lossXent', type=float, default=1, help='lossWeight for Xent')
parser.add_argument('--lossTri', type=float, default=1, help='lossWeight for metric learning')

#prams for mix-match
parser.add_argument('--alpha', default=0.75, type=float)
parser.add_argument('--lambda-u', default=1 , type=float)
parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--val-iteration', type=int, default=100, help='Number of labeled data')
parser.add_argument('--threshold', type=float, default=0.95, help='Threshold setting for Fixmatch')


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

    unlabel_loader = torch.utils.data.DataLoader(
        FixMatchImageLoader(opts, 'unlabel',
                          transform=transforms.Compose([
                              transforms.Resize(opts.imResize),
                              transforms.RandomResizedCrop(opts.imsize),
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomVerticalFlip(),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),]),
                          strong_transform=transforms.Compose([
                              transforms.Resize(opts.imResize),
                              transforms.RandomResizedCrop(opts.imsize),
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomVerticalFlip(),
                              RandAugmentMC(n=2, m=10),
                              transforms.ToTensor(),
                              transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])), 
        batch_size=opts.batchsize, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    print('unlabel_loader done')    
    
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
    class_numbers = train_loader.dataset.classnumber
    model = Res18_basic(class_numbers)
    model = torch.nn.DataParallel(model)    
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))            
    
    if use_gpu:
        model.cuda()
        
    ##########################
    # Set optimizer
    ##########################    
    
    # Set optimizer
    #optimizer = optim.Adam(model.parameters(), lr=opts.lr)
    optimizer = optim.SGD(model.parameters(), lr=opts.lr,  momentum=opts.momentum )
    
    # INSTANTIATE LOSS CLASS
    train_criterion = SemiLoss()
             
    # INSTANTIATE STEP LEARNING SCHEDULER CLASS
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones=[50, 150], gamma=0.1)   
    
    best_acc = 0.0

    writer = SummaryWriter()
    
    for epoch in range(opts.start_epoch, opts.epochs + 1):    
        scheduler.step()
        
        print('start training')
        loss, acc_top1, acc_top5 = train(opts, train_loader, unlabel_loader, model, train_criterion, optimizer, epoch, writer, use_gpu)
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
                
def train(opts, train_loader, unlabel_loader, model, criterion, optimizer, epoch, writer, use_gpu):
    
    losses = AverageMeter()
    losses_x = AverageMeter()
    losses_un = AverageMeter()
    weight_scale = AverageMeter()
    acc_top1 = AverageMeter()
    acc_top5 = AverageMeter()
    avg_loss = 0.0
    avg_top1 = 0.0
    avg_top5 = 0.0
    
    model.train()
    
    nCnt =0 
    steps = (epoch - 1) * opts.val_iteration 
    labeled_train_iter = iter(train_loader)
    unlabeled_train_iter = iter(unlabel_loader)
    
    for batch_idx in range(opts.val_iteration):
        try:
            data = labeled_train_iter.next()
            inputs_x, targets_x = data
        except:
            labeled_train_iter = iter(train_loader)       
            data = labeled_train_iter.next()
            inputs_x, targets_x = data
        try:
            data = unlabeled_train_iter.next()
            inputs_u1, inputs_u2, _ = data
        except:
            unlabeled_train_iter = iter(unlabel_loader)       
            data = unlabeled_train_iter.next()
            inputs_u1, inputs_u2, _ = data         
    
        batch_size = inputs_x.size(0)
        # Transform label to one-hot
        if use_gpu :
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda()
            inputs_u1, inputs_u2 = inputs_u1.cuda(), inputs_u2.cuda()    
        inputs_x, targets_x = Variable(inputs_x), Variable(targets_x)
        inputs_u1, inputs_u2 = Variable(inputs_u1), Variable(inputs_u2)
        
        with torch.no_grad():
            # compute guessed labels of unlabel samples
            embed_u1, pred_u1 = model(inputs_u1)
            pseudo_label = torch.softmax(pred_u1.detach_(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(opts.threshold).float()

            
        optimizer.zero_grad()
        x_criterion = nn.CrossEntropyLoss().cuda()
        u_criterion = nn.CrossEntropyLoss(reduction='none').cuda()

        x_embed, x_pred = model(inputs_x)
        Lx = x_criterion(x_pred, targets_x)
        
        u_embed, u_pred = model(inputs_u2)
        Lu = (u_criterion(u_pred, targets_u) * mask).mean()

        loss = Lx + opts.lambda_u * Lu
        
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_un.update(Lu.item(), inputs_x.size(0))
        
        with torch.no_grad():
            # compute guessed labels of unlabel samples
            embed_x, pred_x1 = model(inputs_x)

        acc_top1b = top_n_accuracy_score(targets_x.data.cpu().numpy(), pred_x1.data.cpu().numpy(), n=1)*100
        acc_top5b = top_n_accuracy_score(targets_x.data.cpu().numpy(), pred_x1.data.cpu().numpy(), n=5)*100    
        acc_top1.update(torch.as_tensor(acc_top1b), inputs_x.size(0))        
        acc_top5.update(torch.as_tensor(acc_top5b), inputs_x.size(0))   
        
        avg_loss += loss.item()
        avg_top1 += acc_top1b
        avg_top5 += acc_top5b  
        
        if batch_idx % opts.log_interval == 0:
            print('Train Epoch:{} [{}/{}] Loss:{:.4f}({:.4f}) Top-1:{:.2f}%({:.2f}%) Top-5:{:.2f}%({:.2f}%) '.format( 
                epoch, batch_idx *inputs_x.size(0), len(train_loader.dataset), losses.val, losses.avg, acc_top1.val, acc_top1.avg, acc_top5.val, acc_top5.avg))            
                
        writer.add_scalar('train_step/loss', loss.item(), steps)
        writer.add_scalar('train_step/loss_x',Lx.item(), steps)
        writer.add_scalar('train_step/loss_unlabel', Lu.item(), steps)
        writer.add_scalar('train_step/acc_top1', acc_top1b, steps)     
        writer.add_scalar('train_step/acc_top5', acc_top5b, steps)    
        
        steps += 1
        nCnt += 1 
        
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
            