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
import wget

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import LambdaLR
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.nn.modules.loss import _WeightedLoss

import torchvision
from torchvision import datasets, models, transforms
#from tensorboardX import SummaryWriter
#import seaborn as sns
#import matplotlib.pyplot as plt

import torch.nn.functional as F

from ImageDataLoader import SimpleImageLoader, FixMatchImageLoader
from models import Res18, Res50, Dense121, Res18_basic
from randaugment import RandAugmentMC

import nsml
from nsml import DATASET_PATH, IS_ON_NSML

from crt import ClassAwareSampler
from efficientnet_pytorch import EfficientNet

NUM_CLASSES = 265
if not IS_ON_NSML:
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
        
def adjust_learning_rate(opts, optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    """
    Linear Warmup.
    """
    if epoch <= 5:
        lr = opts.lr + (0.4-0.03) * (epoch - 1) / 4
    elif epoch < 60:
        lr = 0.4
    elif epoch >= 60 and epoch < 120:
        lr = 0.4 * 0.1
    elif epoch >= 120 and epoch < 160:
        lr = 0.4 * (0.1 ** 2)
    elif epoch >= 160 and epoch < 200:
        lr = 0.4 * (0.1 ** 3)
    else:
        lr = 0.4 * (0.1 ** 4)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
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

def split_ids(path, ratio):
    with open(path) as f:
        ids_l = [[] for i in range(265)]
        ids_u = []
        for i, line in enumerate(f.readlines()):
            if i == 0 or line == '' or line == '\n':
                continue
            line = line.replace('\n', '').split('\t')
            if int(line[1]) >= 0:
                ids_l[int(line[1])].append(int(line[0]))
            else:
                ids_u.append(int(line[0]))
    
    train_ids = []
    val_ids = []

    for labels in ids_l:
        cut = int(ratio*len(labels))
        train_ids += labels[cut:]
        val_ids += labels[:cut]
    
    ids_u += train_ids
    ids_u = np.array(ids_u)
    train_ids = np.array(train_ids)
    val_ids = np.array(val_ids)

    perm1 = np.random.permutation(np.arange(len(train_ids)))
    perm2 = np.random.permutation(np.arange(len(val_ids)))
    train_ids = train_ids[perm1]
    val_ids = val_ids[perm2]

    return train_ids, val_ids, ids_u

def split_ids_original(path, ratio):
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
    ids_u = np.array(ids_u)

    perm = np.random.permutation(np.arange(len(ids_l)))
    cut = int(ratio*len(ids_l))
    train_ids = ids_l[perm][cut:]
    val_ids = ids_l[perm][:cut]
    ids_u = np.concatenate((ids_u, train_ids))

    return train_ids, val_ids, ids_u


class SmoothCrossEntropyLoss(_WeightedLoss):
    def __init__(self, weight=None, reduction='mean', smoothing=0.0):
        super().__init__(weight=weight, reduction=reduction)
        self.smoothing = smoothing
        self.weight = weight
        self.reduction = reduction

    @staticmethod
    def _smooth_one_hot(targets:torch.Tensor, n_classes:int, smoothing=0.0):
        assert 0 <= smoothing < 1
        with torch.no_grad():
            targets = torch.empty(size=(targets.size(0), n_classes),
                    device=targets.device) \
                .fill_(smoothing /(n_classes-1)) \
                .scatter_(1, targets.data.unsqueeze(1), 1.-smoothing)
        return targets

    def forward(self, inputs, targets):
        targets = SmoothCrossEntropyLoss._smooth_one_hot(targets, inputs.size(-1),
            self.smoothing)
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss
### NSML functions
def _infer(model, root_path, test_loader=None):
    if test_loader is None:
        test_loader = torch.utils.data.DataLoader(
            SimpleImageLoader(root_path, 'test',
                               transform=transforms.Compose([
                                   transforms.Resize(opts.imResize),
                                   transforms.CenterCrop(opts.imsize),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                               ])), batch_size=opts.batchsize, shuffle=False, num_workers=4, pin_memory=True)
        print('loaded {} test images'.format(len(test_loader.dataset)))

    outputs = []
    s_t = time.time()
    for idx, image in enumerate(test_loader):
        if torch.cuda.is_available():
            image = image.cuda()
        probs = model(image)
        output = torch.argmax(probs, dim=1)
        output = output.detach().cpu().numpy()
        outputs.append(output)

    outputs = np.concatenate(outputs)
    return outputs

def bind_nsml(model):
    def save(dir_name, *args, **kwargs):
        os.makedirs(dir_name, exist_ok=True)
        state = model.state_dict()
        torch.save(state, os.path.join(dir_name, 'model.pt'))
        print('saved')

    def load(dir_name, *args, **kwargs):
        state = torch.load(os.path.join(dir_name, 'model.pt'))
        model.load_state_dict(state)
        print('loaded')

    def infer(root_path):
        return _infer(model, root_path)

    nsml.bind(save=save, load=load, infer=infer)


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

parser.add_argument('--gpu_ids',default='0', type=str,help='gpu_ids: e.g. 0  0,1,2  0,2')
parser.add_argument('--batchsize', default=128, type=int, help='batchsize')
parser.add_argument('--seed', type=int, default=123, help='random seed')

# basic hyper-parameters
parser.add_argument('--momentum', type=float, default=0.9, metavar='LR', help=' ')
parser.add_argument('--lr', type=float, default=0.03, metavar='LR', help='learning rate (default: 5e-5)')
parser.add_argument('--scheduler', type=int, default=1, metavar='LR', help='0: cosine, 1: multistep, 2: adjust learning rate')

parser.add_argument('--imResize', default=256, type=int, help='')
parser.add_argument('--imsize', default=224, type=int, help='')
parser.add_argument('--lossXent', type=float, default=1, help='lossWeight for Xent')

# arguments for logging and backup
parser.add_argument('--log_interval', type=int, default=10, metavar='N', help='logging training status')
parser.add_argument('--save_epoch', type=int, default=50, help='saving epoch interval')

# hyper-parameters for fixmatch
parser.add_argument('--lambda-u', default=1, type=float)
parser.add_argument('--mu', default=1 , type=int, help="Batch ratio between labeled/unlabeled")
parser.add_argument('--threshold', type=float, default=0.85, help='Threshold setting for Fixmatch')


parser.add_argument('--T', default=0.5, type=float)
parser.add_argument('--smooth', type = int, default=0, help='use smoothcrossentropy loss')

parser.add_argument('--val-iteration', type=int, default=100, help='Number of labeled data')
parser.add_argument('--parser', type=int, default=2)

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
    #model = Res50(NUM_CLASSES)
    if IS_ON_NSML:
        model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=265).cuda()
    else:
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=265).cuda()
        model = torch.nn.DataParallel(model)    

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    n_parameters = sum([p.data.nelement() for p in model.parameters()])
    print('  + Number of params: {}'.format(n_parameters))

    if use_gpu:
        model.cuda()

    ### DO NOT MODIFY THIS BLOCK ###
    if IS_ON_NSML:
        bind_nsml(model)
        if opts.pause:
            nsml.paused(scope=locals())
    ################################
    
    '''
    if IS_ON_NSML:
        print("load our best checkpoint...")
        url = "https://docs.google.com/uc?export=download&id=1J7wlKlRpW_0Qm0vbDKXlmla4IpUsMhfF"
        wget.download(url,'./')
        m = torch.load('./model.pt')
        model.load_state_dict(m)
        print("complete.")
    '''
    
    if opts.mode == 'train':
        model.train()
        # Set dataloader
        if opts.parser == 1:
            train_ids, val_ids, unl_ids = split_ids_original(os.path.join(DATASET_PATH, 'train/train_label'), 0.1)
        if opts.parser >= 2:
            train_ids, val_ids, unl_ids = split_ids(os.path.join(DATASET_PATH, 'train/train_label'), 0.1)
        print('found {} train, {} validation and {} unlabeled images'.format(len(train_ids), len(val_ids), len(unl_ids)))
        
        crtSampler = ClassAwareSampler(data_source=DATASET_PATH + '/train/', ids=train_ids)
        if opts.parser != 3:
            train_loader = torch.utils.data.DataLoader(
                SimpleImageLoader(DATASET_PATH, 'train', train_ids,
                                  transform=transforms.Compose([
                                      transforms.Resize(opts.imResize),
                                      transforms.RandomResizedCrop(opts.imsize),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])),
                                    batch_size=opts.batchsize, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        else:
            train_loader = torch.utils.data.DataLoader(
                SimpleImageLoader(DATASET_PATH, 'train', train_ids,
                                  transform=transforms.Compose([
                                      transforms.Resize(opts.imResize),
                                      transforms.RandomResizedCrop(opts.imsize),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.RandomVerticalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])),
                                    batch_size=opts.batchsize, sampler=crtSampler, num_workers=4, pin_memory=True, drop_last=True)


        print('train_loader done')

        unlabel_loader = torch.utils.data.DataLoader(
            FixMatchImageLoader(DATASET_PATH, 'unlabel', unl_ids,
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
            batch_size=opts.batchsize*opts.mu, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        print('unlabel_loader done')    

        crtSampler_val = ClassAwareSampler(data_source=DATASET_PATH + '/train/', ids=val_ids)
        if True:
            validation_loader = torch.utils.data.DataLoader(
                SimpleImageLoader(DATASET_PATH, 'val', val_ids,
                                   transform=transforms.Compose([
                                       transforms.Resize(opts.imResize),
                                       transforms.CenterCrop(opts.imsize),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])),
                                   batch_size=opts.batchsize, num_workers=4, pin_memory=True, drop_last=False,
                                   shuffle=True)
        else:
            validation_loader = torch.utils.data.DataLoader(
                SimpleImageLoader(DATASET_PATH, 'val', val_ids,
                                   transform=transforms.Compose([
                                       transforms.Resize(opts.imResize),
                                       transforms.CenterCrop(opts.imsize),
                                       transforms.ToTensor(),
                                       transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])),
                                   batch_size=opts.batchsize, num_workers=4, pin_memory=True, drop_last=False,
                                   sampler=crtSampler_val)
        print('validation_loader done')

        # Set optimizer
        #optimizer = optim.Adam(model.parameters(), lr=opts.lr)
        optimizer = optim.SGD(model.parameters(), lr=opts.lr,  momentum=opts.momentum, nesterov=True, weight_decay=0.0001)
        fc_optimizer = optim.SGD(model.parameters(), lr=0.1,  momentum=opts.momentum, nesterov=True, weight_decay=0.0001)


        # INSTANTIATE LOSS CLASS
        train_criterion = SemiLoss()

        iter_num = len(train_ids) // opts.batchsize 
        # INSTANTIATE STEP LEARNING SCHEDULER CLASS
        if opts.scheduler == 0:
            scheduler = get_cosine_schedule_with_warmup(optimizer,0,iter_num * opts.epochs)
        elif opts.scheduler == 1:
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,  milestones=[60, 120, 160], gamma=0.1)
        else:
            scheduler = None # not used

        # Train and Validation 
        best_acc = 0.0
        best_top5 = 0.0
        
        for epoch in range(opts.start_epoch, opts.epochs + 1):
            if epoch != 1:
                loss, train_top1, train_top5 = train(opts, train_loader, unlabel_loader, model, train_criterion, optimizer, epoch, use_gpu, scheduler)
            else:
                loss, train_top1, train_top5 = train(opts, train_loader, unlabel_loader, model, train_criterion, fc_optimizer, epoch, use_gpu, scheduler)

            print('start training')
            if opts.scheduler == 0:
                scheduler.step()
            elif opts.scheduler == 1:
                scheduler.step()
            else:
                adjust_learning_rate(opts, optimizer, epoch)
            

            print('start validation')
            acc_top1, acc_top5, cm = validation(opts, validation_loader, model, epoch, use_gpu)
            is_best = acc_top1 > best_acc
            is_best5 = acc_top5 > best_top5
            
            best_acc = max(acc_top1, best_acc)
            best_top5 = max(acc_top5, best_top5)
            if is_best:
                print('saving top1 best checkpoint...')                
                if IS_ON_NSML:
                    nsml.save(opts.name + '_best')
                else:
                    torch.save(model.state_dict(), os.path.join('runs', opts.name + '_best.pth.tar'))
                '''
                print('saving confusion matrix...')                
                plt.figure()
                sns.heatmap(cm,xticklabels = False, yticklabels=False)
                plt.xlabel("prediction")
                plt.ylabel("True")
                plt.savefig("./confusion_matrix/"+opts.name+"_epoch"+str(epoch)+"_"+str(int(acc_top1))+".png")
                '''
            if is_best5:
                print('saving top5 best checkpoint')
                if IS_ON_NSML:
                    nsml.save(opts.name + '_best5')
                else:
                    torch.save(model.state_dict(), os.path.join('runs', opts.name + '_best5.pth.tar'))
            if (epoch + 1) % opts.save_epoch == 0:
                if IS_ON_NSML:
                    nsml.save(opts.name + '_e{}'.format(epoch))
                else:
                    torch.save(model.state_dict(), os.path.join('runs', opts.name + '_e{}.pth.tar'.format(epoch)))
                    
            if IS_ON_NSML:
                nsml.report(summary={
                    'train__loss__epoch': loss,
                    'train__acc__top1': train_top1,
                    'train__acc__top5': train_top5,
                    'test__acc__top1': acc_top1,
                    'test__acc__top5': acc_top5},
                    step=epoch)

                
def train(opts, train_loader, unlabel_loader, model, criterion, optimizer, epoch, use_gpu, scheduler):
    
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
    if epoch == 1:
        for param in model.parameters():
            param.requires_grad = False
        if IS_ON_NSML:
            model._fc.weight.requires_grad = True
            model._fc.weight.requires_grad = True
        else:
            model.module._fc.weight.requires_grad = True
            model.module._fc.bias.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = True    
    
    nCnt =0
    steps = (epoch - 1) * opts.val_iteration 
    labeled_train_iter = iter(train_loader)
    unlabeled_train_iter = iter(unlabel_loader)
    
    for batch_idx in range(len(train_loader)):
        try:
            data = labeled_train_iter.next()
            inputs_x, targets_x = data
        except:
            labeled_train_iter = iter(train_loader)       
            data = labeled_train_iter.next()
            inputs_x, targets_x = data
        try:
            data = unlabeled_train_iter.next()
            inputs_u1, inputs_u2 = data
        except:
            unlabeled_train_iter = iter(unlabel_loader)       
            data = unlabeled_train_iter.next()
            inputs_u1, inputs_u2 = data         
    
        batch_size = inputs_x.size(0)
        # Transform label to one-hot
        if use_gpu :
            inputs_x, targets_x = inputs_x.cuda(), targets_x.cuda()
            inputs_u1, inputs_u2 = inputs_u1.cuda(), inputs_u2.cuda()    
        inputs_x, targets_x = Variable(inputs_x), Variable(targets_x)
        inputs_u1, inputs_u2 = Variable(inputs_u1), Variable(inputs_u2)
        
        with torch.no_grad():
            # compute guessed labels of unlabel samples
            pred_u1 = model(inputs_u1)
            pseudo_label = torch.softmax(pred_u1.detach_(), dim=-1)
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(opts.threshold).float()

            
        optimizer.zero_grad()
        if opts.smooth == 0:
            x_criterion = nn.CrossEntropyLoss().cuda()
            u_criterion = nn.CrossEntropyLoss(reduction='none').cuda()        
        else:
            x_criterion = SmoothCrossEntropyLoss(smoothing=0.1).cuda()
            u_criterion = SmoothCrossEntropyLoss(reduction='none', smoothing=0.1).cuda()

        x_pred = model(inputs_x)
        Lx = x_criterion(x_pred, targets_x)
        
        u_pred = model(inputs_u2)
        Lu = (u_criterion(u_pred, targets_u) * mask).mean()

        loss = Lx + opts.lambda_u * Lu
        
        # compute gradient and do SGD step
        loss.backward()
        optimizer.step()
        if opts.scheduler == 0:
            scheduler.step()
        
        losses.update(loss.item(), inputs_x.size(0))
        losses_x.update(Lx.item(), inputs_x.size(0))
        losses_un.update(Lu.item(), inputs_x.size(0))
        
        with torch.no_grad():
            # compute guessed labels of unlabel samples
            pred_x1 = model(inputs_x)

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
        '''
        if IS_ON_NSML:
            nsml.report(summary={
                'train__loss': loss.item(),
                'train__lossx': Lx.item(),
                'train__lossunlabel': Lu.item(),
                'train__acctop1': acc_top1b,
                'train__acctop5': acc_top5b},
                step=steps)             
         '''
        steps += 1
        nCnt += 1 
        
    avg_loss =  float(avg_loss/nCnt)
    avg_top1 = float(avg_top1/nCnt)
    avg_top5 = float(avg_top5/nCnt)
    
    return  avg_loss, avg_top1, avg_top5    



def validation(opts, validation_loader, model, epoch, use_gpu):
    model.eval()
    avg_top1= 0.0
    avg_top5 = 0.0
    nCnt =0
    steps = (epoch - 1) * len(validation_loader)
    cm = np.zeros((NUM_CLASSES,NUM_CLASSES))

    with torch.no_grad():
        for batch_idx, data in enumerate(validation_loader):
            inputs, labels = data
            if use_gpu :
                inputs = inputs.cuda()
            inputs = Variable(inputs)
            steps+=1
            nCnt +=1
            preds = model(inputs)

            acc_top1 = top_n_accuracy_score(labels.numpy(), preds.data.cpu().numpy(), n=1)*100
            acc_top5 = top_n_accuracy_score(labels.numpy(), preds.data.cpu().numpy(), n=5)*100
            avg_top1 += acc_top1
            avg_top5 += acc_top5
            '''
            if IS_ON_NSML:
                nsml.report(summary={
                    'test_step__acc_val_top1': acc_top1,
                    'test_step__acc_val_top5': acc_top5},
                    step=steps) 
            '''
            _, outputs = torch.max(preds.data.cpu(), 1)
            #print(labels.numpy())
            #print(outputs.numpy())
            for t, p in zip(labels.numpy(), outputs.numpy()):
                cm[t,p] +=1
                
                
        avg_top1 = float(avg_top1/nCnt)   
        avg_top5= float(avg_top5/nCnt)
        cm = cm.astype(float)/cm.sum(axis=1)[:,np.newaxis]

        print('Test Epoch:{} Top1_acc_val:{:.2f}% Top5_acc_val:{:.2f}% '.format(epoch, avg_top1, avg_top5))
    return avg_top1, avg_top5, cm



if __name__ == '__main__':
    main()


