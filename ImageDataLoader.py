from PIL import Image
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import torch
import random

def default_image_loader(path):
    return Image.open(path).convert('RGB')

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2    

###################################################################################

class FixMatchImageLoader(torch.utils.data.Dataset):
    def __init__(self, rootdir, split, ids=None, ratio = 0, transform=None, strong_transform=None, loader=default_image_loader):
        if split == 'test':
            self.impath = os.path.join(rootdir, 'test_data')
            meta_file = os.path.join(self.impath, 'test_meta.txt')
        else:
            self.impath = os.path.join(rootdir, 'train/train_data')
            meta_file = os.path.join(rootdir, 'train/train_label')

        imnames = []
        imclasses = []
        
        with open(meta_file, 'r') as rf:
            for i, line in enumerate(rf):
                if i == 0:
                    continue
                instance_id, label, file_name = line.strip().split()        
                if int(label) == -1 and (split != 'unlabel' and split != 'test'):
                    continue
                if int(label) != -1 and (split == 'unlabel' or split == 'test'):
                    continue
                if (ids is None) or (int(instance_id) in ids):
                    if os.path.exists(os.path.join(self.impath, file_name)):
                        imnames.append(file_name)
                        if split == 'train' or split == 'val':
                            imclasses.append(int(label))

        self.transform = transform
        self.strong_transform = strong_transform
        self.loader = loader
        self.split = split
        self.imnames = imnames
        self.imclasses = imclasses
        self.ratio = ratio
    
    def __getitem__(self, index):
        filename = self.imnames[index]
        img_name = self.loader(os.path.join(self.impath, filename))
        
        if self.split != 'unlabel':

            label = self.imclasses[index]
            if random.random()< self.ratio:
                return self.strong_transform(img_name), int(label)
            else: 
                return self.transform(img_name), int(label)

        else:        
            img1, img2 = self.transform(img_name), self.strong_transform(img_name)
            return img1, img2       
            
        
    def __len__(self):
        return len(self.imnames)               

class SimpleImageLoader(torch.utils.data.Dataset):
    def __init__(self, rootdir, split, ids=None, transform=None, loader=default_image_loader):
        if split == 'test':
            self.impath = os.path.join(rootdir, 'test_data')
            meta_file = os.path.join(self.impath, 'test_meta.txt')
        else:
            self.impath = os.path.join(rootdir, 'train/train_data')
            meta_file = os.path.join(rootdir, 'train/train_label')

        imnames = []
        imclasses = []
        
        with open(meta_file, 'r') as rf:
            for i, line in enumerate(rf):
                if i == 0:
                    continue
                instance_id, label, file_name = line.strip().split()        
                if int(label) == -1 and (split != 'unlabel' and split != 'test'):
                    continue
                if int(label) != -1 and (split == 'unlabel' or split == 'test'):
                    continue
                if (ids is None) or (int(instance_id) in ids):
                    if os.path.exists(os.path.join(self.impath, file_name)):
                        imnames.append(file_name)
                        if split == 'train' or split == 'val':
                            imclasses.append(int(label))

        self.transform = transform
        self.TransformTwice = TransformTwice(transform)
        self.loader = loader
        self.split = split
        self.imnames = imnames
        self.imclasses = imclasses
    
    def __getitem__(self, index):
        filename = self.imnames[index]
        img = self.loader(os.path.join(self.impath, filename))
        
        if self.split == 'test':
            if self.transform is not None:
                img = self.transform(img)
            return img
        elif self.split != 'unlabel':
            if self.transform is not None:
                img = self.transform(img)
            label = self.imclasses[index]
            return img, label
        else:        
            img1, img2 = self.TransformTwice(img)
            return img1, img2
        
    def __len__(self):
        return len(self.imnames)             
    
class MoCoImageLoader(torch.utils.data.Dataset):
    def __init__(self, rootdir, split, ids=None, transform=None, loader=default_image_loader):
        if split == 'test':
            self.impath = os.path.join(rootdir, 'test_data')
            meta_file = os.path.join(self.impath, 'test_meta.txt')
        else:
            self.impath = os.path.join(rootdir, 'train/train_data')
            meta_file = os.path.join(rootdir, 'train/train_label')

        imnames = []
        imclasses = []
        
        with open(meta_file, 'r') as rf:
            for i, line in enumerate(rf):
                if i == 0:
                    continue
                instance_id, label, file_name = line.strip().split()        
                if os.path.exists(os.path.join(self.impath, file_name)):
                    imnames.append(file_name)
                    if split == 'train' or split == 'val':
                        imclasses.append(int(label))

        self.transform = transform
        self.TransformTwice = TransformTwice(transform)
        self.loader = loader
        self.split = split
        self.imnames = imnames
        self.imclasses = imclasses
    
    def __getitem__(self, index):
        filename = self.imnames[index]
        img = self.loader(os.path.join(self.impath, filename))
        
        return self.transform(img)
        
    def __len__(self):
        return len(self.imnames)             

