from PIL import Image
import os
import os.path
import torch.utils.data
import torchvision.transforms as transforms
import numpy as np
import torch

def default_image_loader(path):
    return Image.open(path).convert('RGB')

class TransformTwice:
    def __init__(self, transform):
        self.transform = transform
    def __call__(self, inp):
        out1 = self.transform(inp)
        out2 = self.transform(inp)
        return out1, out2    
    
class SimpleImageLoader(torch.utils.data.Dataset):
    def __init__(self, args, split, transform=None, loader=default_image_loader):
        rootdir = os.path.join(args.datadir)
        self.impath = os.path.join(args.datadir, 'downimages')
        self.datasets = []
        if split == 'train' : data_file = os.path.join(rootdir, '%s' % args.trainfile)
        elif split == 'test' : data_file = os.path.join(rootdir, '%s' % args.testfile)            
        elif split == 'validation' : data_file = os.path.join(rootdir, '%s' % args.validfile)    
        elif split == 'unlabel' : data_file = os.path.join(rootdir, '%s' % args.unlabelfile)   
        else: print('wrong split information')
            
        meta_file = os.path.join(rootdir, data_file)    
        self.subpath_to_idx_dict = {}
        identity_class= set()
        imnames = []
        imclasses = []
        classindex = []
        
        with open(meta_file, 'r') as rf:     
            for idx, line in enumerate(rf):
                if idx == 0:
                    continue
                instance_id, label, file_name = line.strip().split()        
                category_id = int(label)
                self.subpath_to_idx_dict[file_name] = len(self.datasets)
                self.datasets.append( (os.path.join(self.impath, file_name), int(category_id)) )                    
                imnames.append(file_name)
                imclasses.append(int(label))
                identity_class.add(int(label))

        self.transform = transform
        self.TransformTwice = TransformTwice(transform)
        self.loader = loader
        self.split = split
        self.classnumber = len(identity_class)
        self.imnames = imnames
        self.imclasses = imclasses
    
    def __getitem__(self, index):
        filename = self.imnames[index]
        img_name = self.loader(os.path.join(self.impath, filename))
        
        if self.split != 'unlabel':
            if self.transform is not None:
                img = self.transform(img_name)
            label = self.imclasses[index]            
            return img, int(label)

        else:        
            img1, img2 = self.TransformTwice(img_name)
            label = self.imclasses[index]  
            return img1, img2, int(label)           
            
        
    def __len__(self):
        return len(self.imnames)               
    
class TripletImageLoader(torch.utils.data.Dataset):
    def __init__(self, args, split, transform=None, loader=default_image_loader):
        rootdir = os.path.join(args.datadir)
        self.impath = os.path.join(args.datadir, 'images')
        self.datasets = []

        if split == 'train' : data_file = os.path.join(rootdir, '%s' % args.trainfile)
        elif split == 'test' : data_file = os.path.join(rootdir, '%s' % args.testfile)            
        elif split == 'validation' : data_file = os.path.join(rootdir, '%s' % args.validfile)      
        elif split == 'unlabel' : data_file = os.path.join(rootdir, '%s' % args.unlabelfile)                  
        else: 
            print('wrong split information')
            
        meta_file = os.path.join(rootdir, data_file)    
        self.subpath_to_idx_dict = {}
        identity_class= set()
        imnames = []
        imclasses = []

        classindex = []
        with open(meta_file, 'r') as rf:     
            for idx, line in enumerate(rf):
                if idx == 0:
                    continue
                instance_id, label, file_name = line.strip().split()        
                category_id = int(label)
                self.subpath_to_idx_dict[file_name] = len(self.datasets)
                self.datasets.append( (os.path.join(self.impath, file_name), int(category_id)) )                    
                imnames.append(file_name)
                imclasses.append(int(label))
                identity_class.add(int(label))
    
        self.transform = transform
        self.loader = loader
        self.split = split
        self.classnumber = len(identity_class)
        self.imnames = imnames
        self.imclasses = imclasses
        
    def _get_pos_sample(self, label, index):
        pos_index = np.argwhere(np.asarray(self.imclasses) == label)
        pos_index = pos_index.flatten()
        pos_index = np.setdiff1d(pos_index, index)
        rand = np.random.permutation(len(pos_index))
        result_path = []
        result_label = []
        for ii in range(1):
            tt = ii % len(rand)
            tmp_index = pos_index[rand[tt]]
            result_label.append(self.imclasses[tmp_index])
            result_path.append(self.imnames[tmp_index])
        return result_path, result_label
        
    def _get_neg_sample(self, label):
        neg_index = np.argwhere(np.asarray(self.imclasses) != label)
        neg_index = neg_index.flatten()
        rand = np.random.permutation(len(neg_index))
        result_path = [] 
        result_label = []
        for ii in range(1):    
            tt = ii % len(rand)
            tmp_index = neg_index[rand[tt]]  
            result_label.append(self.imclasses[tmp_index])            
            result_path.append(self.imnames[tmp_index])
        return result_path , result_label   
    
    def __getitem__(self, index):
        filename = self.imnames[index]
        label = self.imclasses[index]   
        pos_path, pos_label = self._get_pos_sample(label, index)
        neg_path, neg_label = self._get_neg_sample(label) 

        anc_name = self.loader(os.path.join(self.impath, filename))
        pos_name = self.loader(os.path.join(self.impath, pos_path[0]))
        neg_name = self.loader(os.path.join(self.impath, neg_path[0]))
        if self.transform is not None:
            img_anc = self.transform(anc_name)
            img_pos = self.transform(pos_name)
            img_neg = self.transform(neg_name)
        return img_anc, int(label), img_pos, int(pos_label[0]), img_neg, int(neg_label[0])
        
    def __len__(self):
        return len(self.imnames)               
    

