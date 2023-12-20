import sys
import argparse
import os
import os.path as osp
import time
import gc
from copy import deepcopy
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data.dataset import random_split
from torch.utils.data import Dataset, TensorDataset, DataLoader, Subset, ConcatDataset
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import prune
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
import torchvision
from torchvision import datasets, transforms,models
from torchvision.io import read_image
from torchvision.datasets.folder import default_loader
from torchvision.datasets.utils import download_url
from torchvision.models import resnet18
import torch.backends.cudnn as cudnn
from tqdm import trange
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import timm
from PIL import Image
from utils import eval_acc,get_logger,count_conflict,remove_parameters, Load_model

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

class CelebaDataset(Dataset):
    """Custom Dataset for loading CelebA face images"""

    def __init__(self, csv_path, img_dir, transform=None):
    
        df = pd.read_csv(csv_path, index_col=0)
        self.img_dir = img_dir
        self.csv_path = csv_path
        self.img_names = df.index.values
        self.y = df['Male'].values
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(os.path.join(self.img_dir,
                                      self.img_names[index]))
        
        if self.transform is not None:
            img = self.transform(img)
        
        label = self.y[index]
        return img, label

    def __len__(self):
        return self.y.shape[0]
    

def ballot(args):

    dataset_name = args.dataname
    save_path = os.path.join('checkpoint',dataset_name)
    if dataset_name == 'cifar100':
        input_transform = transforms.Compose([
                                    transforms.RandomHorizontalFlip(),
                                    transforms.RandomCrop(32,padding=4),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])
        ])
        input_transform_eval = transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5071,0.4866,0.4409],[0.2673,0.2564,0.2762])
        ])
        train_dataset = datasets.CIFAR100(root=os.path.abspath('../data/cifar100'),train=True,transform=input_transform,download=True)
        test_dataset = datasets.CIFAR100(root=os.path.abspath('../data/cifar100'),train=False,transform=input_transform_eval,download=True)
    elif dataset_name == 'tinyimagenet':
        input_transform = transforms.Compose([
                                    transforms.ToTensor(),
        ])
        input_transform_eval = transforms.Compose([
                                    transforms.ToTensor(),
        ])
        train_dataset = datasets.ImageFolder(os.path.abspath('../data/tiny-imagenet-200/train'), input_transform)
        test_dataset = datasets.ImageFolder(os.path.abspath('../data/tiny-imagenet-200/test'), input_transform_eval)
    elif dataset_name == 'celeba':
        datadir = os.path.abspath('../data/celeba')
        df1 = pd.read_csv(os.path.join(datadir,'list_attr_celeba.txt'), sep="\s+", skiprows=1, usecols=['Male'])
        # Make 0 (female) & 1 (male) labels instead of -1 & 1
        df1.loc[df1['Male'] == -1, 'Male'] = 0
        df2 = pd.read_csv(os.path.join(datadir,'list_eval_partition.txt'), sep="\s+", skiprows=0, header=None)
        df2.columns = ['Filename', 'Partition']
        df2 = df2.set_index('Filename')
        df3 = df1.merge(df2, left_index=True, right_index=True)
        df3.to_csv(os.path.join(datadir,'celeba-gender-partitions.csv'))
        df3.loc[df3['Partition'] == 0].to_csv(os.path.join(datadir,'celeba-gender-train.csv'))
        df3.loc[df3['Partition'] == 1].to_csv(os.path.join(datadir,'celeba-gender-valid.csv'))
        df3.loc[df3['Partition'] == 2].to_csv(os.path.join(datadir,'celeba-gender-test.csv'))
        custom_transform = transforms.Compose([transforms.CenterCrop((178, 178)),
                                            transforms.Resize((128, 128)),
                                            #transforms.Grayscale(),                                       
                                            #transforms.Lambda(lambda x: x/255.),
                                            transforms.ToTensor()])
        train_dataset = CelebaDataset(csv_path=os.path.join(datadir,'celeba-gender-train.csv'),
                                    img_dir=os.path.join(datadir,'img_align_celeba/'),
                                    transform=custom_transform)
        test_dataset = CelebaDataset(csv_path=os.path.join(datadir,'celeba-gender-test.csv'),
                                    img_dir=os.path.join(datadir,'img_align_celeba/'),
                                    transform=custom_transform)


    else:
        raise Exception("Dataset: cifar100/TinyImagenet/CelebA")

    def train(net,train_loader):
        net.to(device)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        cnt = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device).float(), targets.to(device).long()

            optimizer.zero_grad()

            benign_outputs = net(inputs)
            loss = criterion(benign_outputs, targets)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            _, predicted = benign_outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            cnt+=1
        acc = 100. * correct / total
        return train_loss, acc

    trial = args.trial
    CLASS_NUM = args.total_cls
    EPOCHS = args.epochs
    method = args.method


    if trial == 0:
        trial = str(time.time())
        root = './checkpoint/'+trial
        os.mkdir(root)
        model = timm.create_model('resnet50',num_classes=CLASS_NUM,pretrained=False)
        model.to(device)
        torch.save(model.state_dict(),osp.join(root,'model_init.pkl'))
        logger = get_logger(osp.join(root,'train_round0.log'))
        logger.info('start training!')

        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,  weight_decay=2e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = MultiStepLR(optimizer, [100, 150, 200], gamma=0.1)
        train_loader = DataLoader(train_dataset,batch_size=32, shuffle=True)
        for epoch in trange(0, EPOCHS):
            train_loss, acc = train(model,train_loader)
            logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch, EPOCHS, train_loss, acc))
            scheduler.step()
        logger.info('finish training!')
        torch.save(model.state_dict(),osp.join(root,'model_round0.pkl'))
    else:
        root = './checkpoint/'+str(trial)
        prune_amount = 0.95
        if method == 'lth':
            model = timm.create_model('resnet50',num_classes=CLASS_NUM,pretrained=False)
            checkpoint = torch.load(osp.join(root,'model_round0.pkl'))
            model.load_state_dict(checkpoint)

            parameters_to_prune = []
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    parameters_to_prune.append((module, "weight"))
                elif isinstance(module, torch.nn.Linear):
                    parameters_to_prune.append((module, "weight"))

            prune_amount = 0.95
            prune.global_unstructured(
                parameters_to_prune,
                pruning_method = prune.L1Unstructured,
                amount = prune_amount,
            )
            custom_mask = {}
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    custom_mask[module_name] = module.weight_mask
                elif isinstance(module, torch.nn.Linear):
                    custom_mask[module_name] = module.weight_mask    
            remove_parameters(model)
            checkpoint = torch.load(osp.join(root,'model_init.pkl'))
            model.load_state_dict(checkpoint)

            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Conv2d):
                    prune.custom_from_mask(module,'weight',custom_mask[module_name])
                elif isinstance(module, torch.nn.Linear):
                    prune.custom_from_mask(module,'weight',custom_mask[module_name])
            remove_parameters(model)

            logger = get_logger(osp.join(root,'train_lth_{}.log'.format(prune_amount)))
            logger.info('start training!')

            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,  weight_decay=2e-4)
            criterion = nn.CrossEntropyLoss()
            scheduler = MultiStepLR(optimizer, [100, 150, 200], gamma=0.1)
            t_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset)*0.9), int(len(train_dataset)*0.1)])
            counter = []
            train_loader = DataLoader(t_dataset,batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset,batch_size=32, shuffle=False)

            for epoch in trange(0, EPOCHS):
                torch.save(model.state_dict(),osp.join(root,'train_lth_{}.pkl'.format(prune_amount)))
                train_loss, acc = train(model,train_loader)
                
                counter.append(count_conflict(model,'train_lth_{}.pkl'.format(prune_amount),train_loader,val_loader,root,num_classes=CLASS_NUM))
                logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch, EPOCHS, train_loss, acc))
                scheduler.step()
            torch.save(counter,osp.join(root,'counter.pkl'))
            logger.info('finish training!')
        if method == 'ballot':
            counter = torch.load(osp.join(root,'counter.pkl'))
            cnt = torch.zeros(CLASS_NUM,2048)
            prune_amount = 0.95
            k = int((1-prune_amount)*len(cnt.view(-1)))
            eta = 1
            for epoch in range(0, EPOCHS):
                grad, fair_grad = counter[epoch]
                grad = F.normalize(grad, dim=0)
                fair_grad = F.normalize(fair_grad, dim=0)
                simi = grad + eta*fair_grad
                top_vals = torch.topk(simi.view(-1), k)
                mask = top_vals[0] != 0
                indices = top_vals[1][mask].cpu()
                cnt.view(-1)[indices]+=1

            custom_mask_fair = torch.zeros(CLASS_NUM,2048)
            indx = torch.topk(cnt.view(-1), k)[1]
            custom_mask_fair.view(-1)[indx] += 1

            model = Load_model(root,'model_init.pkl',device=device)
            custom_mask_fair = custom_mask_fair.to(device)
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    prune.custom_from_mask(module,'weight',custom_mask_fair)  

            logger = get_logger(osp.join(root,'train_lth_fair{}_k{}_eta{}.log'.format(prune_amount,k,eta)))
            logger.info('start training!')

            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,  weight_decay=2e-4)
            criterion = nn.CrossEntropyLoss()
            scheduler = MultiStepLR(optimizer, [100, 150, 200], gamma=0.1)
            train_loader = DataLoader(train_dataset,batch_size=32, shuffle=True)
            for epoch in trange(0, EPOCHS):
                train_loss, acc = train(model,train_loader)
                logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch, EPOCHS, train_loss, acc))
                scheduler.step()
            logger.info('finish training!')
            torch.save(model.state_dict(),osp.join(root,'model_lth_fair{}_k{}_eta{}.pkl'.format(prune_amount,k,eta)))
        if method == 'randomprune':
            cnt = torch.rand(CLASS_NUM,2048)
            k = int((1-prune_amount)*len(cnt.view(-1)))
            eta = 1
            custom_mask_fair = torch.zeros(CLASS_NUM,2048)
            indx = torch.topk(cnt.view(-1), k)[1]
            custom_mask_fair.view(-1)[indx] += 1

            model = Load_model(root,'model_init.pkl',device=device)
            custom_mask_fair = custom_mask_fair.to(device)
            for module_name, module in model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    prune.custom_from_mask(module,'weight',custom_mask_fair)  

            logger = get_logger(osp.join(root,'train_lth_random{}_k{}_eta{}.log'.format(prune_amount,k,eta)))
            logger.info('start training!')
            
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,  weight_decay=2e-4)
            criterion = nn.CrossEntropyLoss()
            scheduler = MultiStepLR(optimizer, [100, 150, 200], gamma=0.1)
            train_loader = DataLoader(train_dataset,batch_size=32, shuffle=True)
            for epoch in trange(0, EPOCHS):
                train_loss, acc = train(model,train_loader)
                logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch, EPOCHS, train_loss, acc)) 
                scheduler.step()
            logger.info('finish training!')
            torch.save(model.state_dict(),osp.join(root,'model_lth_random{}_k{}_eta{}.pkl'.format(prune_amount,k,eta)))


parser = argparse.ArgumentParser(description='CILIATE')
parser.add_argument('--batch_size', default = 32, type = int)
parser.add_argument('--dataname', default = 'cifar100', type = str)
parser.add_argument('--epoch', default = 250, type = int)
parser.add_argument('--trial', default = 0.1, type = float)
parser.add_argument('--total_cls', default = 100, type = int)
parser.add_argument('--method', default = 'fairprune', type = str)
args = parser.parse_args()

if __name__ == "__main__":
    ballot(args)