import os
import os.path as osp
import time
import gc
from copy import deepcopy
import random
import numpy as np

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

from utils import eval_acc,get_logger,count_conflict,remove_parameters, Load_model

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'


dataset_name = 'cifar100'

if dataset_name == 'cifar100':
    save_path = os.path.join('checkpoint',dataset_name)
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

batch_size = 100
test_loader = DataLoader(test_dataset,batch_size=batch_size, shuffle=False)

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

trial = 1687856843.4234667
lth = False
fairprune = False
randomprune = True
 
if trial == 0:
    trial = str(time.time())
    root = './checkpoint/'+trial
    os.mkdir(root)
    model = timm.create_model('resnet50',num_classes=100,pretrained=False)
    model.to(device)
    torch.save(model.state_dict(),osp.join(root,'model_init.pkl'))
    logger = get_logger(osp.join(root,'train_round0.log'))
    logger.info('start training!')

    # optimizer = timm.optim.create_optimizer_v2(model)
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,  weight_decay=2e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = MultiStepLR(optimizer, [100, 150, 200], gamma=0.1)
    EPOCHS = 250
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
    if lth==True:
        model = timm.create_model('resnet50',num_classes=100,pretrained=False)
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

        # optimizer = timm.optim.create_optimizer_v2(model)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,  weight_decay=2e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = MultiStepLR(optimizer, [100, 150, 200], gamma=0.1)
        EPOCHS = 250
        t_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [int(len(train_dataset)*0.9), int(len(train_dataset)*0.1)])
        counter = []
        train_loader = DataLoader(t_dataset,batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset,batch_size=32, shuffle=False)

        for epoch in trange(0, EPOCHS):
            torch.save(model.state_dict(),osp.join(root,'train_lth_{}.pkl'.format(prune_amount)))
            train_loss, acc = train(model,train_loader)
            
            counter.append(count_conflict(model,'train_lth_{}.pkl'.format(prune_amount),train_loader,val_loader,root))
            logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch, EPOCHS, train_loss, acc))
            scheduler.step()
        torch.save(counter,osp.join(root,'counter.pkl'))
        logger.info('finish training!')
    if fairprune == True:
        counter = torch.load(osp.join(root,'counter.pkl'))
        EPOCHS = 250
        cnt = torch.zeros(100,2048)
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

        custom_mask_fair = torch.zeros(100,2048)
        indx = torch.topk(cnt.view(-1), k)[1]
        custom_mask_fair.view(-1)[indx] += 1

        model = Load_model(root,'model_init.pkl',device=device)
        custom_mask_fair = custom_mask_fair.to(device)
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.custom_from_mask(module,'weight',custom_mask_fair)  

        logger = get_logger(osp.join(root,'train_lth_fair{}_k{}_eta{}.log'.format(prune_amount,k,eta)))
        logger.info('start training!')

        # optimizer = timm.optim.create_optimizer_v2(model)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,  weight_decay=2e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = MultiStepLR(optimizer, [100, 150, 200], gamma=0.1)
        EPOCHS = 250
        train_loader = DataLoader(train_dataset,batch_size=32, shuffle=True)
        for epoch in trange(0, EPOCHS):
            train_loss, acc = train(model,train_loader)
            logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch, EPOCHS, train_loss, acc))
            scheduler.step()
        logger.info('finish training!')
        torch.save(model.state_dict(),osp.join(root,'model_lth_fair{}_k{}_eta{}.pkl'.format(prune_amount,k,eta)))
    if randomprune == True:
        EPOCHS = 250
        cnt = torch.rand(100,2048)
        k = int((1-prune_amount)*len(cnt.view(-1)))
        eta = 1
        custom_mask_fair = torch.zeros(100,2048)
        indx = torch.topk(cnt.view(-1), k)[1]
        custom_mask_fair.view(-1)[indx] += 1

        model = Load_model(root,'model_init.pkl',device=device)
        custom_mask_fair = custom_mask_fair.to(device)
        for module_name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                prune.custom_from_mask(module,'weight',custom_mask_fair)  

        logger = get_logger(osp.join(root,'train_lth_random{}_k{}_eta{}.log'.format(prune_amount,k,eta)))
        logger.info('start training!')

        # optimizer = timm.optim.create_optimizer_v2(model)
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,  weight_decay=2e-4)
        criterion = nn.CrossEntropyLoss()
        scheduler = MultiStepLR(optimizer, [100, 150, 200], gamma=0.1)
        EPOCHS = 250
        train_loader = DataLoader(train_dataset,batch_size=32, shuffle=True)
        for epoch in trange(0, EPOCHS):
            train_loss, acc = train(model,train_loader)
            logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch, EPOCHS, train_loss, acc)) 
            scheduler.step()
        logger.info('finish training!')
        torch.save(model.state_dict(),osp.join(root,'model_lth_random{}_k{}_eta{}.pkl'.format(prune_amount,k,eta)))