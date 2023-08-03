import logging
import gc
import numpy as np
import os.path as osp

import torch
import torch.nn as nn
from torch.nn.utils import prune
import timm


def get_logger(filename:str, verbosity=1, name=None):
    '''
    A logger register.
    '''
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    if (logger.hasHandlers()):
        logger.handlers.clear()
    logger.setLevel(level_dict[verbosity])

    # Output to file
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # Output to terminal
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def eval_acc(model,evaldata,seen_cls:int,device):
    '''
    Performance evaluator.
    '''
    model.eval()
    count = 0
    correct = 0
    wrong = 0
    class_pred = np.zeros((seen_cls,seen_cls))
    class_acc = np.zeros(seen_cls)
    for i, (image, label) in enumerate(evaldata):
        gc.collect()
        image = image.to(device)
        label = label.view(-1).to(device)
        p = model(image)
        pred = p[:,:seen_cls].argmax(dim=-1)
        
        correct += sum(pred == label).item()
        wrong += sum(pred != label).item()
        class_pred[label.cpu(),pred.cpu()] += 1
    for i in range(seen_cls):
        class_acc[i] = class_pred[i,i]/class_pred[i,:].sum()
    acc = correct / (wrong + correct)
    CWV = np.square(class_acc-acc).sum()/seen_cls
    MCD = np.max(class_acc) - np.min(class_acc)
    return acc,CWV,MCD,class_acc


def measure_module_sparsity(module, weight=True, bias=False, use_mask=False):

    num_zeros = 0
    num_elements = 0

    if use_mask == True:
        for buffer_name, buffer in module.named_buffers():
            if "weight_mask" in buffer_name and weight == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
            if "bias_mask" in buffer_name and bias == True:
                num_zeros += torch.sum(buffer == 0).item()
                num_elements += buffer.nelement()
    else:
        for param_name, param in module.named_parameters():
            if "weight" in param_name and weight == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()
            if "bias" in param_name and bias == True:
                num_zeros += torch.sum(param == 0).item()
                num_elements += param.nelement()

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity


def measure_global_sparsity(
    model, weight = True,
    bias = False, conv2d_use_mask = False,
    linear_use_mask = False):

    num_zeros = 0
    num_elements = 0

    for module_name, module in model.named_modules():

        if isinstance(module, torch.nn.Conv2d):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=conv2d_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

        elif isinstance(module, torch.nn.Linear):

            module_num_zeros, module_num_elements, _ = measure_module_sparsity(
                module, weight=weight, bias=bias, use_mask=linear_use_mask)
            num_zeros += module_num_zeros
            num_elements += module_num_elements

    sparsity = num_zeros / num_elements

    return num_zeros, num_elements, sparsity


def remove_parameters(model):
    for module_name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
        elif isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except:
                pass
            try:
                prune.remove(module, "bias")
            except:
                pass
    return model


def Load_model(root,path,device='cuda:1',num_classes=100):
    model = timm.create_model('resnet50',num_classes=num_classes,pretrained=False)
    model.to(device)
    checkpoint = torch.load(osp.join(root,path))
    model.load_state_dict(checkpoint)
    return model

def Load_pruned_model(root,path,device='cuda:1',num_classes=100):
    model = timm.create_model('resnet50',num_classes=num_classes,pretrained=False)
    model.to(device)
    checkpoint = torch.load(osp.join(root,path))
    prune.identity(model.fc,'weight')
    model.load_state_dict(checkpoint)
    return model

def count_conflict(model,mname,train_loader,val_loader,root,device='cuda:1',num_classes=100):
    model.eval()
    model2 = timm.create_model('resnet50',num_classes=num_classes,pretrained=False)
    model2.to(device)
    checkpoint = torch.load(osp.join(root,mname))
    model2.load_state_dict(checkpoint)
    
    _,_,_,class_acc = eval_acc(model,val_loader,seen_cls=num_classes,device=device)
    class_acc = torch.tensor(class_acc).long().to(device)+1e-6
    criterion = nn.CrossEntropyLoss()
    fair_cri = nn.CrossEntropyLoss(weight=1/class_acc)
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device).float(), targets.to(device).long()
        output = model(inputs)
        output2 = model2(inputs)
        loss1 = criterion(output, targets)
        loss2 = fair_cri(output2, targets)
#         print(output2)
#         print(loss2)
        loss1.backward()
        loss2.backward()
    
    return (model.fc.weight.grad, model2.fc.weight.grad)