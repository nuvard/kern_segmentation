from __future__ import print_function, division
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union

import pandas as pd
import torch 
import random
import os
from pandas import DataFrame
from PIL import Image
from torch.utils.data.dataloader import default_collate
import sklearn
import shutil
from sklearn.metrics import f1_score, roc_auc_score, precision_score

import numpy as np
from matplotlib import pyplot as plt

import numpy as np
from torchvision import transforms

from torchvision import models
from torch.nn.modules import loss
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torchvision.transforms.functional as TF
from tqdm.auto import tqdm

import os
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms, utils
import torch.nn.functional as tf
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

plt.ion()   # interactive mode
from torchvision.transforms import RandomVerticalFlip, RandomCrop

from albumentations import RandomCrop, Normalize, HorizontalFlip, Resize
from albumentations import Compose
from albumentations.pytorch import ToTensor

import datetime
from utils import *
from prepare_data import *
from model import *
import shutil
import argparse
import numba
import ttach as tta


def criterion(y_pred, y_true, epsilon = 1e-6, num_classes = 6, train=False):
    y_pred = to_numpy(tf.one_hot(y_pred, num_classes))
    y_true = to_numpy(tf.one_hot(y_true, num_classes))
    #if(train==True):
    print(np.sum(y_true, axis=0))
    print(np.sum(y_pred, axis=0))
    tp = np.sum(y_true *y_pred, axis = 0)
    tn = np.sum((1 - y_true) * (1 - y_pred), axis =0)
    fp = np.sum((1 - y_true) * y_pred, axis=0)
    fn = np.sum(y_true * (1 - y_pred), axis=0)
    
    precision = tp / (tp + fp + epsilon)
    recall = tp / (tp + fn + epsilon)
    f1 = 2* (precision*recall) / (precision + recall + epsilon)
    return (np.mean(precision), np.mean(f1))

def train(model, device, train_loader, optimizer, loss_function, epoch, name,  log_path, num_classes=2, wandb_log=0):
    """
    Args: 
        model: Pytorch model instance
        train_loader: Dataset loader 
        other: other is as normal
       
    """
    #model.to(device)
    print(next(model.parameters()).device)
    model.train()
    correct = 0
    best_f1 = 0
    best_loss_so_far = 10
    running_loss = 0
    running_loss
    print(f"wandb {wandb_log}")
    iterator = iter(train_loader)
    
    for idx in tqdm(range(len(train_loader))):
        batch_data = next(iterator)
        data, target = batch_data.images.cuda(), batch_data.labels.cuda()
        optimizer.zero_grad()
        output = model(data).cuda()
        loss = loss_function(output, target).cuda()
        loss.backward()
        optimizer.step()
        pred = output.argmax(dim=1, keepdim=True)
        if (idx==0):
            preds=pred.flatten()
            outputs = output
            targets=target
        else:
            preds = torch.cat((preds, pred.flatten()),0)
            targets = torch.cat((targets, target),0)
            outputs = torch.cat((outputs, output),0)
        running_loss += loss.sum().item()   
        correct += pred.eq(target.view_as(pred)).sum().item()
    running_loss = running_loss/len(train_loader.dataset)
    f1 = f1_score(to_numpy(preds), to_numpy(targets), average="macro") 
    
    ap, f1_my = criterion(preds, targets, train=True)
    #roc = roc_auc_score(y_score = to_numpy(torch.sigmoid(outputs, dim=1)), \
                     #       y_true = to_numpy(torch.nn.functional.one_hot(targets, num_classes)), \
                      #     average = 'macro')
    #ap = precision_score(to_numpy(preds), to_numpy(targets), average="macro")
    
    if wandb_log==1:
        print(1)
        wandb.log({'Train loss': running_loss, 'F1': f1, "F1 (my)": f1_my,\
               'AP': ap}, step=epoch)
   
    print(
        "Train Epoch: {} \tLoss: {:.6f}    F1: {:.4f}    My F1: {:.4f}, AP: {:.4f}".format(
            epoch, running_loss, f1, f1_my, ap
        )
    )
    
    if running_loss < best_loss_so_far:
        best_loss_so_far = loss
        if (wandb_log==1):
            wandb.run.summary['Best train loss'] = loss
            wandb.run.summary['Best epoch'] = epoch
            torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, wandb.run.dir + name +".pth")
            
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, log_path + "trained_models/" + name +".pth")
        
    return len(targets), running_loss

def test_aug(p=0.5, image_size=128):
    return Compose(
      [  
          #HueSaturationValue(hue_shif_limit=5, sat_shift_limit=5, val_shift_limit=5, p=p),
          RandomBrightness(limit=0.1, p=p),
         # GaussNoise(var_limit=20, p=p),
          #ISONoise(p=p),
          RandomSizedCrop(min_max_height=(int(image_size*0.8), int(image_size)), height=image_size, width=image_size, p=1),
        #RandomCrop(image_size, image_size, p=1),
        HorizontalFlip(p=p)
      ]
    )

#@numba.jit(parallel=True)
def augment(data, image_size=224):
    for i in range(data.size()[0]):
    #print(data[0, 0,:,:])
        image_size = data[i].size()[-1]
        transf = test_aug(image_size = image_size)
        to_augment = { "image" : data[i]}
        to_augment["image"] = to_augment["image"].permute(2,1,0).detach().cpu().numpy()
        #print(to_augment)
        augmented = torch.from_numpy(transf(**to_augment)["image"])
        data[i] = augmented.permute(2,0,1)
    return data
    
def test(model, device, test_loader, loss_function, epoch, num_classes=2, wandb_log = 0):
    #print(next(model.parameters()).device)
    #model.to(device)
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    model.eval()
    test_loss = 0
    test_loss
    #correct = 0
    example_images = []
    iterator = []
    #for i in range(3):
    #    iterators.append(iter(test_loader))
    
    test_transforms = tta.Compose(
    [
        tta.HorizontalFlip(),
        #tta.Rotate90(angles=[0, 5]),
        tta.Scale(scales=[0.85, 1, 1.15]),
        #tta.Multiply(factors=[0.99, 1, 1.01]),        
    ]
)
    tta_model = tta.ClassificationTTAWrapper(model, test_transforms)
    
    with torch.no_grad():
        idx=0
        random.seed(42)
        for idx, batch_data in enumerate(tqdm(test_loader)):
            #print(idx)
            #aug_outputs = torch.empty()
            #for i in range(1):
            aug_outputs = []
            data, target = batch_data.images.cuda(), batch_data.labels.cuda()
            """#for i in range(2):
            start.record()
            data.append(batch_data.images.cuda())
            target.append(batch_data.labels.cuda())
            end.record()
            print(batch_data.ids)
            data.append()    """
            
            #for i in range(2):
                #augmented_images = test_transforms.augment_image(dummy_images)
                #data = augment(data)
                # target.cuda()
                # data, target = batch_data.images.cuda(), batch_data.labels.cuda()
                #print("++++++++++++++++++")
                #batch_data = test_loader[idx]
               # out = model(augmented_images).cuda()
               # aug_outputs.append(out)
            #aug_outputs = torch.stack(aug_outputs).cuda()
            #output = torch.stack(aug_outputs)
            #output = torch.mean(aug_outputs, dim=0).cuda()
            output = tta_model(data).cuda()
            
            
            test_loss += loss_function(output, target).cuda().sum().item()
            #print(test_loss.device)
            pred = tf.softmax(output).argmax(dim=1, keepdim=True)
            
            
            
            if (idx==0):
                preds=pred.flatten()
                targets=target
                outputs=output
            else:
                preds = torch.cat((preds, pred.flatten()),0).cuda()
                targets = torch.cat((targets, target),0).cuda()
                outputs = torch.cat((outputs, output),0).cuda()
            
            #torch.cuda.synchronize()
            #print(start.elapsed_time(end))
            #correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
   
            
    f1 = f1_score(to_numpy(preds), to_numpy(targets), average="macro") 
    #roc = roc_auc_score(y_score = to_numpy(torch.softmax(outputs, dim=1)), \
     #                       y_true = to_numpy(torch.nn.functional.one_hot(targets, num_classes)), \
     #                      average = 'macro')
    #ap = precision_score(to_numpy(preds), to_numpy(targets), average="macro")
    ap, f1_my = criterion(preds, targets)
    if(wandb_log==1):
        wandb.log({'Test loss': test_loss, 'Test F1': f1,  "Test F1 (my)": f1_my,\
             'Test AP': ap}, step=epoch)
    print(
        "Test set: Average loss: {:.4f}, F1: {:.4f}\n".format(
            test_loss,
            f1
        )
    )
    return len(targets)

def train_loop(args):
    """
        Does all train/test stuff. Prepares model, data etc.
        
        Args: Dict with:
            lr (float): Learning rate
            batch_size (int): Batch size
            image_size (int): Height of cropped square image in pixels
            path (str): Path to poject, paths to data, notebooks etc set as subdirectories.
            epochs (int): Num of epochs for learning
            num_classes (int): Number of classes
            base (str): Base model, as resnet18 or resnet50
            type (str): Type of learning, as ft (fine-tune), tl (transfer) or ae (autoencoder)
            tags (str): String what makes your model name individual!
            wandb (bool): Log to wandb
              
    """

    LR = float(args.lr)
    BATCH_SIZE = int(args.batch_size)
    IMAGE_SIZE = int(args.image_size)
    
    PATH = args.path
    LOG_PATH = PATH +'logs/'
    DATA_PATH = PATH + 'data/'
    DEVICE = torch.device("cuda")
    
    print(DEVICE)
    
    print('Using device:', DEVICE)
    print(torch.cuda.is_available())

    #Additional Info when using cuda
    if DEVICE.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')
        
    EPOCHS  = int(args.epochs)
    NUM_CLASSES = int(args.num_classes)
    TYPE = args.type
    BASE = args.base
    NAME = args.tags + '_'+ BASE + '_' + TYPE + '_' + str(LR) + '_' + str(NUM_CLASSES)
    WANDB = int(args.wandb)
    WD = float(args.wd)
    B1 = float(args.b1)
    B2 = float(args.b2)
    
    INP_SIZE = int(args.inp_size)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True
    #torch.set_default_tensor_type('torch.cuda.FloatTensor')
    
    if WANDB==1:
        print('wandb!')
        wandb.init(project="kern", name=NAME, dir=LOG_PATH)
        wandb.config.Tags = args.tags.split('#')
    transform = strong_aug(p=0.5, image_size=IMAGE_SIZE)

    print("==> Preparing data")
    

    if (BASE.find('resnet')!=-1):
        model, optimizer, loss = prepare_base_model(lr=LR, device=DEVICE, name=BASE, weight_decay=WD, beta_1=B1, beta_2=B2)
    elif (BASE.find('auto')!=-1):
        model, optimizer, loss = auto_prepare_model(lr=LR, device=DEVICE, name=BASE, inp_size = INP_SIZE, weight_decay=WD, beta_1=B1, beta_2=B2)
    else:
        model, optimizer, loss = prepare_eff_model(lr=LR, device=DEVICE, name=BASE, inp_size = INP_SIZE, weight_decay=WD, beta_1=B1, beta_2=B2, im_size=IMAGE_SIZE)
    model.cuda()
    model = torch.nn.DataParallel(model, device_ids=[0])
    train_loader, test_loader = prepare_dataset(csv_file_uf='data_uf.csv',csv_file_dc='data_dc.csv',
                                        root_dir=DATA_PATH, transform = transform, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, train_prop=0.7, num_workers=8, assign=True)
    torch.save({
            'epoch': 0,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, LOG_PATH + "trained_models/" + NAME +".pth")
    if (WANDB == 1):
        save_files_to_wandb(log_path = LOG_PATH, name=NAME) 
    print("==> Training model")
    for epoch in range(EPOCHS):
        test_len = test(model, DEVICE, test_loader, loss, epoch, num_classes=NUM_CLASSES, wandb_log=WANDB)
        train_len, train_loss = train(model, DEVICE, train_loader, optimizer, loss, epoch, name = NAME, num_classes=NUM_CLASSES, log_path = LOG_PATH, wandb_log = WANDB)
    
    
        
        if WANDB==1:   
            save_files_to_wandb(log_path = LOG_PATH, name=NAME) 
            wandb.config.update({
                "name": NAME,
                "type": TYPE,
                "epochs" : epoch,
                "batch_size" : BATCH_SIZE,
                "img_dim" : IMAGE_SIZE,
                "num_classes" : NUM_CLASSES,
                "n_train" : train_len,
                "n_valid" : test_len,
                "lr" : LR,
                "base_model" : BASE 
                    }, allow_val_change=True)
    test_len = test(model, DEVICE, test_loader, loss, epoch, num_classes=NUM_CLASSES, wandb_log=WANDB)
    
def main():
    parser = argparse.ArgumentParser(description="Train and test loop! Default model: resnet18.                                      NAME = TAGS_BASE_TYPE_LR_NUM_CLASSES.")
    parser.add_argument("--lr", default=1e-3, help="Set up learning rate, default: 1e-3")
    parser.add_argument("--batch_size", default=64, help="Set up batch size, default: 64")
    parser.add_argument("--image_size", default=128, help="Set up image size, default: 128 (for rn18)")
    parser.add_argument("--path", default='/headless/shared/kern_segmentation/', help="Set up working dir, default: /project/")
    parser.add_argument("--tags", default='', help="Make model name individual")
    parser.add_argument("--epochs", default=200, help="Set up num of epochs, default: 100")
    parser.add_argument("--num_classes", default=6, help="Set up num of classes, default: 6 (oil)")
    parser.add_argument("--type", default="ft", help="Log type of learning, default: fine-tune (ft)")
    parser.add_argument("--base", default="resnet18", help="Log type of base model, default: resnet18")
    parser.add_argument("--wandb", default=True, help="Log to wandb, default: True")
    parser.add_argument("--inp_size", default=1280, help="FC layer input size, default:1280")
    parser.add_argument("--b1", default=0.9, help="Beta 1, default: 0.9")
    parser.add_argument("--b2", default=0.999, help="Beta 2, default: 0.999")
    parser.add_argument("--wd", default=1e-2, help="Weight decay, default: 1e-2")
    args = parser.parse_args()
    train_loop(args)
if __name__ == "__main__":
    try:
        torch.multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass
    
    #set_start_method('spawn', force=True)
    main()
