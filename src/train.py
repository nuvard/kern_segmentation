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


def criterion(y_pred, y_true, epsilon = 1e-6, num_classes = 6):
    y_pred = to_numpy(tf.one_hot(y_pred, num_classes))
    y_true = to_numpy(tf.one_hot(y_true, num_classes))
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
    model.train()
    model.to(device)
    correct = 0
    best_f1 = 0
    best_loss_so_far = 10
    running_loss = 0
    print(f"wandb {wandb_log}")
    for idx, batch_data in enumerate(tqdm(train_loader)):
        data, target = batch_data.images.to(device), batch_data.labels.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target)
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
    ap, f1_my = criterion(preds, targets)
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
            wandb.save(os.path.join(wandb.run.dir, name+'.h5'))
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, log_path + "trained_models/" + name +".pth")
    return len(targets), running_loss

def test(model, device, test_loader, loss_function, epoch, num_classes=2, wandb_log = 0):
    model.eval()
    model.to(device)
    test_loss = 0
    correct = 0
    example_images = []
    with torch.no_grad():
        for idx, batch_data in enumerate(tqdm(test_loader)):
            aug_outputs = []
            for i in range(3):
                data, target = batch_data.images.to(device), batch_data.labels.to(device)
                aug_outputs.append(model(data))
            output = torch.stack(aug_outputs)
            output = torch.mean(output, dim=0) 
            test_loss += loss_function(output, target).sum().item()
            pred = output.argmax(dim=1, keepdim=True)
            if (idx==0):
                preds=pred.flatten()
                targets=target
                outputs=output
            else:
                preds = torch.cat((preds, pred.flatten()),0)
                targets = torch.cat((targets, target),0)
                outputs = torch.cat((outputs, output),0)
            correct += pred.eq(target.view_as(pred)).sum().item()
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
        "\nTest set: Average loss: {:.4f}, F1: {:.4f}\n".format(
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
    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
    if WANDB==1:
        print('wandb!')
        wandb.init(project="kern", name=NAME, dir=LOG_PATH)
        wandb.config.Tags = args.tags.split('#')
    transform = strong_aug(p=0.5, image_size=IMAGE_SIZE)

    print("==> Preparing data")
    train_loader, test_loader = prepare_dataset(csv_file_uf=DATA_PATH+'data_uf.csv',csv_file_dc=DATA_PATH+'data_dc.csv',
                                        root_dir=DATA_PATH, transform = transform, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, train_prop=0.8)

    if (BASE.find('resnet')!=-1):
        model, optimizer, loss = prepare_base_model(lr=LR, device=DEVICE, name=BASE, weight_decay=WD, beta_1=B1, beta_2=B2)
    elif (BASE.find('auto')!=-1):
        model, optimizer, loss = auto_prepare_model(lr=LR, device=DEVICE, name=BASE, inp_size = INP_SIZE, weight_decay=WD, beta_1=B1, beta_2=B2)
    else:
        model, optimizer, loss = prepare_eff_model(lr=LR, device=DEVICE, name=BASE, inp_size = INP_SIZE, weight_decay=WD, beta_1=B1, beta_2=B2)

    torch.save({
            'epoch': 0,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, LOG_PATH + "trained_models/" + NAME +".pth")
    if (WANDB == 1):
        save_files_to_wandb(log_path = LOG_PATH, name=NAME) 
    print("==> Training model")
    for epoch in range(EPOCHS):
        train_len, train_loss = train(model, DEVICE, train_loader, optimizer, loss, epoch, name = NAME, num_classes=NUM_CLASSES, log_path = LOG_PATH, wandb_log = WANDB)
        test_len = test(model, DEVICE, test_loader, loss, epoch, num_classes=NUM_CLASSES, wandb_log=WANDB)
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
        
def main():
    parser = argparse.ArgumentParser(description="Train and test loop! Default model: resnet18.                                      NAME = TAGS_BASE_TYPE_LR_NUM_CLASSES.")
    parser.add_argument("--lr", default=1e-3, help="Set up learning rate, default: 1e-3")
    parser.add_argument("--batch_size", default=64, help="Set up batch size, default: 64")
    parser.add_argument("--image_size", default=128, help="Set up image size, default: 128 (for rn18)")
    parser.add_argument("--path", default='/project/', help="Set up working dir, default: /project")
    parser.add_argument("--tags", default='', help="Make model name individual")
    parser.add_argument("--epochs", default=200, help="Set up num of epochs, default: 100")
    parser.add_argument("--num_classes", default=6, help="Set up num of classes, default: 6 (oil)")
    parser.add_argument("--type", default="ft", help="Log type of learning, default: fine-tune (ft)")
    parser.add_argument("--base", default="resnet18", help="Log type of base model, default: resnet18")
    parser.add_argument("--wandb", default=True, help="Log to wandb, default: True")
    parser.add_argument("--inp_size", default=1280, help="FC layer input size, default:1280")
    parser.add_argument("--b1", default=0.9, help="Beta 1, default: 0.9")
    parser.add_argument("--b2", default=0.999, help="Beta 2, default: 0.999")
    parser.add_argument("--wd", default=1e-3, help="Weight decay, default: 1e-3")
    args = parser.parse_args()
    train_loop(args)
if __name__ == "__main__":
    main()
