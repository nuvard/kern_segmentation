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
from dataset import *
import shutil
import argparse

def train(model, device, train_loader, optimizer, loss_function, epoch, name,  log_path, num_classes=2):
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
    roc = roc_auc_score(y_score = to_numpy(torch.softmax(outputs, dim=1)), \
                            y_true = to_numpy(torch.nn.functional.one_hot(targets, num_classes)), \
                           average = 'macro')
    ap = precision_score(to_numpy(preds), to_numpy(targets), average="macro")
    
    wandb.log({'Train loss': running_loss, 'F1': f1, "ROC-AUC": roc,\
               'AP': ap}, step=epoch)
   
    print(
        "Train Epoch: {} \tLoss: {:.6f}    F1: {:.4f}    ROC-AUC: {:.4f}".format(
            epoch, running_loss, f1, roc
        )
    )
    
    if running_loss < best_loss_so_far:
        best_loss_so_far = loss
        wandb.run.summary['Best train loss'] = loss
        wandb.run.summary['Best epoch'] = epoch
        wandb.save(os.path.join(wandb.run.dir, name+'.h5'))
        torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, log_path + "trained_models/" + name +".pth")
    

                
    
def test(model, device, test_loader, loss_function, epoch, num_classes=2):
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
    roc = roc_auc_score(y_score = to_numpy(torch.softmax(outputs, dim=1)), \
                            y_true = to_numpy(torch.nn.functional.one_hot(targets, num_classes)), \
                           average = 'macro')
    ap = precision_score(to_numpy(preds), to_numpy(targets), average="macro")
    
    wandb.log({'Test loss': test_loss, 'Test F1': f1,  "Test ROC-AUC": roc,\
             'Test AP': ap}, step=epoch)
    print(
        "\nTest set: Average loss: {:.4f}, F1: {:.4f}\n".format(
            test_loss,
            f1
        )
    )

def main():
    parser = argparse.ArgumentParser(description='Train and test loop! Default model: resnet18.')
    parser.add_argument("--lr", default=1e-3, help="Set up learning rate, default: 1e-3")
    parser.add_argument("--batch_size", default=64, help="Set up batch size, default: 64")
    parser.add_argument("--image_size", default=128, help="Set up image size, default: 128 (for rn18)")
    parser.add_argument("--path", default='/project/', help="Set up working dir, default: /project")
    parser.add_argument("--name", default='rn-18', help="Set up model name")
    parser.add_argument("--epochs", default=100, help="Set up num of epochs, default: 100")
    parser.add_argument("--num_classes", default=6, help="Set up num of classes, default: 6 (oil)")
    args = parser.parse_args()
    LR = float(args.lr)
    BATCH_SIZE = int(args.batch_size)
    IMAGE_SIZE = int(args.image_size)
    NAME = args.name
    PATH = args.path
    LOG_PATH = PATH +'logs/'
    DATA_PATH = PATH + 'data/'
    DEVICE = torch.device("cpu")
    EPOCHS  = int(args.epochs)
    NUM_CLASSES = int(args.num_classes)
    set_seed()
    wandb.init(project="kern", name=NAME, dir=LOG_PATH)
    print("Super!")
    
    transform = strong_aug(p=0.5, image_size=IMAGE_SIZE)


    dataset = KernDataset(csv_file_uf=DATA_PATH+'data_uf.csv',csv_file_dc=DATA_PATH+'data_dc.csv',
                                        root_dir=DATA_PATH, transform = strong_aug(p=0.5), image_size=IMAGE_SIZE)

    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=DatasetItem.collate,
        num_workers=4,
        worker_init_fn=set_seed()
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=DatasetItem.collate,
        num_workers=4,
        worker_init_fn=set_seed()
    )

    model = models.resnet18(pretrained=True)
    model.fc = nn.Linear(512, 6)
    #print(model)
    temp = model.conv1.weight
    model.conv1 = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1.weight = nn.Parameter(torch.cat((temp,temp),dim=1))
    #print(model.conv1.weight.size())
    #torch.nn.init.xavier_uniform_(model_dc.fc.bias)
    #with torch.no_grad():
    #  model_dc.fc.bias = nn.Parameter(torch.Tensor([0.5, 0.5]))
     
    model = model.to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    loss_function = loss.CrossEntropyLoss()
    torch.save({
            'epoch': 0,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict()
        }, LOG_PATH + "trained_models/" + NAME +".pth")
    save_files_to_wandb(log_path = LOG_PATH, name=NAME) 
    for epoch in range(EPOCHS):
        train(model, DEVICE, train_loader, optimizer, loss_function, epoch, name = NAME, num_classes=NUM_CLASSES, log_path = LOG_PATH)
        save_files_to_wandb(log_path = LOG_PATH, name=NAME) 
        wandb.config.update({
        "name": NAME,
        "type": 'fine-tune',
        "epochs" : epoch,
        "batch_size" : BATCH_SIZE,
        "img_dim" : IMAGE_SIZE,
        "num_classes" : NUM_CLASSES,
        "n_train" : len(train_dataset),
        "n_valid" : len(test_dataset),
        "fc_size" : 512,
        "lr" : LR,
        "base_model" : 'resnet18' 
            }, allow_val_change=True)
        test(model, DEVICE, test_loader, loss_function, epoch, num_classes=NUM_CLASSES)

if __name__ == "__main__":
    main()
