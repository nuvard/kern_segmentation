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
from autoencoder import Autoencoder


"""
Prepares model, loss and optimizer. 

Model base: resnet18
Optimizer: Adam
Loss: Crossentropy

Typical usage example:
model, optimizer, loss = prepare_model(lr=LR, device=DEVICE)

"""

def auto_prepare_model(device, lr=1e-5):
    """
    Args:
      device: torch device (like torch.device("cuda"))
      lr: learning rate, default = 1e-5
      
    Returns: 
      tuple of (model, optimizer, loss)
    """
    
    autoencoder = Autoencoder()
    checkpoint = torch.load('/project/logs/trained_models/autoencoder.pth')
    
    autoencoder.load_state_dict(checkpoint)
    #print(autoencoder)                        
    resnet = models.resnet18(pretrained=True)
    resnet.fc = nn.Sequential(
                      #nn.Dropout(0.2),
                      #nn.ReLU(True),

                      nn.Linear(512, 6)
                      )                      
    #print(model)
    #temp = model.conv1.weight
    #model.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    #model.conv1.weight = nn.Parameter(torch.cat((temp,temp),dim=1))
        
    
    model = nn.Sequential(autoencoder.encoder.to(device), resnet.to(device))
    #print(model)
    optimizer = optim.Adam(model.resnet.parameters(), lr=lr)

    loss_function = loss.CrossEntropyLoss()
    return (model, optimizer, loss_function)


def prepare_model(device, lr=1e-5):
    """
    Args:
      device: torch device (like torch.device("cuda"))
      lr: learning rate, default = 1e-5
      
    Returns: 
      tuple of (model, optimizer, loss)
    """
    
    autoencoder = Autoencoder()
    checkpoint = torch.load('/project/logs/trained_models/autoencoder.pth')
    
    autoencoder.load_state_dict(checkpoint)
    #print(autoencoder)                        
    resnet = models.resnet18(pretrained=True)
    resnet.fc = nn.Sequential(
                      nn.Dropout(0.2),
                      #nn.ReLU(True),

                      nn.Linear(512, 6)
                      )                      
    #print(model)
    #temp = model.conv1.weight
    #model.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
    #model.conv1.weight = nn.Parameter(torch.cat((temp,temp),dim=1))
        
    
    model = nn.Sequential()
    model.add_module('encoder', autoencoder.encoder)
    model.add_module('resnet', resnet.to(device))
    optimizer = optim.Adam(model.resnet.parameters(), lr=lr)

    loss_function = loss.CrossEntropyLoss()
    return (model, optimizer, loss_function)



def prepare_base_model(device, name = 'resnet18', lr=1e-5, beta_1=0.9, beta_2=0.999, weight_decay=1e-3):
    """
    Args:
      device: torch device (like torch.device("cuda"))
      lr: learning rate, default = 1e-5
      
    Returns: 
      tuple of (model, optimizer, loss)
    """              
    model = models.resnet18(pretrained=True)
    model.fc = nn.Sequential(
                      nn.Dropout(0.2),
                      #nn.ReLU(True),

                      nn.Linear(512, 6)
                      )       
                                           
    temp = model.conv1.weight
    
    model.conv1 = nn.Conv2d(6, 32, kernel_size=3, stride=2, padding=1, bias=False)
    model.conv1.weight = nn.Parameter(torch.cat((temp,temp),dim=1))
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta_1,beta_2), weight_decay=weight_decay)

    loss_function = loss.CrossEntropyLoss()
    return (model, optimizer, loss_function)



def prepare_eff_model(device, name ='effitientnet_b0',  lr=1e-5, beta_1=0.9, beta_2=0.999, weight_decay=1e-3, inp_size = 1280):
    """
    Args:
      device: torch device (like torch.device("cuda"))
      lr: learning rate, default = 1e-5
      
    Returns: 
      tuple of (model, optimizer, loss)
    """      
    print("==> Preparing model")
    torch.hub.list('rwightman/gen-efficientnet-pytorch') 
    model =  torch.hub.load('rwightman/gen-efficientnet-pytorch', name, pretrained=True)
    
    model.classifier = nn.Sequential(nn.Dropout(0.2),
                      #nn.ReLU(True),
                      nn.Linear(inp_size, 6)
                      )    
    
    temp = model.conv_stem.weight
    
    model.conv_stem = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv_stem.weight = nn.Parameter(torch.cat((temp,temp),dim=1))
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta_1,beta_2), weight_decay=weight_decay)
    #print(model)
    loss_function = loss.CrossEntropyLoss()
    return (model, optimizer, loss_function)
