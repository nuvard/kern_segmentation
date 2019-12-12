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
from efficientnet_pytorch import EfficientNet
from efficientnet_pytorch.utils import Conv2dStaticSamePadding
from attention_augmented_conv import AugmentedConv
#from attention_augmented_conv import AttentionConv2d
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
    return (model.cuda(), optimizer, loss_function.cuda())


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



def prepare_eff_model(device, name ='effitientnet_b0',  lr=1e-5, beta_1=0.9, beta_2=0.999, weight_decay=1e-3, inp_size = 1280, im_size=224):
    """
    Args:
      device: torch device (like torch.device("cuda"))
      lr: learning rate, default = 1e-5
      
    Returns: 
      tuple of (model, optimizer, loss)
    """      
    print("==> Preparing model")
    if(name.find('_')!=-1):
        torch.hub.list('rwightman/gen-efficientnet-pytorch') 
        model =  torch.hub.load('rwightman/gen-efficientnet-pytorch', name, pretrained=True)
        #print(model) 
        """
        model.global_pool = nn.Sequential(
            nn.Conv2d(1280, 6, kernel_size=1, stride=1, bias=False),
            nn.Dropout(0.2),
                      nn.ReLU(True),
                      nn.AdaptiveAvgPool2d(1))
        
        model.classifier = nn.Sequential(#nn.Dropout(0.2),
                      #nn.ReLU(True),
                      #nn.AdaptiveAvgPool1d(6),
                      nn.Linear(6, 6)    
                      )   
        """
        model.global_pool = nn.Sequential(
                       #AugmentedConv(in_channels=1280, out_channels=200, kernel_size=1, dk=40, dv=4, Nh=1, relative=False, stride=2),
                     #nn.Conv2d(588, 6, kernel_size=1, padding = 1, stride=1, bias=False),
                     nn.BatchNorm2d(1280), 
                     nn.Dropout(p=0.25),
                     nn.AdaptiveAvgPool2d(1)
                     
                     
                      )
        
        model.classifier = nn.Sequential(
           
            #nn.BatchNorm1d(6),
            
                     #nn.ReLU(),
            
            #nn.Dropout(p=0.25),
            nn.BatchNorm1d(1280, eps=1e-05, momentum=0.1),
            nn.Dropout(p=0.5),
            
            nn.ReLU(),
            #nn.AdaptiveAvgPool2d(1),
            nn.Linear(200, 6),
           # nn.AdaptiveAvgPool2d(1),
        ) 
        print("Adding attention");
        temp = model.conv_stem.weight
        #model.conv_stem = AttentionConv2d(in_channels=6, out_channels=64, kernel_size=7, dk=40, dv=4, Nh=4, relative=True, stride=2, padding=3, shape = 24).to(device)
        
        model.conv_stem = nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.conv_stem.weight = nn.Parameter(torch.cat((temp,temp),dim=1))
        temp = model.conv_stem.weight
        model.conv_stem = nn.Sequential(
        AugmentedConv(in_channels=6, out_channels=6, kernel_size=1, dk=40, dv=4, Nh=1, relative=False, stride=1),    
        nn.Conv2d(6, 64, kernel_size=7, stride=2, padding=3, bias=False),)
    # model.cuda()
        #print(model)
    else:
        
        model = EfficientNet.from_pretrained(name) 
        #print(model)

        model._fc = nn.Sequential(nn.Dropout(0.2),
                      #nn.ReLU(True),
                      nn.Linear(inp_size, 6)
                      )    
        #print(model)
        temp = model._conv_stem.weight
    
        model._conv_stem = Conv2dStaticSamePadding(6, 32, image_size=im_size, kernel_size=3, bias=False)
        model._conv_stem.weight = nn.Parameter(torch.cat((temp,temp),dim=1))
    optimizer = optim.Adam(model.parameters(), lr=lr, betas=(beta_1,beta_2), weight_decay=weight_decay)
    #print(model)
    #model.cuda()
    loss_function = loss.CrossEntropyLoss().cuda()
    return (model.cuda(), optimizer, loss_function)
