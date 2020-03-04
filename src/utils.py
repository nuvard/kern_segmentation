
from __future__ import print_function, division
from pathlib import Path

import pandas as pd
import torch 
import random
import os
from pandas import DataFrame
from PIL import Image
import shutil
from matplotlib import pyplot as plt
import numpy as np
from skimage import io, transform
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
plt.ion()   # interactive mode
import datetime
import wandb
from torchvision import transforms

def test_device(device: str):
    torch.ones(1,2,3).to(device)


def set_seed():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.manual_seed(42)

    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    
def apply_class(data, column, class_dict):
    data['class']=data[column].apply(lambda x:class_dict[x] if (x in class_dict) else x) 
    
    
def check(PATH, data):
    to_drop = []
    for i in data.index:
        img_name=data.loc[i, 'Folder']+'/data/'+str(data.loc[i, 'Id'])+'.jpeg'

        #plt.figure()

        if not os.path.isfile(os.path.join(PATH, img_name)):
            to_drop.append(i)
            print("No image " + img_name)
        #plt.show()
    if(len(to_drop)==0):
        print ("All is OK!")

    return data.loc[~data.index.isin(to_drop),]  

def pil2tensor(image1, image2):
    "Convert PIL style `image` array to torch style image tensor."
    print('pil2tensor')
    image2 = image2.resize(image1.size)
    a = torch.from_numpy(np.asarray(image1))
    b = torch.from_numpy(np.asarray(image2))
    concated = torch.cat((a,b),-1)
    return concated

def show_batch_image(image_batch, batch_size=4, name='out.png', a = 0, b = 3 ):
    """
    Show a sample grid image which contains some sample of test set result
    :param image_batch: The output batch of test set
    :param a: - first channel to draw
    :param b: - (b-1) is last channel to draw
    :return: PIL image of all images of the input batch
    """
    print('show!')
    #inv_normalize = transforms.Normalize(
    #mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255],
    #std=[1/0.229, 1/0.224, 1/0.255])
    
    to_pil =  transforms.ToPILImage()
    fs = []
    for i in range(batch_size):
        img = to_pil(image_batch.images[i,a:b,:,:].cpu())
        fs.append(img)
    x, y = fs[0].size
    ncol = int(np.ceil(np.sqrt(batch_size)))
    nrow = int(np.ceil(np.sqrt(batch_size)))
    cvs = Image.new('RGB', (x * ncol, y * nrow))
    for i in range(len(fs)):
        px, py = x * int(i / nrow), y * (i % nrow)
        cvs.paste((fs[i]), (px, py))
    #print(name)
    cvs.save(name, format='png')
    #cvs.show()
    plt.imshow(np.asarray(cvs))


def save_files_to_wandb(log_path, name):
    if not os.path.isdir(os.path.join(wandb.run.dir, 'trained_models')):
        os.mkdir(os.path.join(wandb.run.dir, 'trained_models'))
    files_to_copy = [log_path + "trained_models/" + name +".pth"]
    for fname in files_to_copy:
        dest_path = os.path.join(wandb.run.dir, fname)
        if (fname==dest_path):
            return 1
        if os.path.isdir(fname):
            shutil.copytree(fname, dest_path)
        else:
            shutil.copy2(fname, dest_path)       
        return 0
            
def log_wandb_images(inputs, labels, output, idx):

    inps_cpu = inputs[0][0:3,:,:].detach().cpu().numpy()
    x_image = Image.fromarray(np.uint8(np.round((x + 0.5)*255)))
    input_images.append(wandb.Image(x_image))
    
    wandb.log({"examples": payload}, commit=False)
    
