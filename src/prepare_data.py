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

import numpy as np
from torchvision import transforms

import torchvision.transforms.functional as TF

import os
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


from albumentations import RandomCrop, Normalize, HorizontalFlip, Resize, RandomSizedCrop
from albumentations import Compose
from albumentations.pytorch import ToTensor
import datetime
from utils import *

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from sampler import ImbalancedDatasetSampler




def strong_aug(p=0.5, image_size=128):
    return Compose(

      [ RandomSizedCrop(min_max_height=(int(image_size*0.8), int(image_size)), height=image_size, width=image_size, p=1),
        #RandomCrop(image_size, image_size, p=1),
        HorizontalFlip(p=p),
        
        Normalize([0.4802, 0.4481, 0.3975, 0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262,0.2302, 0.2265, 0.2262])
      ], 
    )

def to_numpy(target):
    return target.detach().cpu().numpy()


@dataclass()
class ItemsBatch:
    images: torch.Tensor
    labels: torch.Tensor
    ids: List[int]
    paths: List[Path]
    items: List["DatasetItem"]


@dataclass()
class DatasetItem:
    image: Union[torch.Tensor, Image.Image]
    label: int
    id: int
    path: Path

    @classmethod
    def collate(cls, items: Sequence["DatasetItem"]) -> ItemsBatch:
        if not isinstance(items, list):
            items = list(items)
            #print(default_collate([item.label for item in items]))
        return ItemsBatch(
            images=default_collate([item.image for item in items]),
            
            labels=default_collate([item.label for item in items]),
            ids=[item.id for item in items],
            paths=[item.path for item in items],
            items=items,
        )

class KernDataset(Dataset):
    """Kern dataset."""

    def __init__(self, csv_file_dc,csv_file_uf, root_dir, transform=None, acc=0.2, image_size = 128):
        np.random.seed(42)
        """
        Args:
            csv_file_dc(string): Path to the csv file with annotations for DL images.
            csv_file_uf(string): Path to the csv file with annotations for UF images
            acc (float): Min with of segment in meters.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
            
        """
        self.acc = acc
        self.data_uf = pd.read_csv(csv_file_uf)
        self.labels_path = csv_file_uf
        self.root_dir = root_dir
        self.transform = transform
        self.data_dc = pd.read_csv(csv_file_dc)
        self.image_size = image_size
    
    def __len__(self):
        return len(self.data_uf)
    
      
    def __getitem__(self, idx):
        #print('idx '+str(idx))
        dc_img_name = os.path.join(self.root_dir,
                                self.data_dc.loc[idx, 'Folder']+'/data/'+str(self.data_dc.loc[idx, 'Id'])+'.jpeg')
        
        uf_img_name = os.path.join(self.root_dir,
                                self.data_uf.loc[idx, 'Folder']+'/data/'+str(self.data_uf.loc[idx, 'Id'])+'.jpeg')
        
        layer_width = self.data_dc.loc[idx, "LayerDown"] - self.data_dc.loc[idx, "LayerTop"]
        image_dc = Image.open(dc_img_name)
        image_uf = Image.open(uf_img_name).resize(image_dc.size)
        
        crop_size = min(image_dc.size[1], int(image_dc.size[1]*(self.acc/layer_width))-1)
            
        image_np = np.concatenate((np.array(image_dc),np.array(image_uf)), axis=-1)
        
        transf = RandomCrop(crop_size,image_dc.size[0] )
        image_np = transf(image=image_np)['image']
        #print('image_size'+str(image_np.shape))
        #print('crop_size' + str( crop_size))
        #print('desired size:' + str(int(1.1*self.image_size+1)))
        if (crop_size<int(self.image_size*1.3)+1):
            transf = Resize(int(1.3*self.image_size), int(image_dc.size[0]*(int(1.3*self.image_size)/crop_size)+1))
            image_np = transf(image=image_np)['image']
        if (transform is not None):
            augmented = self.transform(image = image_np)['image']
            augmented = torch.from_numpy(np.moveaxis(augmented / (255.0 if augmented.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
            
        label = self.data_dc.loc[idx, 'class']
        return DatasetItem(image=augmented, label=label, id=idx, path=dc_img_name)

def prepare_dataset(csv_file_uf, csv_file_dc, root_dir, transform, image_size=128, batch_size=64, num_workers=4, train_prop=0.8 ):
    dataset = KernDataset(csv_file_uf=csv_file_uf,csv_file_dc=csv_file_dc,
                                        root_dir=root_dir, transform = transform, image_size=image_size)

    train_size = int(train_prop * len(dataset))
    test_size = len(dataset) - train_size

    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=ImbalancedDatasetSampler(train_dataset),
        collate_fn=DatasetItem.collate,
        num_workers=num_workers,
        worker_init_fn=set_seed()
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=ImbalancedDatasetSampler(test_dataset),
        collate_fn=DatasetItem.collate,
        num_workers=num_workers,
        worker_init_fn=set_seed()
    )
    return (train_loader, test_loader)
    
    
def train_test_split(data_uf, validation_split=0.3, shuffle=True):
    shuffle_dataset = shuffle
    random_seed = 42
    dataset_size = len(data_uf)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    retun (indices[split:], indices[:split])
