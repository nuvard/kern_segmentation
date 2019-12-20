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
from torch.utils.data.sampler import WeightedRandomSampler, RandomSampler
import numpy as np
#from torchvision import transforms

#import torchvision.transforms.functional as TF

import os
from torch.utils.data import Dataset, DataLoader
from torchvision import utils


from albumentations import RandomCrop, Normalize, HorizontalFlip, Resize, RandomSizedCrop, HueSaturationValue, RandomBrightness, GaussNoise,ISONoise, RandomContrast
from albumentations import Compose
from albumentations.pytorch import ToTensor
import datetime
from utils import *
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")
from sampler import ImbalancedDatasetSampler

import pandas as pd

def strong_aug(p=0.5, image_size=128):
    return Compose(
      [  
          #HueSaturationValue(hue_shif_limit=5, sat_shift_limit=5, val_shift_limit=5, p=p),
          RandomBrightness(limit=0.02, p=p),
          GaussNoise(var_limit=5, p=p),
          #RandomContrast(p=p),
          #ISONoise(p=p),
          RandomSizedCrop(min_max_height=(int(image_size*0.7), int(image_size)), height=image_size, width=image_size, p=0.8),
        #RandomCrop(image_size, image_size, p=1),
        HorizontalFlip(p=p),
        
        Normalize([0.4802, 0.4481, 0.3975, 0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262,0.2302, 0.2265, 0.2262])
      ], 
    )
def test_aug(p=0.5, image_size=128):
    return Compose(
      [  
          #HueSaturationValue(hue_shif_limit=5, sat_shift_limit=5, val_shift_limit=5, p=p),
         # RandomBrightness(limit=0.02, p=p),
         # GaussNoise(var_limit=20, p=p),
          #ISONoise(p=p),
          #RandomSizedCrop(min_max_height=(int(image_size*0.8), int(image_size)), height=image_size, width=image_size, p=1),
        #RandomCrop(image_size, image_size, p=1),
        #HorizontalFlip(p=p),
        
        Normalize([0.4802, 0.4481, 0.3975, 0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262,0.2302, 0.2265, 0.2262])
      ], 
    )

def to_numpy(target):
    return target.detach().cpu().numpy()


@dataclass()
class ItemsBatch:
    images: torch.Tensor
    labels: torch.Tensor
    ids: torch.Tensor
    paths: List[Path]
    items: List["DatasetItem"]


@dataclass()
class DatasetItem:
    image: Union[torch.Tensor, Image.Image]
    label: torch.Tensor
    id: int
    path: Path

    @classmethod
    def collate(cls, items: Sequence["DatasetItem"]) -> ItemsBatch:
        #mp.set_start_method('spawn')
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

    def __init__(self, csv_file_dc, csv_file_uf, root_dir, classes, transform=None, acc=0.2, image_size = 128):
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
        #print(root_dir+csv_file_uf)
        self.acc = acc
        self.data_uf = pd.read_csv(root_dir+csv_file_uf)
        self.labels_path = root_dir+csv_file_uf
        self.classes = pd.read_csv(root_dir+classes)
        self.root_dir = root_dir
        self.transform = transform
        self.data_dc = pd.read_csv(root_dir+csv_file_dc)
        self.image_size = image_size
    
    def __len__(self):
        return len(self.data_uf)
    

      
    def __getitem__(self, idx):
        device = torch.device("cpu")
        #print('idx '+str(idx))
        dc_img_name = os.path.join(self.root_dir,
                                self.data_dc.loc[idx, 'Folder']+'/data/'+str(self.data_dc.loc[idx, 'Id'])+'.jpeg')
        
        uf_img_name = os.path.join(self.root_dir,
                                self.data_uf.loc[idx, 'Folder']+'/data/'+str(self.data_uf.loc[idx, 'Id'])+'.jpeg')
        
        layer_width = self.data_dc.loc[idx, "LayerDown"] - self.data_dc.loc[idx, "LayerTop"]
        image_dc = Image.open(dc_img_name)
        image_uf = Image.open(uf_img_name).resize(image_dc.size)
        crop_size = min(self.image_size,image_dc.size[0], image_dc.size[1], int(image_dc.size[1]*(self.acc/layer_width))-1)
            
        image_np = np.concatenate((np.array(image_dc),np.array(image_uf)), axis=-1)
        transf = RandomCrop(crop_size,crop_size )
        
        image_np = transf(image=image_np)['image']
        if (crop_size<int(self.image_size)):
            transf = Resize(int(self.image_size), int(self.image_size))
            image_np = transf(image=image_np)['image']
        #print('image_size'+str(image_np.shape))
        #print('crop_size' + str( crop_size))
        #print('desired size:' + str(int(1.1*self.image_size+1)))
        
        if (transform is not None):
            augmented = self.transform(image = image_np)['image']
            torch_augmented = torch.from_numpy(np.moveaxis(augmented / (255.0 if augmented.dtype == np.uint8 else 1), -1, 0).astype(np.float32))
            torch_augmented = torch_augmented
            #print(torch_augmented.device)
        label = torch.as_tensor(self.classes.loc[idx,], dtype=torch.float)
        return DatasetItem(image=torch_augmented, label=label, id=idx, path=dc_img_name)
    
def get_label_weights_from_pandas(data):
    labels_list = []
    for i in data['class'].value_counts():
        labels_list.append(1/6)
    #print(labels_list)
    return labels_list
    
def prepare_dataset(csv_file_uf, csv_file_dc, classes, root_dir, transform, image_size=128, batch_size=64, num_workers=8, train_prop=0.7 , assign=False):
    #print('train_'+csv_file_uf)
    train_dataset = KernDataset(csv_file_uf='train_'+csv_file_uf,csv_file_dc='train_'+csv_file_dc, classes = "train_"+classes,
                                        root_dir=root_dir, transform = transform, image_size=image_size)
    test_dataset = KernDataset(csv_file_uf='test_'+csv_file_uf,csv_file_dc='test_'+csv_file_dc, classes = "train_"+classes,
                                        root_dir=root_dir, transform = test_aug(p=0.5), image_size=image_size)
    temp = pd.read_csv(root_dir+'train_'+csv_file_uf)
    
    device = torch.device("cpu")
    #weights = torch.FloatTensor(get_label_weights_from_pandas(temp))
    #train_size = int(train_prop * len(dataset))
    test_size = len(test_dataset)

    #train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    if (assign==True):
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler = RandomSampler(data_source=test_dataset, num_samples=int(len(test_dataset)), replacement=True),
            #sampler=ImbalancedDatasetSampler(dataset=train_dataset, num_samples=int(len(train_dataset)), assign=assign),
            collate_fn=DatasetItem.collate,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=set_seed()
        )
    else:
         train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=False,
            sampler = RandomSampler(data_source=test_dataset, num_samples=int(len(test_dataset)), replacement=True),
            #sampler=ImbalancedDatasetSampler(dataset=train_dataset, num_samples=int(len(train_dataset)), assign=assign),
            collate_fn=DatasetItem.collate,
            num_workers=num_workers,
            pin_memory=True,
            worker_init_fn=set_seed()
        )
            
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler = RandomSampler(data_source=test_dataset, num_samples=int(len(test_dataset)), replacement=True),
        collate_fn=DatasetItem.collate,
        num_workers=num_workers,
        pin_memory=True,
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

    return (indices[split:], indices[:split])
