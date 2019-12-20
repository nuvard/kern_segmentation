import torch
import torch.utils.data
import torchvision
from tqdm import tqdm
import numba
import pandas as pd


class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
    """
   # @numba.jit
    def get_dist(self, dataset):
        label_to_count = {}
        for idx in tqdm(self.indices):
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1
        return label_to_count

      
    def get_label_weights_from_pandas(self, file_dc, file_uf, root_dir='../data/',assign=False):
        
        labels_list = []
        labels_dict = {}
        data_dc = pd.read_csv(root_dir+file_dc)
        data_uf = pd.read_csv(root_dir+file_uf)
        print(data_dc["class"].value_counts())
        for i in data_dc["class"].value_counts():
            labels_list.append(i)
            
        print(labels_list)
        
        data_dc["weight"]=0
        data_uf["weight"]=0
        data_dc["weight"] = data_dc.apply(lambda x: len(data_dc)/labels_list[x["class"]], axis=1)
        data_uf["weight"] = data_dc["weight"]
        if (assign==True):
            data_dc.to_csv(root_dir + 'train_data_dc.csv', index=False)
            data_uf.to_csv(root_dir + 'train_data_uf.csv', index=False) 
        return labels_list
    
    def set_weights(weights,  pandas):
        data["weight"] = 0
        data["weight"] = data.apply(lambda x: 1/labels_list[x["class"]])
        return data
    
    @numba.jit(parallel=True)
    def get_weights(self,pandas, dataset=None, label_to_count=None):
        #print('ok!')
        if dataset is None:
            return[pandas.loc[idx , "weight"] for idx in tqdm(self.indices)]
        else:
            return [1.0 / label_to_count[self._get_label(dataset, idx)]
                    for idx in tqdm(self.indices)]
        
    def __init__(self, dataset,  num_samples=None, indices=None ,assign=False, csv_file_uf='train_data_uf.csv', csv_file_dc='train_data_dc.csv', root_dir = '../data/'):
        
        csv_data_uf = pd.read_csv(root_dir+csv_file_uf)
        csv_data_dc = pd.read_csv(root_dir+csv_file_dc)
        
        print("==> Initialising sampler")
        # if indices is not provided, 
        # all elements in the dataset will be considered
        print("=====> Loading indices")
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
        print("=====> Loading samples")
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
        
        # distribution of classes in the dataset 
        if (assign==True):
            print("=====> Checking distribution")
            label_to_count = self.get_label_weights_from_pandas(csv_file_dc, csv_file_uf, root_dir=root_dir, assign=True)
        print("=====> Assigning weights")
        #csv_data_uf = pd.read_csv(root_dir+csv_file_uf)
        csv_data_dc = pd.read_csv(root_dir+csv_file_dc)
        #print(csv_data_dc.head())
        weights = self.get_weights(pandas=csv_data_dc)
        
        # weight for each sample
        
        #weights = self.get_weights(label_to_count, dataset, pandas)
        self.weights = torch.FloatTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        else:
            return dataset.__getitem__(idx).label
        
    def _get_weight(self, dataset, idx):
        return dataset.__getitem__(idx).weight
        
    def __iter__(self):
        #print("==>Iterating through sampler")
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples