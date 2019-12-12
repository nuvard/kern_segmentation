import torch
import torch.utils.data
import torchvision
from tqdm import tqdm
import numba



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

      
    def get_label_weights_from_pandas(self, data):
        labels_list = []
        for i in data['class'].value_counts():
            labels_list.append(i)
        #print(labels_list)
        return labels_list
    
    def set_weights(self, data):
        data["weight"] = 0
        data["weight"] = data.apply(lambda x: 1/labels_list[x["class"]])
        return data
    
    @numba.jit(parallel=True)
    def get_weights(self, label_to_count, dataset):
        return [1.0 / label_to_count[self._get_label(dataset, idx)]
                    for idx in tqdm(self.indices)]

    def __init__(self, dataset, pandas, indices=None, num_samples=None):
        print("==> Initialising sampler")
        # if indices is not provided, 
        # all elements in the dataset will be considered
        print("=====> Loading indices")
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
        print("=====> Loading samples")
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
        print("=====> Checking distribution")
        # distribution of classes in the dataset 
        label_to_count = self.get_label_weights_from_pandas(pandas)  
        print("=====> Assigning weights")
        # weight for each sample
        
        weights = self.get_weights(label_to_count, dataset)
        self.weights = torch.FloatTensor(weights)

    def _get_label(self, dataset, idx):
        dataset_type = type(dataset)
        if dataset_type is torchvision.datasets.MNIST:
            return dataset.train_labels[idx].item()
        elif dataset_type is torchvision.datasets.ImageFolder:
            return dataset.imgs[idx][1]
        else:
            return dataset.__getitem__(idx).label
                
    def __iter__(self):
        #print("==>Iterating through sampler")
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples