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
  
    def get_label_weights_from_file(self, data_dc, data_uf, root_dir='../data/',assign=False):
        
        labels_list = []
        labels_dict = {}
        for i in data_dc["class"].value_counts():
            labels_list.append(i)
        data_dc["weight"]=0
        data_uf["weight"]=0
        data_dc["weight"] = data_dc.apply(lambda x: len(data_dc)/labels_list[x["class"]], axis=1)
        data_uf["weight"] = data_dc["weight"]
        if (assign==True):
            data_dc.to_csv(self.root_dir + self.file_dc, index=False)
            data_uf.to_csv(self.root_dir + self.file_uf, index=False) 
        return labels_list

    def set_weights_to_file(self):
        """Sets weights in csv-files with uf-data, dc-data, classes (OHE). 

        Weights are applied on "weight" column.

        Args: 
            path: path to data with slash. Example: '../data'
        """
        self.data_uf, self.data_dc, _ = self.set_weights(self.data_uf, self.data_dc, self.classes)
        self.data_dc.to_csv(self.file_dc, index=False)
        self.data_uf.to_csv(self.file_uf, index=False)

    def set_weights(self):
        """Sets weights to kern classes.

        Computes weights of single classes as 1/(count_of_examples*const) 
        and after that sets final weights as mean of class weights to each example. 
        Weights are stored in "weight" column.

        Returns:
            (uf,dc, weights_dict): dataframes with added(changed) "weight" column 
            and dict with weight values.
        """
        weights = self.classes.sum()
        weights_reduced = {}
        weights_list = {}

        for  i,j in weights.items():
            if i in ['алевролит', 'аргиллит', 'песчаник']:
                weights_reduced[i] = j*0.6
            if i=='аргиллит':
                weights_reduced[i] = j*0.7
            if i=='песчаник':
                weights_reduced[i] = j*0.78

        for i, j in (weights.items()):
            #if i=='карбонатная порода' or i=='другое':
            #    weights_list[i]=1
            #else:     
            if i in weights_reduced.keys():
                weights_list[i]=self.classes.sum().sum()/weights_reduced[i]
            else:
                weights_list[i]=self.classes.sum().sum()/(abs(self.classes.sum().sum()-pd.Series(weights_reduced).sum()))
        
        
        def apply_weights(data, weights_dict=self.weights_list):
            """Sets weight as mean of given weights.
            Args: 
                data: row in dataframe
                weights_dict: dict with weight values

            Returns:
                float: computed weight
            """
            weight = 0
            weights = []
            for i,j in data.items():
                #print(i, weights_dict[i])  
                if(j!=0):
                    weights.append(weights_dict[i])
            weight = (np.mean(weights))
            return weight

        uf["weight"]= self.classes.apply(apply_weights, axis=1)
        dc["weight"] = self.classes.apply(apply_weights, axis=1)
        return (uf, dc, weights_list)

    
    @numba.jit(parallel=True)
    def get_weights(self, pandas, dataset=None, label_to_count=None):
        """Gets weights from pandas dataset.
        Args:
            pandas: pandas dataset
            dataset: bool, if not None, uses data in "weights" column.
            label_to_count: dict, contains labels and weights of classes.
        Returns: 
            list with weights
        
        """
        #print('ok!')
        if dataset is None:
            return[pandas.loc[idx , "weight"] for idx in tqdm(self.indices)]
        else:
            return [1.0 / label_to_count[self._get_label(dataset, idx)]
                    for idx in tqdm(self.indices)]
        
    def __init__(self, dataset,  num_samples=None, indices=None,
                 assign=False, set_new=False, 
                 csv_file_uf='train_data_uf.csv', 
                 csv_file_dc='train_data_dc.csv', 
                 classes_file='train_classes.csv',
                 root_dir = '../data/'):
        self.root_dir = root_dir
        self.csv_file_uf = csv_file_uf
        self.csv_file_dc = csv_file_dc
        self.csv_data_uf = pd.read_csv(root_dir+csv_file_uf)
        self.csv_data_dc = pd.read_csv(root_dir+csv_file_dc)
        self.classes = pd.read_csv(root_dir+classes_file)
        print("==> Initialising sampler")
        # if indices is not provided, 
        # all elements in the dataset will be considered
        print("=====> Loading indices")
        self.indices = list(range(len(dataset))) \
            if indices is None else indices
        print("=====> Loading samples")
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
        
        if (set_new==True):
            set_weights_to_file()
        # distribution of classes in the dataset 
        if (assign==True):
            print("=====> Checking distribution")
            label_to_count = self.get_label_weights_from_file(self.csv_data_dc, self.csv_data_uf, root_dir=self.root_dir, assign=True)
        print("=====> Assigning weights")
        #csv_data_uf = pd.read_csv(root_dir+csv_file_uf)
        csv_data_dc = pd.read_csv(root_dir+self.csv_file_dc)
        #print(csv_data_dc.head())
        weights = self.get_weights(pandas=self.csv_data_dc)
        
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