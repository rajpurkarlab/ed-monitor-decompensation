import torch
from torch import nn
import numpy as np
import random
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from data_processing import load_all_features

class BaseDataset(Dataset):
    def __init__(self, df, labels, size=1250, wide_feat=None):
        self.df = df
        self.labels = labels
        self.size = size
        self.wide_feat = wide_feat

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, item):
        labels = self.labels[item]
        vals = self.df[item]
        wide = self.wide_feat[item]
        return {
            "x": torch.tensor(vals, dtype=torch.float),
            "y": torch.tensor(labels, dtype=torch.float),
            "x_wide": torch.tensor(wide, dtype=torch.float)
        } 
    
# By default loads in waveform features as well as `wide` features including demographics, vital signs,
# and signal processing derived waveform features such as PTT and HRV
class DecompensationDataset(pl.LightningDataModule):
    def __init__(self, batch_size, path_tuple, task, lead):
        super().__init__()
        self.batch_size = batch_size
        self.task = task
        self.lead = lead
        self.path_tuple = path_tuple
        
        g = torch.Generator()
        g.manual_seed(0)
        self.generator = g
        
    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        
    def setup(self,stage=None):
        data_tuple = load_all_features(self.path_tuple, self.task, self.lead, get_waves=True, use_inference=False)
        (xtrain_norm, xval_norm, xtest_norm), (xtrain_wide, xval_wide, xtest_wide), (ytrain, yval, ytest) = data_tuple
        print(ytrain[:100])
        data_len = xtrain_norm.shape[-1]
                
        #Set Datasets 
        self.train_dataset = BaseDataset(xtrain_norm,ytrain,size=data_len, wide_feat=xtrain_wide) 
        self.validation_dataset = BaseDataset(xval_norm,yval, size=data_len, wide_feat=xval_wide) 
        self.test_dataset = BaseDataset(xtest_norm,ytest, size=data_len, wide_feat=xtest_wide)

    def train_dataloader(self):
        train_loader = DataLoader(self.train_dataset,
                            batch_size=self.batch_size,
                            shuffle=True, num_workers=4, 
                            worker_init_fn=self.seed_worker, generator=self.generator)
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(self.validation_dataset,
                            batch_size=self.batch_size,
                            shuffle=False, num_workers=4, 
                            worker_init_fn=self.seed_worker, generator=self.generator)       
        return valid_loader
    
    def test_dataloader(self):
        test_loader = DataLoader(self.test_dataset,
                            batch_size=self.batch_size,
                            shuffle=False, num_workers=4, 
                            worker_init_fn=self.seed_worker, generator=self.generator)       
        return test_loader