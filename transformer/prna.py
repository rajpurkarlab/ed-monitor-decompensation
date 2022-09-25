# Setup for pre-trained PRNA transformer model
import argparse
import csv

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from edm.models.transformer_model import load_best_model
from torchmetrics import R2Score, AUROC, F1Score

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Load pre-trained PRNA models
# With fully connected layer pre-trained to classify cardiac abnormalities

class preTrainedPRNA(torch.nn.Module):
    def __init__(self, num_leads, classes, remove_last_layer, embedding_size, layer_sz, model_path, num_wide=0):
        super(preTrainedPRNA, self).__init__()
        self.classes = classes
        self.remove_last_layer = remove_last_layer
        self.num_wide = num_wide
        self.leads = num_leads
        self.model = load_best_model(model_path, deepfeat_sz=embedding_size, remove_last_layer=remove_last_layer)
        
        # if last layer removed, model layers will be organized differently 
        if not remove_last_layer:
            self.model = torch.nn.Sequential(*(list(list(self.model.children())[0].children())[:]))
        
        # make first layer w/ correct # of channels and copy over weights
        layers = list(self.model.children())
        first_layer = list(layers[0].children())[0]
        trained_weights = first_layer.weight
        new_first = nn.Conv1d(self.leads, 128, kernel_size=14, stride=3, padding=2, bias=False)
        new_first.weight = nn.Parameter(torch.tile(trained_weights, (self.leads, 1)))   
        conv_modules = [new_first]
        conv_modules.extend(list(layers[0].children())[1:])
        conv_layers = nn.Sequential(*conv_modules)
        
        # restack network modules with new channel first layer
        modules = [conv_layers]
        modules.extend(layers[1:])
        self.model = torch.nn.Sequential(*modules)
        
        if remove_last_layer:      
            last_layer = list(self.model.children())[-1]
        else:
            last_layer = list(self.model.children())[-2]

        self.fc1 = nn.Linear(last_layer.out_features + num_wide, layer_sz)
        self.fc2 = nn.Linear(layer_sz, classes)
        self.sigmoid = nn.Sigmoid()
        
        #Metrics
        self.train_sys_R2Score, self.train_di_R2Score = R2Score(), R2Score()
        self.val_sys_R2Score, self.val_di_R2Score = R2Score(), R2Score()
        self.test_sys_R2Score, self.test_di_R2Score = R2Score(), R2Score()

    def forward(self, x, wide_feats=None):
        out = self.model(x)       
        if self.num_wide > 0 :
            wide_feats = torch.squeeze(wide_feats).float()
            out = F.relu(self.fc1(torch.cat((wide_feats, out), dim=1)))
        else: 
            out = F.relu(self.fc1(out))  
        out = self.sigmoid(self.fc2(out)) 
        return out

