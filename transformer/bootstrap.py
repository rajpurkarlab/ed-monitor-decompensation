import numpy as np
import pandas as pd
import json

import os 
from torch import nn
import torch
from sklearn.metrics import roc_auc_score

from prna import preTrainedPRNA

prefix_path = "/deep/group/ed-monitor-self-supervised/test_models_v1/ed-monitor-decompensation-clean/"

from data_processing import load_model, run_batch_inference, load_all_features

def bootstrap_deep_models(configs, time, task, data_paths):
    device_str = "cuda"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    print(f"----Starting the {time} timepoint-----")
        
    prna_model_path = "/deep2/group/ed-monitor/models/prna/outputs-wide-64-15sec-bs64/saved_models/ctn/fold_1/ctn.tar"
    path_tuple = data_paths["h5py_file"], data_paths["summary_file"], data_paths["labels_file"], data_paths["data_file"], data_paths["all_splits_file"], data_paths["hrv_ptt_file"]

    if task == "tachycardia":
        models = data_paths["tachycardia_models"]
    if task == "hypotension":
        models = data_paths["hypotension_models"]    
    if task == "hypoxia":
        models = data_paths["hypoxia_models"]    
    
    print(f"starting task : {task}")

    for config in configs:
        if config["wave_type"] == "ECG":
            mod = preTrainedPRNA(1, 1, True, 64, config["wave_size"], prna_model_path)
            model_path = models["ecg_file"]
            lead = "ECG"
        elif config["wave_type"] == "Pleth":
            mod = preTrainedPRNA(1, 1, True, 64, config["wave_size"], prna_model_path)
            model_path = models["pleth_file"]
            lead = "Pleth"
        else:
            mod = preTrainedPRNA(2, 1, True, 64, config["wave_size"], prna_model_path)
            model_path = models["ecg_pleth_file"]
            lead = 'All'

        wave_type, wave_size = config["wave_type"], config["wave_size"]
        print(f"Loading model: {wave_type} with task {task} and size {wave_size}")
        
        data_tuple = load_all_features(path_tuple, task, lead, get_waves=True, use_inference=False)
        
        (xtrain_norm, xval_norm, xtest_norm), (xtrain_wide, xval_wide, xtest_wide), (ytrain, yval, ytest) = data_tuple
        xtrain_norm, xval_norm, xtest_norm = torch.tensor(xtrain_norm).to(device).float(), torch.tensor(xval_norm).to(device).float(), torch.tensor(xtest_norm).to(device).float()    

        deep_model = load_model(model_path, mod, get_output=True)
        deep_model.eval()
        ytestpred = run_batch_inference(64, xtest_norm, deep_model)
        
        aucs = []
        
        # bootstrap across test dataset
        for i in range(10000):
            bootstrap_indices = np.random.choice(range(len(ytest)), size=len(ytest), replace=True)
            bs_pred = [ytestpred[i] for i in bootstrap_indices]
            bs_ytest = [ytest[i] for i in bootstrap_indices]
            
            aucs.append(roc_auc_score(bs_ytest, bs_pred))
         
        print(f"AUROC: {roc_auc_score(ytest, ytestpred)} ---- {np.percentile(aucs, [2.5])}, {np.percentile(aucs, [97.5])}")

def main():
    wave_types = ["ECG", "Pleth", "ECGPleth"]
    configs = []
    wave_size = 4
    
    for wt in wave_types:
        configs.append({"wave_type": wt, "wave_size": wave_size})
            
    tasks = ["tachycardia", "hypotension", "hypoxia"]
    times = ["90min"]
    
    file_path_config = "/deep/group/ed-monitor-self-supervised/test_models_v1/ed-monitor-decompensation-clean/path_configs_new.json"
    with open(file_path_config) as fpc:
        all_paths = json.load(fpc)

    for time in times:
        time_paths = all_paths[time]
        for task in tasks: 
            bootstrap_deep_models(configs, time, task, time_paths)
    
if __name__ == "__main__":
    main()