import numpy as np
import pandas as pd
import json

from prediction_module import DecompensationPrediction
import pytorch_lightning as pl
import os 
from torch import nn
import torch
from pytorch_lightning.callbacks import LearningRateMonitor

from dataset import DecompensationDataset

import wandb
from pytorch_lightning.loggers import WandbLogger
wandb.login()

from prna import preTrainedPRNA

def train_models(config, path_tuple, w_logger, path, num_epochs=50, num_gpus=1, seed=42):  
    pl.seed_everything(seed, workers=True)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath=path,
        filename='models-{epoch:02d}-{valid_loss:.2f}',
        save_top_k=1,
        mode='min')
    
    # Model
    mod = config["model"]

    # Initialize non pre-trained parameters with Glorot / fan_avg.
    if not config["pre-trained"]:
        for p in mod.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    dx = DecompensationDataset(config["batch_size"], path_tuple, 
                                       config["task"], config["lead"], include_base=True, 
                                       include_first_mon=True, include_trend=True, include_cc=True, 
                                       include_ptt=False, include_hrv=False)
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(gpus=num_gpus,max_epochs=num_epochs,callbacks=[checkpoint_callback, lr_monitor], logger = w_logger, deterministic=True, num_sanity_val_steps=0)
    
    task = DecompensationPrediction(mod, config)
    trainer.fit(model=task,datamodule=dx)
    
    # save model
    save_dir = config["save_path"] + config["project"]
    if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
            print("created folder : ", save_dir)
    save_path = save_dir + "/" + config["name"]
    torch.save(mod.state_dict(), save_path)
    
    # run test set
    result = trainer.test(model=task, datamodule=dx, ckpt_path='best')
    return result

def run_configs(config_list, train_fn, path_tuple, epochs=100):
    for config in config_list:
        wandb_logger = WandbLogger(name=config["name"],project=config["project"])

        # make checkpoint folder
        NEWDIR = "./" + config["project"]
        if not os.path.isdir(NEWDIR):
            os.makedirs(NEWDIR)
            print("created folder : ", NEWDIR)

        result = train_fn(config, path_tuple, wandb_logger, NEWDIR, num_epochs=config["epochs"])
        print(result)
        wandb.finish()

        
def main():
    layer_sizes = [4, 8, 16, 32, 64, 128] # size of output embedding layer
    tasks = [('MAP', 'hypotension'), ('SpO2', 'hypoxia'), ('HR', 'tachycardia')]
    leads = [('ECG', 1), ('Pleth', 1), ('All', 2)]
    times = ['60min', '90min', '120min']

    save_path = "/deep/group/ed-monitor-self-supervised/test_models_v1/ed_monitor_decompensation/transformer/saved_models/"
    prna_model_path = "/deep2/group/ed-monitor/models/prna/outputs-wide-64-15sec-bs64/saved_models/ctn/fold_1/ctn.tar"

    file_path_config = "/deep/group/ed-monitor-self-supervised/test_models_v1/ed_monitor_decompensation/path_configs.json"
    with open(file_path_config) as fpc:
        all_paths = json.load(fpc)
    
    config_list = []
    for sz in layer_sizes:
        for time in times:    
            data_paths = all_paths[time]
            path_tuple = data_paths["h5py_file"], data_paths["summary_file"], data_paths["labels_file"], data_paths["data_file"], data_paths["all_splits_file"], data_paths["hrv_ptt_file"]

            for lead in leads:
                mod = preTrainedPRNA(lead[1], 1, True, 64, sz, prna_model_path)
                model_name = str(sz) + "WF_PreTrainedPRNAFeatures_0W_RELU_SGD-m0.9_ep30_step-g0.975-maxlr1e-3_b64"

                for task in tasks:
                    proj= time + task[0] + '_60Sec_0W_NewSplits_NoAbnormalities' + lead[0]
                    task_name = task[1]

                    new_config =  {"project": proj, "name": model_name, 
                                     "batch_size": 64, "model": mod, "epochs":60, "pre-trained": True,
                                     "gamma":0.975, "momentum":0.9, "maxlr":0.001, "minlr":0.00001, "step": 10, "nesterov":False, "wd":1e-3, 
                                     "task": task_name, "lead": lead[0], "save_path": save_path}
                    config_list.append(new_config)
        
    wandb.finish()
    run_configs(config_list, train_models, path_tuple)

    
if __name__ == "__main__":
    main()