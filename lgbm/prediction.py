import numpy as np
import pandas as pd
import json
import lightgbm as lgb

import os 
import torch
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from verstack import LGBMTuner
from collections import defaultdict

prna_model_path = "/deep2/group/ed-monitor/models/prna/outputs-wide-64-15sec-bs64/saved_models/ctn/fold_1/ctn.tar"
prefix_path = "/deep/group/ed-monitor-self-supervised/test_models_v1/ed-monitor-decompensation-clean/"

import sys
sys.path.insert(0, prefix_path)

from transformer.prna import preTrainedPRNA
from transformer.data_processing import load_all_features, filter_by_index

def tune_light_gbm_train(data_tuple):
    """
    tune on training set, data_tuple in the format of (xtrain, xval, xtest), (ytrain, yval, ytest)
    """
    (xtrain, xval, xtest), (ytrain, yval, ytest) = data_tuple
    train_data = lgb.Dataset(xtrain, label=ytrain)  

    print(f"feature size = {xtrain.shape[1]}")
    tuner = LGBMTuner(metric = 'auc', trials = 100, verbosity = 0, visualization = False) 
    tuner.fit(pd.DataFrame(xtrain), pd.Series(ytrain))
    
    params = tuner.best_params
    return params

def tune_light_gbm_val(data_tuple, hparams):
    """
    tune on validation set (get best random state)
    data_tuple in the format of (xtrain, xval, xtest), (ytrain, yval, ytest)
    params in the format of dictionary for input into LGBM
    run 100 models
    """
    (xtrain, xval, xtest), (ytrain, yval, ytest) = data_tuple
    val_aurocs = []
    val_params = []
    params = hparams.copy()
    params['verbose'] = -1
    params['min_child_weight'] = params['min_sum_hessian_in_leaf']
    params['n_jobs'] = params['num_threads']
    params['subsample_freq'] = params['bagging_freq']
    params.pop('bagging_freq', None)
    params.pop("num_threads", None)
    params.pop("min_sum_hessian_in_leaf", None)
    params.pop("verbosity", None)

    for i in range(100): 
        params['random_state'] = i
        lgbm_class = lgb.LGBMClassifier(**params)
        lgbm_class = lgbm_class.fit(xtrain, ytrain, eval_set=[(xval, yval)], 
                                    eval_metric=['auc'], early_stopping_rounds=50, verbose=False)
        val_pred = lgbm_class.predict_proba(xval)[:, 1]
        auroc = roc_auc_score(yval, val_pred)
        val_aurocs.append(auroc)
        val_params.append(params)
        
    index = np.argmax(np.array(val_aurocs))
    best_params = val_params[index]
    best_params['random_state'] = int(index)
    best_auroc = val_aurocs[index]
    print(f"best validation auroc = {best_auroc}")
    
    return best_params

def save_params_metrics_task(time, task, auroc, pr, params):
    with open(prefix_path + time + "_" + task + "_results_params.json", 'w') as f:
        json.dump(params, f)
    with open(prefix_path + time + "_" + task + "_results_auroc.json", 'w') as f:
        json.dump(auroc, f)
    with open(prefix_path + time + "_" + task + "_results_pr.json", 'w') as f:
        json.dump(pr, f)
     
def test_light_gbm(data_tuple, hparams):
    """
    test on the test set (using best random state)
    data_tuple in the format of (xtrain, xval, xtest), (ytrain, yval, ytest)
    params in the format of dictionary for input into LGBM
    """
    (xtrain, xval, xtest), (ytrain, yval, ytest) = data_tuple
    best_params = hparams.copy()
    final_lgbm_class = lgb.LGBMClassifier(**best_params)
    final_lgbm_class = final_lgbm_class.fit(xtrain, ytrain, eval_set=[(xval, yval)], 
                                            eval_metric=['auc'], verbose=False)
    final_pred = final_lgbm_class.predict_proba(xtest)[:, 1]
    auroc = roc_auc_score(ytest, final_pred)
    prscore = average_precision_score(ytest, final_pred)
    return auroc, prscore, final_pred


def main():
    """
    run main script results for every timepoint and task
    """
    device_str = "cuda"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu") 
    
    all_config_path = "/deep/group/ed-monitor-self-supervised/test_models_v1/ed_monitor_decompensation/configs.json"
    with open(all_config_path, 'r') as f:
        config_dict = json.load(f)
    config_names = list(config_dict.keys())
    
    file_path_config = "/deep/group/ed-monitor-self-supervised/test_models_v1/ed_monitor_decompensation/path_configs.json"
    with open(file_path_config) as fpc:
        all_paths = json.load(fpc)
    
    times = ["60min", "90min", "120min"]
    tasks = ["tachycardia", "hypotension", "hypoxia"]
    
    for time in times:
        print(f"----Starting the {time} timepoint-----")
        
        data_paths = all_paths[time]
        path_tuple = data_paths["h5py_file"], data_paths["summary_file"], data_paths["labels_file"], data_paths["data_file"], data_paths["all_splits_file"], data_paths["hrv_ptt_file"]

        for task in tasks:
            auroc_dict, pr_dict, param_dict = defaultdict(list), defaultdict(list), defaultdict(list)
            print(f"starting task : {task}")
            
            for name in config_names:
                config = config_dict[name]
                if config["include_wave"] and config["wave_type"] == "Both":
                    model_type = preTrainedPRNA(1, 1, True, 64, config["wave_size"], prna_model_path)
                    pleth_model = data_paths[task]["pleth_model_file"]
                    ecg_model = data_paths[task]["ecg_model_file"]
                    data_tuple = load_all_features(path_tuple, task, 'All', get_waves=True, use_inference=True, two_models=True, 
                                                   model_type=model_type, model_path=None, pleth_model_path=pleth_model, ecg_model_path=ecg_model)
                else:
                    data_tuple = load_all_features(path_tuple, task, 'All', get_waves=False)
                data_tuple = filter_by_index(data_tuple, config["indices"])
                (xtrain, xval, xtest), (ytrain, yval, ytest) = data_tuple
                
                print(f"Running config: {config['name']}")

                print(f"Light GBM model result: ")
                
                original_params = tune_light_gbm_train(data_tuple)
                best_params = tune_light_gbm_val(data_tuple, original_params)                    
                auroc, pravg, preds = test_light_gbm(data_tuple, best_params)
                
                print(f"For params {best_params}: AUROC = {auroc}, PRScore = {pravg}")

                param_dict[task + '_' + config['name']] = best_params

                auroc_dict[config['name']].append(auroc)
                pr_dict[config['name']].append(pravg)
                        
            save_params_metrics_task(time, task, auroc_dict, pr_dict, param_dict)
        
if __name__ == "__main__":
    main()
