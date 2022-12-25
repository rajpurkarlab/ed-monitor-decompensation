import json 
import torch
import numpy as np
import os
import lightgbm as lgb
import sklearn

prefix_path = "/deep/group/ed-monitor-self-supervised/test_models_v1/ed-monitor-decompensation/"
prna_model_path = "/deep2/group/ed-monitor/models/prna/outputs-wide-64-15sec-bs64/saved_models/ctn/fold_1/ctn.tar"

import sys
sys.path.insert(0, prefix_path)

from transformer.prna import preTrainedPRNA
from transformer.data_processing import load_all_features, filter_by_index
from analysis.features import get_feature_distributions, plot_roc_curve
from analysis.test_characteristics import get_test_characteristics, get_confusion_matrix
from analysis.shap_values import get_shap_analysis
from analysis.calibration import plot_calibration_curve
from analysis.mews import mews_subgroup

def run_single_analysis(configs, time, task, mode, data_paths, thresholds=[0.85, 0.95, 0.99]):
    device_str = "cuda"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")

    print(f"----Starting the {time} timepoint-----")
    
    prna_model_path = "/deep2/group/ed-monitor/models/prna/outputs-wide-64-15sec-bs64/saved_models/ctn/fold_1/ctn.tar"
    path_tuple = data_paths["h5py_file"], data_paths["summary_file"], data_paths["labels_file"], data_paths["data_file"], data_paths["all_splits_file"], data_paths["hrv_ptt_file"], data_paths["mews_labels_file"]

    print(f"starting task : {task}")
    
    params_file = data_paths[task]["params_file"]
    
    with open(params_file) as pf:
        all_params = json.load(pf)

    for config in configs:
        if config["include_wave"] and config["wave_type"] == "Both":
            model_type = preTrainedPRNA(1, 1, True, 64, config["wave_size"], prna_model_path)
            pleth_model = data_paths[task]["pleth_model_file"]
            ecg_model = data_paths[task]["ecg_model_file"]
            data_tuple = load_all_features(path_tuple, task, 'All', get_waves=True, use_inference=True, two_models=True, model_type=model_type, 
                                           model_path=None, pleth_model_path=pleth_model, ecg_model_path=ecg_model, return_mews=True)
        else:
            data_tuple = load_all_features(path_tuple, task, 'All', get_waves=False, return_mews=True)

        print(f"Running config: {config['name']}")

        best_params = all_params[config['name']]

        (xtrain, xval, xtest), (ytrain, yval, ytest), (mtrain, mval, mtest) = data_tuple 
        (xtrain, xval, xtest), (ytrain, yval, ytest) = filter_by_index(((xtrain, xval, xtest), (ytrain, yval, ytest)), config["indices"])
        
        final_lgbm_class = lgb.LGBMClassifier(**best_params)
        final_lgbm_class = final_lgbm_class.fit(xtrain, ytrain, eval_set=[(xval, yval)], eval_metric=['auc'], verbose=False)
        final_pred = final_lgbm_class.predict_proba(xtest)[:, 1]
        
        if mode == 'shap':    
            get_shap_analysis(final_lgbm_class, xtest, config, task, time, num_values=15)
        elif mode == 'confusion':
            get_confusion_matrix(thresholds, final_lgbm_class, xtest, ytest, xval, yval) 
        elif mode == 'characteristic':
            get_test_characteristics(thresholds, final_lgbm_class, xtest, ytest, xval, yval) 
        elif mode == 'calibration_curve':
            plot_name = prefix_path + "lgbm/calibration_plots/" + time + "_" + task + "_" + config['name'] + ".png"
            plot_calibration_curve(ytest, final_pred, plot_name)
        elif mode == 'mews':
            mews_subgroup(mtest, xval, yval, xtest, final_lgbm_class, cutoff=5)
            
def run_pairwise_comparison(config_pair, time, task, data_paths, full_config):

    print(f"----Starting the {time} timepoint-----")
        
    path_tuple = data_paths["h5py_file"], data_paths["summary_file"], data_paths["labels_file"], data_paths["data_file"], data_paths["all_splits_file"], data_paths["hrv_ptt_file"]   
    
    print(f"starting task : {task}")
 
    params_file = prefix_path + "lgbm/saved_hparams/" + time + "_params_" + task + ".json"    
    with open(params_file) as pf:
        all_params = json.load(pf)

    xtests, xvals, lgbm_models, names = [], [], [], []
    for config in config_pair:
        if config["include_wave"] and config["wave_type"] == "Both":
            model_type = preTrainedPRNA(1, 1, True, 64, config["wave_size"], prna_model_path)
            pleth_model = data_paths[task]["pleth_model_file"]
            ecg_model = data_paths[task]["ecg_model_file"]
            data_tuple = load_all_features(path_tuple, task, 'All', get_waves=True, use_inference=True, two_models=True, model_type=model_type, 
                                           model_path=None, pleth_model_path=pleth_model, ecg_model_path=ecg_model)
        else:
            data_tuple = load_all_features(path_tuple, task, 'All', get_waves=False)

        print(f"Running config: {config['name']}")
        best_params = all_params[config['name']]

        data_tuple = filter_by_index(data_tuple, config["indices"])
        (xtrain, xval, xtest), (ytrain, yval, ytest) = data_tuple        
        
        final_lgbm_class = lgb.LGBMClassifier(**best_params)
        final_lgbm_class = final_lgbm_class.fit(xtrain, ytrain, eval_set=[(xval, yval)], eval_metric=['auc'], verbose=False)

        lgbm_models.append(final_lgbm_class)
        xvals.append(xval)
        xtests.append(xtest)
        names.append(config['name'])
            
    config = full_config
    model_type = preTrainedPRNA(1, 1, True, 64, config["wave_size"], prna_model_path)
    pleth_model = data_paths[task]["pleth_model_file"]
    ecg_model = data_paths[task]["ecg_model_file"]
    data_tuple = load_all_features(path_tuple, task, 'All', get_waves=True, use_inference=True, two_models=True, model_type=model_type, 
                                   model_path=None, pleth_model_path=pleth_model, ecg_model_path=ecg_model)
    
    print(f"Running config: {config['name']}")
    best_params = all_params[config['name']]

    (xtrain, xval, xtest), (ytrain, yval, ytest) = filter_by_index(data_tuple, config["indices"])
    print(xtrain[0])

    plot_roc_curve(lgbm_models, xtests, ytest)
    get_feature_distributions(config['features'], lgbm_models[0], lgbm_models[1], xtest, xtests[0], xtests[1], ytest, xvals[0], xvals[1], yval, time, task, names)

def main():
    if len(sys.argv) < 2:
        print("please enter one of the following analysis modes `shap`, `characteristic`, `confusion`, `calibration_curve` or `mews`")
        return 
    modes = sys.argv[1:]
    
    file_path_config = "/deep/group/ed-monitor-self-supervised/test_models_v1/ed-monitor-decompensation/path_configs_new.json"
    with open(file_path_config) as fpc:
        all_paths = json.load(fpc)
        
    best_models_config = "/deep/group/ed-monitor-self-supervised/test_models_v1/ed-monitor-decompensation/best_model_configs.json"
    with open(best_models_config) as bmc:
        best_models = json.load(bmc)
    
    times = ['90min']
    tasks = ["tachycardia", "hypotension", "hypoxia"]
    
    for time in times:
        time_paths = all_paths[time]
        for task in tasks:
            configs = [best_models[time][task]["best"], best_models[time][task]["baseline"]]
            full_config = best_models[time][task]["full"]
            for mode in modes:
                if mode in ['shap', 'characteristic', 'confusion', 'calibration_curve', 'mews']:
                    run_single_analysis(configs, time, task, mode, time_paths)
                elif mode == "comparison": 
                    run_pairwise_comparison(configs, time, task, time_paths, full_config)
                else:
                    print("analysis mode entered was not valid")
                    continue
            
if __name__ == "__main__":
    main()
