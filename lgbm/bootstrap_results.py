import numpy as np
import pandas as pd
import h5py
import json
import lightgbm as lgb
import os 
import torch
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from verstack import LGBMTuner
from collections import defaultdict

prna_model_path = "/deep2/group/ed-monitor/models/prna/outputs-wide-64-15sec-bs64/saved_models/ctn/fold_1/ctn.tar"
prefix_path = "/deep/group/ed-monitor-self-supervised/test_models_v1/ed-monitor-decompensation/"

import sys
sys.path.insert(0, prefix_path)

from transformer.prna import preTrainedPRNA
from transformer.data_processing import load_all_features, filter_by_index

def load_results(time):
    """
    load saved auroc and pr results from test set
    """
    with open(prefix_path + "lgbm/results/" + time + "_tachycardia_results_auroc.json", 'r') as f:
        total_tachy = json.load(f)
    with open(prefix_path + "lgbm/results/" + time + "_tachycardia_results_pr.json", 'r') as f:
        total_tachy_pr = json.load(f)
    with open(prefix_path + "lgbm/results/" + time + "_hypoxia_results_auroc.json", 'r') as f:
        total_hypoxia = json.load(f)
    with open(prefix_path + "lgbm/results/" + time + "_hypoxia_results_pr.json", 'r') as f:
        total_hypoxia_pr = json.load(f)
    with open(prefix_path + "lgbm/results/" + time + "_hypotension_results_auroc.json", 'r') as f:
        total_hypotension = json.load(f)
    with open(prefix_path + "lgbm/results/" + time + "_hypotension_results_pr.json", 'r') as f:
        total_hypotension_pr = json.load(f)
        
    return total_tachy, total_tachy_pr, total_hypoxia, total_hypoxia_pr, total_hypotension, total_hypotension_pr

    
def bootstrap_results(hpams, data_paths, config_names, config_dict, time, task, num_bootstrap=10000):
    """
    bootstrap function for test set bootstrapping of best models given an hpam dictionary
    """
    device_str = "cuda"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")   
    
    print(f"----Bootstrapping the {time} timepoint for the {task} task-----")
    
    path_tuple = data_paths["h5py_file"], data_paths["summary_file"], data_paths["labels_file"], data_paths["data_file"], data_paths["all_splits_file"], data_paths["hrv_ptt_file"]

    all_preds = {}
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

        best_params = hpams[config['name']]
        final_lgbm_class = lgb.LGBMClassifier(**best_params)
        final_lgbm_class = final_lgbm_class.fit(xtrain, ytrain, eval_set=[(xval, yval)], eval_metric=['auc'], verbose=False)
        final_pred = final_lgbm_class.predict_proba(xtest)[:, 1]

        all_preds[config['name']] = final_pred

    bs_auroc_dict = defaultdict(list)
    bs_prscore_dict = defaultdict(list)
    for key in all_preds.keys():
        bs_auroc_dict[key] = []
        bs_prscore_dict[key] = []

    for i in range(num_bootstrap):
        bootstrap_indices = np.random.choice(range(len(ytest)), size=len(ytest), replace=True)
        for key in all_preds.keys():
            preds = [all_preds[key][i] for i in bootstrap_indices]
            labels = [ytest[i] for i in bootstrap_indices]
            auroc = roc_auc_score(labels, preds)
            prscore = average_precision_score(labels, preds)

            bs_auroc_dict[key].append(auroc)
            bs_prscore_dict[key].append(prscore)


    return bs_auroc_dict, bs_prscore_dict

def save_bootstrap(task, time, bs_auroc_dict, bs_prscore_dict):
    """
    save bootstrap results for auroc and pr
    """
    auroc_results = pd.DataFrame.from_dict(bs_auroc_dict)
    pr_results = pd.DataFrame.from_dict(bs_prscore_dict)
    auroc_results.to_csv(prefix_path + "lgbm/" + time + "_" + task + "_auroc_bootstrap_testrun.csv", index=False)
    pr_results.to_csv(prefix_path + "lgbm/" + time + "_" + task + "_prscore_bootstrap_testrun.csv", index=False)


def get_diff_tables(input_bootstrap_file1, time, task, score, combine = False):
    """
    return table of diff to triage+CC
    """
    df = pd.read_csv(input_bootstrap_file1)
    total_results = {}
    
    for col in list(df.columns):
        total_results[col] = df[col] - df['Triage + CC']
    
    results = pd.DataFrame.from_dict(total_results)
    if not combine:
        results.to_csv(prefix_path + "lgbm/" + time + "_" + task + "_" + score + "_TriageCCDiff.csv", index=False)
    else:
        results.to_csv(prefix_path + "lgbm/" + time + "_" + task + "_" + score + "_TriageCCDiff_Combined.csv", index=False)
    return results

        
def get_output_tables_point_CI(input_bootstrap_file1, time, task, score="auroc"):
    """
    save output table as point with confidence interval
    """
    df = pd.read_csv(input_bootstrap_file1)
    
    total_tachy, total_tachy_pr, total_hypoxia, total_hypoxia_pr, total_hypotension, total_hypotension_pr = load_results(time)
  
    total_results = {}
           
    for col in list(df.columns):
        sorted_res = list(df[col]) 
        sorted_res.sort()
        if task == "tachycardia":
            if score == "prscore":
                point = round(total_tachy_pr[col][0], 3)
            else:
                point = round(total_tachy[col][0], 3)
        if task == "hypotension":
            if score == "prscore":
                point = round(total_hypotension_pr[col][0], 3)
            else:
                point = round(total_hypotension[col][0], 3)
        if task == "hypoxia":
            if score == "prscore":
                point = round(total_hypoxia_pr[col][0], 3) 
            else:
                point = round(total_hypoxia[col][0], 3)     
        low = round(np.percentile(list(df[col]) , 2.5), 3)
        high = round(np.percentile(list(df[col]) , 97.5), 3)
        total_results[col] = [str(point) + " " + "(" + str(low) + " - " + str(high) + ")"]
        
    results = pd.DataFrame.from_dict(total_results)
    results.to_csv(prefix_path + "lgbm/" + time + "_" + task + "_" + score + "_CIandPoint.csv", index=False)
    return results

def get_output_tables_point_CI_diff(input_bootstrap_file1, time, task, score, percentiles = [2.5, 97.5]):
    """
    save output of point difference with CI
    """
    df = pd.read_csv(input_bootstrap_file1)

    total_tachy, total_tachy_pr, total_hypoxia, total_hypoxia_pr, total_hypotension, total_hypotension_pr = load_results(time)
    total_results = {}
    
    columns_to_search = list(df.columns)
    for col in columns_to_search[1:]:
        sorted_res = list(df[col]) 
        sorted_res.sort()
        if task == "tachycardia":
            if score == "prscore":
                point = round(total_tachy_pr[col][0] - total_tachy_pr[columns_to_search[0]][0], 3)
            else:
                point = round(total_tachy[col][0] - total_tachy[columns_to_search[0]][0], 3)
        if task == "hypotension":
            if score == "prscore":
                point = round(total_hypotension_pr[col][0] - total_hypotension_pr[columns_to_search[0]][0], 3)
            else:
                point = round(total_hypotension[col][0] - total_hypotension[columns_to_search[0]][0], 3)
        if task == "hypoxia":
            if score == "prscore":
                point = round(total_hypoxia_pr[col][0] - total_hypoxia_pr[columns_to_search[0]][0], 3) 
            else:
                point = round(total_hypoxia[col][0] - total_hypoxia[columns_to_search[0]][0], 3)     
        low = round(np.percentile(list(df[col]) , percentiles[0]), 3)
        high = round(np.percentile(list(df[col]) ,percentiles[1]), 3)
        total_results[col] = [str(point) + " " + "(" + str(low) + " - " + str(high) + ")"]
        
    results = pd.DataFrame.from_dict(total_results)
    results.to_csv(prefix_path + "lgbm/" + time + "_" + task + "_" + score + "_CIandPoint_" + str(percentiles[1] - percentiles[0]) + "_" + "diff.csv", index=False)
    return results

def main():
    """
    run bootstraps for every time and every task (and save results)
    """
    times = ["60min", "90min", "120min"]  
    tasks = ["tachycardia", "hypotension", "hypoxia"]
    scores = ["auroc", "prscore"]
    
    all_config_path = "/deep/group/ed-monitor-self-supervised/test_models_v1/ed-monitor-decompensation/configs.json"
    with open(all_config_path, 'r') as f:
        config_dict = json.load(f)
    config_names = list(config_dict.keys())
    
    file_path_config = "/deep/group/ed-monitor-self-supervised/test_models_v1/ed-monitor-decompensation/path_configs_new.json"
    with open(file_path_config) as fpc:
        all_paths = json.load(fpc)
    
    for time in times:
        data_paths = all_paths[time]        
        for task in tasks:
            params_file = data_paths[task]["params_file"]
            with open(params_file) as pf:
                params = json.load(pf)
                
            bs_auroc_dict, bs_prscore_dict = bootstrap_results(params, data_paths, config_names, config_dict, time, task)
            save_bootstrap(task, time, bs_auroc_dict, bs_prscore_dict)
            for score in scores:
                bootstrap_prefix_path = prefix_path + "lgbm/" + time + "_" + task + "_" + score
                get_diff_tables(bootstrap_prefix_path + "_bootstrap_testrun.csv", time, task, score)
                get_output_tables_point_CI(bootstrap_prefix_path + "_bootstrap_testrun.csv", time, task, score)
                get_output_tables_point_CI_diff(bootstrap_prefix_path + "_TriageCCDiff.csv", time, task, score, percentiles = [2.5, 97.5])

if __name__ == "__main__":
    main()
    