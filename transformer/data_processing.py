import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import json
import h5py

import os 
from torch import nn
import torch
import random 

TACHYCARDIA_CUTOFF = 110
HYPOTENSION_CUTOFF = 65
HYPOXIA_CUTOFF = 90
MEWS_CUTOFF = 4

# labels is a vector of label vectors to be turned into binary labels based off 
# the cutoff threshold value
def binarize(labels, task, mews_cutoff=MEWS_CUTOFF):
    binarized = np.zeros_like(labels)
    binarized[:] = np.nan
    if task == "tachycardia":
        binarized[labels <= TACHYCARDIA_CUTOFF] = 0
        binarized[labels > TACHYCARDIA_CUTOFF] = 1
    elif task == "hypoxia":
        binarized[labels < HYPOXIA_CUTOFF] = 1
        binarized[labels >= HYPOXIA_CUTOFF] = 0
    elif task == "hypotension":
        binarized[labels >= HYPOTENSION_CUTOFF] = 0
        binarized[labels < HYPOTENSION_CUTOFF] = 1
    elif task == "mews":
        binarized[labels < mews_cutoff] = 0
        binarized[labels >= mews_cutoff] = 1
    
    binarized = np.ravel(binarized)
    return binarized

# returns numpy array of relevant wide features
def process_wide_features(wide_feat):
    pat_id = wide_feat.loc[:, ('patient_id')]
    age = wide_feat.loc[:, ("Age")]
    gender = wide_feat.loc[:, ("Gender")]
    acuity = wide_feat.loc[:, ("Triage_acuity")]

    acuity, ac_uniques = pd.factorize(acuity)
    
    acuity = np.where(acuity == -1, np.nan, acuity)
    gender, gen_uniques = pd.factorize(gender)
        
    triage_vitals = wide_feat.loc[:, ("Triage_HR", "Triage_RR", "Triage_SpO2",
                                  "Triage_Temp", "Triage_SBP", "Triage_DBP", "Triage_MAP")]
    parsed = np.stack((pat_id, age, gender, acuity), axis=-1)
    parsed = np.concatenate((parsed, triage_vitals), axis=1)
    
    first_mon = wide_feat.loc[:, ("First_mon_HR", "First_mon_RR", "First_mon_SpO2",
                              "First_mon_SBP", "First_mon_DBP", 
                              "First_mon_MAP", 'First_mon_HRV_1min')]
    parsed = np.concatenate((parsed, first_mon), axis=1)
    
    trend = wide_feat.loc[:, ("Trend_HR", "Trend_RR", "Trend_SpO2",
                              "Trend_SBP", "Trend_DBP", 
                              "Trend_MAP", 'Trend_HRV_1min','Trend_HRV_5min')]
    parsed = np.concatenate((parsed, trend), axis=1)
    
    cc = wide_feat.filter(regex="CC_")
    parsed = np.concatenate((parsed, cc), axis=1)
    
    return parsed

# apply scaling to ecg/pleth waveforms
def scale_input(in_np, scaler_pleth=None, scaler_ecg=None, leads='All'):
    out = in_np.copy()
    if leads == 'All':
        out[:, 0, :] = scaler_pleth.transform(out[:, 0, :])
        out[:, 1, :] = scaler_ecg.transform(out[:, 1, :])
    elif leads == 'Pleth':
        out = out[:, :1, :]
        out[:, 0, :] = scaler_pleth.transform(out[:, 0, :])
    else:
        out = out[:, 1:, :]
        out[:, 0, :] = scaler_ecg.transform(out[:, 0, :])
    return out

# runs inference in mini batches with the given model
def run_batch_inference(batch_size, input_tensor, model, seed=42):
    num_samples = input_tensor.shape[0]
    output_tensor = []
    inferred = 0
    
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed) 
    np.random.seed(seed)
    
    while inferred < num_samples:
        batch_input = input_tensor[inferred:(inferred + batch_size)]
        batch_result = model(batch_input).cpu().detach().numpy().tolist()   
        output_tensor += batch_result
        inferred += batch_size
    
    output_tensor = np.array(output_tensor)    
    return output_tensor

# loads a pre-trained and fine-tuned model for inference 
def load_model(model_path, model_type, get_output=False):
    device_str = "cuda"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")   
    
    model = model_type
    model.load_state_dict(torch.load(model_path))
    model = model.to(device)
    
    if not get_output: # remove last layer, model returns last embedding vector
        model = torch.nn.Sequential(model.model, model.fc1)
    return model

"""
Loads up the data arrays for train, val and test splits 

Returns a data tuple of the form (xtrain, xval, xtest), (ytrain, yval, ytest) if use_inference set to true
xtrain contains in the following order numerics data (triage -> first monitoring), PTT, HRV, Perfusion index, 
and ECG/PPG waveform embeddings (optional if get_waves is False)
If two_models flag is set to false, users will receive a single model's output for waveform embeddings
If set to true, users will receive a concatenation of PPG -> ECG waveforms 

To access raw waveforms, use_inference can be set to false and the output will instead consist of 3 tuples
(xtrain_norm, xval_norm, xtest_norm), (xtrain_wide, xval_wide, xtest_wide), (ytrain, yval, ytest)
Arrays with the suffix `_norm` are tensors of dimension (n_patients, waveform channels, n_timepoints)

y values represent labels, either for tachycardia, hypotension, hypoxia or mews score predictions 
"""
def load_all_features(path_tuple, task, lead, get_waves=False, use_inference=False, two_models=False, model_type=None, 
                      model_path=None, pleth_model_path=None, ecg_model_path=None, mews_cutoff=4):
    device_str = "cuda"
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")   
    
    h5py_file, summary_file, labels_file, data_file, splits_file, hrv_ptt_file = path_tuple
    # input data 
    dfx_pleth = h5py.File(h5py_file, "r").get('waveforms')["Pleth"]["waveforms"][()]
    dfx_ecg = h5py.File(h5py_file, "r").get('waveforms')["II"]["waveforms"][()]
    combined = np.stack((dfx_pleth, dfx_ecg))

    dfx = np.moveaxis(combined, [0, 1, 2], [1, 0, 2])
    dfy = pd.read_csv(summary_file)
    data = pd.read_csv(data_file)
    labels = pd.read_csv(labels_file) 
    hrv_ptt = pd.read_csv(hrv_ptt_file)
    
    #get splits from previous splitter 
    with open(splits_file) as json_file:
        splits = json.load(json_file)

    dfx = dfx[dfy['patient_id'].isin(data['CSN'])]
    labels = labels[labels['CSN'].isin(data['CSN'])]
    dfy = dfy.loc[dfy['patient_id'].isin(data['CSN'])]
    hrv_ptt = hrv_ptt.loc[hrv_ptt['CSN'].isin(data['CSN'])]

    # realign data, labels and dfy indices
    data = data.set_index('CSN')
    data = data.reindex(index=dfy['patient_id'])
    data = data.reset_index()

    labels = labels.set_index('CSN')
    labels = labels.reindex(index=dfy['patient_id'])
    labels = labels.reset_index()
    
    hrv_ptt = hrv_ptt.set_index('CSN')
    hrv_ptt = hrv_ptt.reindex(index=dfy['patient_id'])
    hrv_ptt = hrv_ptt.reset_index().to_numpy()
    
    if task == "tachycardia":
        labels = binarize(np.array(labels['HR']), task)
    elif task == "hypoxia":
        labels = binarize(np.array(labels['SPO2']), task)
    elif task == "hypotension":
        labels = binarize(np.array(labels['MAP']), task)
    elif task == "mews":
        labels = binarize(np.array(labels['MEWS']), task, mews_cutoff=mews_cutoff)
    
    # np.where returns the indices where patient_id is in splits
    xtrain = dfx[np.where(data['patient_id'].isin(splits['train_ids']))] 
    ytrain = labels[np.where(data['patient_id'].isin(splits['train_ids']))]    
        
    xval = dfx[np.where(data['patient_id'].isin(splits['val_ids']))]
    yval = labels[np.where(data['patient_id'].isin(splits['val_ids']))]

    xtest = dfx[np.where(data['patient_id'].isin(splits['test_ids']))]
    ytest = labels[np.where(data['patient_id'].isin(splits['test_ids']))]

    # handle the wide features (numerics data) omitting the first row (patient CSN)
    dfx_wide = process_wide_features(data)
    xtrain_wide = dfx_wide[np.where(data['patient_id'].isin(splits['train_ids']))][:, 1:]
    xval_wide = dfx_wide[np.where(data['patient_id'].isin(splits['val_ids']))][:, 1:]
    xtest_wide = dfx_wide[np.where(data['patient_id'].isin(splits['test_ids']))][:, 1:]

    # grab the waveform derived features 
    hrv_perf_ptt_train = hrv_ptt[np.where(data['patient_id'].isin(splits['train_ids']))][:, 1:]
    hrv_perf_ptt_val = hrv_ptt[np.where(data['patient_id'].isin(splits['val_ids']))][:, 1:]
    hrv_perf_ptt_test = hrv_ptt[np.where(data['patient_id'].isin(splits['test_ids']))][:, 1:]

    xtrain_wide = np.concatenate((xtrain_wide, hrv_perf_ptt_train[:, -1:], hrv_perf_ptt_train[:, :-2], hrv_perf_ptt_train[:, -2:-1]), axis=1)
    xval_wide = np.concatenate((xval_wide, hrv_perf_ptt_val[:, -1:], hrv_perf_ptt_val[:, :-2], hrv_perf_ptt_val[:, -2:-1]), axis=1)
    xtest_wide = np.concatenate((xtest_wide, hrv_perf_ptt_test[:, -1:], hrv_perf_ptt_test[:, :-2], hrv_perf_ptt_test[:, -2:-1]), axis=1)
    
    # Scaling / Normalization (need to change if using ECG or Pleth only)
    scaler_sd_pleth, scaler_sd_ecg = StandardScaler(), StandardScaler()
    scaler_sd_pleth.fit(xtrain[:, 0, :])
    scaler_sd_ecg.fit(xtrain[:, 1, :])
    xtrain_norm = scale_input(in_np = xtrain, scaler_pleth=scaler_sd_pleth, scaler_ecg=scaler_sd_ecg, leads=lead)
    xval_norm = scale_input(in_np = xval, scaler_pleth=scaler_sd_pleth, scaler_ecg=scaler_sd_ecg, leads=lead)
    xtest_norm = scale_input(in_np = xtest, scaler_pleth=scaler_sd_pleth, scaler_ecg=scaler_sd_ecg, leads=lead)
    
    if not get_waves:
        return (xtrain_wide, xval_wide, xtest_wide), (ytrain, yval, ytest)
    if not use_inference:
        return (xtrain_norm, xval_norm, xtest_norm), (xtrain_wide, xval_wide, xtest_wide), (ytrain, yval, ytest)
    
    xtrain_norm_all, xval_norm_all, xtest_norm_all = torch.tensor(xtrain_norm).to(device).float(), torch.tensor(xval_norm).to(device).float(), torch.tensor(xtest_norm).to(device).float()  
    xtrain_norm_pleth, xval_norm_pleth, xtest_norm_pleth = torch.tensor(xtrain_norm[:, :1, :]).to(device).float(), torch.tensor(xval_norm[:, :1, :]).to(device).float(), torch.tensor(xtest_norm[:, :1, :]).to(device).float()   
    xtrain_norm_ecg, xval_norm_ecg, xtest_norm_ecg = torch.tensor(xtrain_norm[:, 1:, :]).to(device).float(), torch.tensor(xval_norm[:, 1:, :]).to(device).float(), torch.tensor(xtest_norm[:, 1:, :]).to(device).float()    
    # otherwise run inference
    
    # check whether we want to use two models and concatenate waveform embeddings 
    if not two_models: 
        extract_model = load_model(model_path, model_type)
        extract_model.eval()
        
        # which lead or combo of leads to use
        if lead == 'All':
            xtrain_embed = run_batch_inference(64, xtrain_norm_all, extract_model)
            xval_embed = run_batch_inference(64, xval_norm_all, extract_model)
            xtest_embed = run_batch_inference(64, xtest_norm_all, extract_model)
        elif lead == 'ECG':
            xtrain_embed = run_batch_inference(64, xtrain_norm_ecg, extract_model)
            xval_embed = run_batch_inference(64, xval_norm_ecg, extract_model)
            xtest_embed = run_batch_inference(64, xtest_norm_ecg, extract_model)
        elif lead == 'Pleth':
            xtrain_embed = run_batch_inference(64, xtrain_norm_pleth, extract_model)
            xval_embed = run_batch_inference(64, xval_norm_pleth, extract_model)
            xtest_embed = run_batch_inference(64, xtest_norm_pleth, extract_model)
            
    # concatenates both ECG and PPG waveform embeddings 
    else:
        extract_model_pleth = load_model(pleth_model_path, model_type)
        extract_model_pleth.eval()

        extract_model_ecg = load_model(ecg_model_path, model_type)
        extract_model_ecg.eval()

        xtrain_embed_pleth = run_batch_inference(64, xtrain_norm_pleth, extract_model_pleth)
        xval_embed_pleth = run_batch_inference(64, xval_norm_pleth, extract_model_pleth)
        xtest_embed_pleth = run_batch_inference(64, xtest_norm_pleth, extract_model_pleth)

        xtrain_embed_ecg = run_batch_inference(64, xtrain_norm_ecg, extract_model_ecg)
        xval_embed_ecg = run_batch_inference(64, xval_norm_ecg, extract_model_ecg)
        xtest_embed_ecg = run_batch_inference(64, xtest_norm_ecg, extract_model_ecg)
        
        xtrain_embed = np.concatenate((xtrain_embed_pleth, xtrain_embed_ecg), axis=1)
        xval_embed = np.concatenate((xval_embed_pleth, xval_embed_ecg), axis=1)
        xtest_embed = np.concatenate((xtest_embed_pleth, xtest_embed_ecg), axis=1)

    all_xtrain = np.concatenate((xtrain_wide, xtrain_embed), axis=1)
    all_xval = np.concatenate((xval_wide, xval_embed), axis=1)
    all_xtest = np.concatenate((xtest_wide, xtest_embed), axis=1)

    return (all_xtrain, all_xval, all_xtest), (ytrain, yval, ytest)

def filter_by_index(data_tuple, indices):
    (xtrain, xval, xtest), (ytrain, yval, ytest) = data_tuple
    
    xtrain_filtered = xtrain[:, indices]
    xval_filtered = xval[:, indices]
    xtest_filtered = xtest[:, indices]
    
    return (xtrain_filtered, xval_filtered, xtest_filtered), (ytrain, yval, ytest)
