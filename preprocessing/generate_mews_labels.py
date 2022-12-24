import h5py
import pandas as pd
import numpy as np
import json

TIMES = ['60min', '90min', '120min']

def generate_mews_labels(h5py_file, data_file, summary_file, time):

    with h5py.File(h5py_file, "r") as f:
        rr_numerics = f.get('numerics_after')["RR"]["vals"][()]
        rr_times = f.get('numerics_after')["RR"]["times"][()]

        hr_numerics = f.get('numerics_after')["HR"]["vals"][()]
        hr_times = f.get('numerics_after')["HR"]["times"][()]

        sbp_numerics = f.get('numerics_after')["NBPs"]["vals"][()]
        sbp_times = f.get('numerics_after')["NBPs"]["times"][()]
        
        print(sbp_numerics)


    # load data_file into a pandas dataframe
    df = pd.read_csv(data_file)
    summary = pd.read_csv(summary_file)

    # align data_file and h5py_file along patient_id
    summary = summary.loc[summary['patient_id'].isin(df['CSN'])]
    df = df.set_index('CSN')
    df = df.reindex(index=summary['patient_id'])
    df = df.reset_index()
    
    print(rr_times.shape)
    print(len(rr_times[0]))
    print(rr_times)
    print(rr_numerics)

    # get the column named "Triage_temp" afrom the dataframe
    triage_temp = df["Triage_Temp"]
    
    # calculate mews score which is defined as the sum of the following:
    # RR score: 2 if RR < 9, 0 if 9 <= RR < 15, 1 if 15 <= RR < 21, 2 if 21 <= RR < 30, 3 if RR >= 30
    # HR score: 2 if HR < 40, 1 if 40 <= HR < 51, 0 if 51 <= HR < 101, 1 if 101 <= HR < 111, 2 if 111 <= HR < 130, 3 if HR >= 130
    # SBP score: 3 if SBP <= 70, 2 if 71 <= SBP < 81, 1 if 81 <= SBP < 101, 0 if 101 <= SBP < 200, 2 if SBP >= 200
    # Triage temperature score: 2 if Triage_temp < 35, 0 if 35 <= Triage_temp < 38.5, 2 if Triage_temp >= 38.5
    # mews score = RR score + HR score + SBP score + Triage temperature score
    mews_score = np.zeros_like(rr_times)
    
    # loop over all patients
    for i in range(rr_times.shape[0]):
        # loop over all time steps
        for j in range(rr_times.shape[1]):
            if rr_numerics[i, j] < 9:
                rr_score = 2
            elif rr_numerics[i, j] < 15:
                rr_score = 0
            elif rr_numerics[i, j] < 21:
                rr_score = 1
            elif rr_numerics[i, j] < 30:
                rr_score = 2
            else:
                rr_score = 3

            if hr_numerics[i, j] < 40:
                hr_score = 2
            elif hr_numerics[i, j] < 51:
                hr_score = 1
            elif hr_numerics[i, j] < 101:
                hr_score = 0
            elif hr_numerics[i, j] < 111:
                hr_score = 1
            elif hr_numerics[i, j] < 130:
                hr_score = 2
            else:
                hr_score = 3

            if sbp_numerics[i, j] <= 70:
                sbp_score = 3
            elif sbp_numerics[i, j] < 81:
                sbp_score = 2
            elif sbp_numerics[i, j] < 101:
                sbp_score = 1
            elif sbp_numerics[i, j] < 200:
                sbp_score = 0
            else:
                sbp_score = 2

            if triage_temp[i] < 35:
                triage_temp_score = 2
            elif triage_temp[i] < 38.5:
                triage_temp_score = 0
            else:
                triage_temp_score = 2

            mews_score[i, j] = rr_score + hr_score + sbp_score + triage_temp_score
            
    print(mews_score)
    
    # get the max mews score for each patient
    mews_score = np.max(mews_score, axis=1)
    
    print(mews_score)
    
    # save mews_score to a csv file 
    np.savetxt(f"labels_{time}_mews.csv", mews_score, delimiter=",")

def main():
    file_path_config = "/deep/group/ed-monitor-self-supervised/test_models_v1/ed-monitor-decompensation-clean/path_configs_new.json"
    with open(file_path_config) as fpc:
        all_paths = json.load(fpc)
        
    for time in TIMES:
        time_paths = all_paths[time]
        h5py_file = time_paths["h5py_file"]
        summary_file = time_paths["summary_file"]
        data_file = time_paths["data_file"]
        
        generate_mews_labels(h5py_file, data_file, summary_file, time)
        
if __name__ == "__main__":
    main()

    
