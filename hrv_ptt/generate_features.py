import numpy as np
import pandas as pd
import h5py
from ptt import get_PTTs
from time_domain_hrv import SDRR_pRR50_RMSSD, HRVI_TINN
from frequency_domain_hrv import plot_waveforms, get_frequency_domain_waveforms, generate_all_fd_hrvs

def generate_ptt(path_tuple):
    h5py_file, summary_file = path_tuple
    
    # PTT 
    pleth = h5py.File(h5py_file, "r").get('waveforms')["Pleth"]["waveforms"][()]
    ii = h5py.File(h5py_file, "r").get('waveforms')["II"]["waveforms"][()]

    csns = pd.read_csv(summary_file).patient_id
    dict_ptts = {'CSN':csns.tolist(), 'ptt':[]}

    for csn in csns:
        dict_ptts['ptt'].append(get_PTTs(csn, pleth, ii, csns, verbose=False))
        
    # save ptt for all patients, not filtered as csv
    ptt_df = pd.DataFrame.from_dict(dict_ptts)
    ptt_df.ptt = ptt_df.ptt.astype(float)
    ptt_df.to_csv('./final_ptt_allpts.csv', index=False)
    
def generate_td_hrv(path_tuple):
    h5py_file, summary_file = path_tuple
    csns = pd.read_csv(summary_file).patient_id

    # Time domain HRV measurements
    RR_int = h5py.File(h5py_file, "r").get('numerics_before')["btbRRInt_ms"]["vals"][()]
    dict_td_hrvs = {'CSN':csns.tolist(), 'SDRR':[], 'pRR50':[], 'RMSSD':[], 'HRVI':[], 'TINN':[]}

    for i in range(len(csns)):
        data = RR_int[i]
        try:
            SDRR, pRR50, RMSSD = SDRR_pRR50_RMSSD(data)
            HRVI, TINN = HRVI_TINN(data)
        except:
            print(f"issue at index {i}")
        dict_td_hrvs['SDRR'].append(SDRR)
        dict_td_hrvs['pRR50'].append(pRR50)
        dict_td_hrvs['RMSSD'].append(RMSSD)
        dict_td_hrvs['HRVI'].append(HRVI)
        dict_td_hrvs['TINN'].append(TINN) 
    
    hrv_td_df = pd.DataFrame.from_dict(dict_td_hrvs)
    hrv_td_df.SDRR = hrv_td_df.SDRR.astype(float)
    hrv_td_df.pRR50 = hrv_td_df.pRR50.astype(float)
    hrv_td_df.RMSSD = hrv_td_df.RMSSD.astype(float)
    hrv_td_df.HRVI = hrv_td_df.HRVI.astype(float)
    hrv_td_df.TINN = hrv_td_df.TINN.astype(float)
    
    # save time domain hrv metrics for all patients, not filtered as csv
    hrv_td_df.to_csv('./final_hrv_allpts.csv', index=False)
    
def generate_fd_hrv(path_tuple):  
    h5py_file, summary_file = path_tuple
    
    # frequency domain hrv measurements
    freq, ecg_fd = get_frequency_domain_waveforms(h5py_file, 60, 125)
    hrv_fd_metrics = generate_all_fd_hrvs(ecg_fd, freq, summary_file)
    
    hrv_fd_df = pd.DataFrame.from_dict(hrv_fd_metrics)
    hrv_fd_df.LF_Abs = hrv_fd_df.LF_Abs.astype(float)
    hrv_fd_df.HF_Abs = hrv_fd_df.HF_Abs.astype(float)
    hrv_fd_df.LF_Rel = hrv_fd_df.LF_Rel.astype(float)
    hrv_fd_df.HF_Rel = hrv_fd_df.HF_Rel.astype(float)
    hrv_fd_df.LF_Peak = hrv_fd_df.LF_Peak.astype(float)
    hrv_fd_df.HF_Peak = hrv_fd_df.HF_Peak.astype(float)
    hrv_fd_df.LFHF_Ratio = hrv_fd_df.LFHF_Ratio.astype(float)
    
    hrv_fd_df.to_csv("./final_hrv_fd_allpts.csv", index=False)

def compile_all(path_tuple, saved_files):
    h5py_file, summary_file = path_tuple
    ptt_file, td_hrv_file, fd_hrv_file = saved_files
    
    # join time domain hrv and ptt with perf
    df_td_hrv = pd.read_csv(td_hrv_file)
    df_ptt = pd.read_csv(ptt_file)
    
    perf = np.nanmean(h5py.File(h5py_file, "r").get('numerics_before')["Perf"]["vals"][()], axis=1)
    df_td_hrv['Perf'] = perf
    df_merged_td_hrv_ptt = pd.merge(df_td_hrv, df_ptt, on='CSN')
    df_merged_td_hrv_ptt.to_csv('./final_hrv_ptt_perf_allpts.csv', index=False)
    
    # merge all ptt, hrv, perf measurements into a single saved file 
    df_fd_hrv = pd.read_csv(fd_hrv_file)
    df_merged_all = pd.merge(df_fd_hrv, df_merged_td_hrv_ptt, on='CSN')

    merged_save_path = "./final_hrv_fd_ptt_perf_allpts.csv"
    df_merged_all.to_csv(merged_save_path, index=False)
    
def main():
    h5py_file = "/deep/group/ed-monitor-self-supervised/v4/downstream.15min.60min.60sec.h5"
    summary_file = "/deep/group/ed-monitor-self-supervised/v4/downstream.15min.60min.60sec.csv"
    
    path_tuple = (h5py_file, summary_file)
    saved_file_paths = './final_ptt_allpts.csv', './final_hrv_allpts.csv', './final_hrv_fd_allpts.csv'
    
    generate_ptt(path_tuple)
    generate_td_hrv(path_tuple)
    generate_fd_hrv(path_tuple)
    
    compile_all(path_tuple, saved_file_paths)
    
if __name__ == "__main__":
    main()
