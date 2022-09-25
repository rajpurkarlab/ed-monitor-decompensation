from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal
import h5py

def plot_waveforms(time, time_wave, freq, freq_wave):
    
    plt.plot (time[:1250], time_wave[1, :1250])
    plt.title('Time Domain ECG Signal - 10 seconds')
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.show()
    
    plt.plot(freq[freq < 0.5], freq_wave[1, freq < 0.5])
    plt.title('Frequency domain ECG Signal < 0.5 Hz')
    plt.xlabel('Frequency in Hz')
    plt.ylabel('Power')
    plt.show()
    
    plt.plot(freq[freq > 0], freq_wave[1, freq > 0])
    plt.title('Frequency domain ECG Signal Hz')
    plt.xlabel('Frequency in Hz')
    plt.ylabel('Power')
    plt.show()

def get_frequency_domain_waveforms(filename, duration, sample_rate):
    
    N = sample_rate * duration 
    time = np.linspace(0, duration, N)
    
    pleth = h5py.File(filename, "r").get('waveforms')["Pleth"]["waveforms"][()]
    ecg = h5py.File(filename, "r").get('waveforms')["II"]["waveforms"][()]
    print(pleth.shape)
    
    freq, ecg_fd = signal.periodogram(ecg, sample_rate)
    freq, pleth_fd = signal.periodogram(ecg, sample_rate)
    
    plot_waveforms(time, ecg, freq, ecg_fd)
    
    return freq, ecg_fd

# many helper fns to produce different freq domain based hrv measurements

# total power should be vlf + lf + hf if short-term recording
# only lf and hf make sense for 1 minute recording
def get_absolute_powers(ecg_fd, freq):
    lf_abs = np.sum(ecg_fd[:, (freq > 0.04) & (freq < 0.15)], axis=1)    
    hf_abs = np.sum(ecg_fd[:, (freq > 0.15) & (freq < 0.4)], axis=1)
    
    return lf_abs, hf_abs
    
def get_relative_powers(absolute_powers, ecg_fd, freq):
    lf_abs, hf_abs = absolute_powers
    
    total_power = np.sum(ecg_fd[:, (freq > 0.04) & (freq < 0.4)], axis=1)
    lf_rel = lf_abs / total_power
    hf_rel = hf_abs / total_power
    
    return lf_rel, hf_rel
    
def get_lf_hf_ratio(absolute_powers):
    lf_abs, hf_abs = absolute_powers
    return lf_abs / hf_abs
    
def get_peak_freq(ecg_fd, freq):
    lf = ecg_fd[:, (freq > 0.04) & (freq < 0.15)]
    lf_freq = freq[(freq > 0.04) & (freq < 0.15)]
    
    hf = ecg_fd[:, (freq > 0.15) & (freq < 0.4)]
    hf_freq = freq[(freq > 0.15) & (freq < 0.4)]
    
    lf_peak = np.take(lf_freq, np.argmax(lf, axis=1))
    hf_peak = np.take(hf_freq, np.argmax(hf, axis=1))
    
    return lf_peak, hf_peak

def generate_all_fd_hrvs(ecg_fd, freq, summary_file):
    csns = pd.read_csv(summary_file).patient_id
    
        
    abs_powers = get_absolute_powers(ecg_fd, freq)
    lf_abs, hf_abs = abs_powers
    
    lf_rel, hf_rel = get_relative_powers(abs_powers, ecg_fd, freq)
    lfhf_ratio = get_lf_hf_ratio(abs_powers)
    lf_peak, hf_peak = get_peak_freq(ecg_fd, freq)
    
    metrics = {'CSN':csns.tolist(), 'LF_Abs': lf_abs, 'HF_Abs': hf_abs, 
               'LF_Rel': lf_rel, 'HF_Rel': hf_rel, 'LF_Peak': lf_peak, 
               'HF_Peak': hf_peak,'LFHF_Ratio': lfhf_ratio}
    return metrics