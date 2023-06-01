import math
import numpy as np
from scipy.signal import find_peaks
from matplotlib import pyplot as plt

def get_PTTs(CSN, pleth, ii, csns, verbose=False):
    """
    function to get pulse transit time from a PPG and ECG raw signal of length 7500 
    (sampled at 125 Hz).
    """
    pleth_search = pleth[np.where(csns == CSN)].reshape(7500,)
    ii_search = ii[np.where(csns == CSN)].reshape(7500,)
    ii_peaks, _ = find_peaks(ii_search, distance=37)
    pleth_peaks, _ = find_peaks(pleth_search, distance=37)
    ii_peaks_norm = ii_peaks * 8 
    pleth_peaks_norm = pleth_peaks * 8
    
    ptts = []
    
    # get RR_ints 
    ii_int = np.array([ii_peaks_norm[i + 1] - ii_peaks_norm[i] for i in range(0, len(ii_peaks_norm) - 1)])

    ii_peaks_used = []
    pleth_peaks_used = []
    # Now let's filter for just the peaks we know are good between ECG and PPG:
    while len(ii_peaks_norm) > 0 and len(pleth_peaks_norm) > 0:
        
        # move pleth_peaks up one if we have an offset of peaks
        while len(ii_peaks_norm) > 0 and len(pleth_peaks_norm) > 0 and ii_peaks_norm[0] > pleth_peaks_norm[0]:
            pleth_peaks_norm = pleth_peaks_norm[1:]
            break
        
        # if we have nothing left, break out
        if len(ii_peaks_norm) == 0 or len(pleth_peaks_norm) == 0:
            break
        
        # if we are too behind
        while len(ii_peaks_norm) > 0 and len(pleth_peaks_norm) > 0 and pleth_peaks_norm[0] - ii_peaks_norm[0] > ii_int[0]:
            ii_peaks_norm = ii_peaks_norm[1:]
            if len(ii_int) > 1:
                ii_int = ii_int[1:]
            break
        
        # now we can get our PTTs
        ptt = pleth_peaks_norm[0] - ii_peaks_norm[0]
        if ptt < ii_int[0] and ptt > 0:
            ii_peaks_used.append(int(ii_peaks_norm[0] / 8))
            pleth_peaks_used.append(int(pleth_peaks_norm[0] / 8))
            ptts.append(ptt)
        
        # Update for next PTT
        ii_peaks_norm = ii_peaks_norm[1:]
        pleth_peaks_norm = pleth_peaks_norm[1:]
        if len(ii_int) > 1:
            ii_int = ii_int[1:]
        else: 
            break
               
    if len(ptts) > 0:
        mean_ptt = np.mean(ptts)
         # PLOT
        if verbose:
            ii_plot = np.array(ii_peaks_used).astype(int)
            pleth_plot = np.array(pleth_peaks_used).astype(int)
            a4_dims = (12, 4)
            fig, (ax1, ax2) = plt.subplots(2, figsize=a4_dims)
            ax1.plot(ii_search)
            ax1.plot(ii_peaks, ii_search[ii_peaks], "x", markersize=7.5)
            ax1.plot(ii_plot, ii_search[ii_plot], "o", alpha=0.4, markersize=5, c = 'red')
            ax2.plot(pleth_search)
            ax2.plot(pleth_peaks, pleth_search[pleth_peaks], "x", markersize=7.5)
            ax2.plot(pleth_plot, pleth_search[pleth_plot], "o", alpha=0.4,markersize=5, c = 'red')
            plt.show()
            # peak_diff = np.abs(len(ii_peaks_norm) - len(pleth_peaks_norm))
            # print(f"difference in number of peaks = {peak_diff}")
            print(f"For patient {CSN}, Mean ptt = {np.mean(ptts)}; len(ptt) = {len(ptts)}")
        return float(mean_ptt)
    else:
        if verbose:
            print("No ptts found")
        return math.nan
    