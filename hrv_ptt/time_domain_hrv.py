import numpy as np
import math
from matplotlib import pyplot as plt

# SDRR, pRR50, RMSSD
def SDRR_pRR50_RMSSD(np_array):
    if np.sum(np.isnan(np_array)) >= 2:
        return math.nan, math.nan, math.nan
    else:
        RR_diff = np.array([np_array[i + 1] - np_array[i] for i in range(0, len(np_array) - 1)])
        pRR50 = np.where(np.absolute(RR_diff) > 50 , 1, 0).sum() / len(RR_diff)
        return np_array.std(), pRR50, np.sqrt(np.mean(np.square(RR_diff)))

# HRVI, TINN
def HRVI_TINN(np_array, verbose=False):
    data = np_array
    if np.sum(np.isnan(np_array)) >= 2:
        return math.nan, math.nan
    else:
        binwidth = int(8)
        counts, edges = np.histogram(data, bins=range(int(np.nanmin(data)), int(np.nanmax(data)) + binwidth, binwidth))
        if verbose:
            plt.stairs(counts, edges, fill=True)
            plt.show()
        if len(counts) == 0:
            return 1.0, 0.0
        else:
            return (len(data) / np.nanmax(counts)), (int(np.nanmax(data)) + binwidth - int(np.nanmin(data)))
