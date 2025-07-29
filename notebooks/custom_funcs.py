##Importing necessary packages
import numpy as np
from SSINS import INS


##Defining two important functions -- the frequency channel selection and averaging functions.

def chan_select(ins, chan_name, shape_dict):

    """
    The frequency channel selection function. Also used to apply masks to your data.

    Args
    -----
    ins:
        INS object to be filtered.
    chan_name:
        channel name from shape dictionary
    shape_dict:
        shape dictionary by which you've divided and labelled the frequency array

    Returns
    -------
    ins_subband:
        INS object whose metric array is the filtered metric array without masks.
    masked_ins:
        INS object whose metric array is the filtered metric array with masks.
    N_bl:
        number of baselines.
    N_freq:
        number of frequencies.
    """

    #Creating band selection mask
    freq_mask = (ins.freq_array >= shape_dict[chan_name][0]) & (ins.freq_array <= shape_dict[chan_name][1])
    subband_chans = np.where(freq_mask)[0]
    N_freq = len(subband_chans)

    #Generating selected subband array
    ins_subband = ins.copy()
    ins_subband.select(freq_chans=subband_chans)
    N_bl = np.max(ins_subband.weights_array)

    #Masking
    masked_ins = ins_subband.copy()
    mask_array = ins_subband.mask_to_flags()[1:]
    masked_data = np.copy(ins_subband.metric_array)
    masked_data[mask_array] = np.nan
    masked_ins.metric_array = masked_data
    
    #Getting rid of the pesky subband lines
    line_mask = np.load('notebooks/Data/linemask.npy')[subband_chans]
    ins_subband.metric_array = ins_subband.metric_array[:, ~line_mask, :]
    ins_subband.freq_array = ins_subband.freq_array[~line_mask]
    masked_ins.metric_array = masked_ins.metric_array[:, ~line_mask, :]
    masked_ins.freq_array = masked_ins.freq_array[~line_mask]

    return ins_subband, masked_ins, N_bl, N_freq


def chan_avg(ins):

    """
    The frequency channel averaging function. Averages the metric array over the frequency axis.

    Args
    ----
    ins:
        INS object to be averaged.

    Returns
    -------
    time_filtered:
        Clean time series.
    freq_averaged:
        Clean and frequency-averaged metric array.
    """
    
    #Calling time and amplitudes, filtering out infs and NaNs by mapping them to -1.
    time = np.array(ins.time_array)
    data = np.array(ins.metric_array[:, :, 0])
    data[~np.isfinite(data)] = -1

    #Averaging over full band/subband per integration
    freq_averaged = np.mean(data, axis=1)

    #Filtering out zeroes
    non_zero_mask = freq_averaged > 0
    time_filtered = time[non_zero_mask]
    freq_averaged = freq_averaged[non_zero_mask]

    return time_filtered, freq_averaged

