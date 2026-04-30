"""

Organizing stuff into a custom package so that I don't have
to keep going back and forth copying and pasting stuff.
Work in (slow) progress.

"""


##Packages
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from tqdm import tqdm
import scipy
from SSINS import Catalog_Plot as cp
from scipy.signal import windows
from scipy import stats as st
from scipy.ndimage import median_filter
from pyuvdata import UVFlag
import os
import numpy.linalg as la


##Defining yaml file importer

def yaml_import(filepath, savepath):

    with open(filepath, 'r') as file:
        new_file = yaml.safe_load(file)
    print('File retrieved from', filepath)
    
    with open(savepath, 'w') as file:
        yaml.dump(new_file, file)
    print('File saved in', savepath)


##Parsing through, finding, and combining the selected h5 files
""" def combine(obsid, tar, searchstrings, directory): """

""" 

    The file search and selection function.

    Args:
    
    obsid -- 

 #Loading pointings dictionary.
    with open(obsid, 'r') as file:
        pointings = yaml.safe_load(file)            

    print('Pointings loaded.')
    
    #Searching for the selected obsids and corresponding pointings.
    keys = [
        key for key, val in pointings.items() 
        if val == tar
        and any(
            s.lower() in key.lower() 
            for s in search_strings
        )
    ]                                               

    print('Accessed obsid. Retrieved', len(keys), 'files.')

    all_files = list(Path(directory).glob('*'))
    
    #Collecting files in the directory.
    matching_files = [
        f for f in tqdm(all_files, desc="Searching files", unit="file")
        if (f.is_file() 
            and "SSINS_data" in f.name
            and any(key.lower() in f.name.lower() for key in keys))
    ]

    matching_files.sort()

    print(len(matching_files))

    #Combining selected files into one large file, and saving on local CEDAR directory.
    combined = INS(matching_files[0], telescope_name='MWA')
    for file in tqdm(matching_files[1:], desc="Combining files", unit="files"):
        current = INS(file, telescope_name='MWA')
        combined += current

    print('Files combined successfully.')

    prefix = '/home/andreili/ssins_env/Week_2/'
    combined.write(prefix, output_type='data', clobber=True)

    print('Combined file saved.') """


##Selecting frequency channels
def chan_select(ins, chan_name, shape_dict):

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
    line_mask = np.load('linemask.npy')[subband_chans]
    ins_subband.metric_array = ins_subband.metric_array[:, ~line_mask, :]
    ins_subband.freq_array = ins_subband.freq_array[~line_mask]
    masked_ins.metric_array = masked_ins.metric_array[:, ~line_mask, :]
    masked_ins.freq_array = masked_ins.freq_array[~line_mask]

    return ins_subband, masked_ins, N_bl, N_freq


##Averaging over a certain channel
def chan_avg(ins):
    
    #Calling time and amplitudes, filtering out infs
    time = np.array(ins.time_array)
    data = np.array(ins.metric_array[:, :, 0])
    data[~np.isfinite(data)] = -1

    #Averaging over full band/subband per integration
    freq_averaged = np.mean(data, axis=1)

    #Filtering out zeroes
    non_zero_mask = freq_averaged > 0
    time_filtered = time[non_zero_mask]
    freq_averaged_filtered = freq_averaged[non_zero_mask]

    return time_filtered, freq_averaged_filtered


#Generating waterfall plots
def waterfall(ins, prefix):
    cp.INS_plot(ins, prefix, file_ext='pdf')


#Generating time series
def timeseries(ins, save_path):

    #Calling time and amplitudes, filtering out infs
    time = np.array(ins.time_array)
    data = np.array(ins.metric_array[:, :, 0])
    data[~np.isfinite(data)] = 0

    #Averaging over full band/subband per integration
    freq_averaged = np.mean(data, axis=1)

    #Filtering out zeroes
    non_zero_mask = freq_averaged != 0
    time_filtered = time[non_zero_mask]
    freq_averaged_filtered = freq_averaged[non_zero_mask]

    print('Data successfully averaged over all freq channels.')

    #Generating scatterplot.
    plt.scatter(time_filtered, freq_averaged_filtered)
    plt.ylabel('Amplitude (arb.)')
    plt.xlabel('Time (Julian date)')
    plt.savefig(save_path)
    print('Plot successfully generated.')


#Getting your background fit
""" def DPSS_fit(time, masked_time, masked_amp, N_win=4, N_terms=20, N_bl=8000, N_freq=75):
    
    The DPSS basis fitting function. Used to fit smooth background to masked data.

    Args:
    time            -- Numpy unmasked time array.
    masked_time     -- Numpy masked time array.
    masked_amp      -- Numpy masked visibility amplitudes array.
    N_win           -- Half-bandwidth in the DPSS basis. See scipy.signal.windows.dpss for documentation.
    N_terms         -- Number of terms you want in the DPSS basis.
    N_bl            -- Number of baselines you're averaging over.
    N_freq          -- Number of frequency bins you're averaging over. 

    #Sorting indices in case they got messed up somewhere.
    sort_ind = np.argsort(masked_time)
    masked_time = np.array(masked_time[sort_ind])
    masked_amp = np.array(masked_amp[sort_ind])

    #Generating uniform time grid
    dt = st.mode(np.diff(masked_time)).mode
    smooth_time = np.arange(time.min(), time.max() + dt, dt)

    #Padding amplitude arrays with NaN
    masked_padded_amp = np.full_like(smooth_time, np.nan, dtype=float)
    masked_indices = np.searchsorted(smooth_time, masked_time)
    masked_padded_amp[masked_indices] = masked_amp

    #Generating DPSS design matrix
    M = len(smooth_time)
    design = windows.dpss(M, N_win, Kmax=N_terms, return_ratios=True)[0].T

    #Constructing least-squares DPSS coefficients
    observed_indices = ~np.isnan(masked_padded_amp)
    splice_design = design[observed_indices, :]
    coefficients = la.solve(splice_design.T @ splice_design, splice_design.T @ masked_padded_amp[observed_indices])
    fit = design @ coefficients     #Main return of the DPSS_fit function
    
    #Finding the noise covariances
    mu = splice_design @ coefficients
    C = 4/np.pi-1
    noise_cov = np.diag((fit ** 2) * C / (N_bl * N_freq))
    splice_noise_cov = np.diag((mu ** 2) * C / (N_bl * N_freq))

    #Finding the parameter covariance
    sigma_p = la.solve(splice_design.T @ la.solve(splice_noise_cov, splice_design), np.eye(splice_design.shape[1]))

    #Finding the fit covariance
    sigma_f = design @ sigma_p @ design.T

    return fit, coefficients, smooth_time, noise_cov, sigma_p, sigma_f """


def DPSS_fit(time, masked_time, masked_amp, N_win=4, N_terms=20, N_bl=8000, N_freq=75, 
             prior_samples=None, prior_power=2.0, min_noise_var=1e-9):
    """
    The DPSS basis fitting function. Used to fit smooth background to masked data.

    Args:
    time            -- Numpy unmasked time array.
    masked_time     -- Numpy masked time array.
    masked_amp      -- Numpy masked visibility amplitudes array.
    N_win           -- Half-bandwidth in the DPSS basis. See scipy.signal.windows.dpss for documentation.
    N_terms         -- Number of terms you want in the DPSS basis.
    N_bl            -- Number of baselines you're averaging over.
    N_freq          -- Number of frequency bins you're averaging over.
    prior_samples   -- Sample matrix (n_samples x n_terms) for DPSS coefficients
    prior_power     -- Exponent for diagonal scaling of prior (default=2.0)
    min_noise_var   -- Minimum noise variance to avoid numerical issues
    """
    # Sorting and grid setup
    sort_ind = np.argsort(masked_time)
    masked_time = masked_time[sort_ind]
    masked_amp = masked_amp[sort_ind]
    dt = st.mode(np.diff(masked_time)).mode
    smooth_time = np.arange(time.min(), time.max() + dt, dt)
    
    # Padding amplitude array with NaNs
    masked_padded_amp = np.full_like(smooth_time, np.nan, dtype=float)
    masked_indices = np.searchsorted(smooth_time, masked_time)
    masked_padded_amp[masked_indices] = masked_amp
    observed_indices = ~np.isnan(masked_padded_amp)
    y_obs = masked_padded_amp[observed_indices]

    # Generate DPSS design matrix
    M = len(smooth_time)
    design = windows.dpss(M, N_win, Kmax=N_terms, return_ratios=True)[0].T
    splice_design = design[observed_indices, :]

    # Constructing priors
    if prior_samples is not None:
        # Compute prior statistics from samples
        c0 = np.mean(prior_samples, axis=0)
        Lambda = np.cov(prior_samples, rowvar=False)
        
        # Add regularization to ensure positive definiteness
        eigvals = np.linalg.eigvalsh(Lambda)
        if np.min(eigvals) <= 0:
            Lambda += np.eye(N_terms) * (np.abs(np.min(eigvals)) + 1e-6)
    else:
        # Default weak prior if no samples provided
        c0 = np.zeros(N_terms)
        Lambda = np.diag(1e6 * (1 + np.arange(N_terms)**prior_power))
    
    # Computing precision matrix
    Gamma = np.linalg.inv(Lambda)  # Prior precision

    # ===== INITIAL FIT FOR NOISE ESTIMATION =====
    # Solve: (A.T A) c = A.T y
    lhs_ols = splice_design.T @ splice_design
    rhs_ols = splice_design.T @ y_obs
    c_init = la.solve(lhs_ols, rhs_ols)
    mu_init = splice_design @ c_init

    # Noise variance from initial fit
    C = 4 / np.pi - 1  # Noise scaling constant
    noise_var = (mu_init**2) * C / (N_bl * N_freq)
    noise_var = np.clip(noise_var, min_noise_var, None)

    # ===== MAP ESTIMATION WITH PRIOR =====
    # Weighted design matrix: W = diag(1/sqrt(noise_var))
    W = 1.0 / np.sqrt(noise_var)
    weighted_design = W[:, None] * splice_design
    weighted_y = W * y_obs

    # MAP equation: (A.T N^{-1} A + Γ) c = A.T N^{-1} y + Γ c0
    lhs_map = weighted_design.T @ weighted_design + Gamma
    rhs_map = weighted_design.T @ weighted_y + Gamma @ c0
    coefficients = la.solve(lhs_map, rhs_map)

    # ===== COVARIANCE MATRICES =====
    # Recompute noise variance using MAP solution
    mu_map = splice_design @ coefficients
    noise_var_map = (mu_map**2) * C / (N_bl * N_freq)
    noise_var_map = np.clip(noise_var_map, min_noise_var, None)
    
    # Parameter covariance: (A.T N^{-1} A + Γ)^{-1}
    W_map = 1.0 / np.sqrt(noise_var_map)
    weighted_design_map = W_map[:, None] * splice_design
    lhs_cov = weighted_design_map.T @ weighted_design_map + Gamma
    sigma_p = la.inv(lhs_cov)

    # Full fit and covariances
    fit = design @ coefficients
    noise_cov = np.diag((fit**2) * C / (N_bl * N_freq))
    sigma_f = design @ sigma_p @ design.T

    return fit, coefficients, smooth_time, noise_cov, sigma_p, sigma_f