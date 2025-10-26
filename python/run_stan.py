import numpy as np
from numpy.polynomial.legendre import legvander
import matplotlib.pyplot as plt
import json
import glob

plt.style.use('seaborn-v0_8')

# ------------------------ Global Variables ------------------------


ABS_DIR = '/users/jmduchar/data/jmduchar/Research/mcgill25/rfi_characterization/'

DATAPATH = ABS_DIR+"data_private/raw_data/"
ANNOTATIONPATH = ABS_DIR+"data_private/annotations/"

ALL_FILES = glob.glob(DATAPATH+"*.npy")
ALL_ANNOTATIONS = glob.glob(ANNOTATIONPATH+"*.npy")


# ------------------------ Helper Functions ------------------------


def time_elapsed(t0, t):
    sec_elapsed = int(t - t0)
    hours = sec_elapsed // 3600
    minutes = (sec_elapsed % 3600) // 60
    seconds = sec_elapsed % 60
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def map_to_unit_interval(u):
    u = np.asarray(u, dtype=np.float64)
    umin, umax = float(np.min(u)), float(np.max(u))
    x = 2.0 * (u - umin) / (umax - umin) - 1.0
    return x


def legendre_design_numpy(u, L):
    x = map_to_unit_interval(u)
    A_full = legvander(x, L - 1)     # shape (T, L) -> columns: P_0 ... P_{L-1}
    return A_full.astype(np.float64, copy=False)


def build_legendre_design_matrix(y, L=8):
    u = np.arange(len(y), dtype=np.float64)
    A = legendre_design_numpy(u, L)
    return A


def obs_pointing_key(path):
    fn = path.split("/")[-1]
    parts = fn.split("_")
    # parts[-2] = obs, parts[-1] = p.ext
    obs = parts[-2]
    p = parts[-1].split(".")[0]
    return (obs, p)


def create_data_dict_hmm(pointing, z, start_idx, stop_idx, sigma=0.33, save_data=False, data_save_path=None):

    all_p = [i for i in ALL_FILES if pointing in i and "bad" not in i]
    all_annotations = [i for i in ALL_ANNOTATIONS if pointing in i and "bad" not in i]

    all_night_pointing = [obs_pointing_key(i) for i in all_p]
    ann_night_pointing = set(obs_pointing_key(i) for i in all_annotations)

    y_unsup_list, y_sup_list = [], []
    s_sup_list = []

    c = 0
    for pdx, (obs,p) in enumerate(all_night_pointing):
        
        if (obs,p) not in ann_night_pointing:  # UNSUP NIGHT
            y_unsup_list.append(z[start_idx[c]-1 : stop_idx[c]])  # guarding against Stan's 1-based inclusive indexing
            
        else:  # SUP NIGHT
            sup_sample = z[start_idx[c]-1 : stop_idx[c]]
            y_sup_list.append(sup_sample)
            
            # Find the corresponding annotations
            for annotation in all_annotations:
                if obs in annotation and p in annotation:
                    labels = np.load(annotation).astype(int)
                
            assert len(labels) == len(sup_sample)
            s_sup_list.append(labels.astype(int))

        c += 1

    # Concatenate
    if len(y_unsup_list) > 0:
        y_unsup = np.concatenate(y_unsup_list, axis=0)
    else:
        y_unsup = np.empty((0,), dtype=float)
        
    if len(y_sup_list) > 0:
        y_sup = np.concatenate(y_sup_list, axis=0)
        s_sup = np.concatenate(s_sup_list, axis=0)
    else:
        y_sup = np.empty((0,), dtype=float)
        s_sup = np.empty((0,), dtype=float)

    # Build dictionary
    data_dict = {
        'N_unsup':         int(len(y_unsup)),
        'y_unsup':         y_unsup.tolist(),
        
        'N_sup':           int(len(y_sup)),
        'y_sup':           y_sup.tolist(),
        's_sup':           s_sup.tolist(),
 
        'sigma':           sigma,
    }

    if save_data:
        with open(
            data_save_path,
            "w"
        ) as f:
            json.dump(data_dict, f, indent=2)

    return data_dict


def create_data_dict_bg(pointing, L, sigma=0.33, save_data=False, data_save_path=None):
    """
    Initialize data dictionary (first pass only)
    """

    all_p = [i for i in ALL_FILES if pointing in i and "bad" not in i]
    all_night_pointing = [obs_pointing_key(i) for i in all_p]

    y_list = []
    A_rows = []
    start_stop = []
    c = 0
    for pdx, (obs,p) in enumerate(all_night_pointing):
        
        sample = np.load(all_p[pdx])
        
        # Remove NAN
        nan_mask = np.isnan(sample)
        sample = sample[~nan_mask]
        
        A = build_legendre_design_matrix(sample, L) 
        
        y_list.append(sample.astype(float))
        A_rows.append(A.astype(float))
        
        a = c
        b = c + len(sample) - 1
        start_stop.append((a, b))
        c += len(sample)

    # Concatenate
    y = np.concatenate(y_list, axis=0)
    A = np.vstack(A_rows)

    # Since this is the first pass, initialize array of ones for w
    w = np.ones_like(y)

    # Stan indexes from 1
    start_idx = [int(a+1) for (a,b) in start_stop]
    stop_idx  = [int(b+1) for (a,b) in start_stop]

    # Build dictionary
    data_dict = {
        'L':         int(L),
        'N':         int(len(y)),
        'y':         y.tolist(),
        'A':         A.tolist(),
        'M':         int(len(start_idx)),
        'w':         w,

        'start_idx': start_idx,
        'stop_idx':  stop_idx,
        
        'sigma':     sigma,
    }

    if save_data:
        with open(
            data_save_path,
            "w"
        ) as f:
            json.dump(data_dict, f, indent=2)

    return data_dict