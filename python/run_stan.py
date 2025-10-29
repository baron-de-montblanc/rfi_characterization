import sys
import time
import numpy as np
from numpy.polynomial.legendre import legvander
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
import json
import glob

plt.style.use('seaborn-v0_8')

# ------------------------ Global Variables ------------------------


ABS_DIR = '/users/jmduchar/data/jmduchar/Research/mcgill25/rfi_characterization/'
STAN_FILE = ABS_DIR+'stan/legendre.stan'
DATA_SAVE_PATH = ABS_DIR+"/data/json/legendre_semisupervised"

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


def create_data_dict(pointing, L, sigma=0.33, save_data=False, save_data_path=None, median_subtract=False):

    all_p = [i for i in ALL_FILES if pointing in i and "bad" not in i]
    all_annotations = [i for i in ALL_ANNOTATIONS if pointing in i and "bad" not in i]

    all_night_pointing = [obs_pointing_key(i) for i in all_p]
    ann_night_pointing = set(obs_pointing_key(i) for i in all_annotations)

    y_unsup_list, y_sup_list = [], []
    A_unsup_rows, A_sup_rows = [], []
    start_stop_unsup, start_stop_sup = [], []
    s_sup_list = []

    c_unsup = 0
    c_sup = 0

    for pdx, (obs,p) in enumerate(all_night_pointing):
        
        sample = np.load(all_p[pdx])
    
        # Remove NAN
        nan_mask = np.isnan(sample)
        sample = sample[~nan_mask]

        if median_subtract:
            sample = sample - np.median(sample)
        
        A = build_legendre_design_matrix(sample, L) 
        
        if (obs,p) not in ann_night_pointing:  # UNSUP NIGHT
            y_unsup_list.append(sample.astype(float))
            A_unsup_rows.append(A.astype(float))
            
            a = c_unsup
            b = c_unsup + len(sample) - 1
            start_stop_unsup.append((a, b))
            c_unsup += len(sample)
            
        else:  # SUP NIGHT
            y_sup_list.append(sample.astype(float))
            A_sup_rows.append(A.astype(float))
            
            # Find the corresponding annotations
            for annotation in all_annotations:
                if obs in annotation and p in annotation:
                    labels = np.load(annotation).astype(int)
                
            assert len(labels) == len(sample)
            s_sup_list.append(labels.astype(int))
            a = c_sup
            b = c_sup + len(sample) - 1
            start_stop_sup.append((a, b))
            c_sup += len(sample)

    # Concatenate
    if len(y_unsup_list) > 0:
        y_unsup = np.concatenate(y_unsup_list, axis=0)
        A_unsup = np.vstack(A_unsup_rows)
    else:
        y_unsup = np.empty((0,), dtype=float)
        A_unsup = np.empty((0, L), dtype=float)
        
    if len(y_sup_list) > 0:
        y_sup = np.concatenate(y_sup_list, axis=0)
        A_sup = np.vstack(A_sup_rows)
        s_sup = np.concatenate(s_sup_list, axis=0)
    else:
        y_sup = np.empty((0,), dtype=float)
        A_sup = np.empty((0, L), dtype=float)
        s_sup = np.empty((0,), dtype=float)

    # Stan indexes from 1
    start_idx_unsup = [int(a+1) for (a,b) in start_stop_unsup]
    stop_idx_unsup  = [int(b+1) for (a,b) in start_stop_unsup]
    start_idx_sup   = [int(a+1) for (a,b) in start_stop_sup]
    stop_idx_sup    = [int(b+1) for (a,b) in start_stop_sup]

    # Build dictionary
    data_dict = {
        'L':               int(L),
        
        'N_unsup':         int(len(y_unsup)),
        'y_unsup':         y_unsup.tolist(),
        'A_unsup':         A_unsup.tolist(),
        'M_unsup':         int(len(start_idx_unsup)),
        'start_idx_unsup': start_idx_unsup,
        'stop_idx_unsup':  stop_idx_unsup,
        
        'N_sup':           int(len(y_sup)),
        'y_sup':           y_sup.tolist(),
        'A_sup':           A_sup.tolist(),
        's_sup':           s_sup.tolist(),
        
        'M_sup':           int(len(start_idx_sup)),
        'start_idx_sup':   start_idx_sup,
        'stop_idx_sup':    stop_idx_sup,
        
        'sigma':           sigma,
    }

    if save_data:
        with open(
            save_data_path,
            "w"
        ) as f:
            json.dump(data_dict, f, indent=2)

    return data_dict


if __name__ == "__main__":
    
    t0 = time.time()

    if len(sys.argv) != 2:
        print("Usage: python run_stan.py <pointing>")
        sys.exit(1)

    pointing = str(sys.argv[1])

    data_dict = create_data_dict(pointing=pointing, L=8, sigma=0.33, save_data=True)
    model = CmdStanModel(stan_file=STAN_FILE)

    # fit the model
    fit = model.sample(
        data=data_dict,
        chains=4, parallel_chains=4,
        adapt_delta=0.995,
        max_treedepth=15,
        show_console=True,
        output_dir="/users/jmduchar/data/jmduchar/Research/mcgill25/rfi_characterization/stan/stan_out/"
    )
    
    print("Time elapsed:", time_elapsed(t0, time.time()))