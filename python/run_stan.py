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

    z_unsup_list, z_sup_list = [], []
    s_sup_list = []
    start_stop_unsup, start_stop_sup = [], []

    c_unsup = 0
    c_sup = 0
    for pdx, (obs,p) in enumerate(all_night_pointing):
        
        if (obs,p) not in ann_night_pointing:  # UNSUP NIGHT
            unsup_sample = z[start_idx[pdx]-1 : stop_idx[pdx]]  # guarding against Stan's 1-based inclusive indexing
            z_unsup_list.append(unsup_sample)

            a = c_unsup
            b = c_unsup + len(unsup_sample) - 1
            start_stop_unsup.append((a, b))
            c_unsup += len(unsup_sample)
            
        else:  # SUP NIGHT
            sup_sample = z[start_idx[pdx]-1 : stop_idx[pdx]]
            z_sup_list.append(sup_sample)
            
            # Find the corresponding annotations
            for annotation in all_annotations:
                if obs in annotation and p in annotation:
                    labels = np.load(annotation).astype(int)
                
            assert len(labels) == len(sup_sample)
            s_sup_list.append(labels.astype(int))
            a = c_sup
            b = c_sup + len(sup_sample) - 1
            start_stop_sup.append((a, b))
            c_sup += len(sup_sample)

    # Concatenate
    if len(z_unsup_list) > 0:
        z_unsup = np.concatenate(z_unsup_list, axis=0)
    else:
        z_unsup = np.empty((0,), dtype=float)
        
    if len(z_sup_list) > 0:
        z_sup = np.concatenate(z_sup_list, axis=0)
        s_sup = np.concatenate(s_sup_list, axis=0)
    else:
        z_sup = np.empty((0,), dtype=float)
        s_sup = np.empty((0,), dtype=float)

    start_idx_unsup = [int(a+1) for (a,b) in start_stop_unsup]
    stop_idx_unsup  = [int(b+1) for (a,b) in start_stop_unsup]
    start_idx_sup   = [int(a+1) for (a,b) in start_stop_sup]
    stop_idx_sup    = [int(b+1) for (a,b) in start_stop_sup]

    # Build dictionary
    data_dict = {
        'N_unsup':         int(len(z_unsup)),
        'z_unsup':         z_unsup.tolist(),
        
        'N_sup':           int(len(z_sup)),
        'z_sup':           z_sup.tolist(),
        's_sup':           s_sup.tolist(),
 
        'sigma':           sigma,

        'start_idx_unsup': start_idx_unsup,
        'stop_idx_unsup':  stop_idx_unsup,
        'start_idx_sup':   start_idx_sup,
        'stop_idx_sup':    stop_idx_sup,
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

    # Initialize priors for Legendre coeffs (first pass)
    mu_mu_X     = np.zeros(L, dtype=float)
    tau_mu_X    = np.full(L, 2.5, dtype=float)
    loc_alpha_X   = np.zeros(L, dtype=float)
    scale_alpha_X = np.ones(L, dtype=float)
    loc_beta_X   = float(np.log(2.0))
    scale_beta_X = 0.35

    # Build dictionary
    data_dict = {
        'L':         int(L),
        'N':         int(len(y)),
        'y':         y.tolist(),
        'A':         A.tolist(),
        'M':         int(len(start_idx)),
        'w':         w.tolist(),

        'start_idx': start_idx,
        'stop_idx':  stop_idx,
        
        'sigma':     sigma,

        'mu_mu_X':        mu_mu_X.tolist(),
        'tau_mu_X':       tau_mu_X.tolist(),
        'loc_alpha_X':    loc_alpha_X.tolist(),
        'scale_alpha_X':  scale_alpha_X.tolist(),
        'loc_beta_X':     loc_beta_X,
        'scale_beta_X':   scale_beta_X,
    }

    if save_data:
        with open(
            data_save_path,
            "w"
        ) as f:
            json.dump(data_dict, f, indent=2)

    return data_dict


def prior_params_from_fit(fit):
    mu_X_draws    = fit.stan_variable("mu_X")           # shape: (draws, L)
    alpha_X_draws = fit.stan_variable("alpha_X")        # shape: (draws, L)
    beta_X_draws  = fit.stan_variable("beta_X")         # shape: (draws,)

    # parametrized using Normal
    mu_mu_X  = mu_X_draws.mean(axis=0)
    tau_mu_X = mu_X_draws.std(axis=0, ddof=1)
    tau_mu_X = np.maximum(tau_mu_X, 1e-3) # floor

    # parametrized using LogNormal
    log_alpha = np.log(np.clip(alpha_X_draws, 1e-12, None))  # clip unusable vals
    loc_alpha_X   = log_alpha.mean(axis=0)
    scale_alpha_X = log_alpha.std(axis=0, ddof=1)
    scale_alpha_X = np.maximum(scale_alpha_X, 1e-3)

    # parametrized using LogNormal
    log_beta = np.log(np.clip(beta_X_draws, 1e-12, None))
    loc_beta_X   = float(log_beta.mean())
    scale_beta_X = float(max(log_beta.std(ddof=1), 1e-3))

    return {
        "mu_mu_X":        mu_mu_X.tolist(),
        "tau_mu_X":       tau_mu_X.tolist(),
        "loc_alpha_X":    loc_alpha_X.tolist(),
        "scale_alpha_X":  scale_alpha_X.tolist(),
        "loc_beta_X":     loc_beta_X,
        "scale_beta_X":   scale_beta_X,
    }