import numpy as np
from numpy.polynomial.legendre import legvander
import json
import glob
import time
from cmdstanpy import CmdStanModel
import argparse
import os


# ------------------------ Global Variables ------------------------


ABS_DIR = '/users/jmduchar/data/jmduchar/Research/mcgill25/rfi_characterization/'
STAN_FILE = ABS_DIR+'stan/legendre.stan'

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


def get_priors(L):

    prior_dict = dict(
        # --- transition priors ---
        alpha_clean       = [10.0, 1.0, 1.0],        # clean -> {clean, rising, blip}
        alpha_rising      = [10.0, 5.0, 1.0],        # rising -> {rising, decay, blip}
        alpha_decay       = [5.0, 1.0, 10.0, 1.0],   # decay  -> {clean, rising, decay, blip}
        alpha_blip        = [10.0, 1.0, 1.0, 1.0],   # blip   -> {clean, rising, decay, blip}

        # --- dynamic parameters ---
        rr_log_mu         = 0.0,        # lognormal mean for rate_rising (exp(0)=1)
        rr_log_sigma      = 1.0,        # wide
        rd_alpha          = 2.0,        # beta(2,2) near-uniform
        rd_beta           = 2.0,

        # --- noise variance ---
        sig_log_mu        = 0.0,        # mean of log(sigma)
        sig_log_sigma     = 2.0,        # covers sigma ~ [0.05, 50]

        # --- blip emission ---
        mu_blip_mean      = 0.0,        # centered
        mu_blip_sd        = 10.0,       # very wide
        k_blip_log_mu     = 0.0,        # lognormal mean for k_blip
        k_blip_log_sigma  = 2.0,        # broad spread

        # --- legendre hyperparameters (lenient, scale-invariant) ---
        mu_X_mean         = [0.0] * L,              # zero-centered
        mu_X_sd           = [5.0] * L,              # wide (allows large coeffs)
        alpha_X_log_mu    = [0.0] * L,              # lognormal mean
        alpha_X_log_sigma = [1.0] * L,              # wide dispersion
        beta_X_log_mu    = float(np.log(2.0)),   # median(beta_X) = 2
        beta_X_log_sigma = 1.0
    )

    return prior_dict


def create_data_dict(
        pointing,
        L,
        save_data           = False, 
        save_data_path      = None, 
        median_subtract     = False,
        data_fraction       = 1,
        annotation_fraction = 1,
        grainsize           = 1,  # for parallelization
    ):

    all_p = sorted([i for i in ALL_FILES if pointing in i and "bad" not in i])
    all_annotations = sorted([i for i in ALL_ANNOTATIONS if pointing in i and "bad" not in i])

    total_num_files = len(all_p)
    cur_data_frac = 1
    while cur_data_frac > data_fraction:
        all_p.pop()
        all_annotations.pop()
        cur_data_frac = len(all_p)/total_num_files

    cur_fraction = len(all_annotations)/len(all_p)
    while cur_fraction > annotation_fraction:
        all_annotations.pop()
        cur_fraction = len(all_annotations)/len(all_p)

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

    # Build data dictionary
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

        'grainsize':       grainsize,
    }

    # Add priors
    prior_dict = get_priors(L)
    data_dict.update(prior_dict)

    if save_data:
        with open(
            save_data_path,
            "w"
        ) as f:
            json.dump(data_dict, f, indent=2)

    return data_dict


def parse_args():
    cpu = os.cpu_count() or 4
    default_threads = max(1, cpu // 4)

    print("Default threads:", default_threads)

    p = argparse.ArgumentParser(
        description="Run Stan HMM with Legendre background, with convenient defaults.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("pointing", help="Pointing ID, e.g. p0")
    p.add_argument("L", type=int, help="Legendre order (L)")
    p.add_argument("--threads-per-chain", type=int, default=default_threads,
                   help="Threads per MCMC chain")
    p.add_argument("--chains", type=int, default=4, help="Number of chains")
    p.add_argument("--parallel-chains", type=int, default=4,
                   help="Number of chains to run in parallel")
    p.add_argument("--adapt-delta", type=float, default=0.95,
                   help="Target acceptance rate")
    p.add_argument("--max-treedepth", type=int, default=15,
                   help="Max treedepth for NUTS")
    p.add_argument("--annotation-fraction", type=float, default=0.0,
                   help="Fraction of annotations for supervised modes")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    t0 = time.time()

    pointing = str(args.pointing)
    L = int(args.L)
    ann_frac = float(args.annotation_fraction)
    sup = {0: "unsupervised", 1: "supervised"}.get(ann_frac, "semisupervised")
        
    DATA_DICT_PATH  = ABS_DIR+f"data/json/legendre_{sup}_{pointing}_L{L}.json"
    data_dict = create_data_dict(
        pointing,
        L,
        save_data           = True,
        save_data_path      = DATA_DICT_PATH,
        median_subtract     = False,
        data_fraction       = 1,
        annotation_fraction = ann_frac,
    )

    model = CmdStanModel(stan_file=STAN_FILE, cpp_options={"STAN_THREADS": "true"})

    fit = model.sample(
        data              = data_dict,
        chains            = args.chains,
        parallel_chains   = args.parallel_chains,
        threads_per_chain = args.threads_per_chain,
        adapt_delta       = args.adapt_delta,
        max_treedepth     = args.max_treedepth,
        show_console      = True,
        output_dir        = ABS_DIR+f"stan/stan_out/legendre_{sup}_{pointing}_L{L}",
    )

    print(f"\nTime elapsed: {time.time() - t0:.1f}s")
