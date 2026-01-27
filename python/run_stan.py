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



# ------------------------ Global Variables (simulation) ------------------------

DATAPATH_SIM = ABS_DIR+"data_andrei_sim/mock_data_wtemps/"
ALL_FILES_SIM = glob.glob(DATAPATH_SIM+"mock_data*.npy")
BEAM_TEMP_FILE = DATAPATH_SIM+"mock_temps.npy"

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


def build_legendre_design_matrix(y, L):
    finite_mask = np.isfinite(y)
    u = np.arange(len(y), dtype=np.float64)
    u = u[finite_mask]  # make time array gappy
    A = legendre_design_numpy(u, L)
    return A


def obs_pointing_key(path):
    fn = path.split("/")[-1]
    parts = fn.split("_")
    # parts[-2] = obs, parts[-1] = p.ext
    obs = parts[-2]
    p = parts[-1].split(".")[0]
    return (obs, p)

def dirichlet_mean(alpha):
    a = np.asarray(alpha, dtype=float)
    return (a / a.sum()).tolist()

def ridge_ls_init(A, y, start_idx, stop_idx, L, lam=1e-6):
    """
    Per-night ridge least squares init for X:
      X_m = argmin ||y_m - A_m X||^2 + lam ||X||^2
    """
    A = np.asarray(A, dtype=float)
    y = np.asarray(y, dtype=float)

    start = np.asarray(start_idx, dtype=int) - 1  # Stan->0 indexing
    stop  = np.asarray(stop_idx, dtype=int) - 1

    M = len(start)
    X = np.zeros((M, L), dtype=float)
    identity = np.eye(L)

    for m, (a, b) in enumerate(zip(start, stop)):
        A_m = A[a:b+1, :]
        y_m = y[a:b+1]
        XtX = A_m.T @ A_m + lam * identity
        X[m, :] = np.linalg.solve(XtX, A_m.T @ y_m)

    return X

# Keep inits strictly inside constraint support (Stan transforms are open intervals)
EPS = 1e-6

def lower_open(x, lo, eps=EPS):
    """Enforce x > lo (strict)."""
    if not np.isfinite(x):
        return float(lo + 10 * eps)
    return float(max(x, lo + eps))

def upper_open(x, hi, eps=EPS):
    """Enforce x < hi (strict)."""
    if not np.isfinite(x):
        return float(hi - 10 * eps)
    return float(min(x, hi - eps))

def clip_open(x, lo, hi, eps=EPS):
    """Enforce lo < x < hi (strict)."""
    if not np.isfinite(x):
        return float((lo + hi) / 2.0)
    return float(np.clip(x, lo + eps, hi - eps))

def safe_simplex(p, eps=1e-12):
    """
    Make a vector strictly positive and sum to 1.
    Works for simplex inits (pi, theta_*).
    """
    p = np.asarray(p, dtype=float)
    p = np.where(np.isfinite(p), p, 0.0)
    p = np.maximum(p, eps)
    s = p.sum()
    if not np.isfinite(s) or s <= 0:
        # fallback to uniform
        p = np.ones_like(p) / p.size
    else:
        p = p / s
    return p.tolist()

def make_chain_init_guarded(data_dict, chain_seed=42, jitter_X=0.05):
    rng = np.random.default_rng(chain_seed)
    L = int(data_dict["L"])

    init = {}

    # ---------- Regression / background priors ----------
    intercept0 = float(data_dict["intercept_mean"])
    slope0     = float(data_dict["slope_mean"])
    scale0     = float(data_dict["scale_mean"])
    shape0     = float(data_dict["shape_mean"])

    # Stan: intercept>0, slope<0, scale>0, shape>0
    init["intercept"] = lower_open(intercept0 + rng.normal(scale=10.0), 0.0)
    init["slope"]     = upper_open(slope0 + rng.normal(scale=1.0), 0.0)
    init["scale"]     = lower_open(scale0 + rng.normal(scale=0.5), 0.0)
    init["shape"]     = lower_open(shape0 + rng.normal(scale=0.2), 0.0)

    # ---------- Per-night Legendre coeffs ----------
    # X_unsup is unconstrained, but we still guard against explosions / NaNs.
    if int(data_dict["M_unsup"]) > 0:
        X_hat = ridge_ls_init(
            data_dict["A_unsup"], data_dict["y_unsup"],
            data_dict["start_idx_unsup"], data_dict["stop_idx_unsup"],
            L=L, lam=1e-6
        )

        # Fallback if LS is non-finite or absurdly large
        if (not np.isfinite(X_hat).all()) or (np.max(np.abs(X_hat)) > 1e12):
            yavg = np.asarray(data_dict["y_tot_nightly_avg"], dtype=float)
            M = int(data_dict["M_unsup"])
            X_hat = np.zeros((M, L), dtype=float)
            if yavg.size == M and np.isfinite(yavg).all():
                X_hat[:, 0] = yavg
            else:
                X_hat[:, 0] = float(np.nanmean(np.asarray(data_dict["y_unsup"], float)))

        X = X_hat + rng.normal(scale=jitter_X, size=X_hat.shape)

        # Optional: clip to prevent A*X overflow at init
        y = np.asarray(data_dict["y_unsup"], dtype=float)
        y_scale = float(np.nanstd(y)) if y.size > 1 else 1.0
        X_clip = max(1e3, 1e4 * y_scale)
        X = np.clip(X, -X_clip, X_clip)

        init["X_unsup"] = X.tolist()

    if int(data_dict["M_sup"]) > 0:
        X_hat = ridge_ls_init(
            data_dict["A_sup"], data_dict["y_sup"],
            data_dict["start_idx_sup"], data_dict["stop_idx_sup"],
            L=L, lam=1e-6
        )
        X = X_hat + rng.normal(scale=jitter_X, size=X_hat.shape)
        init["X_sup"] = X.tolist()

    # ---------- Legendre hyperparameters ----------
    # mu_X unconstrained; alpha_X > 0; beta_X > 0
    if "X_unsup" in init and np.asarray(init["X_unsup"]).size > 0:
        mu_X0 = np.mean(np.asarray(init["X_unsup"], dtype=float), axis=0)
        mu_X0 = np.where(np.isfinite(mu_X0), mu_X0, 0.0)
    else:
        mu_X0 = np.zeros(L)

    init["mu_X"] = (mu_X0 + rng.normal(scale=0.05, size=L)).tolist()

    alpha_X0 = np.exp(np.asarray(data_dict["alpha_X_log_mu"], dtype=float))  # median > 0
    alpha_X  = alpha_X0 * np.exp(rng.normal(scale=0.1, size=L))
    alpha_X  = np.maximum(alpha_X, 1e-6)  # strictly > 0
    init["alpha_X"] = alpha_X.tolist()

    beta_X0 = float(np.exp(data_dict["beta_X_log_mu"]))  # median > 0
    init["beta_X"] = lower_open(beta_X0 * np.exp(rng.normal(scale=0.1)), 0.0)

    # ---------- HMM dynamics ----------
    # Stan: rate_rising > 1.02, 0 < rate_decay < 0.98
    rr0 = float(np.exp(data_dict["rr_log_mu"]))  # median of lognormal
    init["rate_rising"] = lower_open(rr0 * np.exp(rng.normal(scale=0.1)), 1.02)

    rd_mean = float(data_dict["rd_alpha"] / (data_dict["rd_alpha"] + data_dict["rd_beta"]))
    init["rate_decay"] = clip_open(rd_mean + rng.normal(scale=0.05), 0.0, 0.98)

    # ---------- Blip ----------
    # Stan: mu_blip > 0, k_blip > 1
    mu_blip0 = float(data_dict["mu_blip_mean"])
    mu_blip_sd = float(data_dict["mu_blip_sd"])
    init["mu_blip"] = lower_open(mu_blip0 + rng.normal(scale=0.1 * mu_blip_sd), 0.0)

    k0 = float(np.exp(data_dict["k_blip_log_mu"]))
    init["k_blip"] = lower_open(k0 * np.exp(rng.normal(scale=0.2)), 1.0)

    # ---------- Transition matrices (simplexes) ----------
    init["theta_clean"]  = dirichlet_mean(data_dict["alpha_clean"])
    init["theta_rising"] = dirichlet_mean(data_dict["alpha_rising"])
    init["theta_decay"]  = dirichlet_mean(data_dict["alpha_decay"])
    init["theta_blip"]   = dirichlet_mean(data_dict["alpha_blip"])

    # ---------- Initial state (simplex[4]) ----------
    init["pi"] = safe_simplex([0.99, 0.003, 0.003, 0.004])

    return init


def get_priors(L, median_subtract=False):

    prior_dict = dict(
        # --- transition priors ---
        alpha_clean       = [300.0, 1.0, 0.2],        # clean -> {clean, rising, blip}
        alpha_rising      = [1.0, 80.0, 0.2],        # rising -> {rising, decay, blip}
        alpha_decay       = [80.0, 0.5, 1.0, 0.2],   # decay  -> {clean, rising, decay, blip}
        alpha_blip        = [50.0, 1.0, 1.0, 1.0],   # blip   -> {clean, rising, decay, blip}

        # ---- initial state distribution prior ----
        alpha_pi          = [1000,1,1,1],  # strongly prefer to start night in clean state

        # --- dynamic parameters ---
        rr_log_mu         = np.log(1.1),        # lognormal mean for rate_rising (exp(0)=1)
        rr_log_sigma      = 0.2,        # wide
        rd_alpha          = 5.0,        # beta(2,2) near-uniform
        rd_beta           = 2.0,

        # --- blip emission ---
        mu_blip_mean      = 100.0,      # outlier regime
        mu_blip_sd        = 10.0,       # very wide
        k_blip_log_mu     = 0.0,        # lognormal mean for k_blip
        k_blip_log_sigma  = 2.0,        # broad spread

        # --- legendre hyperparameters (lenient, scale-invariant) ---
        mu_X_mean         = [0.0] * L,              # zero-centered
        mu_X_sd           = [5.0] * L,              # wide (allows large coeffs)
        alpha_X_log_mu    = [0.0] * L,              # lognormal mean
        alpha_X_log_sigma = [1.0] * L,              # wide dispersion
        beta_X_log_mu    = float(np.log(2.0)),   # median(beta_X) = 2
        beta_X_log_sigma = 1.0,
    )
    
    if not median_subtract:
        # The zeroth (constant offset) term is way higher than 0! --> encode that
        prior_dict["mu_X_mean"][0] = 750.0
        prior_dict["mu_X_sd"][0]   = 100.0

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
    ann_night_pointing = [obs_pointing_key(i) for i in all_annotations]

    y_unsup_list, y_sup_list = [], []
    A_unsup_rows, A_sup_rows = [], []
    start_stop_unsup, start_stop_sup = [], []
    s_sup_list = []

    c_unsup = 0
    c_sup = 0

    for pdx, (obs,p) in enumerate(all_night_pointing):
        
        sample = np.load(all_p[pdx])
        A = build_legendre_design_matrix(sample, L) 

        # Remove NAN
        nan_mask = np.isnan(sample)
        sample = sample[~nan_mask]

        if median_subtract:
            sample = sample - np.median(sample)
        
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
                
            # Remove NAN locations
            labels = labels[~nan_mask]
            
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

        'sigma':           0.3,
    }

    # Add priors
    prior_dict = get_priors(L, median_subtract=median_subtract)
    data_dict.update(prior_dict)

    if save_data:
        with open(
            save_data_path,
            "w"
        ) as f:
            json.dump(data_dict, f, indent=2)

    return data_dict



def create_data_dict_SIM(
        L,
        save_data           = False, 
        save_data_path      = None, 
        grainsize           = 1,  # for parallelization
    ):

    y_unsup_list = []
    A_unsup_rows = []
    start_stop_unsup = []
    c_unsup = 0
    y_tot_nightly_avg = []

    for pdx, mock_sample in enumerate(ALL_FILES_SIM):
        
        sample = np.load(mock_sample)
        A = build_legendre_design_matrix(sample, L) 

        # Remove NAN
        nan_mask = np.isnan(sample)
        sample = sample[~nan_mask]
        
        y_tot_nightly_avg.append(np.mean(sample))
        y_unsup_list.append(sample.astype(float))
        A_unsup_rows.append(A.astype(float))
        
        a = c_unsup
        b = c_unsup + len(sample) - 1
        start_stop_unsup.append((a, b))
        c_unsup += len(sample)
            
    # Concatenate
    y_unsup = np.concatenate(y_unsup_list, axis=0)
    A_unsup = np.vstack(A_unsup_rows)

    y_sup = np.empty((0,), dtype=float)
    A_sup = np.empty((0, L), dtype=float)
    s_sup = np.empty((0,), dtype=float)

    # Stan indexes from 1
    start_idx_unsup = [int(a+1) for (a,b) in start_stop_unsup]
    stop_idx_unsup  = [int(b+1) for (a,b) in start_stop_unsup]
    start_idx_sup   = []
    stop_idx_sup    = []

    data_dict = {
        'L':                 int(L),

        'M_tot':             int(len(ALL_FILES_SIM)),
        'y_tot_nightly_avg': y_tot_nightly_avg,
        'nightly_temp':      np.load(BEAM_TEMP_FILE).tolist(),
        'night_id_unsup':    np.arange(1, len(ALL_FILES_SIM)+1).tolist(),  # NOTE: only works because we don't have any sup sims
        'night_id_sup':      [],

        'slope_mean':        -12,
        'intercept_mean':    1000,
        'scale_mean':        10,
        'shape_mean':        1.5,
        
        'N_unsup':           int(len(y_unsup)),
        'y_unsup':           y_unsup.tolist(),
        'A_unsup':           A_unsup.tolist(),
        'M_unsup':           int(len(start_idx_unsup)),
        'start_idx_unsup':   start_idx_unsup,
        'stop_idx_unsup':    stop_idx_unsup,
        
        'N_sup':             int(len(y_sup)),
        'y_sup':             y_sup.tolist(),
        'A_sup':             A_sup.tolist(),
        's_sup':             s_sup.tolist(),
        
        'M_sup':             int(len(start_idx_sup)),
        'start_idx_sup':     start_idx_sup,
        'stop_idx_sup':      stop_idx_sup,

        'grainsize':         grainsize,

        'sigma':             0.3,
    }

    # Add rest of priors
    prior_dict = get_priors(L, median_subtract=False)
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


if __name__ == "NOTmain__":  # real data
    args = parse_args()
    t0 = time.time()

    pointing = str(args.pointing)
    L = int(args.L)
    ann_frac = float(args.annotation_fraction)
    sup = {0: "unsupervised", 1: "supervised"}.get(ann_frac, "semisupervised")
    
    print("Default threads:", args.threads_per_chain)
        
    DATA_DICT_PATH  = ABS_DIR+f"data/json/legendre_{sup}_{pointing}_L{L}.json"
    data_dict = create_data_dict(
        pointing,
        L,
        save_data           = True,
        save_data_path      = DATA_DICT_PATH,
        median_subtract     = True,
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


if __name__ == "__main__":  # for simulations
    args = parse_args()
    t0 = time.time()

    L = int(args.L)
    print("Default threads:", args.threads_per_chain)
        
    DATA_DICT_PATH  = ABS_DIR+"data/json/legendre_simulation.json"
    data_dict = create_data_dict_SIM(
        L,
        save_data           = False, 
        save_data_path      = None, 
        grainsize           = 1,  # for parallelization
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
