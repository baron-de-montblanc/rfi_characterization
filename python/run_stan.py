import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from cmdstanpy import CmdStanModel
import json
from scipy.stats import norm, cauchy
import glob
import corner
import time

plt.style.use('seaborn-v0_8')


# =====================================================
#
# Run Stan HMM and plot useful outputs
#
# =====================================================


# Global variables
ABS_DIR = '/users/jmduchar/data/jmduchar/Research/mcgill25/rfi_characterization/'
STAN_FILE = ABS_DIR+'stan/three_state.stan'
DATAPATH = ABS_DIR+"data_private/"
ALL_FILES = glob.glob(DATAPATH+"*.npy")


def single_night(idx):
    """
    Process data for a single night and generate plots.
    
    Args:
        idx (int): Index of the file in ALL_FILES to process.
    """
    
    filename = ALL_FILES[idx]
    model = CmdStanModel(stan_file=STAN_FILE)

    obs, pointing = (filename.split("/")[-1].split(".")[0].split("_")[-2], 
                    filename.split("/")[-1].split(".")[0].split("_")[-1])
    print("\nPROCESSING SINGLE NIGHT:",obs,pointing,"\n")
    
    data = np.load(filename)
    data_dict = {
        'N': len(data),
        'y': data.tolist()
    }

    # fit the model
    fit = model.sample(
        data=data_dict, 
        show_console=True,
    #     iter_sampling=1000,
    #     iter_warmup=100,
    #     adapt_delta=0.99,
    #     max_treedepth=15,
        chains=4,
    )

    print("FIT DIAGNOSIS:")
    print(fit.diagnose())
    
    plot_data_vs_pred(data, fit, pointing, obs)
    corner_plot(fit, pointing, obs)
    
    
    
def entire_pointing(pointing='p0'):
    """
    Process data for all files corresponding to a given pointing and generate plots.
    
    Args:
        pointing (str): The pointing string to filter files.
    """
    
    print("\nPROCESSING ENTIRE POINTING:",pointing,"\n")
    
    all_p = [i for i in ALL_FILES if pointing in i]
    model = CmdStanModel(stan_file=STAN_FILE)
    
    data = []
    for d in all_p:
        data.extend(np.load(d))
    data = np.asarray(data)
    data_dict = {
        'N': len(data),
        'y': data.tolist()
    }
    
    fit = model.sample(
        data=data_dict, 
        show_console=True,
    #     iter_sampling=1s000,
    #     iter_warmup=100,
    #     adapt_delta=0.99,
    #     max_treedepth=15,
        chains=4,
    )
    
    print("FIT DIAGNOSIS:")
    print(fit.diagnose())
    
    plot_data_vs_pred(data, fit, pointing)
    corner_plot(fit, pointing)
    
    
    
# =====================================================
#
# Helper functions
#
# =====================================================



def plot_data_vs_pred(data, fit, pointing, obs=None):
    """
    Plot the observed data and the most probable hidden states.

    Args:
        data (array): The observed data.
        fit (CmdStanFit): The fitted model object.
        pointing (str): The pointing string.
        obs (str, optional): The observation string.
    """
    
    viterbi = fit.stan_variable('viterbi')
    predictions = np.mean(viterbi, axis=0)

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(10, 6))

    axs[0].plot(data)
    axs[0].set_title(f"{obs} {pointing}")
    # axs[0].set_yscale("symlog")
    axs[0].set_ylabel("amplitude")

    axs[1].plot(predictions)
    axs[1].set_title("Most probable hidden states (avg across sampling iterations)")
    axs[1].set_yticks([1,2,3])
    axs[1].set_ylabel("1-clean  2-RFI (rising)  3-RFI (decaying)")

    plt.xlabel("Time step")
    plt.tight_layout()
    if obs:
        plt.savefig(ABS_DIR+f"figures/three_state_{obs}_{pointing}.png", dpi=300, bbox_inches='tight')
    else:
        plt.savefig(ABS_DIR+f"figures/three_state_{pointing}.png", dpi=300, bbox_inches='tight')
        
    plt.clf()
    
    
    
def corner_plot(fit, pointing, obs=None):
    
    theta_clean = fit.stan_variable('theta_clean')
    theta_rising = fit.stan_variable('theta_rising')
    theta_decay = fit.stan_variable('theta_decay')
    rate_rising = fit.stan_variable('rate_rising')[:,np.newaxis]
    rate_decay = fit.stan_variable('rate_decay')[:,np.newaxis]
    sigma = fit.stan_variable('sigma')[:,np.newaxis]

    # Stack these samples into a single array for corner plot
    samples = np.hstack([theta_clean, theta_rising, theta_decay, rate_rising, rate_decay, sigma])

    # Create the corner plot
    fig = corner.corner(samples, labels=["theta_clean[0]",
                                         "theta_clean[1]", 
                                         "theta_rising[0]", 
                                         "theta_rising[1]",
                                         "theta_decay[0]",
                                         "theta_decay[1]",
                                         "theta_decay[2]",
                                         "rate_rising",
                                         "rate_decay",
                                         "sigma",
                                        ], show_titles=True)
    if obs:
        plt.suptitle(f"Corner plot for {obs} {pointing}", fontsize=24)
        plt.savefig(ABS_DIR+f"figures/corner_{obs}_{pointing}.png", dpi=400, bbox_inches='tight')
    else:
        plt.suptitle(f"Corner plot for {pointing}", fontsize=24)
        plt.savefig(ABS_DIR+f"figures/corner_{pointing}.png", dpi=400, bbox_inches='tight')
    plt.clf()
    
    

# ====================================================
#
# Main procedure
#
# ====================================================


if __name__ == "__main__":
    
    t0 = time.time()
    
    if len(sys.argv) != 3:
        print("Usage: python run_stan.py <mode> <file_idx_or_pointing>")
        sys.exit(1)
        
    mode = sys.argv[1]
    file_idx_or_pointing = sys.argv[2]
    
    if mode not in ["single", "all"]:
        print("Error: Invalid mode. Choose 'single' or 'all'.")
        sys.exit(1)
    
    if mode == "single":
        single_night(int(file_idx_or_pointing))
    elif mode == "all":
        entire_pointing(file_idx_or_pointing)

    print("Time elapsed:", time.time()-t0, "seconds.")