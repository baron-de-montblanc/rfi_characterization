# Utility functions
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import corner
from scipy.stats import norm, mode, t
import arviz as az
plt.style.use('seaborn-v0_8')

def get_night(
        data,
        target_obsid,
    ):
    """
    Given a dictionary mapping obsids to pointings, and the starting obsid,
    return all adjacent obsids for that night and pointing.

    Parameters:
        data (dict): Dictionary of obsid strings mapped to integer values.
        target_obsid (str): The obsid to search around.

    Returns:
        list: A list of adjacent obsids (including the target) from the same night
    """
    sorted_obsids = sorted(data.keys(), key=int)
    target_index = sorted_obsids.index(target_obsid)
    target_value = data[target_obsid]
    result = [target_obsid]

    # Go left
    i = target_index - 1
    while i >= 0 and data[sorted_obsids[i]] == target_value:
        result.insert(0, sorted_obsids[i])
        i -= 1

    # Go right
    i = target_index + 1
    while i < len(sorted_obsids) and data[sorted_obsids[i]] == target_value:
        result.append(sorted_obsids[i])
        i += 1

    return target_value, result  # return the pointing and all associated obsids for that night


def get_pointing(
        data,
        pointing,
    ):
    """
    Given a dictionary mapping obsids to pointings, and the desired pointing,
    return *all* obsids for that pointing.

    Parameters:
        data (dict): Dictionary of obsid strings mapped to integer values.
        pointing (int): The desired pointing

    Returns:
        list: Complete list of obsids from that poiting
    """
    result = []
    for i in data:
        if data[i] == pointing:
            result.append(i)

    return result  # return all associated obsids for that pointing


def get_ref_obsids(
        data,
    ):
    """
    Given a data dictionary that maps OBSIDs to pointings,
    get the list of all 'reference' OBSIDs; i.e. the first
    OBSID per pointing.
    """
    # Sort the OBSIDs numerically (they should already be sorted, but just in case)
    sorted_items = sorted(data.items(), key=lambda x: int(x[0]))

    ref_obsids = []
    prev_pointing = None

    for obsid, pointing in sorted_items:
        if pointing != prev_pointing:
            ref_obsids.append(obsid)
            prev_pointing = pointing

    return ref_obsids


def plot_supervised_inputs(
        data_dict, 
        pointing,
        save_path=None,
    ):
    
    fig, axes = plt.subplots(2, 1, figsize=(8, 4),
                            gridspec_kw={'height_ratios': [2, 1]},
                            )

    axes[0].scatter(range(len(data_dict['y_sup'])),data_dict['y_sup'], s=4) 
    axes[0].set_title("Input Supervised Data & Annotations")
    axes[0].set_ylabel("SSINS amplitude")
    # axes[0].set_yscale("symlog")

    axes[1].plot(data_dict['s_sup'], color=sns.color_palette()[2])
    axes[1].set_title("1-Clean  2-Rising  3-Decaying  4-Blip")
    axes[1].set_yticks([1,2,3,4])

    if save_path is not None:
        plt.savefig(os.path.join(save_path, f"supervised_inputs_{pointing}.png"), dpi=300, bbox_inches='tight')

    plt.tight_layout()
    plt.show()


def plot_data_vs_pred(
        data_dict, 
        fit, 
        pointing, 
        cut_idx=None,
        save_path=None,
    ):
    """
    Plot the observed data and the most probable hidden states.

    Args:
        data_dict
        fit (CmdStanFit): The fitted model object.
        pointing (str): The pointing string; eg 'p0'
    """
    data = np.asarray(data_dict['y_unsup'])
    viterbi = fit.stan_variable('viterbi')
    predictions = mode(viterbi).mode

    if cut_idx is not None:

        start_stop_idx = data_dict['start_stop_idx']
        obs_pointing = data_dict['obs_pointing']
        
        cut_lower, cut_upper = start_stop_idx[cut_idx]
        obs, point = obs_pointing[cut_idx]
        
        data = data[cut_lower:cut_upper]
        predictions = predictions[cut_lower:cut_upper]
        
        title = f"night of {obs}, {point}"
        savetitle = f"{obs}_{point}"

    else:
        title = pointing
        savetitle = pointing

    fig, axs = plt.subplots(nrows=2, ncols=1, sharex=True, figsize=(8, 4),
                           gridspec_kw={'height_ratios': [2, 1]},
                           )

    axs[0].scatter(range(len(data)), data, s=4)
    axs[0].set_title(f"SSINS amplitude (median- and background-subtracted) for {title}")
    # axs[0].set_yscale("symlog")
    axs[0].set_ylabel("SSINS amplitude")

    axs[1].plot(predictions, color=sns.color_palette()[2])
    axs[1].set_title("Most probable hidden states (mode across all chains and iterations)")
    axs[1].set_yticks([1,2,3,4])
    axs[1].set_ylim(0.75,4.25)
    # axs[1].set_ylabel("1-Clean  2-Rising  3-Decaying  4-Blip")

    plt.xlabel("Time step")
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, f"data_vs_pred_{savetitle}.png"), dpi=300, bbox_inches='tight')
        
    plt.show()


def transition_corner_plot(
        fit,
        save_path=None,
    ):

    # Collect all theta draws into one big array
    thetas = np.hstack([
        fit.stan_variable("theta_clean"),   # shape (4000, 3)
        fit.stan_variable("theta_rising"),  # shape (4000, 3)
        fit.stan_variable("theta_decay"),   # shape (4000, 4)
        fit.stan_variable("theta_blip"),    # shape (4000, 4)
    ])

    labels = [
        r"clean $\to$ clean", r"clean $\to$ rising", r"clean $\to$ blip",
        r"rising $\to$ rising", r"rising $\to$ decay", r"rising $\to$ blip",
        r"decay $\to$ clean", r"decay $\to$ rising", r"decay $\to$ decay", r"decay $\to$ blip",
        r"blip $\to$ clean", r"blip $\to$ rising", r"blip $\to$ decay", r"blip $\to$ blip"
    ]


    # Trim labels to the right length:
    labels = labels[:thetas.shape[1]]

    # Make corner plot
    fig = corner.corner(
        thetas,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 10},
        bins=40
    )

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "transition_corner.png"), dpi=300, bbox_inches='tight')

    plt.show()


def emission_corner_plot(
        fit,
        save_path=None,
    ):
    params = np.column_stack([
        fit.stan_variable("rate_rising"),
        fit.stan_variable("rate_decay"),
        fit.stan_variable("mu_blip"),
    #     fit.stan_variable("sigma"),
        fit.stan_variable("tau_blip"),
    ])

    labels = [
        r"$\mathrm{rate}_{rising}$",
        r"$\mathrm{rate}_{decay}$",
        r"$\mu_{blip}$",
    #     r"$\sigma$",
        r"$\tau_{blip}$"
    ]

    # Make corner plot
    fig = corner.corner(
        params,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 10},
        bins=40
    )

    if save_path is not None:
        plt.savefig(os.path.join(save_path, "emission_corner.png"), dpi=300, bbox_inches='tight')

    plt.show()


def robust_range(arr, lo=0.5, hi=99.5, pad=0.05):
    if arr.size == 0:
        return (-1, 1)
    a, b = np.percentile(arr, [lo, hi])
    m = (b - a)
    return (a - pad*m, b + pad*m)


def overlay(ax, samples, pdf, title, label, bins='fd'):
    ax.hist(samples, bins=100, density=True, alpha=0.75)
    xmin, xmax = robust_range(samples)
    x = np.linspace(xmin, xmax, 600)
    if title!="Blip":
        empirical_std = np.std(samples, ddof=1)
        ax.plot(x, norm.pdf(x, loc=0.0, scale=empirical_std), 
                lw=2, label=f"Empirical N(0, $\sigma$ = {empirical_std:.2f})",
                linestyle='--', color=sns.color_palette()[1],
               )
    ax.plot(x, pdf(x), lw=2, label=label, color=sns.color_palette()[2])
    ax.set_title(title)
    ax.set_xlabel("SSINS Amplitude" if "residual" not in title.lower() else "Residual")
    ax.set_ylabel("Density")
    ax.set_yscale("log")
    ax.legend(loc="upper right", frameon=False)


def plot_prediction_hist(
        data_dict,
        fit,
        save_path=None,
    ):

    data = np.asarray(data_dict['y_unsup'])
    viterbi = fit.stan_variable('viterbi')
    predictions = mode(viterbi).mode

    clean_mask = predictions == 1
    rising_mask = predictions == 2
    decay_mask = predictions == 3
    blip_mask = predictions == 4

    sigma = 0.33
    rate_rising = float(np.mean(fit.stan_variable("rate_rising")))
    rate_decay  = float(np.mean(fit.stan_variable("rate_decay")))
    mu_blip     = float(np.mean(fit.stan_variable("mu_blip")))
    tau_blip    = float(np.mean(fit.stan_variable("tau_blip")))

    # ---- data & states ----
    y = np.asarray(data_dict['y_unsup'])
    viterbi = fit.stan_variable('viterbi')
    pred = mode(viterbi).mode

    clean_mask = pred == 1
    rising_mask = pred == 2
    decay_mask  = pred == 3
    blip_mask   = pred == 4

    # shift y by one to get y_{t-1}
    y_tm1 = np.roll(y, 1)      # rolls right, so y_tm1[t] = y[t-1]
    y_tm1[0] = np.nan          # first element has no predecessor

    # residuals only where states are rising/decay
    r_resid = y[rising_mask] - rate_rising * y_tm1[rising_mask]
    d_resid = y[decay_mask]  - rate_decay  * y_tm1[decay_mask]
    r_resid = r_resid[~np.isnan(r_resid)]
    d_resid = d_resid[~np.isnan(d_resid)]

    clean_y = y[clean_mask]
    blip_y  = y[blip_mask]

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs = axs.ravel()

    # Clean: y_t ~ N(0, sigma)
    overlay(
        axs[0], clean_y,
        lambda x: norm.pdf(x, loc=0.0, scale=sigma),
        "Clean",
        fr"Model N(0, $\sigma$ = {sigma:.2f})"
    )

    # Rising: residuals y_t - rate_rising*y_{t-1} ~ N(0, sigma)
    overlay(
        axs[1], r_resid,
        lambda x: norm.pdf(x, loc=0.0, scale=sigma),
        "Rising (residuals)",
        fr"Model $\epsilon \sim$ N(0, $\sigma$ = {sigma:.2f})"
    )

    # Decay: residuals y_t - rate_decay*y_{t-1} ~ N(0, sigma)
    overlay(
        axs[2], d_resid,
        lambda x: norm.pdf(x, loc=0.0, scale=sigma),
        "Decay (residuals)",
        fr"Model $\epsilon \sim$ N(0, $\sigma$ = {sigma:.2f})"
    )

    # Blip: y_t ~ Student-t(ν=3, μ, τ)
    overlay(
        axs[3], blip_y,
        lambda x: t.pdf(x, df=3, loc=mu_blip, scale=tau_blip),
        "Blip",
        fr"t($\nu$ = 3, $\mu$ = {mu_blip:.2f}, $\tau$ = {tau_blip:.2f})"
    )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(os.path.join(save_path, "label_hist_overlay.png"), dpi=300, bbox_inches='tight')

    plt.show()


def build_idata(
    data_dict,
    fit
):

    # Convert to arrays (ArviZ prefers numpy arrays)
    y_unsup = np.asarray(data_dict["y_unsup"], dtype=float)          # shape (N_unsup,)
    A_unsup = np.asarray(data_dict["A_unsup"], dtype=float)          # shape (N_unsup, L)
    y_sup   = np.asarray(data_dict["y_sup"], dtype=float)            # shape (N_sup,)
    A_sup   = np.asarray(data_dict["A_sup"], dtype=float)            # shape (N_sup, L)
    s_sup   = np.asarray(data_dict["s_sup"], dtype=int)              # shape (N_sup,)

    L       = int(data_dict["L"])
    N_unsup = int(data_dict["N_unsup"])
    N_sup   = int(data_dict["N_sup"])
    sigma   = float(data_dict["sigma"])

    # If present in JSON:
    mu_X    = np.asarray(data_dict["mu_X"], dtype=float)             # shape (L,)
    alpha_X = np.asarray(data_dict["alpha_X"], dtype=float)          # shape (L,)
    beta_X  = np.asarray(data_dict["beta_X"], dtype=float)           # shape (L,)

    # ---- Nice labels for dimensions ----
    state_names = ["clean", "rising", "decay", "blip"]

    coords = {
        "t_unsup": np.arange(N_unsup),
        "t_sup":   np.arange(N_sup),
        "ell":     np.arange(L),
        "to_clean":  ["clean", "rising", "blip"],   # theta_clean has 3 outcomes
        "to_rising": ["rising", "decay", "blip"],   # theta_rising has 3 outcomes
        "to_decay":  state_names,                   # 4 outcomes
        "to_blip":   state_names,                   # 4 outcomes
        "state":     state_names,
    }

    # Map dims to both posterior vars (from Stan) and the observed/constant data you pass in
    dims = {
        # observed data
        "y_unsup": ["t_unsup"],
        "y_sup":   ["t_sup"],
        "s_sup":   ["t_sup"],
        # constants (if you include them)
        "A_unsup": ["t_unsup", "ell"],
        "A_sup":   ["t_sup", "ell"],
        "mu_X":    ["ell"],
        "alpha_X": ["ell"],
        "beta_X":  ["ell"],
        # common posterior variables from your Stan model
        "X":             ["ell"],
        "viterbi":       ["t_unsup"],
        "theta_clean":   ["to_clean"],
        "theta_rising":  ["to_rising"],
        "theta_decay":   ["to_decay"],
        "theta_blip":    ["to_blip"],
    }

    # ---- Build InferenceData ----
    idata = az.from_cmdstanpy(
        posterior=fit,
        observed_data={
            "y_unsup": y_unsup,
            "y_sup":   y_sup,
            "s_sup":   s_sup,
        },
        constant_data={
            "A_unsup": A_unsup,
            "A_sup":   A_sup,
            "L":       L,
            "N_unsup": N_unsup,
            "N_sup":   N_sup,
            "sigma":   sigma,
            "mu_X":    mu_X,
            "alpha_X": alpha_X,
            "beta_X":  beta_X,
        },
        coords=coords,
        dims=dims,
        # If you saved warmup and want it loaded too:
        # save_warmup=True,
    )

    return idata
