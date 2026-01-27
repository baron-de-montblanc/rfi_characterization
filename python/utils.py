# Utility functions
import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import corner
from scipy.stats import norm, cauchy, mode, t
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def emission_corner_plot(
        fit,
        save_path=None,
    ):
    params = np.column_stack([
        fit.stan_variable("rate_rising"),
        fit.stan_variable("rate_decay"),
        fit.stan_variable("mu_blip"),
#         fit.stan_variable("sigma"),
        fit.stan_variable("tau_blip"),
    ])

    labels = [
        r"$\mathrm{rate}_{rising}$",
        r"$\mathrm{rate}_{decay}$",
        r"$\mu_{blip}$",
#         r"$\sigma$",
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
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def legendre_corner_plot(
        fit,
        save_path=None,
        supervision_level="unsup",
        night=0,
    ):
    params = fit.stan_variable(f"X_{supervision_level}")[:,night,:]

    # Make corner plot
    fig = corner.corner(
        params,
        # labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 10},
        bins=40
    )

    fig.suptitle(f"Legendre basis coefficients for night {night}", fontsize=16)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

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


def get_residuals(
        data_dict,
        fit,
        sup_level='unsup',
    ):

    X = fit.stan_variables()[f"X_{sup_level}"]
    A = np.asarray(data_dict[f"A_{sup_level}"])
    y = np.asarray(data_dict[f"y_{sup_level}"])

    start = np.asarray(data_dict[f"start_idx_{sup_level}"]) - 1  # 0-based indexing
    stop  = np.asarray(data_dict[f"stop_idx_{sup_level}"])       # Python slice is exclusive

    draws, M_sup, L = X.shape
    N_sup = len(y)

    mu_draws = np.zeros((draws, N_sup))

    for m in range(M_sup):
        a, b = start[m], stop[m]
        mu_draws[:, a:b] = X[:, m, :] @ A[a:b, :].T

    bg = mu_draws.mean(axis=0)  # (N_sup,)
    res = y - bg

    return res


def plot_prediction_hist(
        data_dict,
        fit,
        save_path=None,
        sup_level="unsup",
        sigma=0.3,
    ):
    rate_rising = float(np.mean(fit.stan_variable("rate_rising")))
    rate_decay  = float(np.mean(fit.stan_variable("rate_decay")))
    mu_blip     = float(np.mean(fit.stan_variable("mu_blip")))
    tau_blip    = float(np.mean(fit.stan_variable("tau_blip")))

    # ---- data & states ----
    res = get_residuals(data_dict, fit, sup_level)

    if sup_level == "unsup":
        viterbi = fit.stan_variable('viterbi')
        pred = mode(viterbi).mode
    else:
        pred = np.asarray(data_dict['s_sup'])

    clean_mask = pred == 1
    rising_mask = pred == 2
    decay_mask  = pred == 3
    blip_mask   = pred == 4

    # shift y by one to get y_{t-1}
    res_tm1 = np.roll(res, 1)      # rolls right, so y_tm1[t] = y[t-1]
    res_tm1[0] = np.nan          # first element has no predecessor

    # residuals only where states are rising/decay
    r_resid = res[rising_mask] - rate_rising * res_tm1[rising_mask]
    d_resid = res[decay_mask]  - rate_decay  * res_tm1[decay_mask]
    r_resid = r_resid[~np.isnan(r_resid)]
    d_resid = d_resid[~np.isnan(d_resid)]

    clean_res = res[clean_mask]
    blip_res  = res[blip_mask]

    fig, axs = plt.subplots(2, 2, figsize=(10, 6))
    axs = axs.ravel()

    # Clean: y_t ~ N(0, sigma)
    overlay(
        axs[0], clean_res,
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
        axs[3], blip_res,
        lambda x: t.pdf(x, df=3, loc=mu_blip, scale=tau_blip),
        "Blip",
        fr"t($\nu$ = 3, $\mu$ = {mu_blip:.2f}, $\tau$ = {tau_blip:.2f})"
    )

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

