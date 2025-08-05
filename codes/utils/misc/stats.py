import numpy as np
import pandas as pd
import scipy.stats as st


def cohen_d(x, y):
    nx = len(x)
    ny = len(y)
    dof = nx + ny - 2
    return (np.mean(x) - np.mean(y)) / np.sqrt(((nx-1)*np.std(x, ddof=1) ** 2 + (ny-1)*np.std(y, ddof=1) ** 2) / dof)


def compute_dprime(arr1, arr2):
    """
    Compute d-prime across subjects, for each time point and pixel.

    Parameters:
    - arr1, arr2: numpy arrays of shape (n_subjects, n_time, x, y)

    Returns:
    - d_prime: numpy array of shape (n_time, x, y)
    """
    mean1 = np.nanmean(arr1, axis=0)  # shape: (n_time, x, y)
    mean2 = np.nanmean(arr2, axis=0)

    std1 = np.nanstd(arr1, axis=0)
    std2 = np.nanstd(arr2, axis=0)

    pooled_std = np.sqrt(0.5 * (std1 ** 2 + std2 ** 2))

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        d_prime = (mean1 - mean2) / pooled_std
    d_prime[np.isnan(d_prime)] = 0  # or use np.nan_to_num

    return d_prime  # shape: (n_time, x, y)


def first_above_threshold_frame(d, threshold, start_frame):
    """
    Find first time index >= start_frame where d > threshold, per pixel.

    Parameters:
    - d: array of shape (time, x, y)
    - threshold: scalar
    - start_frame: int, index of first frame to consider

    Returns:
    - first_frame: array of shape (x, y), with frame index or -1 if not found
    """
    # Restrict to time slice from start_frame onward
    d_sub = d[start_frame:]  # shape: (T', x, y)

    # Apply threshold
    mask = d_sub > threshold  # shape: (T', x, y)

    # Find first True along time axis
    first_frame = np.argmax(mask, axis=0).astype(float)  # shape: (x, y)

    # Handle cases where threshold is never exceeded
    no_true = ~np.any(mask, axis=0)
    first_frame[no_true] = np.nan

    return first_frame


def psth_context_stats(df, grid_spot):
    ts = df.time.unique()
    ts = ts[ts > 0]
    n_ts = len(ts)
    spot_list = []
    ts_list = []
    mean_rew = []
    mean_norew = []
    std_rew = []
    std_norew = []
    mean_diff = []
    dprime = []
    dof_list = []
    w_values = []
    p_values = []
    p_values_corr = []
    for spot in grid_spot:
        for tested_ts in ts:
            rew_data = df.loc[(df.time == tested_ts) & (df.epoch == 'rewarded') & (df.cell_type == spot)]['activity'].values
            mean_rew.append(np.mean(rew_data))
            std_rew.append(np.std(rew_data))

            non_rew_data = df.loc[(df.time == tested_ts) & (df.epoch == 'non-rewarded') & (df.cell_type == spot)]['activity'].values
            mean_norew.append(np.mean(non_rew_data))
            std_norew.append(np.std(non_rew_data))

            mean_diff.append(np.mean(rew_data) - np.mean(non_rew_data))
            dof_list.append(len(non_rew_data) - 1)

            dprime.append((np.nanmean(rew_data) - np.nanmean(non_rew_data)) / np.sqrt(0.5 * (np.var(rew_data) + np.var(non_rew_data))))

            stat_res = st.ttest_rel(rew_data, non_rew_data)
            spot_list.append(spot)
            ts_list.append(tested_ts)
            w_values.append(stat_res[0])
            p_values.append(stat_res[1])
            p_values_corr.append(stat_res[1] * n_ts)
    stats_results = pd.DataFrame()
    stats_results['Spot'] = spot_list
    stats_results['Time'] = ts_list
    stats_results['RewMean'] = mean_rew
    stats_results['RewSTD'] = std_rew
    stats_results['NoRewMean'] = mean_norew
    stats_results['NoRewSTD'] = std_norew
    stats_results['MeanDiff'] = mean_diff
    stats_results['DoF'] = dof_list
    stats_results['Dprime'] = dprime
    stats_results['T'] = w_values
    stats_results['p'] = p_values
    stats_results['p corr'] = p_values_corr

    return stats_results

