import numpy as np
import pandas as pd
import scipy.stats as st
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


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

            if len(rew_data) == len(non_rew_data):
                stat_res = st.ttest_rel(rew_data, non_rew_data)
            else:
                stat_res = st.ttest_ind(rew_data, non_rew_data)

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


def exponential_decay(t, A, tau, C):
    """
    Exponential decay function: y = A * exp(-t/tau) + C

    Parameters:
    - t: time
    - A: amplitude of decay
    - tau: time constant (what we want to estimate)
    - C: asymptotic value
    """
    return A * np.exp(-t / tau) + C


def fit_transition_adaptation(data_to_plot, transition_type='To W+', fit_start=1, include_bin_minus_one=True):
    """
    Fit exponential decay starting from bin -1 (as t=0) through positive bins.

    Parameters:
    - data_to_plot: DataFrame with columns ['time_bin', 'lick_flag', 'transition']
    - transition_type: 'To W+' or 'To W-'
    - fit_start: starting bin for positive bins (default 1)
    - include_bin_minus_one: if True, include bin -1 as t=0 in the fit

    Returns:
    - tau: time constant in bins (multiply by 10 for seconds)
    - params: all fitted parameters [A, tau, C]
    - fit_quality: R-squared value
    """
    # Get bin -1
    bin_minus_one = data_to_plot[(data_to_plot['time_bin'] == -1) &
                                 (data_to_plot['transition'] == transition_type)]

    # Get positive bins for fitting
    data_positive = data_to_plot[(data_to_plot['time_bin'] >= fit_start) &
                                 (data_to_plot['transition'] == transition_type)].copy()

    if include_bin_minus_one and len(bin_minus_one) > 0:
        # Combine bin -1 with positive bins
        data_subset = pd.concat([bin_minus_one, data_positive])
    else:
        data_subset = data_positive

    # Sort by time_bin
    data_subset = data_subset.sort_values('time_bin')

    x_original = data_subset['time_bin'].values
    y = data_subset['lick_flag'].values

    # Transform to continuous time starting from 0
    # bin -1 -> t=0 (10 seconds before transition)
    # bin 1 -> t=1 (10 seconds after transition, so 20 seconds from start)
    # bin 2 -> t=2, bin 3 -> t=3, etc.
    # We need to account for the actual time represented
    x_transformed = np.where(x_original < 0, 0, x_original)

    # Initial parameter guesses
    y_start = y[0]  # Value at bin -1
    y_end = y[-1]
    A_init = y_start - y_end
    tau_init = 3.0
    C_init = y_end

    try:
        # Fit the exponential decay using transformed x
        params, covariance = curve_fit(
            exponential_decay,
            x_transformed,
            y,
            p0=[A_init, tau_init, C_init],
            bounds=([-np.inf, 0.1, -np.inf], [np.inf, 50, np.inf]),
            maxfev=10000
        )

        A_fit, tau_fit, C_fit = params

        # Calculate R-squared
        y_pred = exponential_decay(x_transformed, *params)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)

        # Return original bin values along with fitted parameters
        return tau_fit, params, r_squared, x_original, y

    except Exception as e:
        print(f"Fitting failed for {transition_type}: {str(e)}")
        return None, None, None, x_original, y


def analyze_both_transitions(data_to_plot, plot=True, start_fit=1, include_bin_minus_one=True):
    """
    Analyze both transition types and optionally plot results.

    Returns dictionary with results for both transitions.
    """
    results = {}

    for transition in ['To W+', 'To W-']:
        tau, params, r_squared, x, y = fit_transition_adaptation(
            data_to_plot, transition, start_fit, include_bin_minus_one)

        if tau is not None:
            results[transition] = {
                'tau_bins': tau,
                'tau_seconds': tau * 10,
                'amplitude': params[0],
                'asymptote': params[2],
                'r_squared': r_squared,
                'x': x,
                'y': y,
                'params': params
            }

            print(f"\n{transition}:")
            print(f"  Time constant (τ): {tau:.2f} bins = {tau * 10:.1f} seconds")
            print(f"  Amplitude (A): {params[0]:.3f}")
            print(f"  Asymptote (C): {params[2]:.3f}")
            print(f"  R²: {r_squared:.3f}")

    if plot and len(results) > 0:
        fig, ax = plt.subplots(1, 1, figsize=(5, 3))
        colors = {'To W+': (0 / 255, 135 / 255, 0 / 255),
                  'To W-': (129 / 255, 0 / 255, 129 / 255)}

        for transition, res in results.items():
            # Get all data for this transition (including negative time bins)
            data_subset = data_to_plot[data_to_plot['transition'] == transition].copy()
            data_subset = data_subset.sort_values('time_bin')

            # Apply appropriate shift: +0.5 for negative bins, -0.5 for positive bins
            time_bins = data_subset['time_bin'].values
            x_all = np.where(time_bins < 0,
                             (time_bins + 0.5) * 10,
                             (time_bins - 0.5) * 10)
            y_all = data_subset['lick_flag'].values

            # Plot all data points (before and after transition)
            ax.plot(x_all, y_all, 'o-', color=colors[transition],
                    markersize=10, linewidth=2,
                    label=f'{transition} (τ={res["tau_seconds"]:.1f}s)', zorder=3)

            # Plot fitted curve
            # Generate curve in transformed space (where bin -1 is at t=0, bin 1+ at their values)
            fitted_bins = res['x']
            fitted_bins_transformed = np.where(fitted_bins < 0, 0, fitted_bins)

            # Create dense curve in transformed space
            x_fit_transformed = np.linspace(0, fitted_bins_transformed.max(), 100)
            y_fit = exponential_decay(x_fit_transformed, *res['params'])

            # Transform back to original bin space for display
            # t=0 -> bin -1, t=1 -> bin 1, t=2 -> bin 2, etc.
            x_fit_bins = np.where(x_fit_transformed == 0, -1, x_fit_transformed)

            # Apply display shift
            x_fit_shifted = np.where(x_fit_bins < 0,
                                     (x_fit_bins + 0.5) * 10,
                                     (x_fit_bins - 0.5) * 10)

            ax.plot(x_fit_shifted, y_fit, '--', color=colors[transition],
                    linewidth=2, alpha=0.7, zorder=2)

        # Add vertical line at transition (x=0)
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, zorder=1)

        # Format axes
        ax.set_xlabel('Time around transition (s)', fontsize=12)
        ax.set_ylabel('Lick Probability', fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(-60, 60)
        ax.set_xticks([-50, -40, -30, -20, -10, 0, 10, 20, 30, 40, 50])

        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.legend(frameon=False, loc='best', fontsize=10)

        plt.tight_layout()

    return results, fig


def subtract_baseline(df, start, stop):
    df = df.copy()

    # Create a trial identifier: increment when time resets/decreases
    df['trial_id'] = (df['Time'].diff() < 0).cumsum()

    # Calculate baseline (mean activity from start to stop) for each trial
    baseline = df[(df['Time'] >= start) & (df['Time'] <= stop)].groupby(
        'trial_id'
    )['MeanDiff'].mean().reset_index()

    # Rename the activity column to baseline
    baseline.rename(columns={'MeanDiff': 'baseline'}, inplace=True)

    # Merge baseline back to the original table
    df = df.merge(baseline, on='trial_id', how='left')

    # Subtract baseline from activity
    df['MeanDiff_corr'] = df['MeanDiff'] - df['baseline']

    # Drop columns before they get annoying
    df = df.drop(['baseline', 'trial_id'], axis=1)

    return df
