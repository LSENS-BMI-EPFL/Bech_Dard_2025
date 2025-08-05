import json
from codes.utils.misc.stats import *
import seaborn as sns
import matplotlib.pyplot as plt
from codes.utils.misc.plot_on_grid import plot_grid_on_allen
from codes.utils.misc.fig_saving import save_fig


def figure2cd(data_table, saving_path, saving_formats):
    data_table['shuffle_dist_sub'] = data_table['shuffle_dist_sub'].apply(json.loads)

    avg_df = data_table.groupby(by=['context', 'trial_type', 'opto_grid_ml', 'opto_grid_ap']).agg(
        data=('data_mean', list),
        data_sub=('data_mean_sub', list),
        data_mean=('data_mean', 'mean'),
        data_mean_sub=(
            'data_mean_sub', 'mean'),
        shuffle_dist=('shuffle_dist', 'sum'),
        shuffle_dist_sub=(
            'shuffle_dist_sub', 'sum'),
        percentile_avg=('percentile', 'mean'),
        percentile_avg_sub=(
            'percentile_sub', 'mean'),
        n_sigma_avg=('n_sigma', 'mean'),
        n_sigma_avg_sub=(
            'n_sigma_sub', 'mean'))

    avg_df['d_sub'] = avg_df.apply(lambda x: abs(cohen_d(x.shuffle_dist_sub, x.data_sub)), axis=1)
    avg_df = avg_df.reset_index()

    dprime_palette = 'inferno'
    seismic_palette = sns.diverging_palette(265, 10, s=100, l=40, sep=30, n=200, center="light", as_cmap=True)

    fig, axes = plt.subplots(2, 3, figsize=(8, 6))
    fig.suptitle(f'Opto grid control subtracted')
    fig1, axes1 = plt.subplots(2, 3, figsize=(8, 6))
    fig1.suptitle(f'd prime subtracted')

    for name, group in avg_df.groupby(by=['context', 'trial_type']):
        if 'whisker_trial' in name:
            outcome = 'outcome_w'
            col = 2
        elif 'auditory_trial' in name:
            outcome = 'outcome_a'
            col = 1
        else:
            outcome = 'outcome_n'
            col = 0

        row = 0 if 'rewarded' in name else 1

        group.rename(columns={'opto_grid_ml': 'x', 'opto_grid_ap': 'y'}, inplace=True)
        fig, axes[row, col] = plot_grid_on_allen(group, outcome=f"data_mean_sub", palette=seismic_palette, facecolor=None,
                                                 edgecolor=None, vmin=-0.4,
                                                 vmax=0.4, dotsize=250, fig=fig, ax=axes[row, col], result_path=None)
        fig.tight_layout()

        fig1, axes1[row, col] = plot_grid_on_allen(group, outcome="d_sub", palette=dprime_palette,
                                                   vmin=0.5, facecolor=None, edgecolor=None,
                                                   vmax=2.2, dotsize=250, fig=fig1,
                                                   ax=axes1[row, col], result_path=None)
        fig1.tight_layout()

    cols = ['No stim', 'Auditory', 'Whisker']
    rows = ['Rewarded', 'No rewarded']

    for axes in [axes, axes1]:
        for a, col in zip(axes[0], cols):
            a.set_title(col)
        for a, row in zip(axes[:, 0], rows):
            a.set_ylabel(row)

    names = ['Figure2C', 'Figure2D']
    for idx, panel in enumerate([fig, fig1]):
        save_fig(panel, saving_path, names[idx], formats=saving_formats)

    plt.close('all')
