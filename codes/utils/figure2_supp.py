import os
import json
from codes.utils.misc.stats import *
import matplotlib.pyplot as plt
from codes.utils.misc.plot_on_grid import plot_grid_on_allen
from codes.utils.misc.fig_saving import save_fig
from codes.utils.misc.plot_utils import *


def plot_figure2_supp1ab(data_table, saving_path, saving_formats):
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
        fig, axes[row, col] = plot_grid_on_allen(group, outcome=f"data_mean_sub", palette=seismic_palette,
                                                 facecolor=None,
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

    names = ['Figure2_supp1A', 'Figure2_supp1B']
    for idx, panel in enumerate([fig, fig1]):
        save_fig(panel, saving_path, names[idx], formats=saving_formats)

    plt.close('all')


def plot_figure2_supp1cde(muscimol, ringer, saving_path, sites, names, saving_formats):
    for idx, site in enumerate(sites):
        muscimol_path = os.path.join(muscimol, f'{site}_Average mice behavior', 'context_days_full_table.csv')
        ringer_path = os.path.join(ringer, f'{site}_general_average_across_days.csv')
        muscimol_df = pd.read_csv(muscimol_path)
        ringer_df = pd.read_csv(ringer_path)

        muscimol_df['drug'] = 'Muscimol'
        ringer_df['drug'] = 'Ringer'
        df = pd.concat((muscimol_df, ringer_df))

        df = df.loc[df.artificial_day != 2]
        df = df.drop(['Unnamed: 0', 'session_id', 'day', 'context', 'context_background'], axis=1)

        hue_name = ['Rewarded', 'Non-Rewarded']
        context_palette = {
            'catch_palette': {hue_name[0]: 'darkgray', hue_name[1]: 'lightgrey'},
            'wh_palette': {hue_name[0]: 'green', hue_name[1]: 'darkmagenta'},
            'aud_palette': {hue_name[0]: 'mediumblue', hue_name[1]: 'cornflowerblue'}
        }

        figure, (ax0, ax1) = plt.subplots(1, 2, figsize=(4, 4), sharey=True)
        for outcome, palette_key in zip(['outcome_n', 'outcome_a', 'outcome_w'],
                                        ['catch_palette', 'aud_palette', 'wh_palette']):
            plot_with_point_and_strip(data=df.loc[df.drug == 'Muscimol'], x_name='artificial_day', y_name=outcome,
                                      hue='context_rwd_str', palette=context_palette, ax=ax1, palette_key=palette_key,
                                      link_mice=False)

            plot_with_point_and_strip(data=df.loc[df.drug == 'Ringer'], x_name='artificial_day', y_name=outcome,
                                      hue='context_rwd_str', palette=context_palette, ax=ax0, palette_key=palette_key,
                                      link_mice=False)
        ax0.set_title('Ringer injection')
        ax1.set_title('Muscimol injection')
        for ax in [ax0, ax1]:
            ax.get_legend().set_visible(False)
            ax_set(ax, ylim=[-0.1, 1.05], xlabel='Day', ylabel='Lick probability')
        figure.suptitle(site)
        figure.tight_layout()

        save_fig(figure, saving_path, f'Figure2_supp1{names[idx]}_left', formats=saving_formats)

        # Separate artificial_day 0 and 1 into different DataFrames
        day_0 = df[df['artificial_day'] == 0].set_index(['mouse_id', 'context_rwd_str', 'drug'])
        day_1 = df[df['artificial_day'] == 1].set_index(['mouse_id', 'context_rwd_str', 'drug'])

        # Ensure the indices are aligned before subtraction
        difference = day_1[['outcome_a', 'outcome_w', 'outcome_n']] - day_0[['outcome_a', 'outcome_w', 'outcome_n']]

        # Reset the index and rename columns
        difference = difference.reset_index()
        difference.columns = ['mouse_id', 'context_rwd_str', 'drug', 'outcome_a_diff', 'outcome_w_diff',
                              'outcome_n_diff']

        figure, axes = plt.subplots(1, 3, figsize=(5, 4), sharey=False)
        ax = 0
        titles = ['catch trials', 'auditory trials', 'whisker trials']
        for outcome, palette_key in zip(['outcome_n_diff', 'outcome_a_diff', 'outcome_w_diff'],
                                        ['wh_palette', 'wh_palette', 'wh_palette']):
            sns.stripplot(data=difference, x='drug', order=['Ringer', 'Muscimol'], y=outcome, hue='context_rwd_str',
                          palette=context_palette[palette_key], legend=False, dodge=True, ax=axes.flatten()[ax],
                          size=10)
            sns.pointplot(data=difference, x='drug', order=['Ringer', 'Muscimol'], y=outcome, hue='context_rwd_str',
                          palette=context_palette[palette_key],
                          legend=False, ax=axes.flatten()[ax], linestyle='none', estimator=np.nanmean, alpha=0.5,
                          dodge=True)
            axes.flatten()[ax].set_ylim(-1, 1)
            axes.flatten()[ax].set_title(titles[ax])
            axes.flatten()[ax].set_ylabel('Delta lick probability')
            axes.flatten()[ax].axhline(y=0, c='k', linestyle='--')
            ax += 1
        sns.despine()
        figure.suptitle(site)
        figure.tight_layout()
        save_fig(figure, saving_path, f'Figure2_supp1{names[idx]}_right', formats=saving_formats)


