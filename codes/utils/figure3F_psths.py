import os.path

import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from codes.utils.misc.stats import psth_context_stats
from codes.utils.misc.fig_saving import save_fig
from codes.utils.misc.table_saving import save_table


def figure3f_psth(table, saving_path, name, formats=['png']):
    table = table.drop(['event', 'roi', 'behavior_type', 'behavior_day'], axis=1)

    # Select time
    table = table.loc[(table.time > -0.09) & (table.time < 0.160)]

    # Average within session
    session_df = table.groupby(['time', 'mouse_id', 'session_id', 'cell_type', 'trial_type', 'epoch'],
                               as_index=False).agg('mean')

    # Average within mouse
    mouse_df = session_df.copy()
    mouse_df = mouse_df.drop(['session_id'], axis=1)
    mouse_df = mouse_df.groupby(['time', 'mouse_id', 'cell_type', 'trial_type', 'epoch'], as_index=False).agg(
        'mean')

    # Add AP / ML coordinates
    coord_mouse_df = mouse_df.copy(deep=True)
    coord_mouse_df['AP'] = mouse_df['cell_type'].apply(lambda x: ast.literal_eval(x)[0])
    coord_mouse_df['ML'] = mouse_df['cell_type'].apply(lambda x: ast.literal_eval(x)[1])
    data_to_plot = coord_mouse_df.loc[
        ((coord_mouse_df.trial_type == 'whisker_hit_trial') & (coord_mouse_df.epoch == 'rewarded')) |
        ((coord_mouse_df.trial_type == 'whisker_miss_trial') & (coord_mouse_df.epoch == 'non-rewarded'))
        ]

    # Select grid points
    selected_spots = ['(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 1.5)', '(-1.5, 0.5)', '(2.5, 2.5)', '(0.5, 4.5)']
    data_to_plot = data_to_plot.loc[data_to_plot.cell_type.isin(selected_spots)]

    # Do the stats W+ vs W-
    stat_results = psth_context_stats(df=data_to_plot, grid_spot=selected_spots)
    save_table(df=stat_results, saving_path=saving_path, name=f'{name}_stats')

    # Plot PSTHs activity
    fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(4, 6))
    color_palette = [(129 / 255, 0 / 255, 129 / 255), (0 / 255, 135 / 255, 0 / 255)]
    for ax_idx, ax in enumerate(axes.flatten()):
        data = data_to_plot.loc[data_to_plot.cell_type == selected_spots[ax_idx]]
        sns.lineplot(data, x='time', y='activity', hue='epoch', hue_order=['non-rewarded', 'rewarded'],
                     palette=color_palette, legend=False, ax=ax)
        sns.despine()
        ax.axvline(x=0, ymin=0, ymax=1, c='orange', linestyle='--')
        ax.set_xlim(-0.1, 0.180)
        ax.set_ylim(-0.01, 0.052)
        ax.set_yticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05], labels=['0.0', '1.0', '2.0', '3.0', '4.0', '5.0'])
        ax.set_ylabel('DF/F0 (%)')
        ax.set_xlabel('Time (ms)')
        ax.set_title(f'{ast.literal_eval(selected_spots[ax_idx])[0]} / {ast.literal_eval(selected_spots[ax_idx])[1]}')

    fig.suptitle('Whisker trials by context')
    fig.tight_layout()

    save_fig(fig, saving_path=saving_path, figure_name=name, formats=formats)

    # Plot Dprime and Mean diff for supplementary
    saving_folder = os.path.join(os.path.dirname(saving_path), 'supplementary', 'figure3_supp1', 'Reviewing')
    if not os.path.exists(saving_folder):
        os.makedirs(saving_folder)

    # Fig 1 (Mean diff):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    c_palette = ['orange', 'darkorange', 'royalblue', 'blue', 'purple', 'red']
    sns.lineplot(stat_results, x='Time', y='MeanDiff', hue='Spot', hue_order=selected_spots,
                 palette=c_palette, legend=True, ax=ax)
    ax.axvline(x=0, ymin=0, ymax=1, c='orange', linestyle='--')
    ax.set_xlabel('Time (ms)')
    ax.set_xlim(-0.1, 0.180)
    ax.set_yticks([0.00, 0.01, 0.02, 0.03], labels=['0.0', '1.0', '2.0', '3.0'])
    ax.set_ylabel('Delta activity (% dF / F)')
    ax.set_ylim(-0.01, 0.035)
    sns.despine()
    fig.tight_layout()
    save_fig(fig, saving_path=saving_folder, figure_name=f'{name}_supp_3_delta_gran_avg', formats=formats)

    # Fig 2 (D prime):
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    sns.lineplot(stat_results, x='Time', y='Dprime', hue='Spot', hue_order=selected_spots,
                 palette=c_palette, legend=True, ax=ax)
    ax.axvline(x=0, ymin=0, ymax=1, c='orange', linestyle='--')
    ax.axhline(y=2, xmin=0, xmax=1, c='grey', linestyle='--')
    ax.axhline(y=0, xmin=0, xmax=1, c='black', linestyle='-', linewidth=1)
    ax.set_xlabel('Time (ms)')
    ax.set_xlim(-0.1, 0.180)
    ax.set_ylabel("D'")
    ax.set_ylim(-2, 5)
    sns.despine()
    fig.tight_layout()
    save_fig(fig, saving_path=saving_folder, figure_name=f'{name}_supp_3_dprime', formats=formats)

