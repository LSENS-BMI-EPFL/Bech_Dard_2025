import ast
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from codes.utils.misc.plot_average_widefield_timecourse import plot_average_wf_timecourse
from codes.utils.misc.stats import psth_context_stats
from codes.utils.misc.fig_saving import save_fig


def wf_timecourse_context_switch(data, saving_path, formats=['png'], scale=(-0.15, 0.015), halfrange=0.010):
    # Keep only correct trial types:
    trial_types = ['rewarded', 'non-rewarded']

    # Plot
    plot_average_wf_timecourse(data, trial_types, saving_path, formats=formats, scale=scale, diff_range=halfrange)


def dprime_by_mouse(table, saving_path, name, formats):
    table = table.drop(['event', 'roi', 'behavior_type', 'behavior_day'], axis=1)

    # Select time
    table = table.loc[(table.time > -0.09) & (table.time < 0.160)]

    # Average within session
    session_df = table.groupby(['time', 'mouse_id', 'session_id', 'cell_type', 'trial_type', 'epoch'],
                               as_index=False).agg('mean')

    # Add AP / ML coordinates
    coord_df = session_df.copy(deep=True)
    coord_df['AP'] = session_df['cell_type'].apply(lambda x: ast.literal_eval(x)[0])
    coord_df['ML'] = session_df['cell_type'].apply(lambda x: ast.literal_eval(x)[1])
    df = coord_df.loc[
        ((coord_df.trial_type == 'whisker_hit_trial') & (coord_df.epoch == 'rewarded')) |
        ((coord_df.trial_type == 'whisker_miss_trial') & (coord_df.epoch == 'non-rewarded'))
        ]

    # Select grid points
    selected_spots = ['(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 1.5)', '(-1.5, 0.5)', '(2.5, 2.5)', '(0.5, 4.5)']
    df_sel = df.loc[df.cell_type.isin(selected_spots)]

    # Loop on mice
    mice_stats = []
    for mouse in df_sel.mouse_id.unique():
        mouse_df = df_sel.loc[df_sel.mouse_id == mouse]
        mouse_stats = psth_context_stats(df=mouse_df, grid_spot=selected_spots)
        mouse_stats['mouse'] = mouse
        mice_stats.append(mouse_stats)
    mice_stats = pd.concat(mice_stats)

    # Look for d' above 2 for each spot and each mouse
    mice_stats = mice_stats.loc[mice_stats.Time > 0.02]

    # Significant level :
    wh_df_sig = (mice_stats[mice_stats['Dprime'] > 2]
                 .sort_values(['mouse', 'Time'])
                 .groupby(['mouse', 'Spot'], as_index=False)
                 .first()
                 .sort_values(['mouse', 'Time']))

    # Find the average order
    avg_df = wh_df_sig.drop('mouse', axis=1).groupby(['Spot'], as_index=False).agg('mean')
    avg_df = avg_df.sort_values(['Time'])

    # Make the figure
    fig, ax = plt.subplots(1, 1, figsize=(3, 4))
    sns.barplot(wh_df_sig, x='Spot', order=avg_df['Spot'].to_list(), y='Time', ax=ax)
    sns.stripplot(wh_df_sig, x='Spot', order=avg_df['Spot'].to_list(), y='Time', ax=ax)
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_ylabel('Time (s)')
    ax.set_xlabel('Sorted areas')
    fig.tight_layout()

    save_fig(fig, saving_path=saving_path, figure_name=name, formats=formats)
