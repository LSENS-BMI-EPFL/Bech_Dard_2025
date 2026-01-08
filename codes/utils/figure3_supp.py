import ast
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from codes.utils.misc.plot_average_widefield_timecourse import plot_average_wf_timecourse, \
    plot_wf_timecourse_aud_wh_diff
from codes.utils.misc.stats import psth_context_stats
from codes.utils.misc.fig_saving import save_fig


def process_group(grouped_df, arr_name, output_col_name, bl=10, remove_baseline=False):
    processed = []
    for array in grouped_df[arr_name]:
        if remove_baseline:
            # Subtract average of first bl timepoints for each sample (row)
            baseline = np.nanmean(array[:, :bl], axis=1, keepdims=True)  # shape (samples, 1)
            corrected = array - baseline
        else:
            corrected = array
        processed.append(corrected)

    # Stack into 3D array: (n_arrays, samples, time)
    stacked = np.stack(processed, axis=0)  # shape: (group_size, samples, time)

    # Average over the first axis (arrays)
    averaged = np.nanmean(stacked, axis=0)  # shape: (samples, time)

    return pd.Series({output_col_name: averaged})


def wf_timecourse_context_switch(data, saving_path, formats=['png'], scale=(-0.10, 0.010), halfrange=0.010):
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


def wh_psth_by_trial_index(df, sorted_areas, save_folder, name, formats, pre_stim=5):
    session_tables = df.copy()
    grouped_session_tables = (session_tables.groupby(['context', 'mouse_id', 'session_id', 'whisker_index'],
                                                     as_index=False).
                              apply(process_group, arr_name='activity_array', output_col_name='session_avg',
                                    bl=pre_stim,
                                    remove_baseline=True, include_groups=False).reset_index(drop=True))

    mice_tables = grouped_session_tables.copy(deep=True)
    mice_tables = mice_tables.drop(['session_id'], axis=1)
    grouped_mice_tables = (mice_tables.groupby(['context', 'mouse_id', 'whisker_index'], as_index=False).
                           apply(process_group, arr_name='session_avg', output_col_name='mouse_avg',
                                 include_groups=False).reset_index(drop=True))

    long_rows = []
    for _, row in grouped_mice_tables.iterrows():
        array = row['mouse_avg']  # shape: (samples, time)
        n_samples, n_time = array.shape
        for sample in range(n_samples):
            for t in range(n_time):
                long_rows.append({
                    'roi': sorted_areas[sample],
                    'time': (t - pre_stim) / 100,
                    'dff': array[sample, t],
                    'whisker_index': row['whisker_index'],
                    'mouse_id': row['mouse_id'],
                    'context': row['context']
                })
    long_df = pd.DataFrame(long_rows)

    # Select ROIs
    selected_rois = ['(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 1.5)', '(2.5, 2.5)', '(-1.5, 0.5)', '(0.5, 4.5)']

    # ROI x trial index split context
    fig = sns.relplot(data=long_df.loc[long_df.roi.isin(selected_rois)],
                      x='time', y='dff', kind='line',
                      row='roi',
                      col='whisker_index',
                      hue='context',
                      hue_order=[0, 1],
                      palette=['darkmagenta', 'green'],
                      legend=False,
                      height=2, aspect=0.8,
                      facet_kws={'sharey': True})

    # Add vertical line at x=0 for each subplot
    for ax in fig.axes.flatten():
        ax.axvline(x=0, ymin=0, ymax=1, color='orange', linestyle='--')
    fig.set_titles(row_template="{row_name}", col_template="{col_name}")
    fig.tight_layout()
    save_fig(fig, saving_path=save_folder, figure_name=f'{name}_psths_v1', formats=formats)

    # ROI x context split by trial index
    fig = sns.relplot(data=long_df.loc[long_df.roi.isin(selected_rois)], x='time', y='dff', kind='line',
                      col='roi', row='context', hue='whisker_index', height=2, aspect=0.8)
    fig.tight_layout()
    fig.set_titles(row_template="{row_name}", col_template="{col_name}")
    save_fig(fig, saving_path=save_folder, figure_name=f'{name}_psths_v2', formats=formats)

    # Peak value figure
    long_df_sel = long_df.loc[(long_df.time > 0) & (long_df.time < 0.130) & long_df.roi.isin(selected_rois)]
    long_df_sel_max = long_df_sel.loc[
        long_df_sel.groupby(['roi', 'mouse_id', 'whisker_index', 'context'], as_index=False)['dff'].idxmax()['dff']]

    fig = sns.relplot(long_df_sel_max, x='whisker_index', y='dff', hue='context', hue_order=[0, 1],
                      palette=['darkmagenta', 'green'], col='roi', kind='line', aspect=0.8, height=3, legend=False)
    fig.figure.suptitle(f'0-120ms PSTH peak across trials', size=12)
    fig.tight_layout()
    save_fig(fig, saving_path=save_folder, figure_name=f'{name}_psths_peak', formats=formats)


def wf_timecourse_auditory_to_whisker(aud_data, whisker_data, saving_path, formats=['png'],
                                      halfrange=0.03):
    # Keep only correct trial types in W+ context:
    wh_trial_types = ['rewarded_whisker_hit_trial']
    aud_trial_types = ['rewarded_auditory_hit_trial']

    # PLot difference
    plot_wf_timecourse_aud_wh_diff(aud_data, whisker_data, aud_trial_types, wh_trial_types,
                                   saving_path, formats=formats,
                                   diff_range=halfrange)


