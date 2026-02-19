import os.path

import ast
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

from codes.utils.misc.plot_average_widefield_timecourse import plot_average_wf_timecourse, \
    plot_wf_timecourse_aud_wh_diff
from codes.utils.misc.stats import psth_context_stats, compute_dprime_over_time
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


def wf_timecourse_context_switch(data, saving_path, formats=['png'], scale=(-0.015, 0.015), halfrange=0.010):
    # Keep only correct trial types:
    trial_types = ['rewarded', 'non-rewarded']

    # Plot
    plot_average_wf_timecourse(data, trial_types, saving_path, formats=formats, scale=scale, diff_range=halfrange)


def dprime_by_mouse(table, thr, saving_path, name, formats):
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
    wh_df_sig = (mice_stats[mice_stats['Dprime'] > thr]
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
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylabel(f"Time (s) (D' > {thr})")
    ax.set_xlabel('Sorted areas')
    fig.tight_layout()

    save_fig(fig, saving_path=saving_path, figure_name=f'{name}_thr{thr}', formats=formats)


def dprime_by_session(table, thr, saving_path, name, formats):
    table = table.drop(['event', 'roi', 'behavior_type', 'behavior_day'], axis=1)

    # Select time
    table = table.loc[(table.time > -0.09) & (table.time < 0.160)]

    # Add AP / ML coordinates
    coord_df = table.copy(deep=True)
    coord_df['AP'] = table['cell_type'].apply(lambda x: ast.literal_eval(x)[0])
    coord_df['ML'] = table['cell_type'].apply(lambda x: ast.literal_eval(x)[1])
    df = coord_df.loc[
        ((coord_df.trial_type == 'whisker_hit_trial') & (coord_df.epoch == 'rewarded')) |
        ((coord_df.trial_type == 'whisker_miss_trial') & (coord_df.epoch == 'non-rewarded'))
        ]

    # Select grid points
    selected_spots = ['(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 1.5)', '(-1.5, 0.5)', '(2.5, 2.5)', '(0.5, 4.5)']
    df_sel = df.loc[df.cell_type.isin(selected_spots)]

    # Loop on session
    session_stats_list = []
    for session in df_sel.session_id.unique():
        session_df = df_sel.loc[df_sel.session_id == session]
        session_stats = psth_context_stats(df=session_df, grid_spot=selected_spots)
        session_stats['mouse'] = session[0:5]
        session_stats['session'] = session
        session_stats_list.append(session_stats)
    session_stats = pd.concat(session_stats_list)

    # Look for d' above 2 for each spot and each mouse
    session_stats = session_stats.loc[session_stats.Time > 0.02]

    # Significant level :
    wh_df_sig = (session_stats[session_stats['Dprime'] > thr]
                 .sort_values(['mouse', 'session', 'Time'])
                 .groupby(['mouse', 'session', 'Spot'], as_index=False)
                 .first()
                 .sort_values(['mouse', 'session', 'Time']))

    # Find the average order
    mouse_avg = wh_df_sig.drop('session', axis=1).groupby(['mouse', 'Spot'], as_index=False).agg('mean')
    avg_df = mouse_avg.drop('mouse', axis=1).groupby(['Spot'], as_index=False).agg('mean')
    avg_df = avg_df.sort_values(['Time'])

    # Make the figure
    fig, ax = plt.subplots(1, 1, figsize=(5, 4))

    # Plot individual session points with increased dodge
    sns.stripplot(wh_df_sig, x='Spot', order=avg_df['Spot'].to_list(), y='Time',
                  hue='mouse', dodge=True, legend=False, ax=ax, size=4, alpha=0.6)

    # Plot mouse averages
    sns.pointplot(mouse_avg, x='Spot', order=avg_df['Spot'].to_list(), y='Time',
                  hue='mouse', dodge=True, legend=False, linestyle='none',
                  markers='o', markersize=6, ax=ax)

    # Plot grand average on the right side of each spot cluster
    offset = 0.5  # Adjust this to move grand average further right

    for i, spot in enumerate(avg_df['Spot'].to_list()):
        spot_data = avg_df[avg_df['Spot'] == spot]
        ax.plot(i + offset, spot_data['Time'].values[0],
                marker='D', markersize=8, color='black',
                markeredgewidth=1.5, markeredgecolor='black',
                markerfacecolor='white', zorder=10)

    ax.spines[['top', 'right']].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_ylabel(f"Time (s) (D' > {thr})")
    ax.set_xlabel('Sorted areas')
    ax.set_xlim(-0.5, len(avg_df) - 0.5 + offset + 0.2)  # Extend x-limit to accommodate offset
    fig.tight_layout()

    save_fig(fig, saving_path=saving_path, figure_name=f'{name}_thr{thr}', formats=formats)


def mean_diff_session(table, saving_path, name, formats=['png', 'svg']):
    table = table.drop(['event', 'roi', 'behavior_type', 'behavior_day'], axis=1)

    # Select time
    table = table.loc[(table.time > -0.09) & (table.time < 0.160)]

    # Add AP / ML coordinates
    coord_df = table.copy(deep=True)
    coord_df['AP'] = table['cell_type'].apply(lambda x: ast.literal_eval(x)[0])
    coord_df['ML'] = table['cell_type'].apply(lambda x: ast.literal_eval(x)[1])
    df = coord_df.loc[
        ((coord_df.trial_type == 'whisker_hit_trial') & (coord_df.epoch == 'rewarded')) |
        ((coord_df.trial_type == 'whisker_miss_trial') & (coord_df.epoch == 'non-rewarded'))
        ]

    # Select grid points
    selected_spots = ['(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 1.5)', '(-1.5, 0.5)', '(2.5, 2.5)', '(0.5, 4.5)']
    df_sel = df.loc[df.cell_type.isin(selected_spots)]

    # Loop on session to get the mean diff and stats session based
    session_stats_list = []
    for session in df_sel.session_id.unique():
        session_df = df_sel.loc[df_sel.session_id == session]
        session_stats = psth_context_stats(df=session_df, grid_spot=selected_spots)
        session_stats['mouse'] = session[0:5]
        session_stats['session'] = session
        session_stats_list.append(session_stats)
    session_stats = pd.concat(session_stats_list)
    session_stats_mouse_avg = session_stats.drop('session', axis=1)
    session_stats_mouse_avg = session_stats_mouse_avg.groupby(['mouse', 'Time', 'Spot'], as_index=False).agg('mean')

    # Average within session
    df_sel_avg = df_sel.groupby(['time', 'mouse_id', 'session_id', 'cell_type', 'trial_type', 'epoch'],
                                as_index=False).agg('mean')
    # Loop on mice
    mice_stats = []
    for mouse in df_sel_avg.mouse_id.unique():
        mouse_df = df_sel_avg.loc[df_sel_avg.mouse_id == mouse]
        mouse_stats = psth_context_stats(df=mouse_df, grid_spot=selected_spots)
        mouse_stats['mouse'] = mouse
        mice_stats.append(mouse_stats)
    mice_stats = pd.concat(mice_stats)

    # FIGURES
    new_save_folder = os.path.join(saving_path, 'mouse_summary')
    if not os.path.exists(new_save_folder):
        os.makedirs(new_save_folder)
    c_palette = ['orange', 'darkorange', 'royalblue', 'blue', 'purple', 'red']
    context_palette = ['darkmagenta', 'green']
    for mouse in df_sel.mouse_id.unique():
        print(f'Mouse: {mouse}')
        mouse_table = df_sel.loc[df_sel.mouse_id == mouse]
        n_cols = len(selected_spots) + 2
        n_rows = len(mouse_table.session_id.unique()) + 1
        fig_size = (int(n_cols * 3), int(n_rows * 3))
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, sharex=True, figsize=fig_size)
        for ax in axes.flatten():
            ax.set_xlim(-0.1, 0.180)
            ax.set_ylim(-0.01, 0.052)
            ax.set_yticks([0.00, 0.01, 0.02, 0.03, 0.04, 0.05], labels=['0.0', '1.0', '2.0', '3.0', '4.0', '5.0'])
            ax.set_ylabel('DF/F0 (%)')
            ax.set_xlabel('Time (s)')
            ax.axvline(x=0, ymin=0, ymax=1, c='orange', linestyle='--')

        row_idx = 0
        for session in mouse_table.session_id.unique():
            print(f'Session: {session}')
            df_plot = mouse_table.loc[mouse_table.session_id == session]
            col_idx = 0
            for spot in selected_spots:
                df_subplot = df_plot.loc[df_plot.cell_type == spot]
                sns.lineplot(df_subplot, x='time', y='activity', hue='epoch', hue_order=['non-rewarded', 'rewarded'],
                             palette=context_palette, legend=False, ax=axes[row_idx, col_idx])
                sns.despine()
                if row_idx == 0:
                    axes[row_idx, col_idx].set_title(f'{spot}')
                col_idx += 1
            row_idx += 1

        mouse_avg_table = mouse_table.groupby(['mouse_id', 'session_id', 'cell_type', 'trial_type', 'epoch', 'time'],
                                              as_index=False).agg('mean')
        col_idx = 0
        for spot in selected_spots:
            df_subplot = mouse_avg_table.loc[mouse_avg_table.cell_type == spot]
            sns.lineplot(df_subplot, x='time', y='activity', hue='epoch', hue_order=['non-rewarded', 'rewarded'],
                         palette=context_palette, estimator='mean', errorbar=None, legend=False,
                         linewidth=2, ax=axes[-1, col_idx])
            sns.lineplot(df_subplot, x='time', y='activity', hue='epoch', hue_order=['non-rewarded', 'rewarded'],
                         palette=context_palette, estimator=None, units='session_id', legend=False,
                         linewidth=0.5, alpha=0.3, ax=axes[-1, col_idx])
            axes[-1, col_idx].set_title(f'{mouse} average')
            sns.despine()
            col_idx += 1

        row_idx = 0
        for session in mouse_table.session_id.unique():
            df_subplot = session_stats.loc[session_stats.session == session]
            sns.lineplot(df_subplot, x='Time', y='MeanDiff', hue='Spot', hue_order=selected_spots,
                         palette=c_palette, legend=False, ax=axes[row_idx, -2])
            sns.despine()
            axes[row_idx, -2].set_ylabel('Delta DF/F0 (%)')
            row_idx += 1

        row_idx = 0
        for session in mouse_table.session_id.unique():
            df_subplot = session_stats.loc[session_stats.session == session]
            sns.lineplot(df_subplot, x='Time', y='Dprime', hue='Spot', hue_order=selected_spots,
                         palette=c_palette, legend=False, ax=axes[row_idx, -1])
            sns.despine()
            axes[row_idx, -1].set_ylim(-1, 3)
            axes[row_idx, -1].set_yticks(np.arange(-1, 4),  labels=['-1', '0', '1', '2', '3'])
            axes[row_idx, -1].set_ylabel("D'")
            row_idx += 1

        df_subplot = session_stats_mouse_avg.loc[session_stats_mouse_avg.mouse == mouse]
        sns.lineplot(df_subplot, x='Time', y='MeanDiff', hue='Spot', hue_order=selected_spots,
                     palette=c_palette, legend=False, ax=axes[-1, -2])
        axes[-1, -2].set_ylabel('Delta DF/F0 (%)')
        axes[-1, -2].set_title(f'{mouse} average')
        sns.despine()

        df_subplot = mice_stats.loc[mice_stats.mouse == mouse]
        sns.lineplot(df_subplot, x='Time', y='Dprime', hue='Spot', hue_order=selected_spots,
                     palette=c_palette, legend=True, ax=axes[-1, -1])
        sns.despine()
        axes[-1, -1].set_title(f"{mouse} D'")
        axes[-1, -2].set_ylabel("D'")
        axes[-1, -1].set_ylim(-3, 6)
        axes[-1, -1].set_yticks(np.arange(-3, 7), labels=['-3', '-2', '-1', '0', '1', '2', '3', '4', '5', '6'])

        fig.suptitle(f'{mouse}')
        fig.tight_layout()

        save_fig(fig, saving_path=new_save_folder, figure_name=f'{name}_{mouse}_summary', formats=formats)

    # Figure 1 : n mice (cols) x n sessions (rows)
    fig, axes = plt.subplots(session_stats.groupby('mouse', as_index=False)['session'].nunique()['session'].max(),
                             len(session_stats.mouse.unique()), sharex=True, sharey=True, figsize=(14, 14))
    for col_idx, mouse in enumerate(session_stats.mouse.unique()):
        mouse_df = session_stats.loc[session_stats.mouse == mouse]
        for row_idx, session in enumerate(mouse_df.session.unique()):
            df_plot = mouse_df.loc[mouse_df.session == session]
            sns.lineplot(df_plot, x='Time', y='MeanDiff', hue='Spot', hue_order=selected_spots,
                         palette=c_palette, legend=False, ax=axes[row_idx, col_idx])
            sns.despine()

            axes[row_idx, col_idx].axvline(x=0, ymin=0, ymax=1, c='orange', linestyle='--')
            axes[row_idx, col_idx].set_xlabel('Time (ms)')
            axes[row_idx, col_idx].set_xlim(-0.1, 0.180)
            axes[row_idx, col_idx].set_ylabel('Delta activity')
            axes[row_idx, col_idx].set_ylim(-0.01, 0.045)
            axes[row_idx, col_idx].set_title(f'{session}')
    fig.tight_layout()
    save_fig(fig, saving_path=new_save_folder, figure_name=f'{name}_indiv_mice_session', formats=formats)

    # Figure 2 : n mice (cols)
    fig, axes = plt.subplots(1, len(session_stats_mouse_avg.mouse.unique()), sharex=True, figsize=(14, 4))
    for idx, mouse in enumerate(session_stats_mouse_avg.mouse.unique()):
        df_plot = session_stats_mouse_avg.loc[session_stats_mouse_avg.mouse == mouse]
        sns.lineplot(df_plot, x='Time', y='MeanDiff', hue='Spot', hue_order=selected_spots,
                     palette=c_palette, legend=False, ax=axes[idx])
        sns.despine()

        axes[idx].axvline(x=0, ymin=0, ymax=1, c='orange', linestyle='--')
        axes[idx].set_xlabel('Time (ms)')
        axes[idx].set_xlim(-0.1, 0.180)
        axes[idx].set_ylim(-0.01, 0.045)
        axes[idx].set_ylabel('Delta activity')
        axes[idx].set_title(f'{mouse}')
    fig.tight_layout()
    save_fig(fig, saving_path=new_save_folder, figure_name=f'{name}_indiv_mice', formats=formats)


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
                                      halfrange=0.025):
    # Keep only correct trial types in W+ context:
    wh_trial_types = ['rewarded_whisker_hit_trial']
    aud_trial_types = ['rewarded_auditory_hit_trial']

    # PLot difference
    plot_wf_timecourse_aud_wh_diff(aud_data, whisker_data, aud_trial_types, wh_trial_types,
                                   saving_path, formats=formats,
                                   diff_range=halfrange)


def dlc_psths(side_dlc, top_dlc, save_folder, name, formats=['png']):
    # JAW
    jaw_trace = 'jaw_y'  # could be 'jaw_angle'
    cols = [jaw_trace, 'time', 'mouse_id', 'session_id', 'context', 'trial_type', 'correct_choice']
    # Filter columns
    jaw_df = side_dlc[cols].copy(deep=True).reset_index(drop=True)
    # Keep only correct trial in relevant time window
    jaw_df = jaw_df.loc[(jaw_df.correct_choice == 1)
                        & (jaw_df.time >= -0.1)
                        & (jaw_df.time <= 0.2)].reset_index(drop=True)
    # Add trial id column
    jaw_df['trial_id'] = (jaw_df['time'].diff() < 0).cumsum()
    # Add baseline column
    jaw_df['baseline'] = jaw_df.groupby('trial_id')[jaw_trace].transform(
        lambda x: x[jaw_df.loc[x.index, 'time'] < 0].mean()
    )
    # Baseline subtraction to the data
    jaw_df[jaw_trace] = jaw_df[jaw_trace] - jaw_df['baseline']
    # Average within session
    jaw_df_avg = jaw_df.groupby(['time', 'mouse_id', 'session_id', 'context', 'trial_type', 'correct_choice'],
                                as_index=False).agg('mean')
    # Average within mouse
    mouse_jaw_df = jaw_df_avg.drop('session_id', axis=1).groupby(['time', 'mouse_id', 'context',
                                                                  'trial_type', 'correct_choice'],
                                                                 as_index=False).agg('mean')
    print(f"{len(mouse_jaw_df.mouse_id.unique())} mice for jaw trace")

    # WHISKERS
    cols = ['whisker_angle', 'time', 'mouse_id', 'session_id', 'context', 'trial_type',
            'correct_choice']
    # Filter columns
    whisker_df = top_dlc[cols].copy(deep=True).reset_index(drop=True)
    # Keep only correct trial in relevant time window
    whisker_df = whisker_df.loc[(whisker_df.correct_choice == 1)
                                & (whisker_df.time >= -0.1)
                                & (whisker_df.time <= 0.2)].reset_index(drop=True)
    dt = np.median(1 / np.diff(whisker_df.time.unique()))
    # Add trial id column
    whisker_df['trial_id'] = (whisker_df['time'].diff() < 0).cumsum()
    # Add baseline column
    whisker_df['baseline'] = whisker_df.groupby('trial_id')['whisker_angle'].transform(
        lambda x: x[whisker_df.loc[x.index, 'time'] < 0].mean()
    )
    # Baseline subtraction to the data
    whisker_df['whisker_angle'] = whisker_df['whisker_angle'] - whisker_df['baseline']
    # Whisker velocity:
    whisker_df['whisker_velocity'] = np.zeros_like(whisker_df['whisker_angle'])
    whisker_df.loc[1:, 'whisker_velocity'] = np.diff(whisker_df['whisker_angle'])
    whisker_df['whisker_speed'] = np.abs(whisker_df['whisker_velocity']) * dt

    # Average within session
    whisker_df_avg = whisker_df.groupby(['time', 'mouse_id', 'session_id', 'context', 'trial_type', 'correct_choice'],
                                        as_index=False).agg('mean')
    # Average within mouse
    mouse_whisker_df = whisker_df_avg.drop('session_id', axis=1).groupby(['time', 'mouse_id', 'context',
                                                                          'trial_type', 'correct_choice'],
                                                                         as_index=False).agg('mean')
    print(f"{len(mouse_jaw_df.mouse_id.unique())} mice for whisker trace")

    # Get trial type
    whttype = [ttype for ttype in mouse_jaw_df.trial_type.unique() if 'whisker' in ttype]
    audttype = [ttype for ttype in mouse_jaw_df.trial_type.unique() if 'auditory' in ttype]

    # Colors
    context_palette = ['darkmagenta', 'green']

    # FIGURE : STIM ALIGNED PSTHs
    fig, axes = plt.subplots(2, 3, figsize=(9, 6), sharex=True)
    # Jaw Y
    sns.lineplot(mouse_jaw_df.loc[mouse_jaw_df.trial_type.isin(whttype)], x='time', y=jaw_trace, hue='context',
                 hue_order=['non-rewarded', 'rewarded'], palette=context_palette, legend=False, ax=axes[0, 0])
    sns.lineplot(mouse_jaw_df.loc[mouse_jaw_df.trial_type.isin(audttype)], x='time', y=jaw_trace, hue='context',
                 hue_order=['non-rewarded', 'rewarded'], palette=context_palette, legend=False, ax=axes[1, 0])

    # Whisker angle
    sns.lineplot(mouse_whisker_df.loc[mouse_whisker_df.trial_type.isin(whttype)], x='time', y='whisker_angle',
                 hue='context', hue_order=['non-rewarded', 'rewarded'], palette=context_palette, legend=False,
                 ax=axes[0, 1])
    sns.lineplot(mouse_whisker_df.loc[mouse_whisker_df.trial_type.isin(audttype)], x='time', y='whisker_angle',
                 hue='context', hue_order=['non-rewarded', 'rewarded'], palette=context_palette, legend=False,
                 ax=axes[1, 1])

    # Whisker speed
    sns.lineplot(mouse_whisker_df.loc[(mouse_whisker_df.trial_type.isin(whttype))
                                      & (mouse_whisker_df.time > -0.1)],
                 x='time', y='whisker_speed', hue='context', hue_order=['non-rewarded', 'rewarded'],
                 palette=context_palette, legend=False, ax=axes[0, 2])
    sns.lineplot(mouse_whisker_df.loc[(mouse_whisker_df.trial_type.isin(audttype))
                                      & (mouse_whisker_df.time > -0.1)],
                 x='time', y='whisker_speed', hue='context', hue_order=['non-rewarded', 'rewarded'],
                 palette=context_palette, legend=False, ax=axes[1, 2])

    sns.despine()
    for ax in axes[0, :].flatten():
        ax.axvline(x=0, ymin=0, ymax=1, linestyle='-', c='orange')
    for ax in axes[1, :].flatten():
        ax.axvline(x=0, ymin=0, ymax=1, linestyle='-', c='blue')
    for ax in axes[:, 0].flatten():
        ax.set_ylabel('(mm)')
    for ax in axes[:, 1].flatten():
        ax.set_ylabel('(deg)')
        ax.set_ylim(-4, 12)
    for ax in axes[:, 2].flatten():
        ax.set_ylabel('(deg / sec)')
        ax.set_ylim(0, 800)
    axes[0, 0].set_title('Jaw opening')
    axes[0, 1].set_title('Whisker angle')
    axes[0, 2].set_title('Whisker speed')

    fig.tight_layout()

    save_fig(fig, saving_path=save_folder, figure_name=f'{name}', formats=formats)

    # D-prime for WHISKER trials - whisker angle and speed
    dprime_whisker = compute_dprime_over_time(
        df=mouse_whisker_df.loc[mouse_whisker_df.trial_type.isin(whttype)],
        group_cols=['time', 'context', 'trial_type', 'correct_choice'],
        value_cols=['whisker_angle', 'whisker_speed']
    )

    # D-prime for AUDITORY trials - whisker angle and speed
    dprime_auditory_whisker = compute_dprime_over_time(
        df=mouse_whisker_df.loc[mouse_whisker_df.trial_type.isin(audttype)],
        group_cols=['time', 'context', 'trial_type', 'correct_choice'],
        value_cols=['whisker_angle', 'whisker_speed']
    )

    # D-prime for WHISKER trials - jaw opening
    dprime_whisker_jaw = compute_dprime_over_time(
        df=mouse_jaw_df.loc[mouse_jaw_df.trial_type.isin(whttype)],
        group_cols=['time', 'context', 'trial_type', 'correct_choice'],
        value_cols=[jaw_trace]
    )

    # D-prime for AUDITORY trials - jaw opening
    dprime_auditory_jaw = compute_dprime_over_time(
        df=mouse_jaw_df.loc[mouse_jaw_df.trial_type.isin(audttype)],
        group_cols=['time', 'context', 'trial_type', 'correct_choice'],
        value_cols=[jaw_trace]
    )

    # FIGURE : d-primes
    fig, axes = plt.subplots(2, 3, figsize=(9, 5), sharex=True)

    # Row 0: Whisker trials
    axes[0, 0].plot(dprime_whisker_jaw['time'], dprime_whisker_jaw[f'dprime_{jaw_trace}'])
    axes[0, 0].set_ylabel(f"D' {jaw_trace}")
    axes[0, 0].set_title('Jaw opening')
    axes[0, 0].axvline(x=0, color='orange', linestyle='-')

    axes[0, 1].plot(dprime_whisker['time'], dprime_whisker['dprime_whisker_angle'])
    axes[0, 1].set_ylabel("D' whisker angle")
    axes[0, 1].set_title('Whisker angle')
    axes[0, 1].axvline(x=0, color='orange', linestyle='-')

    axes[0, 2].plot(dprime_whisker['time'], dprime_whisker['dprime_whisker_speed'])
    axes[0, 2].set_ylabel("D' whisker speed")
    axes[0, 2].set_title('Whisker speed')
    axes[0, 2].axvline(x=0, color='orange', linestyle='-')

    # Row 1: Auditory trials
    axes[1, 0].plot(dprime_auditory_jaw['time'], dprime_auditory_jaw[f'dprime_{jaw_trace}'])
    axes[1, 0].set_ylabel(f"D' {jaw_trace}")
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].axvline(x=0, color='blue', linestyle='-')

    axes[1, 1].plot(dprime_auditory_whisker['time'], dprime_auditory_whisker['dprime_whisker_angle'])
    axes[1, 1].set_ylabel("D' whisker angle")
    axes[1, 1].set_xlabel('Time (s)')
    axes[1, 1].axvline(x=0, color='blue', linestyle='-')

    axes[1, 2].plot(dprime_auditory_whisker['time'], dprime_auditory_whisker['dprime_whisker_speed'])
    axes[1, 2].set_ylabel("D' whisker speed")
    axes[1, 2].set_xlabel('Time (s)')
    axes[1, 2].axvline(x=0, color='blue', linestyle='-')

    sns.despine()

    for ax in axes.flatten():
        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.axhline(y=1, color='r', linestyle='--', alpha=0.3)
        ax.set_ylim(-1, 5)

    fig.tight_layout()

    save_fig(fig, saving_path=save_folder, figure_name=f'{name}_dprime', formats=formats)

