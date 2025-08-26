import os
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from codes.utils.misc.fig_saving import save_fig


def figure3j(auditory_df, whisker_df, saving_path, name, formats=['png', 'svg']):
    t_start = 0.05
    t_stop = 0.12

    # DROP
    auditory_df = auditory_df.drop(['event', 'roi', 'behavior_type', 'behavior_day'], axis=1)
    whisker_df = whisker_df.drop(['event', 'roi', 'behavior_type', 'behavior_day'], axis=1)

    # DATA:
    # Whisker
    wh_session_df = whisker_df.groupby(['time', 'mouse_id', 'session_id', 'cell_type', 'trial_type', 'epoch'],
                                       as_index=False).agg(np.nanmean)
    wh_mouse_df = wh_session_df.copy()
    wh_mouse_df = wh_mouse_df.drop(['session_id'], axis=1)
    wh_mouse_df = wh_mouse_df.groupby(['time', 'mouse_id', 'cell_type', 'trial_type', 'epoch'],
                                      as_index=False).agg(np.nanmean)
    wh_coord_mouse_df = wh_mouse_df.copy(deep=True)
    wh_coord_mouse_df['AP'] = wh_mouse_df['cell_type'].apply(lambda x: ast.literal_eval(x)[0])
    wh_coord_mouse_df['ML'] = wh_mouse_df['cell_type'].apply(lambda x: ast.literal_eval(x)[1])
    correct_wh_df = wh_coord_mouse_df.loc[
        ((wh_coord_mouse_df.trial_type == 'whisker_hit_trial') & (wh_coord_mouse_df.epoch == 'rewarded')) | (
                    (wh_coord_mouse_df.trial_type == 'whisker_miss_trial') & (
                        wh_coord_mouse_df.epoch == 'non-rewarded'))]
    # Whisker peak
    wh_avg_table_response = correct_wh_df.loc[(correct_wh_df.time > t_start) & (correct_wh_df.time < t_stop)]
    wh_max_response_table = wh_avg_table_response.loc[
        wh_avg_table_response.groupby(['mouse_id', 'cell_type', 'trial_type', 'epoch', 'AP', 'ML'])[
            "activity"].idxmax()]

    # Auditory
    aud_session_df = auditory_df.groupby(['time', 'mouse_id', 'session_id', 'cell_type', 'trial_type', 'epoch'],
                                         as_index=False).agg(np.nanmean)
    aud_mouse_df = aud_session_df.copy()
    aud_mouse_df = aud_mouse_df.drop(['session_id'], axis=1)
    aud_mouse_df = aud_mouse_df.groupby(['time', 'mouse_id', 'cell_type', 'trial_type', 'epoch'],
                                        as_index=False).agg(np.nanmean)
    aud_coord_mouse_df = aud_mouse_df.copy(deep=True)
    aud_coord_mouse_df['AP'] = aud_coord_mouse_df['cell_type'].apply(lambda x: ast.literal_eval(x)[0])
    aud_coord_mouse_df['ML'] = aud_coord_mouse_df['cell_type'].apply(lambda x: ast.literal_eval(x)[1])
    correct_aud_df = aud_coord_mouse_df.loc[aud_coord_mouse_df.trial_type == 'auditory_hit_trial']
    aud_avg_table_response = correct_aud_df.loc[(correct_aud_df.time > t_start) & (correct_aud_df.time < t_stop)]
    aud_max_response_table = aud_avg_table_response.loc[
        aud_avg_table_response.groupby(['mouse_id', 'cell_type', 'trial_type', 'epoch', 'AP', 'ML'])[
            "activity"].idxmax()]

    full_peak_df = pd.concat([wh_max_response_table, aud_max_response_table], ignore_index=True)
    selected_spots = ['(-2.5, 5.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 1.5)', '(-1.5, 0.5)', '(2.5, 2.5)',
                      '(0.5, 4.5)']
    sel_full_peak_df = full_peak_df.loc[full_peak_df.cell_type.isin(selected_spots)]

    colors = {'(-1.5, 0.5)': 'pink',
              '(-1.5, 3.5)': 'darkorange',
              '(-1.5, 4.5)': 'orange',
              '(0.5, 4.5)': 'red',
              '(1.5, 1.5)': 'blue',
              '(2.5, 2.5)': 'mediumorchid',
              '(-2.5, 5.5)': 'cyan'}

    # By context
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True, sharex=True)
    for idx, context in enumerate(list(sel_full_peak_df.epoch.unique())):
        whisker_trial = 'whisker_hit_trial' if context == 'rewarded' else 'whisker_miss_trial'
        for c_idx, roi in enumerate(list(sel_full_peak_df.cell_type.unique())):
            color = colors.get(roi)
            xval = sel_full_peak_df.loc[(sel_full_peak_df.epoch == context) & (sel_full_peak_df.cell_type == roi) & (
                        sel_full_peak_df.trial_type == 'auditory_hit_trial'), 'activity'].values[:]
            yval = sel_full_peak_df.loc[(sel_full_peak_df.epoch == context) & (sel_full_peak_df.cell_type == roi) & (
                        sel_full_peak_df.trial_type == whisker_trial), 'activity'].values[:]
            axes.flatten()[idx].scatter(xval, yval, color=color, label=roi)
            axes.flatten()[idx].plot(np.arange(0, 0.055, 1 / 100), np.arange(0, 0.055, 1 / 100), c='k', linestyle='--')
            axes.flatten()[idx].spines[['top', 'right']].set_visible(False)
            axes.flatten()[idx].set_title(f'{"W-" if "non" in context else "W+"}')
            axes.flatten()[idx].set_xlim(0, 0.055)
            axes.flatten()[idx].set_ylim(0, 0.055)
            axes.flatten()[idx].set_yticks(np.arange(0, 0.06, 1 / 100))
            axes.flatten()[idx].set_xticks(np.arange(0, 0.06, 1 / 100))
    for ax in axes.flatten():
        ax.set_xlabel('Auditory')
        ax.set_ylabel('Whisker')
    fig.tight_layout()

    save_fig(fig, saving_path=saving_path, figure_name=f'{name}_context', formats=formats)

    # By trial type
    fig, axes = plt.subplots(1, 2, figsize=(6, 3), sharey=True, sharex=True)
    for c_idx, roi in enumerate(list(sel_full_peak_df.cell_type.unique())):
        color = colors.get(roi)
        xval = sel_full_peak_df.loc[(sel_full_peak_df.epoch == 'non-rewarded') & (sel_full_peak_df.cell_type == roi) & (
                    sel_full_peak_df.trial_type == 'auditory_hit_trial'), 'activity'].values[:]
        yval = sel_full_peak_df.loc[(sel_full_peak_df.epoch == 'rewarded') & (sel_full_peak_df.cell_type == roi) & (
                    sel_full_peak_df.trial_type == 'auditory_hit_trial'), 'activity'].values[:]
        axes.flatten()[0].scatter(xval, yval, color=color)
        axes.flatten()[0].set_title(f'Auditory')
        xval = sel_full_peak_df.loc[(sel_full_peak_df.epoch == 'non-rewarded') & (sel_full_peak_df.cell_type == roi) & (
                    sel_full_peak_df.trial_type == 'whisker_miss_trial'), 'activity'].values[:]
        yval = sel_full_peak_df.loc[(sel_full_peak_df.epoch == 'rewarded') & (sel_full_peak_df.cell_type == roi) & (
                    sel_full_peak_df.trial_type == 'whisker_hit_trial'), 'activity'].values[:]
        axes.flatten()[1].scatter(xval, yval, color=color, label=roi)
        axes.flatten()[1].set_title(f'Whisker')
    for idx, ax in enumerate(axes.flatten()):
        ax.set_xlabel('W-')
        ax.set_ylabel('W+')
        ax.plot(np.arange(0, 0.055, 1 / 100), np.arange(0, 0.055, 1 / 100), c='k', linestyle='--')
        ax.set_xlim(0, 0.055)
        ax.set_ylim(0, 0.055)
        axes.flatten()[idx].set_yticks(np.arange(0, 0.06, 1 / 100))
        axes.flatten()[idx].set_xticks(np.arange(0, 0.06, 1 / 100))
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()

    save_fig(fig, saving_path=saving_path, figure_name=f'{name}_trials', formats=formats)

