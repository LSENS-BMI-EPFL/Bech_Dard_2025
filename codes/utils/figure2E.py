import os
import ast
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from codes.utils.misc.plot_on_grid import plot_grid_on_allen


def figure2e(data_table, saving_path):

    data_table = data_table.drop(['stim_onset', 'lick_time'], axis=1)
    whisker_lick_df = data_table.loc[(data_table.context == 1) & (data_table.trial_type == 'whisker_trial')]
    auditory_lick_df = data_table.loc[data_table.trial_type == 'auditory_trial']

    # Group by mouse
    whisker_piezo_lick_df_mouse = whisker_lick_df.groupby(by=['mouse_id', 'trial_type', 'opto_stim_coord'], as_index=False).agg(np.nanmean)
    auditory_piezo_lick_df_mouse = auditory_lick_df.groupby(by=['mouse_id', 'trial_type', 'opto_stim_coord'], as_index=False).agg(np.nanmean)

    # Compute delta contact and dprime for each mouse
    # Whisker
    reference = whisker_piezo_lick_df_mouse[whisker_piezo_lick_df_mouse['opto_stim_coord'] == '(-5.0, 5.0)'][['mouse_id', 'rt']].rename(columns={'rt': 'ctrl_rt'})
    whisker_piezo_lick_df_mouse = whisker_piezo_lick_df_mouse.merge(reference, on='mouse_id', how='left')
    whisker_piezo_lick_df_mouse['rt_diff'] = whisker_piezo_lick_df_mouse['rt'] - whisker_piezo_lick_df_mouse['ctrl_rt']
    wh_ctrl_rt = whisker_piezo_lick_df_mouse.loc[whisker_piezo_lick_df_mouse.opto_stim_coord == '(-5.0, 5.0)', 'rt'].values
    whdprime = []
    for spot in whisker_piezo_lick_df_mouse.opto_stim_coord.unique():
        data = whisker_piezo_lick_df_mouse.loc[whisker_piezo_lick_df_mouse.opto_stim_coord == spot, 'rt'].values
        whdprime.append(np.abs(np.nanmean(wh_ctrl_rt) - np.nanmean(data)) / np.sqrt(0.5 * (np.nanstd(wh_ctrl_rt) ** 2 + np.nanstd(data) ** 2)))
    whdprime_df = pd.DataFrame({'opto_stim_coord': whisker_piezo_lick_df_mouse.opto_stim_coord.unique().tolist(), 'dprime': whdprime})

    # Auditory
    reference = auditory_piezo_lick_df_mouse[auditory_piezo_lick_df_mouse['opto_stim_coord'] == '(-5.0, 5.0)'][['mouse_id', 'rt']].rename(columns={'rt': 'ctrl_rt'})
    auditory_piezo_lick_df_mouse = auditory_piezo_lick_df_mouse.merge(reference, on='mouse_id', how='left')
    auditory_piezo_lick_df_mouse['rt_diff'] = auditory_piezo_lick_df_mouse['rt'] - auditory_piezo_lick_df_mouse['ctrl_rt']
    aud_ctrl_rt = auditory_piezo_lick_df_mouse.loc[auditory_piezo_lick_df_mouse.opto_stim_coord == '(-5.0, 5.0)', 'rt'].values
    audprime = []
    for spot in auditory_piezo_lick_df_mouse.opto_stim_coord.unique():
        data = auditory_piezo_lick_df_mouse.loc[auditory_piezo_lick_df_mouse.opto_stim_coord == spot, 'rt'].values
        audprime.append(np.abs(np.nanmean(aud_ctrl_rt) - np.nanmean(data)) / np.sqrt(0.5 * (np.nanstd(aud_ctrl_rt) ** 2 + np.nanstd(data) ** 2)))
    audprime_df = pd.DataFrame({'opto_stim_coord': auditory_piezo_lick_df_mouse.opto_stim_coord.unique().tolist(), 'dprime': audprime})

    # Average across mice for mean plot on the grid
    # Whisker
    wh_piezo_rt_df = whisker_piezo_lick_df_mouse.copy()
    wh_piezo_rt_df = wh_piezo_rt_df.drop('mouse_id', axis=1)
    wh_piezo_rt_df = wh_piezo_rt_df.groupby(by=['trial_type', 'opto_stim_coord'], as_index=False).agg(np.nanmean)

    # Auditory
    aud_piezo_rt_df = auditory_piezo_lick_df_mouse.copy()
    aud_piezo_rt_df = aud_piezo_rt_df.drop('mouse_id', axis=1)
    aud_piezo_rt_df = aud_piezo_rt_df.groupby(by=['trial_type', 'opto_stim_coord'], as_index=False).agg(np.nanmean)

    # Put this rt on the grid for each context
    seismic_palette = sns.diverging_palette(265, 10, s=100, l=40, sep=30, n=200, center="light", as_cmap=True)

    # Whisker
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    wh_piezo_rt_df['y'] = wh_piezo_rt_df['opto_stim_coord'].apply(lambda x: ast.literal_eval(x)[0])
    wh_piezo_rt_df['x'] = wh_piezo_rt_df['opto_stim_coord'].apply(lambda x: ast.literal_eval(x)[1])
    plot_grid_on_allen(wh_piezo_rt_df.loc[wh_piezo_rt_df.opto_stim_coord != '(-5.0, 5.0)'].copy(), outcome='rt_diff',
                       palette=seismic_palette, facecolor=None, edgecolor='black', result_path=None, dotsize=175,
                       vmin=-0.25, vmax=0.25, fig=fig, ax=ax)
    fig.suptitle('Whisker trials')
    fig.tight_layout()
    fig.savefig(os.path.join(saving_path, 'Figure2E_whisker_effect.png'), dpi=400)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    whdprime_df['y'] = whdprime_df['opto_stim_coord'].apply(lambda x: ast.literal_eval(x)[0])
    whdprime_df['x'] = whdprime_df['opto_stim_coord'].apply(lambda x: ast.literal_eval(x)[1])
    plot_grid_on_allen(whdprime_df.loc[whdprime_df.opto_stim_coord != '(-5.0, 5.0)'].copy(), outcome='dprime',
                       palette='binary', facecolor=None, edgecolor='black', result_path=None, dotsize=175,
                       vmin=0.5, vmax=2.2, fig=fig, ax=ax)
    fig.suptitle('Whisker trials')
    fig.tight_layout()
    fig.savefig(os.path.join(saving_path, 'Figure2E_whisker_dprime.png'), dpi=400)

    # Auditory
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    aud_piezo_rt_df['y'] = aud_piezo_rt_df['opto_stim_coord'].apply(lambda x: ast.literal_eval(x)[0])
    aud_piezo_rt_df['x'] = aud_piezo_rt_df['opto_stim_coord'].apply(lambda x: ast.literal_eval(x)[1])
    plot_grid_on_allen(aud_piezo_rt_df.loc[aud_piezo_rt_df.opto_stim_coord != '(-5.0, 5.0)'].copy(), outcome='rt_diff',
                       palette=seismic_palette, facecolor=None, edgecolor='black', result_path=None, dotsize=175,
                       vmin=-0.25, vmax=0.25, fig=fig, ax=ax)
    fig.suptitle('Auditory trials')
    fig.tight_layout()
    fig.savefig(os.path.join(saving_path, 'Figure2E_auditory_effect.png'), dpi=400)

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    audprime_df['y'] = audprime_df['opto_stim_coord'].apply(lambda x: ast.literal_eval(x)[0])
    audprime_df['x'] = audprime_df['opto_stim_coord'].apply(lambda x: ast.literal_eval(x)[1])
    plot_grid_on_allen(audprime_df.loc[audprime_df.opto_stim_coord != '(-5.0, 5.0)'].copy(), outcome='dprime',
                       palette='binary', facecolor=None, edgecolor='black', result_path=None, dotsize=175,
                       vmin=0.5, vmax=2.2, fig=fig, ax=ax)
    fig.suptitle('Auditory trials')
    fig.tight_layout()
    fig.savefig(os.path.join(saving_path, 'Figure2E_auditory_dprime.png'), dpi=400)

    plt.close('all')

