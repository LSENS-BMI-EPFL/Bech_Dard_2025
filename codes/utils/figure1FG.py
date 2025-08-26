import os
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
from codes.utils.misc.fig_saving import save_fig
from codes.utils.misc.table_saving import save_table, df_to_latex


def plot_figure1fg(data_table, saving_path, name, saving_formats):

    color_palette = [(129 / 255, 0 / 255, 129 / 255), (0 / 255, 135 / 255, 0 / 255)]
    pplot_size = 5
    lw = 2
    fig_width, fig_height = 4, 2.6

    # Plot
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(fig_width, fig_height), sharey=True, width_ratios=[2, 1])
    sns.pointplot(data_table, y='WHR', x='Whisker_trial', hue='Transition', palette=color_palette, legend=False, ax=ax0,
                  dodge=False, markersize=pplot_size, linewidth=lw)
    for mouse in data_table.Mouse_ID.unique():
        mouse_df = data_table.loc[data_table.Mouse_ID == mouse]
        sns.pointplot(mouse_df.loc[mouse_df.Whisker_trial.isin([-1, 1])], y='WHR', x='Whisker_trial', hue='Transition',
                      palette=color_palette, legend=False, linestyle='-', linewidth=lw, dodge=0.2, ax=ax1)
    sns.despine()
    positions = [tick for tick in ax0.get_xticks()]
    ax0.axvline(x=np.median(positions), ymin=0, ymax=1, c='k', linestyle='--')
    positions = [tick for tick in ax1.get_xticks()]
    ax1.axvline(x=np.median(positions), ymin=0, ymax=1, c='k', linestyle='--')
    ax0.set_ylim(0, 1.05)
    ax0.set_ylabel('Lick probability')
    ax0.set_xlabel('Whisker trial')
    ax1.set_xlabel('Whisker trial')
    fig.suptitle(f'Average from {len(data_table.Mouse_ID.unique())} mice')
    fig.tight_layout()

    save_fig(fig, saving_path, figure_name=name, formats=saving_formats)

    # Table
    avg_data_table = data_table.copy().drop('Mouse_ID', axis=1).groupby(['Transition', 'Whisker_trial'],
                                                                        as_index=False).agg(['mean', 'std'])
    rename_dict = {'to_non_rewarded': 'to W-',
                   'to_rewarded': 'to W+',
                   }
    avg_data_table['Transition'] = avg_data_table['Transition'].replace(rename_dict)
    avg_data_table['N'] = len(data_table.Mouse_ID.unique())
    avg_data_table.rename(columns={'Whisker_trial': 'Whisker trial index'}, inplace=True)
    df_to_latex(df=avg_data_table, filename=os.path.join(saving_path, 'Figure1F_table.tex'),
                caption='Figure1F', label='')

    # Statistics:
    to_rewarded_pre = data_table.loc[(data_table.Whisker_trial == -1) &
                                     (data_table.Transition == 'to_rewarded')].WHR.values[:]
    to_rewarded_post = data_table.loc[(data_table.Whisker_trial == 1) &
                                      (data_table.Transition == 'to_rewarded')].WHR.values[:]
    to_nn_rewarded_pre = data_table.loc[(data_table.Whisker_trial == -1) &
                                        (data_table.Transition == 'to_non_rewarded')].WHR.values[:]
    to_nn_rewarded_post = data_table.loc[(data_table.Whisker_trial == 1) &
                                         (data_table.Transition == 'to_non_rewarded')].WHR.values[:]

    to_rewarded_p = st.ttest_rel(to_rewarded_post, to_rewarded_pre)
    to_nn_rewarded_p = st.ttest_rel(to_nn_rewarded_post, to_nn_rewarded_pre)

    print(f'To R+ p-val: {to_rewarded_p}')
    print(f'To R- p-val: {to_nn_rewarded_p}')

    print(
        f"W- to W+ : pre {np.round(np.nanmean(to_rewarded_pre) * 100, 1)} +/- {np.round(np.nanstd(to_rewarded_pre) * 100, 1)}, "
        f"post {np.round(np.nanmean(to_rewarded_post) * 100, 1)} +/- {np.round(np.nanstd(to_rewarded_post) * 100, 1)}, "
        f"W+ to W- : pre {np.round(np.nanmean(to_nn_rewarded_pre) * 100, 1)} +/- {np.round(np.nanstd(to_nn_rewarded_pre) * 100, 1)}, "
        f"post {np.round(np.nanmean(to_nn_rewarded_post) * 100, 1)} +/- {np.round(np.nanstd(to_nn_rewarded_post) * 100, 1)}")

    stats_dict = {
        'Transition': ['to W+', 'to W-'],
        'mean-pre': [np.nanmean(to_rewarded_pre), np.nanmean(to_nn_rewarded_pre)],
        'mean-post': [np.nanmean(to_rewarded_post), np.nanmean(to_nn_rewarded_post)],
        'std-pre': [np.nanstd(to_rewarded_pre), np.nanstd(to_nn_rewarded_pre)],
        'std-post': [np.nanstd(to_rewarded_post), np.nanstd(to_nn_rewarded_post)],
        'N': [len(to_rewarded_post), len(to_nn_rewarded_post)],
        'T': [to_rewarded_p[0], to_nn_rewarded_p[0]],
        'p': [to_rewarded_p[1], to_nn_rewarded_p[1]],
        'pcorr': [to_rewarded_p[1] * 2, to_nn_rewarded_p[1] * 2],
        'significant': [p < 0.05 for p in [to_rewarded_p[1] * 2, to_nn_rewarded_p[1] * 2]]
    }
    stats_table = pd.DataFrame.from_dict(stats_dict)
    save_table(stats_table, saving_path, f'{name}_stat_results', format=['csv'])

    # Convert to LaTeX with booktabs and improved names
    pd.set_option("display.float_format", "{:.2e}".format)
    df_to_latex(df=stats_table, filename=os.path.join(saving_path, 'Figure1G_table.tex'),
                caption='Figure1G', label='', form={col: lambda x: f"${x:.2e}$" for col in ['p', 'pcorr']})

