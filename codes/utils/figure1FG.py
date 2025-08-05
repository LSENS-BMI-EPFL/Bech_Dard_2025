import numpy as np
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
from codes.utils.misc.fig_saving import save_fig


def plot_figure1fg(data_table, saving_path, name, saving_formats):

    color_palette = [(129 / 255, 0 / 255, 129 / 255), (0 / 255, 135 / 255, 0 / 255)]
    pplot_size = 5
    lw = 2
    fig_width, fig_height = 4, 2.35

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
    fig.tight_layout()

    save_fig(fig, saving_path, figure_name=name, formats=saving_formats)

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