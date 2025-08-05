import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from codes.utils.misc.fig_saving import save_fig


def plot_figure1d(data_table, saving_path, name, saving_formats):
    # Drop useless columns
    data_table = data_table.drop(['Unnamed: 0', 'day', 'context_background'], axis=1)

    # Average over session for each mouse
    avg_data_table = data_table.copy()
    avg_data_table = avg_data_table.drop('session_id', axis=1)
    avg_data_table = avg_data_table.groupby(['mouse_id', 'context', 'context_rwd_str', 'artificial_day'],
                                            as_index=False).agg(np.nanmean)

    # Plot
    trials = ['outcome_a', 'outcome_w', 'outcome_n']
    c_palettes = [['cyan', 'blue'], ['darkmagenta', 'green'], ['grey', 'black']]
    pplot_size = 5
    scatter_alpha = 0.5
    scatter_size = 3
    fig_width, fig_height = 2, 2.5
    x_label_dict = {0: 'ON', 1: 'OFF', 2: 'ON'}

    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height), sharey=True)
    for idx, trial in enumerate(trials):
        sns.pointplot(avg_data_table, x='artificial_day', y=trial, hue='context_rwd_str', palette=c_palettes[idx],
                      legend=False, ax=ax, dodge=True, markersize=pplot_size)
        sns.stripplot(avg_data_table, x='artificial_day', y=trial, hue='context_rwd_str', palette=c_palettes[idx],
                      legend=False, ax=ax, dodge=True, s=scatter_size, alpha=scatter_alpha)
        ax.set_ylabel('Lick probability')
        ax.set_xlabel('Context')
        ax.set_ylim(0, 1.05)
        sns.despine()
    positions = [tick for tick in ax.get_xticks()]
    new_label = [x_label_dict[int(i.get_text())] for i in ax.get_xticklabels()]
    ax.xaxis.set_major_locator(FixedLocator(positions))
    ax.set_xticklabels(new_label, rotation=30)
    fig.tight_layout()

    save_fig(fig, saving_path, figure_name=name, formats=saving_formats)
