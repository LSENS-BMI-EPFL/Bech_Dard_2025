import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator
from codes.utils.misc.fig_saving import save_fig
from codes.utils.misc.table_saving import save_table, df_to_latex
pd.set_option("display.float_format", "{:.2e}".format)


def plot_figure1e(data_table, saving_path, name, saving_formats):
    # Drop useless columns
    data_table = data_table.drop(['day', 'context_background'], axis=1)

    # Count
    n_sess = len(data_table.session_id.unique())
    n_mice = len(data_table.mouse_id.unique())

    # Average over session for each mouse
    avg_data_table = data_table.copy()
    avg_data_table = avg_data_table.drop('session_id', axis=1)
    avg_data_table = avg_data_table.groupby(['mouse_id', 'context', 'context_rwd_str', 'artificial_day'],
                                            as_index=False).agg('mean')

    # Plot
    trials = ['outcome_a', 'outcome_w', 'outcome_n']
    c_palettes = [['cyan', 'blue'], ['darkmagenta', 'green'], ['grey', 'black']]
    pplot_size = 5
    scatter_alpha = 0.5
    scatter_size = 3
    fig_width, fig_height = 2, 2.6
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
    fig.suptitle(f'Average from {n_sess} sessions, {n_mice} mice')
    fig.tight_layout()

    save_fig(fig, saving_path, figure_name=name, formats=saving_formats)

    # Table with averaged values
    gran_average_df = avg_data_table.copy()
    gran_average_df = gran_average_df.drop('mouse_id', axis=1)
    gran_average_df = gran_average_df.groupby(['context', 'context_rwd_str', 'artificial_day'],
                                              as_index=False).agg('mean')
    gran_average_df.rename(columns={'outcome_a': 'auditory_mean',
                                    'outcome_w': 'whisker_mean',
                                    'outcome_n': 'catch_mean'}, inplace=True)

    gran_std_df = avg_data_table.copy()
    gran_std_df = gran_std_df.drop('mouse_id', axis=1)
    gran_std_df = gran_std_df.groupby(['context', 'context_rwd_str', 'artificial_day'], as_index=False).agg('std')
    gran_std_df.rename(columns={'outcome_a': 'auditory_std',
                                'outcome_w': 'whisker_std',
                                'outcome_n': 'catch_std'}, inplace=True)
    stats_df = pd.merge(gran_average_df, gran_std_df, left_on=['context_rwd_str', 'context', 'artificial_day'],
                        right_on=['context_rwd_str', 'context', 'artificial_day'], how='right')
    stats_df['N'] = len(avg_data_table.mouse_id.unique())

    # Convert to LaTeX with booktabs and improved names
    stats_df = stats_df.drop('context', axis=1)
    rename_dict = {0: 'ON',
                   1: 'OFF',
                   2: 'ON'
                   }
    stats_df['artificial_day'] = stats_df['artificial_day'].replace(rename_dict)
    stats_df.rename(columns={'context_rwd_str': 'Context',
                             'artificial_day': 'Context sound',
                             'auditory_mean': 'Auditory mean',
                             'whisker_mean': 'Whisker mean',
                             'catch_mean': 'Catch mean',
                             'auditory_std': 'Auditory std',
                             'whisker_std': 'Whisker std',
                             'catch_std': 'Catch std'}, inplace=True)

    save_table(df=stats_df, saving_path=saving_path, name='Figure1E_table', format=['csv'])
    df_to_latex(df=stats_df, filename=os.path.join(saving_path, 'Figure1E_table.tex'), caption='Figure1E', label='')
