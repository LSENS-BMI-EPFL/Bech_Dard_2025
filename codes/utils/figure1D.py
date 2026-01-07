import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from codes.utils.misc.fig_saving import save_fig
from codes.utils.misc.table_saving import save_table, df_to_latex
pd.set_option("display.float_format", "{:.2e}".format)


def plot_figure1d(data_table, saving_path, name, saving_formats):
    # Drop useless columns
    data_table = data_table.drop(['day', 'artificial_day', 'context_rwd_str'], axis=1)

    # Count
    n_sess = len(data_table.session_id.unique())
    n_mice = len(data_table.mouse_id.unique())

    # Average over session for each mouse
    avg_data_table = data_table.copy()
    avg_data_table = avg_data_table.drop('session_id', axis=1)
    avg_data_table = avg_data_table.groupby(['mouse_id', 'context', 'context_background'], as_index=False).agg(
        'mean')

    color_palette = ['sienna', 'pink']
    pplot_size = 5
    scatter_alpha = 0.5
    scatter_size = 3
    fig_width, fig_height = 3, 2.6

    # Plot
    trials = ['outcome_a', 'outcome_w', 'outcome_n']
    titles = ['Auditory', 'Whisker', 'Catch']
    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), sharey=True)
    for idx, trial in enumerate(trials):
        sns.pointplot(avg_data_table, y=trial, hue='context_background', palette=color_palette, legend=False,
                      ax=axes.flatten()[idx], dodge=0.7, markersize=pplot_size)
        sns.stripplot(avg_data_table, y=trial, hue='context_background', palette=color_palette, legend=False,
                      ax=axes.flatten()[idx], dodge=True, s=scatter_size, alpha=scatter_alpha)
        axes.flatten()[idx].set_title(titles[idx])
        axes.flatten()[idx].set_ylabel('Lick probability')
        axes.flatten()[idx].set_ylim(0, 1.05)
        sns.despine()
    fig.suptitle(f'Average from {n_sess} sessions, {n_mice} mice')
    fig.tight_layout()

    save_fig(fig, saving_path, figure_name=name, formats=saving_formats)

    # Table with averaged values
    gran_average_df = avg_data_table.copy()
    gran_average_df = gran_average_df.drop(['mouse_id', 'context'], axis=1)
    gran_average_df = gran_average_df.groupby(['context_background'],
                                              as_index=False).agg('mean')
    gran_average_df.rename(columns={'outcome_a': 'auditory_mean',
                                    'outcome_w': 'whisker_mean',
                                    'outcome_n': 'catch_mean'}, inplace=True)

    gran_std_df = avg_data_table.copy()
    gran_std_df = gran_std_df.drop(['mouse_id', 'context'], axis=1)
    gran_std_df = gran_std_df.groupby(['context_background'], as_index=False).agg('std')
    gran_std_df.rename(columns={'outcome_a': 'auditory_std',
                                'outcome_w': 'whisker_std',
                                'outcome_n': 'catch_std'}, inplace=True)
    stats_df = pd.merge(gran_average_df, gran_std_df, left_on=['context_background'],
                        right_on=['context_background'], how='right')
    stats_df['N'] = len(avg_data_table.mouse_id.unique())

    # Convert to LaTeX with booktabs and improved names
    stats_df.rename(columns={'context_background': 'Context sound',
                             'auditory_mean': 'Auditory mean',
                             'whisker_mean': 'Whisker mean',
                             'catch_mean': 'Catch mean',
                             'auditory_std': 'Auditory std',
                             'whisker_std': 'Whisker std',
                             'catch_std': 'Catch std'}, inplace=True)
    df_to_latex(df=stats_df, filename=os.path.join(saving_path, 'Figure1D_table.tex'), caption='Figure1D', label='')
    save_table(df=stats_df, saving_path=saving_path, name='Figure1D_table', format=['csv'])
