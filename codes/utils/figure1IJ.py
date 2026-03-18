import os
import seaborn as sns
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel
import matplotlib.pyplot as plt
from codes.utils.misc.fig_saving import save_fig
from codes.utils.misc.table_saving import save_table


def plot_baseline_differences(side_dlc_data, top_dlc_data, save_path, supp_save_path, figname,
                              fig_formats=['png', 'svg']):

    s_path = os.path.join(supp_save_path, 'figure1_supp2')
    if not os.path.exists(s_path):
        os.makedirs(s_path)

    print('Data formatting')
    uncentered_combined_side_data = side_dlc_data.copy(deep=True)
    uncentered_combined_top_data = top_dlc_data.copy(deep=True)

    # DATA : processing
    uncentered_combined_side_data['jaw_angle'] = 90 - uncentered_combined_side_data['jaw_angle']
    uncentered_combined_side_data['trial_count'] = (uncentered_combined_side_data['time'].diff().abs() > 1).cumsum()
    uncentered_combined_side_data['jaw_speed'] = uncentered_combined_side_data.groupby(
        by=['trial_count']).apply(
        lambda x: np.pad(abs(np.diff(x.jaw_y.to_numpy())), (1, 0), 'constant',
                         constant_values=np.nan)).explode().reset_index()[0].values
    uncentered_combined_side_data['jaw_speed'] = uncentered_combined_side_data['jaw_speed'] * 200

    uncentered_combined_top_data['whisker_speed'] = uncentered_combined_top_data['whisker_velocity'].abs() * 200
    uncentered_combined_top_data['trial_count'] = (uncentered_combined_top_data['time'].diff().abs() > 1).cumsum()

    # DATA : time selection (quiet window)
    uncentered_combined_side_data = uncentered_combined_side_data[(uncentered_combined_side_data.time < 0)]
    uncentered_combined_top_data = uncentered_combined_top_data[(uncentered_combined_top_data.time < 0)]

    # DATA : correct choice
    uncentered_combined_top_data['correct_choice'] = uncentered_combined_top_data['correct_choice'].astype(bool)
    uncentered_combined_side_data['correct_choice'] = uncentered_combined_side_data['correct_choice'].astype(bool)

    # DATA : readable legends
    uncentered_combined_side_data['legend'] = uncentered_combined_side_data.apply(
        lambda x: f"{x.context} - {'correct' if x.correct_choice == 1 else 'incorrect'}", axis=1)
    uncentered_combined_side_data['stim_type'] = uncentered_combined_side_data.apply(
        lambda x: x.trial_type.split("_")[0], axis=1)
    uncentered_combined_side_data = uncentered_combined_side_data.loc[
        uncentered_combined_side_data.trial_type.str.contains('trial')]

    uncentered_combined_top_data['legend'] = uncentered_combined_top_data.apply(
        lambda x: f"{x.context} - {'correct' if x.correct_choice == 1 else 'incorrect'}", axis=1)
    uncentered_combined_top_data['stim_type'] = uncentered_combined_top_data.apply(lambda x: x.trial_type.split("_")[0],
                                                                                   axis=1)
    uncentered_combined_top_data = uncentered_combined_top_data.loc[
        uncentered_combined_top_data.trial_type.str.contains('trial')]

    # DATA : final average and merge side and top
    data = uncentered_combined_side_data.groupby(
        by=['mouse_id', 'session_id', 'context', 'context_background', 'trial_type', 'correct_choice', 'legend',
            'stim_type', 'trial_count']).agg(
        {'jaw_y': 'mean', 'jaw_speed': 'mean', 'pupil_area': 'mean'}).reset_index()

    data = data.merge(uncentered_combined_top_data.groupby(
        by=['mouse_id', 'session_id', 'context', 'context_background', 'trial_type', 'correct_choice', 'legend',
            'stim_type', 'trial_count']).agg({'whisker_angle': 'mean', 'whisker_speed': 'mean'}).reset_index()[
                          ['trial_count', 'whisker_angle', 'whisker_speed']], on='trial_count')

    data = data.melt(
        id_vars=['mouse_id', 'session_id', 'context', 'trial_type', 'correct_choice', 'legend', 'stim_type',
                 'trial_count'], value_vars=['jaw_y', 'jaw_speed', 'pupil_area', 'whisker_angle', 'whisker_speed'],
        var_name='bodypart')

    data['correct_choice'] = data.correct_choice.astype(bool)
    data['lick'] = data['legend'].map(
        {'non-rewarded - incorrect': 1, 'non-rewarded - correct': 0, 'rewarded - correct': 1,
         'rewarded - incorrect': 0}).astype(bool)

    # DATA : only whisker trials
    data = data[data.stim_type == 'whisker']
    n_comparisons = 20

    # Ensure 'value' column is float throughout
    data['value'] = pd.to_numeric(data['value'], errors='coerce')

    # CONTEXT EFFECT
    print(' ')
    print('Context effect ... ')
    context_data = data.drop(['trial_type', 'correct_choice', 'legend', 'stim_type', 'lick'], axis=1).groupby(
        by=['mouse_id', 'session_id', 'context', 'bodypart'], as_index=False).agg('mean')
    context_data = context_data.drop('session_id', axis=1).groupby(
        by=['mouse_id', 'context', 'bodypart'], as_index=False).agg('mean')

    # Stats context general effect
    stats = []
    for name, group in context_data.groupby(by='bodypart'):
        correct = group.loc[group.context == 'rewarded'].dropna()
        incorrect = group.loc[group.context == 'non-rewarded'].dropna()
        if correct.shape[0] != incorrect.shape[0]:
            correct = correct[correct.mouse_id.isin(incorrect.mouse_id)]
        correct_vals = correct['value'].values.astype(float)
        incorrect_vals = incorrect['value'].values.astype(float)

        t, p = ttest_rel(correct_vals, incorrect_vals)
        results = {
            'bodypart': name,
            'dof': correct.mouse_id.unique().shape[0] - 1,
            'mean_correct': correct['value'].mean(),
            'std_correct': correct['value'].std(),
            'mean_incorrect': incorrect['value'].mean(),
            'std_incorrect': incorrect['value'].std(),
            't': t,
            'p': np.round(p, 8),
            'p_corr': p * n_comparisons,
            'alpha': 0.05,
            'alpha_corr': 0.05 / n_comparisons,
            'significant': p * n_comparisons < 0.05,
            'd_prime': abs((correct['value'].mean() - incorrect['value'].mean())) / np.std(
                correct['value'].to_numpy() - incorrect['value'].to_numpy())
        }
        stats += [results]
    stats = pd.DataFrame(stats)
    save_table(stats, s_path, 'Figure1_supp2BC_context_stats')
    print('Stats done')

    # Figure 1 - figure supplement 2BC context
    fig, axes = plt.subplots(1, len(context_data.bodypart.unique()), figsize=(12, 3))
    for ax, part in zip(axes.flat, context_data.bodypart.unique()):
        subset = context_data[context_data.bodypart == part].dropna()
        ax.set_title(f"{part} \nn = {stats.loc[(stats.bodypart == part), 'dof'].to_numpy()[0] + 1}")
        ax.spines[['top', 'right']].set_visible(False)

        g = sns.pointplot(subset,
                          x='context',
                          y='value',
                          hue='context',
                          legend=False,
                          order=['non-rewarded', 'rewarded'],
                          palette=['#6E188A', '#348A18'],
                          estimator='mean',
                          errorbar=('ci', 95),
                          markers='o',
                          linestyle='none',
                          dodge=True,
                          ax=ax
                          )

        pivoted = subset.pivot(index='mouse_id', columns='context', values='value')
        pivoted = pivoted.dropna()
        for _, row in pivoted.iterrows():
            ax.plot([0.1, 0.9], row.values, color='gray', alpha=0.4, linewidth=3)

        if stats.loc[stats.bodypart == part, 'significant'].any():
            star_loc = max(ax.get_ylim())
            ax.scatter(.5,
                       stats.loc[(stats.bodypart == part), 'significant'].map({True: 1}).to_numpy() * star_loc * 0.9,
                       marker='*', s=100, c='k')

        ax.margins(x=0.25)
    fig.tight_layout()
    save_fig(fig, s_path, figname + '_supp2BC_context', fig_formats)
    print('Plots done')

    # LICK EFFECT
    print(' ')
    print('Lick effect ...')
    lick_data = data.drop(['trial_type', 'correct_choice', 'legend', 'stim_type', 'context'], axis=1).groupby(
        by=['mouse_id', 'session_id', 'lick', 'bodypart'], as_index=False).agg('mean')
    lick_data = lick_data.drop('session_id', axis=1).groupby(
        by=['mouse_id', 'lick', 'bodypart'], as_index=False).agg('mean')

    # Stats lick / no-lick general effect
    stats = []
    for name, group in lick_data.groupby(by='bodypart'):
        correct = group.loc[group.lick == True].dropna()
        incorrect = group.loc[group.lick == False].dropna()
        if correct.shape[0] != incorrect.shape[0]:
            correct = correct[correct.mouse_id.isin(incorrect.mouse_id)]
        correct_vals = correct['value'].values.astype(float)
        incorrect_vals = incorrect['value'].values.astype(float)

        t, p = ttest_rel(correct_vals, incorrect_vals)
        results = {
            'bodypart': name,
            'dof': correct.mouse_id.unique().shape[0] - 1,
            'mean_correct': correct['value'].mean(),
            'std_correct': correct['value'].std(),
            'mean_incorrect': incorrect['value'].mean(),
            'std_incorrect': incorrect['value'].std(),
            't': t,
            'p': np.round(p, 8),
            'p_corr': p * n_comparisons,
            'alpha': 0.05,
            'alpha_corr': 0.05 / n_comparisons,
            'significant': p * n_comparisons < 0.05,
            'd_prime': abs((correct['value'].mean() - incorrect['value'].mean())) / np.std(
                correct['value'].to_numpy() - incorrect['value'].to_numpy())
        }
        stats += [results]
    stats = pd.DataFrame(stats)
    save_table(stats, s_path, 'Figure1_supp2BC_lick_stats')
    print('Stats done')

    fig, axes = plt.subplots(1, len(lick_data.bodypart.unique()), figsize=(12, 3))
    for ax, part in zip(axes.flat, lick_data.bodypart.unique()):
        subset = lick_data[lick_data.bodypart == part].dropna()
        ax.set_title(f"{part} \nn = {stats.loc[(stats.bodypart == part), 'dof'].to_numpy()[0] + 1}")
        ax.spines[['top', 'right']].set_visible(False)

        g = sns.pointplot(subset,
                          x='lick',
                          y='value',
                          hue='lick',
                          legend=False,
                          order=[False, True],
                          palette=['#a0a0a0', '#000000'],
                          estimator='mean',
                          errorbar=('ci', 95),
                          markers='o',
                          linestyle='none',
                          dodge=True,
                          ax=ax
                          )

        pivoted = subset.pivot(index='mouse_id', columns='lick', values='value')
        pivoted = pivoted.dropna()
        for _, row in pivoted.iterrows():
            ax.plot([0.1, 0.9], row.values, color='gray', alpha=0.4, linewidth=3)

        if stats.loc[stats.bodypart == part, 'significant'].any():
            star_loc = max(ax.get_ylim())
            ax.scatter(.5,
                       stats.loc[(stats.bodypart == part), 'significant'].map({True: 1}).to_numpy() * star_loc * 0.9,
                       marker='*', s=100, c='k')

        ax.margins(x=0.25)
    fig.tight_layout()
    save_fig(fig, s_path, figname + '_supp2BC_lick', fig_formats)
    print('Plots done')

    # CONTEXT - LICK INTERACTION EFFECT
    print(' ')
    print('Context - Lick interaction effect ...')
    lick_vs_context_data = data.drop(['trial_type', 'correct_choice', 'legend', 'stim_type'], axis=1).groupby(
        by=['mouse_id', 'session_id', 'context', 'lick', 'bodypart'], as_index=False).agg('mean')
    lick_vs_context_data = lick_vs_context_data.drop('session_id', axis=1).groupby(
        by=['mouse_id', 'context', 'lick', 'bodypart'], as_index=False).agg('mean')
    lick_vs_context_data['legend'] = lick_vs_context_data.apply(
        lambda x: f"{x.context} - {'lick' if x.lick else 'no-lick'}", axis=1)

    stats = []
    for name, group in lick_vs_context_data.groupby(by=['bodypart', 'context']):
        correct = group.loc[group.lick == True].dropna()
        incorrect = group.loc[group.lick == False].dropna()
        if correct.shape[0] != incorrect.shape[0]:
            correct = correct[correct.mouse_id.isin(incorrect.mouse_id)]
        correct_vals = correct['value'].values.astype(float)
        incorrect_vals = incorrect['value'].values.astype(float)

        t, p = ttest_rel(correct_vals, incorrect_vals)
        results = {
            'bodypart': name[0],
            'context': name[1],
            'dof': correct.mouse_id.unique().shape[0] - 1,
            'mean_correct': correct['value'].mean(),
            'std_correct': correct['value'].std(),
            'mean_incorrect': incorrect['value'].mean(),
            'std_incorrect': incorrect['value'].std(),
            't': t,
            'p': np.round(p, 8),
            'p_corr': p * n_comparisons,
            'alpha': 0.05,
            'alpha_corr': 0.05 / n_comparisons,
            'significant': p * n_comparisons < 0.05,
            'd_prime': abs((correct['value'].mean() - incorrect['value'].mean())) / np.std(
                correct_vals - incorrect_vals)
        }
        stats += [results]
    stats = pd.DataFrame(stats)
    save_table(stats, save_path, 'Figure1IJ_stats')
    print('Stats done')

    palette = {'non-rewarded - no-lick': '#C5A2D0',
               'non-rewarded - lick': '#6E188A',
               'rewarded - no-lick': '#ADD0A2',
               'rewarded - lick': '#348A18'}

    # Define the fixed categorical order for the x-axis
    legend_order = ['non-rewarded - no-lick', 'non-rewarded - lick', 'rewarded - no-lick', 'rewarded - lick']

    reference = 'non-rewarded - no-lick'
    norm_df = []
    for i, row in lick_vs_context_data.iterrows():
        mouse_id = row.mouse_id
        bodypart = row.bodypart
        ref_val = lick_vs_context_data.loc[
            (lick_vs_context_data.mouse_id == mouse_id) &
            (lick_vs_context_data.bodypart == bodypart) &
            (lick_vs_context_data.legend == reference), 'value'].to_numpy()
        if len(ref_val) > 0:
            row = row.copy()
            row['value'] = float(row['value']) - float(ref_val[0])
        norm_df += [row]
    norm_df = pd.DataFrame(norm_df)
    norm_df['value'] = pd.to_numeric(norm_df['value'], errors='coerce')

    lick_vs_context_data = norm_df

    fig, axes = plt.subplots(1, len(lick_vs_context_data.bodypart.unique()), figsize=(12, 3))
    for ax, part in zip(axes.flat, lick_vs_context_data.bodypart.unique()):
        ax.set_title(f"{part} \nn = {stats.loc[(stats.bodypart == part), 'dof'].unique()[0] + 1}")
        ax.margins(x=0.25)
        ax.spines[['top', 'right']].set_visible(False)

        subset = lick_vs_context_data[lick_vs_context_data.bodypart == part].dropna()
        subset = subset.copy()
        subset['legend'] = subset.apply(lambda x: f"{x.context} - {'lick' if x.lick else 'no-lick'}", axis=1)

        sns.pointplot(subset,
                      x='legend',
                      y='value',
                      hue='legend',
                      order=legend_order,
                      hue_order=legend_order,
                      palette=palette,
                      estimator='mean',
                      errorbar=('ci', 95),
                      markers='o',
                      linestyle='none',
                      dodge=False,
                      ax=ax
                      )
        legend = ax.get_legend()
        if legend is not None:
            legend.set_visible(False)
        ax.set_xlabel('')
        ax.set_xticklabels([])

        # Use string category labels for ax.plot and ax.scatter (categorical x-axis)
        for c in lick_vs_context_data.context.unique():
            no_lick_label = f"{c} - no-lick"
            lick_label = f"{c} - lick"

            pivoted = subset.loc[subset.context == c].pivot(index='mouse_id', columns='lick', values='value')
            pivoted = pivoted.dropna()
            for _, row in pivoted.iterrows():
                ax.plot([no_lick_label, lick_label], row.values, color='gray', alpha=0.4, linewidth=3)

            is_significant = stats.loc[
                (stats.bodypart == part) & (stats.context == c), 'significant'
            ].values
            if len(is_significant) > 0 and is_significant[0]:
                star_loc = max(ax.get_ylim()) * 0.9
                # Place star between the two labels for this context
                ax.annotate('*', xy=(lick_label, star_loc),
                            ha='center', va='bottom', fontsize=14, color='k')

    fig.tight_layout()
    save_fig(fig, save_path, figname + 'IJ', fig_formats)
    print('Plots done')
