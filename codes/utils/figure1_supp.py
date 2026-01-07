import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from codes.utils.misc.fig_saving import save_fig
from codes.utils.misc.table_saving import save_table


def plot_figure1_supp1a(data_table, saving_path, name, saving_formats):
    colors = ['green' if data_table.w_context_expert.iloc[i] else 'grey' for i in range(len(data_table))]
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.scatter(data_table.w_contrast_mean.values, data_table.d_prime.values, color=colors)
    ax.spines[['top', 'right']].set_visible(False)
    ax.axvline(x=0.375, ymin=0, ymax=1, c='r', linestyle='--')
    ax.set_xlabel('Contrast')
    ax.set_ylabel("d'")
    fig.tight_layout()

    save_fig(fig, saving_path, figure_name=name, formats=saving_formats)


def plot_figure1_supp1b(data_table, saving_path, name, saving_formats):
    # Block histogram
    count_df = data_table.drop(['epoch length'], axis=1).groupby(['mouse_id', 'session_id'], as_index=False).count()

    # Histogram
    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    ax.hist(count_df.epoch.values, bins=np.arange(0, 50, 2), color='k')
    ax.spines[['top', 'right']].set_visible(False)
    ax.axvline(x=np.nanmean(count_df.epoch.values), ymin=0, ymax=1, c='r', linestyle='--')
    ax.set_xlabel('Block count')
    ax.set_ylabel("Session count")
    fig.tight_layout()

    save_fig(fig, saving_path, figure_name=name, formats=saving_formats)


def plot_figure1_supp1c(data_table, saving_path, name, saving_formats):
    # Block duration
    session_df = data_table.groupby(['mouse_id', 'session_id', 'epoch'], as_index=False).agg('mean')
    mouse_df = session_df.copy()
    mouse_df = mouse_df.drop(['session_id'], axis=1)
    mouse_df = mouse_df.groupby(['mouse_id', 'epoch'], as_index=False).agg('mean')

    # Duration
    color_palette = [(129 / 255, 0 / 255, 129 / 255), (0 / 255, 135 / 255, 0 / 255)]
    pplot_size = 10
    scatter_size = 5
    scatter_alpha = 0.5

    # Duration stats
    rew_d = mouse_df.loc[mouse_df.epoch == 'rewarded', 'epoch length'].values[:]
    nnrew_d = mouse_df.loc[mouse_df.epoch == 'non-rewarded', 'epoch length'].values[:]
    stat_res = st.ttest_rel(rew_d, nnrew_d)
    duration_stats = {'W+ mean': np.nanmean(rew_d),
                      'W+ std': np.nanstd(rew_d),
                      'W- mean': np.nanmean(nnrew_d),
                      'W- std': np.nanstd(nnrew_d),
                      'T': stat_res[0],
                      'pval': stat_res[1],
                      'd-prime': (np.nanmean(rew_d) - np.nanmean(nnrew_d)) / np.sqrt(
                          0.5 * (np.nanstd(rew_d) ** 2 + np.nanstd(nnrew_d) ** 2))
                      }
    stats_df = pd.DataFrame(duration_stats, index=[0])
    stats_df['Significant'] = [True if val < 0.05 else False for val in stats_df['pval'].values.tolist()]
    save_table(df=stats_df, saving_path=saving_path, name=f'{name}_stats', format=['csv'])

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    sns.pointplot(mouse_df, y='epoch length', hue='epoch', palette=color_palette, legend=False, ax=ax, dodge=0.7,
                  markersize=pplot_size)
    sns.stripplot(mouse_df, y='epoch length', hue='epoch', palette=color_palette, legend=False, ax=ax, dodge=True,
                  s=scatter_size, alpha=scatter_alpha)
    if stat_res[1] < 0.05:
        ax.plot(0, 1.02 * max(mouse_df['epoch length']), '*', c='k', markersize=10)
    ax.set_ylim(160, 240)
    ax.set_ylabel('Block duration (s)')
    ax.set_xlabel('Context')
    sns.despine()
    fig.tight_layout()

    save_fig(fig, saving_path, figure_name=name, formats=saving_formats)


def plot_figure1_supp1d(data_table, saving_path, name, saving_formats):
    # Plot
    color_palette = [(129 / 255, 0 / 255, 129 / 255), (0 / 255, 135 / 255, 0 / 255)]
    pplot_size = 5
    lw = 2

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(3.35, 2.25), sharey=True, width_ratios=[2, 1])
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


def plot_figure1_supp1e(data_table, saving_path, name, saving_formats):
    stim_data_table = data_table.loc[data_table.trial_type.isin(["auditory_trial", "whisker_trial"])]
    avg_df = stim_data_table.groupby(['mouse_id', 'trial_type', 'context'], as_index=False).agg('mean')

    # Stats
    ttypes = []
    mean_r = []
    mean_nnr = []
    std_r = []
    std_nnr = []
    dprime = []
    tval = []
    pval = []

    for ttype in avg_df.trial_type.unique():
        r_data = avg_df.loc[(avg_df.trial_type == ttype) & (avg_df.context == 'Rewarded')].computed_reaction_time.values
        nnr_data = avg_df.loc[
            (avg_df.trial_type == ttype) & (avg_df.context == 'Non-Rewarded')].computed_reaction_time.values
        r_mean = np.nanmean(r_data)
        nnr_mean = np.nanmean(nnr_data)
        r_std = np.nanstd(r_data)
        nnr_std = np.nanstd(nnr_data)
        d = (r_mean - nnr_mean) / np.sqrt(0.5 * (r_std ** 2 + nnr_std ** 2))
        stat_res = st.ttest_rel(r_data, nnr_data)

        ttypes.append(ttype)
        mean_r.append(np.round(r_mean, 3))
        mean_nnr.append(np.round(nnr_mean, 3))
        std_r.append(np.round(r_std, 3))
        std_nnr.append(np.round(nnr_std, 3))
        dprime.append(d)
        tval.append(stat_res[0])
        pval.append(stat_res[1] * len(avg_df.trial_type.unique()))

    reaction_stats = {'trial': ttypes,
                      'R+ mean': mean_r,
                      'R+ std': std_r,
                      'R- mean': mean_nnr,
                      'R- std': std_nnr,
                      'T': tval,
                      'pval': pval,
                      'd-prime': dprime
                      }
    reaction_stats = pd.DataFrame(reaction_stats)
    pd.set_option("display.float_format", "{:.2e}".format)
    reaction_stats['Significant'] = [True if val < 0.05 else False for val in reaction_stats['pval'].values.tolist()]
    save_table(df=reaction_stats, saving_path=saving_path, name=f'{name}_stats', format=['csv'])

    # Set plot parameters.
    color_palette = [(129 / 255, 0 / 255, 129 / 255), (0 / 255, 135 / 255, 0 / 255)]
    pplot_size = 10
    scatter_size = 5
    scatter_alpha = 0.5

    fig, ax = plt.subplots(1, 1, figsize=(3, 3))
    sns.pointplot(avg_df, y='computed_reaction_time', x='trial_type', hue='context', palette=color_palette,
                  legend=False, ax=ax, dodge=0.7, markersize=pplot_size, linestyle='none')
    sns.stripplot(avg_df, y='computed_reaction_time', x='trial_type', hue='context', palette=color_palette,
                  legend=False, ax=ax, dodge=True, s=scatter_size, alpha=scatter_alpha)
    ax.set_ylabel('Spout contact time (s)')
    ax.set_xlabel('Trial')
    sns.despine()
    fig.tight_layout()

    save_fig(fig, saving_path, figure_name=name, formats=saving_formats)


def dprime_criterion(data_table, saving_path, name, saving_formats):
    # Count
    n_sess = len(data_table.session_id.unique())
    n_mice = len(data_table.mouse_id.unique())

    # Keep relevant column
    cols = ['mouse_id', 'session_id', 'trial_id', 'trial_type',
            'outcome_w', 'outcome_a', 'outcome_n', 'context', 'context_background']
    data_table_sel = data_table[cols]

    session_list = []
    mouse_list = []
    context_list = []
    dprime_list = []
    criterion_list = []
    for session in data_table_sel.session_id.unique():
        df = data_table_sel.loc[data_table_sel.session_id == session].copy()
        df['block'] = df['trial_id'].transform(lambda x: x // 20)
        df_block = df.groupby(['mouse_id', 'session_id', 'block', 'trial_type', 'context', 'context_background'],
                              as_index=False).agg('mean')

        for context in df_block.context.unique():
            catch_data = df_block.loc[df_block.context == context].outcome_n.dropna().values[:]
            wh_data = df_block.loc[df_block.context == context].outcome_w.dropna().values[:]

            # dprime
            dprime = (np.mean(wh_data) - np.mean(catch_data)) / np.sqrt(0.5 * (np.var(wh_data) + np.var(catch_data)))

            # criterion
            z_wh = st.norm.ppf(np.mean(wh_data))
            z_catch = st.norm.ppf(np.mean(catch_data))
            criterion = -0.5 * (z_wh + z_catch)

            session_list.append(session)
            mouse_list.append(df.mouse_id.unique()[0])
            dprime_list.append(dprime)
            criterion_list.append(criterion)
            context_list.append(context)

    results_df = pd.DataFrame({'mouse': mouse_list,
                               'session': session_list,
                               'dprime': dprime_list,
                               'criterion': criterion_list,
                               'context': context_list})

    avg_results = results_df.drop('session', axis=1).groupby(['mouse', 'context'], as_index=False).agg('mean')

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(5, 2.5))
    sns.pointplot(avg_results, y='dprime', hue='context', hue_order=[0, 1], palette=['darkmagenta', 'green'],
                  legend=False, dodge=True, ax=ax0)
    sns.stripplot(avg_results, y='dprime', hue='context', hue_order=[0, 1], palette=['darkmagenta', 'green'],
                  legend=False, dodge=True, ax=ax0)
    sns.pointplot(avg_results, y='criterion', hue='context', hue_order=[0, 1], palette=['darkmagenta', 'green'],
                  legend=False, dodge=True, ax=ax1)
    sns.stripplot(avg_results, y='criterion', hue='context', hue_order=[0, 1], palette=['darkmagenta', 'green'],
                  legend=False, dodge=True, ax=ax1)
    ax0.set_ylabel("Whisker D'")
    ax0.set_xlabel('Context')
    ax1.set_ylabel("Whisker Criterion")
    ax1.set_xlabel('Context')
    sns.despine()
    fig.tight_layout()

    save_fig(fig, saving_path, figure_name=name, formats=saving_formats)
