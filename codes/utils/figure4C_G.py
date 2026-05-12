import itertools
import os
import re
import sys
sys.path.append(os.getcwd())
import glob
import warnings

import matplotlib.pyplot as plt
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['svg.fonttype'] = 'none'
import yaml
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, rgb2hex, hex2color
from matplotlib.lines import Line2D            
from itertools import product

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics.pairwise import paired_distances, cosine_similarity

from codes.utils.misc.plot_on_grid import get_wf_scalebar, reduce_im_dimensions, plot_grid_on_allen, generate_reduced_image_df
from codes.utils.misc.plot_on_allen import plot_wf_single_frame

from pathlib import Path
from scipy.stats import ttest_1samp, linregress

import random
np.random.seed(982874)
random.seed(982874)

def load_wf_opto_data(data_path):
    total_df = []
    
    data_files = glob.glob(os.path.join(data_path, "*", "results.parquet.gzip")) 
    for file in data_files:
        df = [pd.read_parquet(Path(file, compression='gzip'))]
        total_df += df

    return pd.concat(total_df, ignore_index=True)


def load_opto_data(nwb_files, opto_result_path):
    single_mouse_result_files = glob.glob(os.path.join(opto_result_path, "*", "opto_data.json"))
    mice=[]
    for file in single_mouse_result_files:
        mice += [file.split("\\")[-2]]

    opto_df = []
    for file in single_mouse_result_files:
        d= pd.read_json(file)
        d['mouse_id'] = [file.split("\\")[-2] for i in range(d.shape[0])]
        opto_df += [d]
    opto_df = pd.concat(opto_df)
    opto_df = opto_df.loc[opto_df.opto_grid_ap!=3.5]

    opto_df = opto_df.reset_index(drop=True)
    opto_df['opto_stim_coord'] = opto_df.apply(lambda x: tuple([x.opto_grid_ap, x.opto_grid_ml]), axis=1)
    return opto_df.loc[opto_df.mouse_id.isin(mice)]


def load_opto_data_old(group, data_path):
    
    single_mouse_result_files = glob.glob(os.path.join(data_path, "*", "opto_data.json"))
    opto_df = []
    for file in single_mouse_result_files:
        d= pd.read_json(file)
        d['mouse_name'] = [file.split("/")[-2] for i in range(d.shape[0])]
        opto_df += [d]
    opto_df = pd.concat(opto_df)
    opto_df = opto_df.loc[opto_df.opto_grid_ap!=3.5]
    opto_avg_df = opto_df.groupby(by=['context', 'trial_type', 'opto_grid_ml', 'opto_grid_ap']).agg(
                                                                                 data=('data_mean', list),
                                                                                 data_sub=('data_mean_sub', list),
                                                                                 data_mean=('data_mean', 'mean'),
                                                                                 data_mean_sub=(
                                                                                 'data_mean_sub', 'mean'),
                                                                                 shuffle_dist=('shuffle_dist', 'sum'),
                                                                                 shuffle_dist_sub=(
                                                                                 'shuffle_dist_sub', 'sum'),
                                                                                 percentile_avg=('percentile', 'mean'),
                                                                                 percentile_avg_sub=(
                                                                                 'percentile_sub', 'mean'),
                                                                                 n_sigma_avg=('n_sigma', 'mean'),
                                                                                 n_sigma_avg_sub=(
                                                                                 'n_sigma_sub', 'mean'))
    opto_avg_df['shuffle_mean'] = opto_avg_df.apply(lambda x: np.mean(x.shuffle_dist), axis=1)
    opto_avg_df['shuffle_std'] = opto_avg_df.apply(lambda x: np.std(x.shuffle_dist), axis=1)
    opto_avg_df['shuffle_mean_sub'] = opto_avg_df.apply(lambda x: np.mean(x.shuffle_dist_sub), axis=1)
    opto_avg_df['shuffle_std_sub'] = opto_avg_df.apply(lambda x: np.std(x.shuffle_dist_sub), axis=1)

    opto_avg_df = opto_avg_df.reset_index()
    opto_avg_df['opto_stim_coord'] = opto_avg_df.apply(lambda x: tuple([x.opto_grid_ap, x.opto_grid_ml]), axis=1)
    return opto_avg_df


    
def plot_example_stim_images(nwb_files, result_path): ## this needs nwb wrappers

    df = []
    for nwb_file in nwb_files:
        bhv_data = bhv_utils.build_standard_behavior_table([nwb_file])
        if bhv_data.trial_id.duplicated().sum()>0:
            bhv_data['trial_id'] = bhv_data.index.values

        bhv_data = bhv_data.loc[(bhv_data.early_lick==0) & (bhv_data.opto_grid_ap!=3.5)]
        bhv_data['opto_stim_coord'] = bhv_data.apply(lambda x: f"({x.opto_grid_ap}, {x.opto_grid_ml})",axis=1)
        wf_timestamps = nwb_read.get_widefield_timestamps(nwb_file, ['ophys', 'dff0'])
        session_id = nwb_read.get_session_id(nwb_file)
        mouse_id = nwb_read.get_mouse_id(nwb_file)
        print(f"--------- {session_id} ---------")
        for loc in bhv_data.opto_stim_coord.unique():
            if loc not in ["(-1.5, 3.5)", "(1.5, 1.5)", "(-1.5, 0.5)", "(-5.0, 5.0)"]:
                continue

            opto_data = bhv_data.loc[bhv_data.opto_stim_coord==loc]
            opto_data['mouse_id'] = mouse_id
            opto_data['session_id'] = session_id
            trials = opto_data.start_time
            wf_image = get_frames_by_epoch(nwb_file, trials, wf_timestamps, start=40, stop=60)
            opto_data['wf_image'] = [wf_image[i] for i in range(wf_image.shape[0])]
            df += [opto_data]

    df = pd.concat(df)
    df['wf_image_sub'] = df.apply(lambda x: x['wf_image'] - np.nanmean(x['wf_image'][:10], axis=0),axis=1)
    mouse_avg = df.groupby(by=['mouse_id', 'context', 'trial_type', 'opto_stim_coord']).agg({'wf_image_sub': lambda x: np.nanmean(np.stack(x), axis=0)}).reset_index()
    avg = mouse_avg.groupby(by=['context', 'trial_type', 'opto_stim_coord']).agg({'wf_image_sub': lambda x: np.nanmean(np.stack(x), axis=0)}).reset_index()

    for c, group in avg.groupby('context'):
        for loc in group.opto_stim_coord.unique():
            print(c, loc)
            im_seq = group.loc[(group.trial_type=='whisker_trial') & (group.opto_stim_coord==loc), 'wf_image_sub'].to_numpy()[0]
            save_path = os.path.join(result_path, 'rewarded' if c else 'non-rewarded', f"{loc}_stim")
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            for i in range(9, 16):
                fig, ax = plt.subplots()
                plot_single_frame(im_seq[i], f"Frame {i-10}", fig=fig, ax=ax, norm=True, colormap='hotcold', vmin=-0.03, vmax=0.03)
                fig.savefig(os.path.join(save_path, f'whisker_stim_frame_{i-10}.png'))

            im_seq = group.loc[(group.trial_type=='no_stim_trial') & (group.opto_stim_coord==loc), 'wf_image_sub'].to_numpy()[0]

            for i in range(9, 16):
                fig, ax = plt.subplots()
                plot_single_frame(im_seq[i], f"Frame {i-10}", fig=fig, ax=ax, norm=True, colormap='hotcold', vmin=-0.03, vmax=0.03)
                fig.savefig(os.path.join(save_path, f'no_stim_frame_{i-10}.png'))


def Figure4_D_Figure4_supp2_D(control_df, pc_df, result_path):

    roi_list = ['(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 1.5)', '(-1.5, 0.5)', '(0.5, 4.5)', '(2.5, 2.5)']

    fig, ax = plt.subplots(2,6, figsize=(24,8))
    fig.suptitle('PC1')
    fig1, ax1 = plt.subplots(2,6, figsize=(24,8))
    fig1.suptitle('PC2')
    fig2, ax2 = plt.subplots(2,6, figsize=(24,8))
    fig2.suptitle('PC3')

    for i, stim in enumerate(roi_list):


        for j, (name, subgroup) in enumerate(control_df.groupby('context')):
            if name=='rewarded':
                whisker = ['#83f28f', '#348A18']
            else:
                whisker = ['#D9C4EC', '#6E188A']

            if i==0:
                ax[0, i].set_title(name)
                ax1[0, i].set_title(name)
                ax2[0, i].set_title(name)

            trial = 'whisker_trial'
            group = subgroup[subgroup.trial_type == 'whisker_trial']

            sns.lineplot(group, 
                            x='time', 
                            y='PC 1', 
                            hue='legend', 
                            hue_order = ['(-5.0, 5.0) - no lick', '(-5.0, 5.0) - lick'], 
                            palette=whisker, estimator='mean', errorbar=('ci', 95), ax=ax[j, i])
            sns.lineplot(group, 
                            x='time', 
                            y='PC 2', 
                            hue='legend', 
                            hue_order = ['(-5.0, 5.0) - no lick', '(-5.0, 5.0) - lick'], 
                            palette=whisker, estimator='mean', errorbar=('ci', 95), ax=ax1[j, i])                    
            sns.lineplot(group, 
                            x='time', 
                            y='PC 3', 
                            hue='legend', 
                            hue_order = ['(-5.0, 5.0) - no lick', '(-5.0, 5.0) - lick'], 
                            palette=whisker, estimator='mean', errorbar=('ci', 95), ax=ax2[j, i])                    

            group = pc_df.loc[(pc_df.context==name) & (pc_df.trial_type==trial) & (pc_df.opto_stim_coord==stim)]


            sns.lineplot(group, 
                            x='time', 
                            y='PC 1', 
                            color='royalblue', estimator='mean', errorbar=('ci', 95), ax=ax[j, i])
            sns.lineplot(group, 
                            x='time', 
                            y='PC 2', 
                            color='royalblue', estimator='mean', errorbar=('ci', 95), ax=ax1[j, i])                    
            sns.lineplot(group, 
                            x='time', 
                            y='PC 3', 
                           color='royalblue', estimator='mean', errorbar=('ci', 95), ax=ax2[j, i]) 
                           
            ax[j,i].set_ylim(-35,35)
            ax[j,i].set_ylabel('PC 1')

            ax1[j,i].set_ylim(-15,10)
            ax1[j,i].set_ylabel('PC 2')

            ax2[j,i].set_ylim(-15,5)
            ax2[j,i].set_ylabel('PC 3')

    save_path = os.path.join(result_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f"Figure4_supp2_D_top.png"))
    fig.savefig(os.path.join(save_path, f"Figure4_supp2_D_top.svg"))
    fig1.tight_layout()
    fig1.savefig(os.path.join(save_path, f"Figure4_supp2_D_mid.png"))
    fig1.savefig(os.path.join(save_path, f"Figure4_supp2_D_mid.svg"))
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_path, f"Figure4_supp2_D_bot.png"))
    fig2.savefig(os.path.join(save_path, f"Figure4_supp2_D_bot.svg"))


def compute_angle_stim_lick(control_df, pc_df, result_path):

    control = control_df.loc[(control_df.time>=0) & (control_df.trial_type=='whisker_trial')].drop(
        columns=['mouse_id', 'legend']
    ).groupby(by=['context', 'trial_type', 'opto_stim_coord', 'lick_flag', 'time']).apply('mean').reset_index()
    stim = pc_df.loc[(pc_df.time>=0) & (pc_df.trial_type=='whisker_trial') & (pc_df.opto_stim_coord!='(-5.0, 5.0)')]
    for pc in ['PC 1', 'PC 2', 'PC 3']:
        control[pc] = control[pc] + stim[pc].min()
        stim[pc] = stim[pc] + stim[pc].min()

    result_df = []
    for name, group in stim.groupby(by=['mouse_id', 'context', 'opto_stim_coord']):
        context = group.context.unique()[0]
        lick_sim = np.diag(cosine_similarity(group['PC 3'].reset_index(drop=True).reset_index().to_numpy(), control.loc[(control.context==context) & (control.lick_flag==1), 'PC 3'].reset_index(drop=True).reset_index().to_numpy()))
        nolick_sim = np.diag(cosine_similarity(group['PC 3'].reset_index(drop=True).reset_index().to_numpy(), control.loc[(control.context==context) & (control.lick_flag==0), 'PC 3'].reset_index(drop=True).reset_index().to_numpy()))

        for pc in ['PC 1', 'PC 2', 'PC 3']:
            v1 = group[pc].to_numpy().flatten()/np.linalg.norm(group[pc].to_numpy().flatten())
            v2 = control.loc[(control.context==context) & (control.lick_flag==1), pc].to_numpy().flatten()/np.linalg.norm(control.loc[(control.context==context) & (control.lick_flag==1), pc].to_numpy().flatten())
            v3 = control.loc[(control.context==context) & (control.lick_flag==0), pc].to_numpy().flatten()/np.linalg.norm(control.loc[(control.context==context) & (control.lick_flag==0), pc].to_numpy().flatten())
            angle_lick = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
            angle_nolick = np.degrees(np.arccos(np.clip(np.dot(v1, v3), -1.0, 1.0)))
            angle_control = np.degrees(np.arccos(np.clip(np.dot(v2, v3), -1.0, 1.0)))

            result={
                'mouse_id': name[0],
                'pc': pc,
                'context': context,
                'opto_stim_coord': group.opto_stim_coord.unique()[0],
                'PC3_similarity': (lick_sim - nolick_sim).sum(),
                'angle_lick': angle_lick,
                'angle_nolick': angle_nolick,
                'angle_diff': angle_lick - angle_nolick,
                'angle_control': angle_control
            }
            result_df += [result]
    result_df = pd.DataFrame(result_df)
    save_path = os.path.join(result_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    result_df.to_csv(os.path.join(save_path, 'results_angle.csv'))
    return result_df


def Figure4_E(angle_df, result_path):
    save_path = os.path.join(result_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    roi_list = ['(-1.5, 0.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(0.5, 4.5)', '(1.5, 1.5)', '(2.5, 2.5)']
    group = angle_df[(angle_df.opto_stim_coord.isin(roi_list)) & (angle_df.pc=="PC 3")]
    pc = "PC 3"

    fig, ax = plt.subplots(figsize=(4,3))
    sns.barplot(
        data=group,
        x='opto_stim_coord',
        y='angle_lick',
        order=['(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 1.5)', '(-1.5, 0.5)', '(2.5, 2.5)', '(0.5, 4.5)'],
        hue='context',
        hue_order=['non-rewarded', 'rewarded'],
        palette=['purple', 'green'],
        alpha=0.5,
        estimator='mean',
        errorbar=('ci', 95),
        err_kws=dict(color='black', lw=1),
        ax=ax,
    )
    sns.stripplot(
        data=group,
        x='opto_stim_coord',
        y='angle_lick',
        order=['(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 1.5)', '(-1.5, 0.5)', '(2.5, 2.5)', '(0.5, 4.5)'],
        hue='context',
        hue_order=['non-rewarded', 'rewarded'],
        palette=['purple', 'green'],
        s= 5,  
        jitter=True, 
        dodge=True,
        ax=ax,
    )
    sns.despine()
    ax.set_xticks(['(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 1.5)', '(-1.5, 0.5)', '(2.5, 2.5)', '(0.5, 4.5)'], ['wS1', 'wS2', 'wM', 'RSC', 'ALM', 'tjS1'])
    ax.legend_.remove()
    fig.tight_layout()
    fig.savefig(os.path.join(save_path, f"Figure4_E.png"))
    fig.savefig(os.path.join(save_path, f"Figure4_E.svg"))


def Figure4_F_G_correlations(opto_df, angle_df, result_path):
    mouse_color = ["#003049", "#d62828", "#f77f00", "#fcbf49"]
    roi_color = {
        '(-1.5, 3.5)': '#ff8c00', 
        '(-1.5, 4.5)': "#ffa500", 
        '(1.5, 1.5)': "#0000ff", 
        '(-1.5, 0.5)': '#6495ed', 
        '(2.5, 2.5)': "#ff0000", 
        '(0.5, 4.5)': "#ba55d3"
    }
    save_path = os.path.join(result_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)    
    opto_df = opto_df[opto_df.trial_type=='whisker_trial']
    angle_df['colors'] = angle_df.apply(lambda x: roi_color[x.opto_stim_coord] if x.opto_stim_coord in roi_color.keys() else "#808080", axis=1)

    avg_angle_df = angle_df.drop(columns=['mouse_id']).groupby(by=['pc', 'context', 'opto_stim_coord', 'colors'], as_index=False, sort=False).agg('mean')
    avg_opto_df = opto_df.loc[opto_df.trial_type=='whisker_trial'].drop(
        columns=['mouse_id', 'index', 'opto_grid_ml', 'opto_grid_ap', 'shuffle_mean', 'shuffle_std', 'context_background',
       'shuffle_mean_sub', 'shuffle_std_sub', 'shuffle_dist', 'shuffle_dist_sub', 'data_mean', 'percentile',
       'percentile_sub', 'n_sigma', 'n_sigma_sub', 'p', 'p_corr', 'p_sub', 'p_corr_sub', 'opto_grid_ml_wf', 'opto_grid_ap_wf', 'trial_type',]).groupby(
        by=['context', 'opto_stim_coord'], as_index=False, sort=False).agg('mean')

    for pc in ['PC 3']:
        fig, ax = plt.subplots(1,2, figsize=(8,4))
        fig.suptitle(f"Average {pc} Plick correlation\n n=4 mice")
        for i, c in enumerate(avg_angle_df.context.unique()):
            ax[i].set_title(c)

            colorlist = avg_angle_df.sort_values(['context', 'opto_stim_coord']).loc[(avg_angle_df.pc==pc) & (angle_df.context==c), "colors"]
            angle = avg_angle_df.sort_values(['context', 'opto_stim_coord']).loc[(avg_angle_df.pc==pc) & (angle_df.context==c), "angle_lick"]
            opto = avg_opto_df.sort_values(['context', 'opto_stim_coord']).loc[avg_opto_df.context==c, "data_mean_sub"]
            z = np.polyfit(angle, opto, 1)  
            y_hat = np.poly1d(z)(angle)
            model = linregress(angle, opto)
            text = f"$R^2 = {model.rvalue ** 2:0.3f}$\n$p={model.pvalue:0.3e}$"

            ax[i].scatter(angle, opto, color=colorlist)
            ax[i].plot(angle, y_hat, c='k', ls='-', lw=2)
            ax[i].text(0.05, 0.95, text, transform=ax[i].transAxes, fontsize=10, verticalalignment='top')

            ax[i].set_ylim([-0.5, 0.3])
            ax[i].set_xlim([0, 15])
            ax[i].spines[['right', 'top']].set_visible(False)
        fig.savefig(os.path.join(save_path, f'Figure4_F_G_right.png'))
        fig.savefig(os.path.join(save_path, f'Figure4_F_G_right.svg'))


def Figure4_F_G_map(avg_angle_df, result_path):
    save_path = os.path.join(result_path)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    for name, group in avg_angle_df.groupby('context'):
        for pc in ['PC 1', 'PC 2', 'PC 3']:
            fig, ax = plt.subplots(figsize=(4,4))
            im_df = generate_reduced_image_df(group.loc[group.pc==pc, 'angle_lick'].to_numpy()[np.newaxis, :], [eval(coord) for coord in group.loc[group.pc==pc, 'opto_stim_coord']])
            im_df = im_df.rename(columns={'dff0': 'angle'})
            plot_grid_on_allen(im_df, outcome='angle', palette='viridis', facecolor=None, edgecolor=None, dotsize=440, result_path=None, vmin=0, vmax=10, fig=fig, ax=ax)
            ax.set_axis_off()
            fig.savefig(os.path.join(save_path, f'Figure4_{"F" if name=="rewarded" else "G"}_left.png'), dpi=400)


def Figure4_supp2_BC(pca, result_path):
    
    # Explained variance ratio
    exp_var = [val * 100 for val in pca.explained_variance_ratio_]
    plot_y = [sum(exp_var[:i+1]) for i in range(len(exp_var))]
    plot_x = range(1, len(plot_y) + 1)
    fig,ax = plt.subplots(figsize=(7,4))
    ax.plot(plot_x, plot_y, marker="o", color="#9B1D20")
    for x, y in zip(plot_x, plot_y):
        plt.text(x, y + 3, f"{y:.1f}%", ha="center", va="bottom")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Cumulative percentage of variance explained")
    ax.set_ylim([50,100])
    ax.set_xticks(plot_x)
    ax.grid(axis="y")
    ax.spines[['top', 'right']].set_visible(False)
    for ext in ['.png', '.svg']:
        fig.savefig(Path(result_path, f'Figure4_supp2_B{ext}'))

    ## Plot biplots to see which variables (i.e. rois) contain most of the explained variance (?)

    coeff = np.transpose(pca.components_)

    ## Plot loadings in allen ccf
    labels=['(-0.5, 0.5)', '(-0.5, 1.5)', '(-0.5, 2.5)', '(-0.5, 3.5)',
       '(-0.5, 4.5)', '(-0.5, 5.5)', '(-1.5, 0.5)', '(-1.5, 1.5)',
       '(-1.5, 2.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(-1.5, 5.5)',
       '(-2.5, 0.5)', '(-2.5, 1.5)', '(-2.5, 2.5)', '(-2.5, 3.5)',
       '(-2.5, 4.5)', '(-2.5, 5.5)', '(-3.5, 0.5)', '(-3.5, 1.5)',
       '(-3.5, 2.5)', '(-3.5, 3.5)', '(-3.5, 4.5)', '(-3.5, 5.5)',
       '(0.5, 0.5)', '(0.5, 1.5)', '(0.5, 2.5)', '(0.5, 3.5)', '(0.5, 4.5)',
       '(0.5, 5.5)', '(1.5, 0.5)', '(1.5, 1.5)', '(1.5, 2.5)', '(1.5, 3.5)',
       '(1.5, 4.5)', '(1.5, 5.5)', '(2.5, 0.5)', '(2.5, 1.5)', '(2.5, 2.5)',
       '(2.5, 3.5)', '(2.5, 4.5)', '(2.5, 5.5)']
    fig, ax = plt.subplots(1, 3, figsize=(7, 12))
    fig.suptitle("PC1-3 loadings")

    for i in range(3):
        im_pca = generate_reduced_image_df(coeff[np.newaxis, :, i], [eval(label) for label in labels])
        im_pca.drop(im_pca[(im_pca.x==5.5)&(im_pca.y==2.5)].index, inplace=True)        
        plot_grid_on_allen(im_pca, outcome='dff0', palette='seismic', facecolor=[], edgecolor=[], result_path=None, dotsize=340, vmin=-im_pca.dff0.abs().max(), vmax=im_pca.dff0.abs().max(), norm=None, fig=fig, ax= ax.flat[i])
        ax.flat[i].set_axis_off()
        ax.flat[i].set_title(f"PC {i+1}")
    fig.savefig(os.path.join(result_path, f"Figure4_supp2_C.png"))


def dimensionality_reduction(data_path, output_path):
    # result_path = Path(output_path, 'PCA_150')
    result_path = Path(os.getcwd(), 'figures')

    if not os.path.exists(result_path):
        os.makedirs(result_path)

    coords_list = {'wS1': "(-1.5, 3.5)", 'wS2': "(-1.5, 4.5)", 'wM1': "(1.5, 1.5)", 'wM2': "(2.5, 1.5)", 'RSC': "(-0.5, 0.5)", "RSC_2": "(-1.5, 0.5)",
            'ALM': "(2.5, 2.5)", 'tjS1':"(0.5, 4.5)", 'tjM1':"(1.5, 3.5)", 'control': "(-5.0, 5.0)"}


    group = 'controls' if 'control' in str(output_path) else 'VGAT'

    opto_data_path = fr'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/optogenetic_results/{group}'
    opto_df = load_opto_data(group, opto_data_path)
    opto_df = opto_df[~opto_df.opto_stim_coord.astype(str).isin(["(1.5, 5.5)", "(2.5, 4.5)", "(2.5, 5.5)"])]
    
    total_df = load_wf_opto_data(data_path)
    total_df = total_df[~total_df.opto_stim_coord.isin(["(1.5, 5.5)", "(2.5, 4.5)", "(2.5, 5.5)"])]
    total_df.context = total_df.context.map({0:'non-rewarded', 1:'rewarded'})
    total_df['time'] = [[np.linspace(-1,3.98,250)] for i in range(total_df.shape[0])]
    total_df['legend']= total_df.apply(lambda x: f"{x.opto_stim_coord} - {'lick' if x.lick_flag==1 else 'no lick'}",axis=1)

    d = {c: lambda x: x.unique()[0] for c in ['opto_stim_loc', 'legend']}
    d['time'] = lambda x: list(x)[0][0]
    for c in ['(-0.5, 0.5)', '(-0.5, 1.5)', '(-0.5, 2.5)', '(-0.5, 3.5)', '(-0.5, 4.5)', '(-0.5, 5.5)', '(-1.5, 0.5)',
       '(-1.5, 1.5)', '(-1.5, 2.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(-1.5, 5.5)', '(-2.5, 0.5)', '(-2.5, 1.5)', '(-2.5, 2.5)',
       '(-2.5, 3.5)', '(-2.5, 4.5)', '(-2.5, 5.5)', '(-3.5, 0.5)', '(-3.5, 1.5)', '(-3.5, 2.5)', '(-3.5, 3.5)', '(-3.5, 4.5)',
       '(-3.5, 5.5)', '(0.5, 0.5)', '(0.5, 1.5)', '(0.5, 2.5)', '(0.5, 3.5)', '(0.5, 4.5)', '(0.5, 5.5)', '(1.5, 0.5)', 
       '(1.5, 1.5)', '(1.5, 2.5)', '(1.5, 3.5)', '(1.5, 4.5)', '(2.5, 0.5)', '(2.5, 1.5)', '(2.5, 2.5)',
        '(2.5, 3.5)']:
        d[f"{c}"]= lambda x: np.nanmean(np.stack(x), axis=0)
          
    mouse_df = total_df.groupby(by=['mouse_id', 'context', 'trial_type', 'opto_stim_coord', 'lick_flag']).agg(d).reset_index()
    mouse_df = mouse_df.melt(id_vars=['mouse_id', 'context', 'trial_type', 'legend', 'opto_stim_coord', 'lick_flag', 'time'],
                                 value_vars=['(-0.5, 0.5)', '(-0.5, 1.5)', '(-0.5, 2.5)', '(-0.5, 3.5)', '(-0.5, 4.5)', '(-0.5, 5.5)', '(-1.5, 0.5)',
       '(-1.5, 1.5)', '(-1.5, 2.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(-1.5, 5.5)', '(-2.5, 0.5)', '(-2.5, 1.5)', '(-2.5, 2.5)',
       '(-2.5, 3.5)', '(-2.5, 4.5)', '(-2.5, 5.5)', '(-3.5, 0.5)', '(-3.5, 1.5)', '(-3.5, 2.5)', '(-3.5, 3.5)', '(-3.5, 4.5)',
       '(-3.5, 5.5)', '(0.5, 0.5)', '(0.5, 1.5)', '(0.5, 2.5)', '(0.5, 3.5)', '(0.5, 4.5)', '(0.5, 5.5)', '(1.5, 0.5)', 
       '(1.5, 1.5)', '(1.5, 2.5)', '(1.5, 3.5)', '(1.5, 4.5)', '(2.5, 0.5)', '(2.5, 1.5)', '(2.5, 2.5)', '(2.5, 3.5)'],
                                 var_name='roi',
                                 value_name='dff0').explode(['time', 'dff0'])
    mouse_df = mouse_df[(mouse_df.time>=-0.15)&(mouse_df.time<=0.15)]
    
    # Use control stim location, whisker and catch trials, with lick and no-lick separate to compute the pc space
    avg_df = mouse_df.drop(columns=['mouse_id']).groupby(by=['context', 'trial_type', 'lick_flag', 'legend', 'opto_stim_coord', 'roi', 'time']).agg(lambda x: np.nanmean(x)).reset_index()
    avg_df.time = avg_df.time.round(2)
    subset_df = avg_df[(avg_df.trial_type.isin(['whisker_trial', 'no_stim_trial'])) & (avg_df.opto_stim_coord=="(-5.0, 5.0)")].pivot(index=['context','trial_type', 'legend', 'time'], columns='roi', values='dff0')
    avg_data_for_pca = subset_df.to_numpy()

    # Standardize average data for training: Based on trials with light on control location 
    scaler = StandardScaler()
    fit_scaler = scaler.fit(avg_data_for_pca)
    avg_data_for_pca = fit_scaler.transform(avg_data_for_pca)
    pca = PCA(n_components=15)
    results = pca.fit(np.nan_to_num(avg_data_for_pca))

    # Plot coefficients, biplots and variance explained
    Figure4_supp2_BC(pca, result_path)

    subset_df = mouse_df[
        (mouse_df.trial_type.isin(['whisker_trial', 'no_stim_trial'])) & 
        (mouse_df.opto_stim_coord=="(-5.0, 5.0)")].pivot(
        index=['mouse_id', 'context', 'trial_type', 'lick_flag', 'legend', 'opto_stim_coord', 'time'], columns='roi', values='dff0')
    avg_data_for_pca = subset_df.to_numpy()
    avg_data_for_pca = fit_scaler.transform(avg_data_for_pca)
    principal_components = pca.transform(np.nan_to_num(avg_data_for_pca))

    control_df = pd.DataFrame(data=principal_components, index=subset_df.index)
    control_df.columns = [f"PC {i+1}" for i in range(0, principal_components.shape[1])]
    control_df = control_df.reset_index()

    # Project whisker and catch trials 
    mouse_df = total_df.groupby(by=['mouse_id', 'context', 'trial_type', 'opto_stim_coord']).agg(d).reset_index()
    mouse_df = mouse_df.melt(id_vars=['mouse_id', 'context', 'trial_type', 'opto_stim_coord', 'time'],
                                value_vars=['(-0.5, 0.5)', '(-0.5, 1.5)', '(-0.5, 2.5)', '(-0.5, 3.5)', '(-0.5, 4.5)', '(-0.5, 5.5)', '(-1.5, 0.5)',
    '(-1.5, 1.5)', '(-1.5, 2.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(-1.5, 5.5)', '(-2.5, 0.5)', '(-2.5, 1.5)', '(-2.5, 2.5)',
    '(-2.5, 3.5)', '(-2.5, 4.5)', '(-2.5, 5.5)', '(-3.5, 0.5)', '(-3.5, 1.5)', '(-3.5, 2.5)', '(-3.5, 3.5)', '(-3.5, 4.5)',
    '(-3.5, 5.5)', '(0.5, 0.5)', '(0.5, 1.5)', '(0.5, 2.5)', '(0.5, 3.5)', '(0.5, 4.5)', '(0.5, 5.5)', '(1.5, 0.5)', 
    '(1.5, 1.5)', '(1.5, 2.5)', '(1.5, 3.5)', '(1.5, 4.5)', '(2.5, 0.5)', '(2.5, 1.5)', '(2.5, 2.5)', '(2.5, 3.5)'],
                                var_name='roi',
                                value_name='dff0').explode(['time', 'dff0'])

    mouse_df = mouse_df.reset_index()
    mouse_df = mouse_df[(mouse_df.time>=-0.15)&(mouse_df.time<=0.15)]
    subset_df = mouse_df[mouse_df.trial_type.isin(['whisker_trial', 'no_stim_trial'])].pivot(
        index=['mouse_id', 'context','trial_type', 'opto_stim_coord', 'time'], columns='roi', values='dff0')

    avg_data_for_pca = subset_df.to_numpy()
    avg_data_for_pca = fit_scaler.transform(avg_data_for_pca)
    principal_components = pca.transform(np.nan_to_num(avg_data_for_pca))

    pc_df = pd.DataFrame(data=principal_components, index=subset_df.index)
    pc_df.columns = [f"PC {i+1}" for i in range(0, principal_components.shape[1])]
    subset_df = subset_df.join(pc_df).reset_index()
    pc_df = pc_df.reset_index()

    Figure4_D_Figure4_supp2_D(control_df, pc_df[pc_df.opto_stim_coord!="(-5.0, 5.0)"], result_path)

    angle_df = compute_angle_stim_lick(control_df, pc_df[pc_df.opto_stim_coord!="(-5.0, 5.0)"], result_path)
    Figure4_E(angle_df, result_path)

    avg_angle_df = angle_df.drop(columns='mouse_id').groupby(by=['pc', 'context', 'opto_stim_coord'], as_index=False, sort=False).agg('mean')
    # Figure4_F_G_map(avg_angle_df, result_path)
    Figure4_F_G_correlations(opto_df, angle_df, result_path)


def main(data_path, output_path):


    # plot_example_stim_images(nwb_files, output_path)
    dimensionality_reduction(data_path, output_path)


if __name__ == "__main__":

    for file in ['context_sessions_wf_opto']: #, 'context_sessions_wf_opto_controls', 'context_sessions_wf_opto_photoactivation'
   
        # output_path = os.path.join('//sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Pol_Bech', 'Pop_results',
        #                             'Context_behaviour', 'optogenetic_widefield_results', 'controls' if 'controls' in str(config_file) else 'VGAT')
        # 
        data_path = os.path.join('//sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Pol_Bech', 'Pop_results',
                                 'Context_behaviour', 'optogenetic_widefield_results', 'controls' if 'controls' in str(file) else 'VGAT')          
        output_path = os.path.join('//sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Pol_Bech', 'Pop_results', 'bech_dard_4C_test'
                                    'optogenetic_widefield_results', 'controls' if 'controls' in str(file) else 'VGAT')        
        # output_path = os.path.join('//sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Pol_Bech', 'Pop_results',
        #                             'Context_behaviour', 'optogenetic_widefield_results', 'photoactivation')
        # output_path = haas_pathfun(output_path)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        main(data_path, output_path)