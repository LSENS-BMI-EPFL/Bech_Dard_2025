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
import scipy
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, rgb2hex, hex2color
from matplotlib.lines import Line2D            
from itertools import product

import nwb_wrappers.nwb_reader_functions as nwb_read
import nwb_utils.utils_behavior as bhv_utils
from nwb_utils import utils_misc
from nwb_utils.utils_plotting import lighten_color, remove_top_right_frame

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics.pairwise import paired_distances

from utils.wf_plotting_utils import reduce_im_dimensions, plot_grid_on_allen, generate_reduced_image_df
from utils.dlc_utils import *
from utils.widefield_utils import *
from utils.haas_utils import *
import utils.behaviour_plot_utils as plot_utils


def load_opto_data(group):
    opto_results = fr'//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Pop_results/Context_behaviour/optogenetic_results/{group}'
    opto_results = haas_pathfun(opto_results)
    single_mouse_result_files = glob.glob(os.path.join(opto_results, "*", "opto_data.json"))
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


    
def plot_example_stim_images(nwb_files, result_path):

    from utils.wf_plotting_utils import plot_single_frame
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


def plot_timecourses_by_outcome(long_df, result_path):
    fig, ax = plt.subplots(3, 4, figsize=(8,6))
    fig.suptitle("PC timecourses")
    g = sns.relplot(data=long_df[long_df.trial_type=='whisker_trial'], x='time', y='data', hue='context', 
                    hue_order=['rewarded', 'non-rewarded'], 
                    palette=['#348A18', '#6E188A'], 
                    col='PC', row='opto_stim_coord', kind='line', estimator='mean', errorbar=('ci', 95), linewidth=2, facet_kws=dict(sharey=False))
    for k in range(g.axes.shape[0]):
        g.axes[k, 0].set_ylim(-30,30)
        g.axes[k, 1].set_ylim(-15,5)
        g.axes[k, 2].set_ylim(-15,5)

    g.figure.savefig(Path(result_path, 'whisker_PC_timecourses_by_region.png'))
    g.figure.savefig(Path(result_path, 'whisker_PC_timecourses_by_region.svg'))
    
    fig, ax = plt.subplots(3, 4, figsize=(8,6))
    fig.suptitle("PC timecourses")
    g = sns.relplot(data=long_df[long_df.trial_type=='no_stim_trial'], x='time', y='data', hue='context', 
                    hue_order=['rewarded', 'non-rewarded'], 
                    palette=['#348A18', '#6E188A'], 
                    col='PC', row='opto_stim_coord', kind='line', estimator='mean', errorbar=('ci', 95), linewidth=2, facet_kws=dict(sharey=False))
    for k in range(g.axes.shape[0]):
        g.axes[k, 0].set_ylim(-30,30)
        g.axes[k, 1].set_ylim(-15,5)
        g.axes[k, 2].set_ylim(-15,5)

    g.figure.savefig(Path(result_path, 'catch_PC_timecourses_by_region.png'))
    g.figure.savefig(Path(result_path, 'catch_PC_timecourses_by_region.svg'))

    g = sns.relplot(data=long_df[long_df.trial_type=='whisker_trial'], x='time', y='data', hue='opto_stim_coord', 
                    hue_order=["(-5.0, 5.0)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)", "(1.5, 3.5)", "(-0.5, 0.5)"], 
                    palette=['#000000', '#011f4b', '#03396c', '#005b96', '#6497b1', '#b3cde0', '#ffccd5', '#c9184a', '#590d22'], 
                    col='context', row='PC', kind='line', estimator='mean', errorbar=('ci', 95), linewidth=2, facet_kws=dict(sharey=False))
    for k in range(g.axes.shape[1]):
        g.axes[0, k].set_ylim(-30,30)
        g.axes[1, k].set_ylim(-15,5)
        g.axes[2, k].set_ylim(-15,5)
    g.figure.savefig(Path(result_path, 'whisker_PC_timecourses_by_trial_outcome.png'))
    g.figure.savefig(Path(result_path, 'whisker_PC_timecourses_by_trial_outcome.svg'))

    g = sns.relplot(data=long_df[long_df.trial_type=='no_stim_trial'], x='time', y='data', hue='opto_stim_coord',       
                    hue_order=["(-5.0, 5.0)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)", "(1.5, 3.5)", "(-0.5, 0.5)"], 
                    palette=['#000000', '#011f4b', '#03396c', '#005b96', '#6497b1', '#b3cde0', '#ffccd5', '#c9184a', '#590d22'], 
                    col='context', row='PC', kind='line', estimator='mean', errorbar=('ci', 95), linewidth=2, facet_kws=dict(sharey=False))
    for k in range(g.axes.shape[1]):
        g.axes[0, k].set_ylim(-30,30)
        g.axes[1, k].set_ylim(-15,5)
        g.axes[2, k].set_ylim(-15,5)
    g.figure.savefig(Path(result_path, 'catch_PC_timecourses_by_trial_outcome.png'))
    g.figure.savefig(Path(result_path, 'catch_PC_timecourses_by_trial_outcome.svg'))


def plot_projected_pc_timecourses(subset_df, color_dict, result_path):
    
    lines = ['#000000', '#011f4b', '#03396c', '#005b96', '#6497b1', '#b3cde0', '#ffccd5', '#c9184a', '#590d22']
    handles = [Line2D([0], [0], color=c, lw=4) for c in lines]

    for trial in subset_df.trial_type.unique():
        fig, ax= plt.subplots(1,2, figsize=(8,4), sharey=True, sharex=True)
        fig1, ax1 = plt.subplots(1,2, figsize=(8,4), sharey=True, sharex=True)
        fig2, ax2 = plt.subplots(1,2, figsize=(8,4), sharey=True, sharex=True)

        for i, (name, group) in enumerate(subset_df[subset_df.trial_type==trial].groupby(by=['context'])):
            for stim in color_dict.keys():
                ax.flat[i].scatter(group.loc[group.opto_stim_coord==stim, 'PC 1'], group.loc[group.opto_stim_coord==stim, 'PC 2'], c=group.loc[group.opto_stim_coord==stim, 'time'], cmap=color_dict[stim], label=stim, s=10)
                ax.flat[i].scatter(group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 1'], group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 2'], s=10, facecolors='none', edgecolors='r')
                ax1.flat[i].scatter(group.loc[group.opto_stim_coord==stim, 'PC 2'], group.loc[group.opto_stim_coord==stim, 'PC 3'], c=group.loc[group.opto_stim_coord==stim, 'time'], cmap=color_dict[stim], label=stim, s=10)
                ax1.flat[i].scatter(group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 2'], group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 3'], s=10, facecolors='none', edgecolors='r')
                ax2.flat[i].scatter(group.loc[group.opto_stim_coord==stim, 'PC 1'], group.loc[group.opto_stim_coord==stim, 'PC 3'], c=group.loc[group.opto_stim_coord==stim, 'time'], cmap=color_dict[stim], label=stim, s=10)
                ax2.flat[i].scatter(group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 1'], group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 3'], s=10, facecolors='none', edgecolors='r')

            ax.flat[i].set_xlim(-35,35)
            ax.flat[i].set_xlabel('PC 1')
            ax.flat[i].set_ylim(-15,5)
            ax.flat[i].set_ylabel('PC 2')
            ax.flat[i].set_title(f"{'Rewarded' if name=='rewarded' else 'Non-rewarded'} {trial}")

            ax1.flat[i].set_xlim(-15,5)
            ax1.flat[i].set_xlabel('PC 2')
            ax1.flat[i].set_ylim(-15,5)
            ax1.flat[i].set_ylabel('PC 3')
            ax1.flat[i].set_title(f"{'Rewarded' if name=='rewarded' else 'Non-rewarded'} {trial}")

            ax2.flat[i].set_xlim(-35,35)
            ax2.flat[i].set_xlabel('PC 1')
            ax2.flat[i].set_ylim(-15,5)
            ax2.flat[i].set_ylabel('PC 3')
            ax2.flat[i].set_title(f"{'Rewarded' if name=='rewarded' else 'Non-rewarded'} {trial}")

        save_path = os.path.join(result_path, 'timecourses')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        fig.legend(handles, ["(-5.0, 5.0)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)", "(1.5, 3.5)", "(-0.5, 0.5)"], loc='outside upper right')
        fig.savefig(Path(save_path, f"{trial}_dimensionality_reduction_PC1vsPC2.png"))
        fig.savefig(Path(save_path, f"{trial}_dimensionality_reduction_PC1vsPC2.svg"))
        fig1.legend(handles, ["(-5.0, 5.0)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)", "(1.5, 3.5)", "(-0.5, 0.5)"], loc='outside upper right')
        fig1.savefig(Path(save_path, f"{trial}_dimensionality_reduction_PC2vsPC3.png"))
        fig1.savefig(Path(save_path, f"{trial}_dimensionality_reduction_PC2vsPC3.svg"))

        fig2.legend(handles, ["(-5.0, 5.0)", "(-1.5, 3.5)", "(-1.5, 4.5)", "(1.5, 1.5)", "(2.5, 1.5)", "(2.5, 2.5)", "(0.5, 4.5)", "(1.5, 3.5)", "(-0.5, 0.5)"], loc='outside upper right')
        fig2.savefig(Path(save_path, f"{trial}_dimensionality_reduction_PC1vsPC3.png"))
        fig2.savefig(Path(save_path, f"{trial}_dimensionality_reduction_PC1vsPC3.svg"))


def plot_trajectories_by_region(subset_df, color_dict, result_path):
    for trial in subset_df.trial_type.unique():
        for stim in color_dict.keys():
            save_path = os.path.join(result_path, stim, 'by_region')
            if not os.path.exists(save_path):
                os.makedirs(save_path, exist_ok=True)

            if stim == "(-5.0, 5.0)":
                continue

            fig, ax= plt.subplots(1,2, figsize=(8,4), sharey=True, sharex=True)
            fig.suptitle(stim)
            fig1, ax1 = plt.subplots(1,2, figsize=(8,4), sharey=True, sharex=True)
            fig1.suptitle(stim)
            fig2, ax2 = plt.subplots(1,2, figsize=(8,4), sharey=True, sharex=True)
            fig2.suptitle(stim)
            
            fig3 = plt.figure(figsize=(4,4))
            fig3.suptitle(f"{trial} - stim {stim}")
            ax3 = fig3.add_subplot(1, 1, 1, projection='3d')

            for i, (name, group) in enumerate(subset_df[subset_df.trial_type==trial].groupby(by=['context'])):
                if name=='rewarded':
                    color = LinearSegmentedColormap.from_list('', ['#FFFFFF', '#348A18'], N=subset_df.time.unique().shape[0])
                    control = LinearSegmentedColormap.from_list('', ['#FFFFFF', '#000000'], N=subset_df.time.unique().shape[0])
                else:
                    color = LinearSegmentedColormap.from_list('', ['#FFFFFF', '#6E188A'], N=subset_df.time.unique().shape[0])
                    control = LinearSegmentedColormap.from_list('', ['#FFFFFF', '#808080'], N=subset_df.time.unique().shape[0])

                ax.flat[i].scatter(group.loc[group.opto_stim_coord==stim, 'PC 1'], group.loc[group.opto_stim_coord==stim, 'PC 2'], c=group.loc[group.opto_stim_coord==stim, 'time'], cmap=color, label=name[0], s=10)
                ax.flat[i].scatter(group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 1'], group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 2'], s=10, facecolors='none', edgecolors='r')
                ax.flat[i].scatter(group.loc[group.opto_stim_coord=="(-5.0, 5.0)", 'PC 1'], group.loc[group.opto_stim_coord=="(-5.0, 5.0)", 'PC 2'], c=group.loc[group.opto_stim_coord=="(-5.0, 5.0)", 'time'], cmap=control, label=name[0], s=10)
                ax.flat[i].scatter(group.loc[(group.opto_stim_coord=="(-5.0, 5.0)") & (group.time==0), 'PC 1'], group.loc[(group.opto_stim_coord=="(-5.0, 5.0)") & (group.time==0), 'PC 2'], s=10, facecolors='none', edgecolors='r')
                
                ax1.flat[i].scatter(group.loc[group.opto_stim_coord==stim, 'PC 2'], group.loc[group.opto_stim_coord==stim, 'PC 3'], c=group.loc[group.opto_stim_coord==stim, 'time'], cmap=color, label=name[0], s=10)
                ax1.flat[i].scatter(group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 2'], group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 3'], s=10, facecolors='none', edgecolors='r')
                ax1.flat[i].scatter(group.loc[group.opto_stim_coord=="(-5.0, 5.0)", 'PC 2'], group.loc[group.opto_stim_coord=="(-5.0, 5.0)", 'PC 3'], c=group.loc[group.opto_stim_coord=="(-5.0, 5.0)", 'time'], cmap=control, label=name[0], s=10)
                ax1.flat[i].scatter(group.loc[(group.opto_stim_coord=="(-5.0, 5.0)") & (group.time==0), 'PC 2'], group.loc[(group.opto_stim_coord=="(-5.0, 5.0)") & (group.time==0), 'PC 3'], s=10, facecolors='none', edgecolors='r')
                
                ax2.flat[i].scatter(group.loc[group.opto_stim_coord==stim, 'PC 1'], group.loc[group.opto_stim_coord==stim, 'PC 3'], c=group.loc[group.opto_stim_coord==stim, 'time'], cmap=color, label=name[0], s=10)
                ax2.flat[i].scatter(group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 1'], group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 3'], s=10, facecolors='none', edgecolors='r')
                ax2.flat[i].scatter(group.loc[group.opto_stim_coord=="(-5.0, 5.0)", 'PC 1'], group.loc[group.opto_stim_coord=="(-5.0, 5.0)", 'PC 3'], c=group.loc[group.opto_stim_coord=="(-5.0, 5.0)", 'time'], cmap=control, label=name[0], s=10)
                ax2.flat[i].scatter(group.loc[(group.opto_stim_coord=="(-5.0, 5.0)") & (group.time==0), 'PC 1'], group.loc[(group.opto_stim_coord=="(-5.0, 5.0)") & (group.time==0), 'PC 3'], s=10, facecolors='none', edgecolors='r')

                y = group.loc[group.opto_stim_coord==stim, 'PC 1']
                x = group.loc[group.opto_stim_coord==stim, 'PC 2']
                z = group.loc[group.opto_stim_coord==stim, 'PC 3']

                ax3.plot(x, y, z, c=color(15))
                ax3.scatter(group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 2'],
                            group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 1'],
                            group.loc[(group.opto_stim_coord==stim) & (group.time==0), 'PC 3'],
                            c='r')
                y = group.loc[group.opto_stim_coord=="(-5.0, 5.0)", 'PC 1']
                x = group.loc[group.opto_stim_coord=="(-5.0, 5.0)", 'PC 2']
                z = group.loc[group.opto_stim_coord=="(-5.0, 5.0)", 'PC 3']

                ax3.plot(x, y, z, c=control(15))
                ax3.scatter(group.loc[(group.opto_stim_coord=="(-5.0, 5.0)") & (group.time==0), 'PC 2'],
                            group.loc[(group.opto_stim_coord=="(-5.0, 5.0)") & (group.time==0), 'PC 1'],
                            group.loc[(group.opto_stim_coord=="(-5.0, 5.0)") & (group.time==0), 'PC 3'],
                            c='r')
                ax.flat[i].set_xlim(-35, 35)
                ax.flat[i].set_ylim(-15, 5)
                ax.flat[i].set_xlabel('PC 1')
                ax.flat[i].set_ylabel('PC 2')
                ax.flat[i].set_title(f"{name}")

                ax1.flat[i].set_ylim(-15, 5)
                ax1.flat[i].set_ylim(-15, 5)
                ax1.flat[i].set_xlabel('PC 2')
                ax1.flat[i].set_ylabel('PC 3')
                ax1.flat[i].set_title(f"{name}")

                ax2.flat[i].set_xlim(-35,35)
                ax2.flat[i].set_ylim(-15, 5)
                ax2.flat[i].set_xlabel('PC 1')
                ax2.flat[i].set_ylabel('PC 3')
                ax2.flat[i].set_title(f"{name}")

                ax3.set_xlim(-15,5)
                ax3.set_ylim(-35,35)
                ax3.set_zlim(-15,5)
                ax3.set_xlabel('PC 2')
                ax3.set_ylabel('PC 1')
                ax3.set_zlabel('PC 3')

            fig.savefig(Path(save_path, f"{trial}_dimensionality_reduction_PC1vsPC2.png"))
            fig.savefig(Path(save_path, f"{trial}_dimensionality_reduction_PC1vsPC2.svg"))

            fig1.savefig(Path(save_path, f"{trial}_dimensionality_reduction_PC2vsPC3.png"))
            fig1.savefig(Path(save_path, f"{trial}_dimensionality_reduction_PC2vsPC3.svg"))

            fig2.savefig(Path(save_path, f"{trial}_dimensionality_reduction_PC1vsPC3.png"))
            fig2.savefig(Path(save_path, f"{trial}_dimensionality_reduction_PC1vsPC3.svg"))
            
            fig3.savefig(Path(save_path, f"{trial}_dimensionality_reduction_3d.png"))
            fig3.savefig(Path(save_path, f"{trial}_dimensionality_reduction_3d.svg"))


def plot_trial_based_pca(control_df, pc_df, result_path):
    roi_list = ['(-0.5, 0.5)', '(-1.5, 0.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(0.5, 4.5)', '(1.5, 1.5)', '(1.5, 3.5)', '(2.5, 2.5)']

    for stim in roi_list:
        fig, ax = plt.subplots(1,2, figsize=(8,4))
        fig.suptitle('PC1')
        fig1, ax1 = plt.subplots(1,2, figsize=(8,4))
        fig1.suptitle('PC2')
        fig2, ax2 = plt.subplots(1,2, figsize=(8,4))
        fig2.suptitle('PC3')

        for i, (name, subgroup) in enumerate(control_df.groupby('context')):
            if name=='rewarded':
                whisker = LinearSegmentedColormap.from_list('', ['#FFFFFF', '#348A18'], N=subgroup.time.unique().shape[0])
                catch = LinearSegmentedColormap.from_list('', ['#FFFFFF', '#000000'], N=subgroup.time.unique().shape[0])
            else:
                whisker = LinearSegmentedColormap.from_list('', ['#FFFFFF', '#6E188A'], N=subgroup.time.unique().shape[0])
                catch = LinearSegmentedColormap.from_list('', ['#FFFFFF', '#000000'], N=subgroup.time.unique().shape[0])

            ax[i].set_title(name)
            ax1[i].set_title(name)
            ax2[i].set_title(name)

            trial = 'whisker_trial'
            group = subgroup[subgroup.trial_type == 'whisker_trial']

            sns.lineplot(group, 
                            x='time', 
                            y='PC 1', 
                            hue='legend', 
                            hue_order = ['(-5.0, 5.0) - no lick', '(-5.0, 5.0) - lick'], 
                            palette=['gray', 'black'], estimator='mean', errorbar=('ci', 95), ax=ax[i])
            sns.lineplot(group, 
                            x='time', 
                            y='PC 2', 
                            hue='legend', 
                            hue_order = ['(-5.0, 5.0) - no lick', '(-5.0, 5.0) - lick'], 
                            palette=['gray', 'black'], estimator='mean', errorbar=('ci', 95), ax=ax1[i])                    
            sns.lineplot(group, 
                            x='time', 
                            y='PC 3', 
                            hue='legend', 
                            hue_order = ['(-5.0, 5.0) - no lick', '(-5.0, 5.0) - lick'], 
                            palette=['gray', 'black'], estimator='mean', errorbar=('ci', 95), ax=ax2[i])                    

            group = pc_df.loc[(pc_df.context==name) & (pc_df.trial_type==trial) & (pc_df.opto_stim_coord==stim)]


            sns.lineplot(group, 
                            x='time', 
                            y='PC 1', 
                            color='royalblue', estimator='mean', errorbar=('ci', 95), ax=ax[i])
            sns.lineplot(group, 
                            x='time', 
                            y='PC 2', 
                            color='royalblue', estimator='mean', errorbar=('ci', 95), ax=ax1[i])                    
            sns.lineplot(group, 
                            x='time', 
                            y='PC 3', 
                           color='royalblue', estimator='mean', errorbar=('ci', 95), ax=ax2[i]) 
                           
            ax[i].set_ylim(-35,35)
            ax[i].set_ylabel('PC 1')

            ax1[i].set_ylim(-15,10)
            ax1[i].set_ylabel('PC 2')

            ax2[i].set_ylim(-15,5)
            ax2[i].set_ylabel('PC 3')

            save_path = os.path.join(result_path, stim)
            if not os.path.exists(save_path):
                os.makedirs(save_path)

        fig.tight_layout()
        fig.savefig(os.path.join(save_path, f"PC1_control_vs_stim_trial.png"))
        fig.savefig(os.path.join(save_path, f"PC1_control_vs_stim_trial.svg"))
        fig1.tight_layout()
        fig1.savefig(os.path.join(save_path, f"PC2_control_vs_stim_trial.png"))
        fig1.savefig(os.path.join(save_path, f"PC2_control_vs_stim_trial.svg"))
        fig2.tight_layout()
        fig2.savefig(os.path.join(save_path, f"PC3_control_vs_stim_trial.png"))
        fig2.savefig(os.path.join(save_path, f"PC3_control_vs_stim_trial.svg"))


def boxplot_quantification_trials(control_df, pc_df, result_path):
    from sklearn.metrics.pairwise import cosine_similarity
    control_df['lick_flag'] = control_df.apply(lambda x: 0 if 'no lick' in x.legend else 1, axis=1)
    control = control_df.loc[(control_df.time>=0) & (control_df.trial_type=='whisker_trial')].groupby(by=['context', 'lick_flag', 'time']).apply('mean').reset_index()
    stim = pc_df.loc[(pc_df.time>=0) & (pc_df.trial_type=='whisker_trial') & (pc_df.opto_stim_coord!='(-5.0, 5.0)')]
    for pc in ['PC 1', 'PC 2', 'PC 3']:
        control[pc] = control[pc] + stim[pc].min()
        stim[pc] = stim[pc] + stim[pc].min()

    result_df = []
    for name, group in stim.groupby(by=['mouse_id', 'context', 'opto_stim_coord']):
        ctrl = control_df.loc[(control_df.mouse_id==name[0]) & (control_df.trial_type=='whisker_trial') & (control_df.time>=0) &(control_df.context==name[1]), ['lick_flag', 'time', 'PC 1', 'PC 2', 'PC 3']]
        context = group.context.unique()[0]
        x_max = control.loc[(control.context==context) & (control.lick_flag==1), 'PC 3'].values[-1]
        x_min = control.loc[(control.context==context) & (control.lick_flag==0), 'PC 3'].values[-1]
        x_std = (group['PC 3'].values[-1]-x_min)/(x_max-x_min)
        lick_sim = np.diag(cosine_similarity(group['PC 3'].reset_index(drop=True).reset_index().to_numpy(), control.loc[(control.context==context) & (control.lick_flag==1), 'PC 3'].reset_index(drop=True).reset_index().to_numpy()))
        nolick_sim = np.diag(cosine_similarity(group['PC 3'].reset_index(drop=True).reset_index().to_numpy(), control.loc[(control.context==context) & (control.lick_flag==0), 'PC 3'].reset_index(drop=True).reset_index().to_numpy()))

        # pc = ['PC 1', 'PC 2', 'PC 3']
        pc=['PC 3']
        v1 = group[pc].to_numpy().flatten()/np.linalg.norm(group[pc].to_numpy().flatten())
        v2 = control.loc[(control.context==context) & (control.lick_flag==1), pc].to_numpy().flatten()/np.linalg.norm(control.loc[(control.context==context) & (control.lick_flag==1), pc].to_numpy().flatten())
        v3 = control.loc[(control.context==context) & (control.lick_flag==0), pc].to_numpy().flatten()/np.linalg.norm(control.loc[(control.context==context) & (control.lick_flag==0), pc].to_numpy().flatten())
        angle_lick = np.degrees(np.arccos(np.clip(np.dot(v1, v2), -1.0, 1.0)))
        angle_nolick = np.degrees(np.arccos(np.clip(np.dot(v1, v3), -1.0, 1.0)))
        angle_control = np.degrees(np.arccos(np.clip(np.dot(v2, v3), -1.0, 1.0)))

        mse_control = np.sum((control.loc[(control.context==context) & (control.lick_flag==1), 'PC 3'].values - control.loc[(control.context==context) & (control.lick_flag==0), 'PC 3'].values)**2)/group.shape[0]
        result={
            'context': context,
            'opto_stim_coord': group.opto_stim_coord.unique()[0],
            'PC3_mse_lick': ((np.sum((group['PC 3'].values - control.loc[(control.context==context) & (control.lick_flag==1), 'PC 3'].values)**2))/group.shape[0])/mse_control,
            'PC3_mse_no_lick': ((np.sum((group['PC 3'].values - control.loc[(control.context==context) & (control.lick_flag==0), 'PC 3'].values)**2))/group.shape[0])/mse_control,
            'PC3_r_lick': np.corrcoef(group['PC 3'].values, control.loc[(control.context==context) & (control.lick_flag==1), 'PC 3'].values)[0,1],
            'PC3_r_no_lick': np.corrcoef(group['PC 3'].values, control.loc[(control.context==context) & (control.lick_flag==0), 'PC 3'].values)[0,1],
            'PC3_distance': x_std * (1 - (-1)) + (-1),
            'PC3_similarity': (lick_sim - nolick_sim).sum(),
            'PC3_angle_lick': angle_lick,
            'PC3_angle_nolick': angle_nolick,
            'PC3_angle_diff': angle_lick - angle_nolick,
            'PC3_angle_control': angle_control
        }
        result_df += [result]
    result_df = pd.DataFrame(result_df)

    for pc in ['PC3']:
        result_df[f'{pc}_mse'] = result_df[f'{pc}_mse_no_lick'] - result_df[f'{pc}_mse_lick']
        result_df[f'{pc}_r'] = result_df[f'{pc}_r_lick'] - result_df[f'{pc}_r_no_lick']

    subset_df = result_df[result_df.opto_stim_coord.isin(['(-0.5, 0.5)', '(-1.5, 0.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(1.5, 3.5)', '(0.5, 4.5)', '(1.5, 1.5)', '(2.5, 2.5)'])]
    comparisons = subset_df.opto_stim_coord.unique().shape[0]*2

    save_path = os.path.join(result_path, 'quantification')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    stats_df = []
    for name, group in subset_df.groupby(by=['opto_stim_coord', 'context']):
        from scipy.stats import ttest_1samp
        t, p = ttest_1samp(group.PC3_mse, 0)
        stats = {
            'metric': 'mse',
            'coord': name[0],
            'context': name[1],
            'dof': group.shape[0]-1,
            'mean': group.PC3_mse.mean(),
            'std': group.PC3_mse.std(),
            't': t,
            'p': p,
            'alpha': 0.05,
            'alpha_corr': 0.05/comparisons,
            'p_corr': p*comparisons,
            'significant': True if p<=0.05/comparisons else False,
            'd': abs(group.PC3_mse.mean()/group.PC3_mse.std())
        }

        stats_df += [stats]
    stats_df = pd.DataFrame(stats_df)
    stats_df.to_csv(os.path.join(save_path, 'stats.csv'))

    g = sns.catplot(
        subset_df,
        x='context',
        y='PC3_mse',
        hue='context',
        hue_order=['non-rewarded', 'rewarded'],
        palette=['purple', 'green'],
        col='opto_stim_coord',
        kind='violin',
        inner="point"
    )
    for ax in g.axes.flat:
        ax.axhline(y=0, xmin=0, xmax=1, ls='--', c='gray')

    g.figure.savefig(os.path.join(save_path, 'PC3_mse_trials.png'))

    g = sns.catplot(
        subset_df,
        x='context',
        y='PC3_r',
        hue='context',
        hue_order=['non-rewarded', 'rewarded'],
        palette=['purple', 'green'],
        col='opto_stim_coord',
        kind='violin',
        inner="point"
    )

    g.figure.savefig(os.path.join(save_path, 'PC3_r_trials.png'))

    g = sns.catplot(
        subset_df,
        y='PC3_distance',
        x='context',
        hue='context',
        hue_order=['non-rewarded', 'rewarded'],
        palette=['purple', 'green'],
        col='opto_stim_coord',
        kind='violin',
        inner="point"
    )

    g.figure.savefig(os.path.join(save_path, 'PC3_distance_trials.png'))


def dimensionality_reduction(nwb_files, output_path):
    result_path = Path(output_path, 'PCA_150')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    coords_list = {'wS1': "(-1.5, 3.5)", 'wS2': "(-1.5, 4.5)", 'wM1': "(1.5, 1.5)", 'wM2': "(2.5, 1.5)", 'RSC': "(-0.5, 0.5)", "RSC_2": "(-1.5, 0.5)",
            'ALM': "(2.5, 2.5)", 'tjS1':"(0.5, 4.5)", 'tjM1':"(1.5, 3.5)", 'control': "(-5.0, 5.0)"}


    group = 'controls' if 'control' in str(output_path) else 'VGAT'
    opto_avg_df = load_opto_data(group)

    total_df = load_wf_opto_data(nwb_files, output_path)
    total_df.context = total_df.context.map({0:'non-rewarded', 1:'rewarded'})
    total_df['time'] = [[np.linspace(-1,3.98,250)] for i in range(total_df.shape[0])]
    total_df['legend']= total_df.apply(lambda x: f"{x.opto_stim_coord} - {'lick' if x.lick_flag==1 else 'no lick'}",axis=1)

    d = {c: lambda x: x.unique()[0] for c in ['opto_stim_loc', 'legend']}
    d['time'] = lambda x: list(x)[0][0]
    for c in ['(-0.5, 0.5)', '(-0.5, 1.5)', '(-0.5, 2.5)', '(-0.5, 3.5)', '(-0.5, 4.5)', '(-0.5, 5.5)', '(-1.5, 0.5)',
       '(-1.5, 1.5)', '(-1.5, 2.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(-1.5, 5.5)', '(-2.5, 0.5)', '(-2.5, 1.5)', '(-2.5, 2.5)',
       '(-2.5, 3.5)', '(-2.5, 4.5)', '(-2.5, 5.5)', '(-3.5, 0.5)', '(-3.5, 1.5)', '(-3.5, 2.5)', '(-3.5, 3.5)', '(-3.5, 4.5)',
       '(-3.5, 5.5)', '(0.5, 0.5)', '(0.5, 1.5)', '(0.5, 2.5)', '(0.5, 3.5)', '(0.5, 4.5)', '(0.5, 5.5)', '(1.5, 0.5)', 
       '(1.5, 1.5)', '(1.5, 2.5)', '(1.5, 3.5)', '(1.5, 4.5)', '(1.5, 5.5)', '(2.5, 0.5)', '(2.5, 1.5)', '(2.5, 2.5)', '(2.5, 3.5)', '(2.5, 4.5)', '(2.5, 5.5)', 
       'A1', 'ALM', 'tjM1', 'tjS1', 'RSC', 'wM1', 'wM2', 'wS1', 'wS2']:
        d[f"{c}"]= lambda x: np.nanmean(np.stack(x), axis=0)
          
    mouse_df = total_df.groupby(by=['mouse_id', 'context', 'trial_type', 'opto_stim_coord', 'lick_flag']).agg(d).reset_index()
    mouse_df = mouse_df.melt(id_vars=['mouse_id', 'context', 'trial_type', 'legend', 'opto_stim_coord', 'lick_flag', 'time'],
                                 value_vars=['(-0.5, 0.5)', '(-0.5, 1.5)', '(-0.5, 2.5)', '(-0.5, 3.5)', '(-0.5, 4.5)', '(-0.5, 5.5)', '(-1.5, 0.5)',
       '(-1.5, 1.5)', '(-1.5, 2.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(-1.5, 5.5)', '(-2.5, 0.5)', '(-2.5, 1.5)', '(-2.5, 2.5)',
       '(-2.5, 3.5)', '(-2.5, 4.5)', '(-2.5, 5.5)', '(-3.5, 0.5)', '(-3.5, 1.5)', '(-3.5, 2.5)', '(-3.5, 3.5)', '(-3.5, 4.5)',
       '(-3.5, 5.5)', '(0.5, 0.5)', '(0.5, 1.5)', '(0.5, 2.5)', '(0.5, 3.5)', '(0.5, 4.5)', '(0.5, 5.5)', '(1.5, 0.5)', 
       '(1.5, 1.5)', '(1.5, 2.5)', '(1.5, 3.5)', '(1.5, 4.5)', '(1.5, 5.5)', '(2.5, 0.5)', '(2.5, 1.5)', '(2.5, 2.5)', '(2.5, 3.5)', '(2.5, 4.5)', '(2.5, 5.5)',],
                                 var_name='roi',
                                 value_name='dff0').explode(['time', 'dff0'])
    mouse_df = mouse_df[(mouse_df.time>=-0.15)&(mouse_df.time<=0.15)]
    
    avg_df = mouse_df.groupby(by=['context', 'trial_type', 'lick_flag', 'legend', 'opto_stim_coord', 'roi', 'time']).agg(lambda x: np.nanmean(x)).reset_index()
    avg_df.time = avg_df.time.round(2)
    subset_df = avg_df[(avg_df.trial_type.isin(['whisker_trial', 'no_stim_trial'])) & (avg_df.opto_stim_coord=="(-5.0, 5.0)")].pivot(index=['context','trial_type', 'legend', 'time'], columns='roi', values='dff0')
    avg_data_for_pca = subset_df.to_numpy()
    labels = subset_df.keys()

    # Standardize average data for training: Based on trials with light on control location 
    scaler = StandardScaler()
    fit_scaler = scaler.fit(avg_data_for_pca)
    avg_data_for_pca = fit_scaler.transform(avg_data_for_pca)
    pca = PCA(n_components=15)
    results = pca.fit(np.nan_to_num(avg_data_for_pca))
    principal_components = pca.transform(np.nan_to_num(avg_data_for_pca))

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
    # Plot coefficients, biplots and variance explained
    plot_pca_stats(pca, result_path)

    project_data = 'trials'
    # Project whisker and catch trials 
    total_df['time'] = total_df.time = total_df.apply(lambda x: np.asarray(x.time[0]), axis=1)
    df = total_df.melt(id_vars=['mouse_id', 'session_id', 'trial_id', 'context', 'trial_type', 'opto_stim_coord', 'legend', 'time'],
                                value_vars=['(-0.5, 0.5)', '(-0.5, 1.5)', '(-0.5, 2.5)', '(-0.5, 3.5)', '(-0.5, 4.5)', '(-0.5, 5.5)', '(-1.5, 0.5)',
    '(-1.5, 1.5)', '(-1.5, 2.5)', '(-1.5, 3.5)', '(-1.5, 4.5)', '(-1.5, 5.5)', '(-2.5, 0.5)', '(-2.5, 1.5)', '(-2.5, 2.5)',
    '(-2.5, 3.5)', '(-2.5, 4.5)', '(-2.5, 5.5)', '(-3.5, 0.5)', '(-3.5, 1.5)', '(-3.5, 2.5)', '(-3.5, 3.5)', '(-3.5, 4.5)',
    '(-3.5, 5.5)', '(0.5, 0.5)', '(0.5, 1.5)', '(0.5, 2.5)', '(0.5, 3.5)', '(0.5, 4.5)', '(0.5, 5.5)', '(1.5, 0.5)', 
    '(1.5, 1.5)', '(1.5, 2.5)', '(1.5, 3.5)', '(1.5, 4.5)', '(1.5, 5.5)', '(2.5, 0.5)', '(2.5, 1.5)', '(2.5, 2.5)', '(2.5, 3.5)', '(2.5, 4.5)', '(2.5, 5.5)',],
                                var_name='roi',
                                value_name='dff0').explode(['time', 'dff0'])
    df = df.reset_index()
    df = df[(df.time>=-0.15)&(df.time<=0.15)]
    subset_df = df[df.trial_type.isin(['whisker_trial', 'no_stim_trial'])].pivot(
        index=['mouse_id', 'session_id', 'trial_id', 'context','trial_type', 'opto_stim_coord', 'legend', 'time'], columns='roi', values='dff0')    
            
    avg_data_for_pca = subset_df.to_numpy()
    avg_data_for_pca = fit_scaler.transform(avg_data_for_pca)
    principal_components = pca.transform(np.nan_to_num(avg_data_for_pca))

    pc_df = pd.DataFrame(data=principal_components, index=subset_df.index)
    pc_df.columns = [f"PC {i+1}" for i in range(0, principal_components.shape[1])]
    subset_df = subset_df.join(pc_df).reset_index()
    pc_df = pc_df.reset_index()
    ## Plot PC timecourses

    color_dict = {"(-5.0, 5.0)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#000000'], N=subset_df.time.unique().shape[0]),
                "(-1.5, 3.5)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#011f4b'], N=subset_df.time.unique().shape[0]),
                "(-1.5, 4.5)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#03396c'], N=subset_df.time.unique().shape[0]),
                "(1.5, 1.5)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#005b96'], N=subset_df.time.unique().shape[0]),
                "(2.5, 1.5)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#6497b1'], N=subset_df.time.unique().shape[0]),
                "(2.5, 2.5)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#b3cde0'], N=subset_df.time.unique().shape[0]),
                "(0.5, 4.5)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#ffccd5'], N=subset_df.time.unique().shape[0]),
                "(1.5, 3.5)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#c9184a'], N=subset_df.time.unique().shape[0]),
                "(-0.5, 0.5)": LinearSegmentedColormap.from_list('', ['#FFFFFF', '#590d22'], N=subset_df.time.unique().shape[0])
    }

    long_df = subset_df.melt(id_vars=['mouse_id', 'context', 'trial_type', 'opto_stim_coord', 'time'], 
                            value_vars=['PC 1', 'PC 2', 'PC 3'], 
                            var_name='PC', 
                            value_name='data').explode(['time', 'data'])
    long_df = long_df[long_df.opto_stim_coord.isin(list(coords_list.values()))]
    plot_timecourses_by_outcome(long_df, result_path)

    ## Plot projected time courses onto PCx vs PCy
    plot_projected_pc_timecourses(subset_df.groupby(by=['context', 'trial_type', 'opto_stim_coord', 'time']).agg(lambda x: np.nanmean(x)).reset_index(), color_dict, result_path)

    ## Control vs stim one by one projections
    plot_trajectories_by_region(subset_df.groupby(by=['context', 'trial_type', 'opto_stim_coord', 'time']).agg(lambda x: np.nanmean(x)).reset_index(), color_dict, result_path)

    # control_df = pc_df[pc_df.opto_stim_coord=="(-5.0, 5.0)"]
    pc_df = pc_df.groupby(by=['mouse_id', 'context', 'trial_type', 'opto_stim_coord', 'time'], as_index=False, sort=False).agg('mean')
    plot_trial_based_pca(control_df, pc_df[pc_df.opto_stim_coord!="(-5.0, 5.0)"], result_path)
    boxplot_quantification_trials(control_df, pc_df[pc_df.opto_stim_coord!="(-5.0, 5.0)"], result_path)


def main(nwb_files, output_path):
    # combine_data(nwb_files, output_path)
    plot_example_stim_images(nwb_files, output_path)
    dimensionality_reduction(nwb_files, output_path)


if __name__ == "__main__":

    for file in ['context_sessions_wf_opto']: #, 'context_sessions_wf_opto_controls', 'context_sessions_wf_opto_photoactivation'
        config_file = f"//sv-nas1.rcp.epfl.ch/Petersen-Lab/analysis/Pol_Bech/Session_list/{file}.yaml"
        config_file = haas_pathfun(config_file)

        with open(config_file, 'r', encoding='utf8') as stream:
            config_dict = yaml.safe_load(stream)

        nwb_files = [haas_pathfun(p.replace("\\", '/')) for p in config_dict['Session path']]
        
        output_path = os.path.join('//sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Pol_Bech', 'Pop_results',
                                    'Context_behaviour', 'optogenetic_widefield_results', 'controls' if 'controls' in str(config_file) else 'VGAT')        
        # output_path = os.path.join('//sv-nas1.rcp.epfl.ch', 'Petersen-Lab', 'analysis', 'Pol_Bech', 'Pop_results',
        #                             'Context_behaviour', 'optogenetic_widefield_results', 'photoactivation')
        output_path = haas_pathfun(output_path)

        if not os.path.exists(output_path):
            os.makedirs(output_path)

        main(nwb_files, output_path)