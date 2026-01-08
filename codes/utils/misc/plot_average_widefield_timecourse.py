import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from codes.utils.misc.plot_on_allen import plot_wf_avg, plot_wf_single_frame
from codes.utils.misc.stats import compute_dprime, first_above_threshold_frame


def plot_average_wf_timecourse(data, trial_types, saving_path, formats=['png'],
                               scale=(-0.035, 0.035), diff_range=0.015):
    rewarded_idx = np.where(['non-rewarded' not in trial for trial in trial_types])[0]
    rewarded_key = np.array(trial_types)[rewarded_idx].tolist()[0]
    non_rewarded_idx = np.where(['non-rewarded' in trial for trial in trial_types])[0]
    non_rewarded_key = np.array(trial_types)[non_rewarded_idx].tolist()[0]

    mice_avg_data_dict = dict()
    for trial in trial_types:
        mice_avg_data_dict.setdefault(trial, [])

    for mouse_id, trial_type_data_dict in data.items():
        print(f'Mouse: {mouse_id}')
        for trial_type, mouse_data in trial_type_data_dict.items():
            print(f'Trial type: {trial_type}, {len(mouse_data)} sessions')
            if trial_type not in trial_types:
                print('Trial type not correct')
                continue
            mouse_avg_data = [mouse_data[sess][1] for sess in range(len(mouse_data))]
            mouse_avg_data = np.stack(mouse_avg_data)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mouse_avg_data = np.nanmean(mouse_avg_data, axis=0)
            mice_avg_data_dict[trial_type].append(mouse_avg_data)
    rew_data_dict = mice_avg_data_dict[rewarded_key]
    norew_data_dict = mice_avg_data_dict[non_rewarded_key]

    for trial_type, data in mice_avg_data_dict.items():
        data = np.stack(data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            data = np.nanmean(data, axis=0)
        mice_avg_data_dict[trial_type] = data

    # TIMECOURSES FOR EACH CONTEXT
    for key, data in mice_avg_data_dict.items():
        plot_wf_avg(avg_data=data, output_path=saving_path, n_frames_post_stim=12, n_frames_averaged=2, key=key,
                    center_frame=10,
                    c_scale=scale, figname=f'all_mice_{key}', save_formats=formats, subdir=key)

    if 'supplementary' not in saving_path:
        # TIMECOURSES D'
        rew_data = np.stack(rew_data_dict)
        norew_data = np.stack(norew_data_dict)
        d_prime = compute_dprime(rew_data, norew_data)
        plot_wf_avg(avg_data=d_prime, output_path=saving_path, n_frames_post_stim=12, n_frames_averaged=2,
                    key='dprime', center_frame=10,
                    colormap='viridis', c_scale=(0, 3.5), figname='all_mice_dprime', save_formats=formats, subdir='dprime')

        if os.path.basename(saving_path) == '3F':
            # FIRST TIME D' ABOVE 2
            first_frame = first_above_threshold_frame(d=d_prime, threshold=2, start_frame=12)
            first_frame = first_frame / 100 + 2 / 100
            filtered_first_frame = gaussian_filter(first_frame, sigma=1)
            t_sig_save_path = os.path.join(saving_path, 'dprime')
            if not os.path.exists(t_sig_save_path):
                os.makedirs(t_sig_save_path)
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            plot_wf_single_frame(frame=filtered_first_frame, title='Figure3H_map', figure=fig, ax_to_plot=ax, suptitle=' ',
                                 saving_path=os.path.dirname(saving_path), save_formats=['png'], colormap='binary_r',
                                 vmin=0.03, vmax=0.08, cbar_shrink=0.7, separated_plots=True, nan_c='white')


def plot_wf_timecourse_aud_wh_diff(aud_data, wh_data, aud_trials, wh_trials, saving_path, formats=['png'],
                                   diff_range=0.03):
    # WHISKER DATA
    mice_avg_data_dict = dict()
    for trial in wh_trials:
        mice_avg_data_dict.setdefault(trial, [])

    for mouse_id, trial_type_data_dict in wh_data.items():
        print(f'Mouse: {mouse_id}')
        for trial_type, mouse_data in trial_type_data_dict.items():
            if trial_type not in wh_trials:
                continue
            print(f'Trial type: {trial_type}, {len(mouse_data)} sessions')
            mouse_avg_data = [mouse_data[sess][1] for sess in range(len(mouse_data))]
            mouse_avg_data = np.stack(mouse_avg_data)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mouse_avg_data = np.nanmean(mouse_avg_data, axis=0)
            mice_avg_data_dict[trial_type].append(mouse_avg_data)

    for trial_type, data in mice_avg_data_dict.items():
        data = np.stack(data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            data = np.nanmean(data, axis=0)
        mice_avg_data_dict[trial_type] = data

    # AUDITORY DATA
    mice_avg_data_dict_auditory = dict()
    for trial in aud_trials:
        mice_avg_data_dict_auditory.setdefault(trial, [])

    for mouse_id, trial_type_data_dict in aud_data.items():
        print(f'Mouse: {mouse_id}')
        for trial_type, mouse_data in trial_type_data_dict.items():
            print(f'Trial type: {trial_type}, {len(mouse_data)} sessions')
            if trial_type not in aud_trials:
                print('Trial type not correct')
                continue
            mouse_avg_data = [mouse_data[sess][1] for sess in range(len(mouse_data))]
            mouse_avg_data = np.stack(mouse_avg_data)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                mouse_avg_data = np.nanmean(mouse_avg_data, axis=0)
            mice_avg_data_dict_auditory[trial_type].append(mouse_avg_data)

    for trial_type, data in mice_avg_data_dict_auditory.items():
        data = np.stack(data)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            data = np.nanmean(data, axis=0)
        mice_avg_data_dict_auditory[trial_type] = data

    data_to_plot = mice_avg_data_dict['rewarded_whisker_hit_trial'] - mice_avg_data_dict_auditory[
        'rewarded_auditory_hit_trial']

    plot_wf_avg(avg_data=data_to_plot, output_path=saving_path, n_frames_post_stim=12, n_frames_averaged=2,
                key='Reviewing', center_frame=10, halfrange=diff_range, colormap='seismic',
                figname=f'all_mice_whisker_auditory_diff',
                save_formats=formats, subdir='whisker_auditory_diff')

