from codes.utils.misc.plot_average_widefield_timecourse import plot_average_wf_timecourse


def figure3f(data, saving_path, formats=['png']):
    # Keep only correct trial types:
    trial_types = ['rewarded_whisker_hit_trial', 'non-rewarded_whisker_miss_trial']

    # Plot
    plot_average_wf_timecourse(data, trial_types, saving_path, formats=formats)

