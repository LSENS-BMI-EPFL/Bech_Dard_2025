from codes.utils.misc.plot_average_widefield_timecourse import plot_average_wf_timecourse


def figure3d(data, saving_path, formats=['png']):
    trial_types = list(data[list(data.keys())[0]].keys())

    # Keep only correct:
    trial_types = [trial for trial in trial_types if 'hit' in trial]

    # Plot
    plot_average_wf_timecourse(data, trial_types, saving_path, formats=formats)

        