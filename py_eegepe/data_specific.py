from pathlib import Path
import numpy as np
import subprocess
import pickle


class git_label():
    def __init__(self):
        try:
            self.label_ = subprocess.check_output(["git", "describe", "--tag"]).strip().decode("utf-8")
        except:
            self.label_ = 'vx.x.x'
            print('Not git tag found! Setting to {}\n'.format(self.label_))

    def __str__(self):
        return str(self.label_)

    @property
    def label(self):
        return self.label_

    @property
    def tag(self):
        return self.label_.split('-')[0]


git_label_ = git_label()  # this is a global...


def specifier(dataset, subject, ix_run):
    """
    Specifies where the data of interest is, and what form it takes relative to the dataset folder.

    Returns the path relative to dataset for a given subject, and the name of the file to be loaded.
    """
    subject_path = ''
    run_name = ''
    if str(dataset) == 'dataset_example':
        subject_path = Path('data/{}/EEG'.format(subject))
        if ix_run is None:
            run_name = None
        else:
            run_name = Path('example{:02d}.bdf'.format(ix_run))
    elif str(dataset) == '2017-04-21_LIINC_go_nogo':
        subject_path = Path('data/{}/EEG'.format(subject))
        if ix_run is None:
            run_name = None
        else:
            run_name = Path('aud-{:02d}.bdf'.format(ix_run))
    elif str(dataset) == '2019-02-29_healthy_brain_network':
        subject_path = Path('data/{}/EEG/raw/eeglab_format/'.format(subject))
        run_name = Path('RestingState.set')
    else:
        raise Exception("dataset mispecified.\nsubject_path: {}\nrun_name: {}".format(subject_path, run_name))

    return subject_path, run_name


def specifier_proc(dataset, subject, ix_run, es):
    """
    Like specifier, but returns the equivalent folder structure for processing.
    Unlike for specifier, these folders do not need to exist a-priori. For convenience, they generally take a similar
    structure to what is written for the specifier function.
    """
    if str(dataset) == 'dataset_example':
        subject_path = Path('proc_{}/{}/EEG'.format(git_label_.tag, subject))
        if ix_run is None:
            run_name = Path('exampleAS{}'.format(es))
        else:
            run_name = Path('example{:02d}{}'.format(ix_run, es))

    elif str(dataset) == '2017-04-21_LIINC_go_nogo':
        subject_path = Path('proc_{}/{}/EEG'.format(git_label_.tag, subject))
        if ix_run is None:
            run_name = Path('aud-nn{}'.format(es))
        else:
            run_name = Path('aud-{:02d}{}'.format(ix_run, es))
    elif str(dataset) == '2019-02-29_healthy_brain_network':
        subject_path = Path('proc_{}/{}/EEG/raw/eeglab_format/'.format(git_label_.tag, subject))
        run_name = Path('RestingState-nn{}'.format(es))
    else:
        raise Exception("dataset mispecified")

    return subject_path, run_name


def _sublist(datadir, dataset):
    """
    Return the list of subjects that we are interested in for a given dataset.
    """
    d = datadir / dataset

    d_proc = specifier_proc(dataset, '', None, '')[0].parts[0]
    d_data = specifier(dataset, '', None)[0].parts[0]
    dabs_data = d / d_data
    f_sub_list = d / d_proc / 'sub_list.dat'

    if dabs_data.exists():
        if str(dataset) == '2017-04-21_LIINC_go_nogo':
            p = d.glob('*{}/*/'.format(d_data))
            sub_list = [p_.parts[-1] for p_ in p if p_.is_dir()]
            # Excluding subject for different montage + data has different naming convention + can't find events
            sub_list.remove('2011-07-01_EEG_Eyelink(BC)')

        elif str(dataset) == '2019-02-29_healthy_brain_network':
            p = d.glob('*{}/*/'.format(d_data))
            sub_list = [p_.parts[-1] for p_ in p if p_.is_dir()]
            sub_list_temp = sub_list.copy()
            for ix in range(len(sub_list_temp)):
                f = dabs_data / sub_list_temp[ix] / 'EEG' / 'raw' / 'eeglab_format' / 'RestingState.set'
                if not f.exists():
                    sub_list.remove(sub_list_temp[ix])

        if not f_sub_list.parents[0].exists():
            f_sub_list.parents[0].mkdir()

        with open(str(f_sub_list), 'wb') as handle:
            pickle.dump(sub_list, handle)

    else:
        try:
            with open(str(f_sub_list), 'rb') as handle:
                sub_list = pickle.load(handle)
        except EnvironmentError:
            print('Data not found, so tried to load pre-made subject list, but that was also not found.')

    return sub_list


def preprocessor(dataset, f_i, eeg_setting=None):
    """
    Data pre-processing goes here for any given dataset.
    This should isolate the signal of interest (s, e.g. occipital alpha)
    and specify the events at which phase should be estimated. Also returns time (t) and sampling rate fs.

    The eeg_setting field that specifies which signal is of itnerest comes from opt_init in paradigm.
    """
    if str(dataset) == '2017-04-21_LIINC_go_nogo':
        # same dataset as 2017-04-21_EEG_linbi, and computationally identical loading (just cleaned up a bit)
        import mne
        raw = mne.io.read_raw_edf(str(f_i), preload=True)

        # should check the impact of this, but we just assume it's minimal for now
        # since this dataset is 2k it's a bit hard to work with directly
        # also note that changing this could mean the network arch. has to change
        fs = 512
        raw.resample(fs)

        ch_ev = np.where([x == 'Status' for x in raw.ch_names])[0][0]
        raw._data[ch_ev, :] = raw._data[ch_ev, :] - np.min(raw._data[ch_ev, :])
        raw._data[ch_ev, raw._data[ch_ev, :] == np.max(raw._data[ch_ev, :])] = -1

        events = mne.find_events(raw)

        rel_events = events[:, 1] == 0
        events = events[rel_events]

        # We don't care about actual recovery of event phase for this data set application - we only care about how
        # well we are recovering. Then subsample the events so that we can apply a stronger acausal filter.
        # Note that we also don't predict at these events, but at a shifted point for evaluation (see core.core_shared)
        events = events[1::2, :]

        n_ch = 64
        montage = mne.channels.read_montage('biosemi64')
        montage.selection = montage.selection[:n_ch]
        if '2011-07-01_EEG_Eyelink(BC)' in str(f_i):
            # this subject has a weird montage setup
            raw.drop_channels(raw.ch_names[n_ch:])
        else:
            raw.rename_channels(dict(zip(raw.ch_names, montage.ch_names)))
        raw.set_montage(montage)

        if eeg_setting == 'occipital_alpha_hjorth':
            eeg_focus_mne = raw.copy().pick_channels(['POz'])
            eeg_ref_mne = raw.copy().pick_channels(['PO3', 'PO4', 'Pz', 'Oz'])
        elif eeg_setting == 'motor_mu_hjorth':
            eeg_focus_mne = raw.copy().pick_channels(['C3'])
            eeg_ref_mne = raw.copy().pick_channels(['C1', 'C5', 'FC3', 'CP3'])
        elif eeg_setting == 'motor_beta_hjorth':
            eeg_focus_mne = raw.copy().pick_channels(['C3'])
            eeg_ref_mne = raw.copy().pick_channels(['C1', 'C5', 'FC3', 'CP3'])
        else:
            raise Exception("eeg_setting incorrect")

        s, t = eeg_focus_mne[:, :]
        s_ref, _ = eeg_ref_mne[:, :]
        s = s - np.mean(s_ref, 0)
    elif str(dataset) == '2019-02-29_healthy_brain_network':
        import mne
        raw = mne.io.read_raw_eeglab(str(f_i), preload=True)

        fs = 512
        raw.resample(fs)

        mon_ = mne.channels.make_standard_montage('GSN-HydroCel-129')
        raw.set_montage(mon_)
        # raw.plot_sensors()
        #  make up some events - where we hypothetically want to predict phase:
        events = np.arange(10*fs, raw.n_times - 10*fs, 6 * fs).reshape(-1, 1).astype(int)

        if eeg_setting == 'occipital_alpha_car':
            raw_ref, _ = mne.set_eeg_reference(raw, ref_channels='average')
            eeg_focus_mne = raw_ref.copy().pick_channels(['E75'])
            # eeg_ref_mne = raw.copy().pick_channels(['E74', 'E82', 'E70', 'E83', 'E71', 'E76'])
        # elif eeg_setting == 'occipital_mu_hjorth':
        #     raise Exception('not implemented')
        elif eeg_setting == 'motor_mu_hjorth':
            raise Exception('not implemented')
        elif eeg_setting == 'motor_beta_hjorth':
            raise Exception('not implemented')
        else:
            raise Exception("eeg_setting incorrect")

        s, t = eeg_focus_mne[:, :]
        # s_ref, _ = eeg_ref_mne[:, :]
        # s = s - np.mean(s_ref, 0)

        # TODO: TEMPORARY! C-FIR results were quite poor
        #  test with a lowpass in case it's the line noise
        # from scipy import signal
        # b, a = signal.firwin(int(fs/2), 50, fs=fs), 1
        # s = signal.filtfilt(b, a, s)

    # elif str(dataset) == '2017-04-21_EEG_linbi':
    #     less clear (computationally identical) version of event selection - used for manuscript
    #     import mne
    #     raw = mne.io.read_raw_edf(str(f_i), preload=True)
    #     n_ch = 64
    #
    #     # should check that biosemi inbuilt is the same as the biosemi file
    #     montage = mne.channels.read_montage('biosemi64')
    #     montage.selection = montage.selection[:n_ch]
    #
    #     if '2011-07-01_EEG_Eyelink(BC)' in str(f_i):
    #         # this subject has a weird montage setup
    #         raw.drop_channels(raw.ch_names[n_ch:])
    #     else:
    #         raw.rename_channels(dict(zip(raw.ch_names, montage.ch_names)))
    #
    #     raw.set_montage(montage)
    #
    #     # ideally want to do a scalp laplacian here
    #
    #     # should check the impact of this, but we just assume it's minimal for now
    #     # since this dataset is 2k it's a bit hard to work with directly
    #     # also note that changing this could mean the network arch. has to change
    #     fs = 512
    #     raw.resample(fs)
    #
    #     events = mne.find_events(raw)
    #     rel_events = events[:, 1] == 16128
    #     events = events[rel_events]
    #     events = events[1::2, :]
    #
    #     # do hjorth
    #     if eeg_setting == 'occipital_alpha_hjorth':
    #         eeg_focus_mne = raw.copy().pick_channels(['POz'])
    #         eeg_ref_mne = raw.copy().pick_channels(['PO3', 'PO4', 'Pz', 'Oz'])
    #     elif eeg_setting == 'motor_mu_hjorth':
    #         eeg_focus_mne = raw.copy().pick_channels(['C3'])
    #         eeg_ref_mne = raw.copy().pick_channels(['C1', 'C5', 'FC3', 'CP3'])
    #     elif eeg_setting == 'motor_beta_hjorth':
    #         eeg_focus_mne = raw.copy().pick_channels(['C3'])
    #         eeg_ref_mne = raw.copy().pick_channels(['C1', 'C5', 'FC3', 'CP3'])
    #     else:
    #         raise Exception("eeg_setting incorrect")
    #
    #     s, t = eeg_focus_mne[:, :]
    #     s_ref, _ = eeg_ref_mne[:, :]
    #     s = s - np.mean(s_ref, 0)


    else:
        raise Exception("dataset mispecified in preproc")
        # maybe can now clear raw from memory
    return s, t, events, fs
