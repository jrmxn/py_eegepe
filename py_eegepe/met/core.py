# from pathlib import Path
from scipy import signal
import numpy as np
from . import met_fir, met_net, met_reg, met_ukf, met_shared
from .. import data_loader
from .. import sp


# from .. import sp
# import matplotlib.pyplot as plt


def core_shared(pro, meta, predict_ev, use_ground_truth, preproc_type, predict_amplitude,
                use_above_alphasnr=None, ix_shift_input=0):
    """
    Split up data into epochs for training. It's 'shared' because it gets used by both training and testing.

    if ix_shift_input < 0, then we are still trying to predict the phase at the event
    but we are shifting the data that we are predicting it from to the left
    e.g. if before we were trying to predict phase at ix 0, with ix_shift_input = 0, we are trying to do this with
    data up to and including ix 0. If we set ix_shift_input = -10, then we still predict phase at ix = 0, but now our
    last input data point will stop at -10. This doesn't have to be handled in a special way by the network, but does
    need to be approached with care when using FIR methods
    assert ix_shift_input == 0, "ix_shift_input==0 is fully implemented, but untested. Was set to: {}".format(ix_shift_input)

    This is pretty key - by default scaling is 2 for training and 1 for testing.
    For actual prediction we want it to be 1 for training and 0 for testing.
    It sets how many filter lengths before an event is allowed to be seen by training/testing
    """

    if predict_ev['scaling_type'] == 'alpha_filter':
        ix_back = predict_ev['scaling'] * meta['alpha_filter']['order']
    else:
        raise Exception('ix_back not defined?')

    s = pro['s']
    t = pro['t']

    if use_ground_truth:
        assert np.shape(pro['s_alpha_gt'])[0] == np.shape(pro['s'])[0], 's_alpha_gt is the wrong shape'
        s_alpha_acausal = pro['s_alpha_gt']
    else:
        assert np.shape(pro['s_alpha_acausal'])[0] == np.shape(pro['s'])[0], 's_alpha_acausal is the wrong shape'
        s_alpha_acausal = pro['s_alpha_acausal']

    if predict_amplitude:
        h_alpha_acausal = signal.hilbert(s_alpha_acausal)
    else:
        h_alpha_acausal = np.exp(1j * np.angle(signal.hilbert(s_alpha_acausal)))

    event_ix = [meta['events'][ix][0] for ix in range(len(meta['events']))]
    event_ix_input = [ix + ix_shift_input for ix in event_ix]
    s_preproc = met_shared.met_preproc(s, meta['fs'], met_try=preproc_type)
    x_ev, _ = met_shared.slice_signal(event_ix_input, s_preproc, t, ix_back=ix_back)
    h_ev, _ = met_shared.slice_signal(event_ix, h_alpha_acausal, t, ix_back=ix_back)

    # get alpha to neighbour snr, and then take only trials that are higher than the provided percentile
    s_mat = np.zeros((len(x_ev), int(1 + meta['fs']/2)))
    for ix in range(len(x_ev)):
        f_spec, s_mat[ix, :] = sp.welch(x_ev[ix], meta['fs'], int(meta['fs']))
    f_ix_lower = np.all(np.stack((f_spec >= 5.5, f_spec <= 8)), axis=0)
    f_ix_upper = np.all(np.stack((f_spec >= 13, f_spec <= 15.5)), axis=0)
    f_ix_edges = np.any(np.stack((f_ix_lower, f_ix_upper)), axis=0)
    f_ix_middle = np.all(np.stack((f_spec > 8, f_spec < 13)), axis=0)
    p_alpha = np.mean(s_mat[:, f_ix_middle], axis=1)
    p_edges = np.mean(s_mat[:, f_ix_edges], axis=1)
    p_ratio = p_alpha / p_edges

    if use_above_alphasnr:  # should never be true at test time
        p_ratio_th = np.percentile(p_ratio, use_above_alphasnr)
        ix_keep = p_ratio > p_ratio_th
        x_ev = [x_ev[ix] for ix in range(len(ix_keep)) if ix_keep[ix]]
        h_ev = [h_ev[ix] for ix in range(len(ix_keep)) if ix_keep[ix]]

    return x_ev, h_ev, p_ratio


def core_arch_select(pro, meta, predict_ev, arch, overwrite_met, f_arch, opt_met=None,
                     ix_shift_input=0, predict_amplitude=False, use_ground_truth_training=False,
                     use_above_alphasnr=None):
    """
    Takes in data and a specified model architecture to use, and returns the actual model.
    Makes a call to core_shared with the data to split it up as needed for training.
    Note that the returned model is augment with some information here - most importantly, a handle to the
    method function so that for the evaluation call, the correct test function is used.

    :param pro:
    :param meta:
    :param predict_ev:
    :param arch:
    :param overwrite_met:
    :param f_arch:
    :param opt_met:
    :param ix_shift_input:
    :param predict_amplitude:
    :param use_ground_truth_training:
    :return:
    """
    model_info = dict()
    model_info['ix_shift_input'] = ix_shift_input
    model_info['predict_amplitude'] = predict_amplitude
    model_info['preproc'] = opt_met.preproc

    # if arch.split('_')[0] == 'fir':
    #     model_info['preproc'] = 'subsmooth'
    # elif arch.split('_')[0] == 'net':
    #     model_info['preproc'] = 'diff'
    # else:
    #     raise Exception("arch mispecified: {}".format(arch))

    x_ev_train, h_ev_train, p_ratio = core_shared(pro, meta, predict_ev, use_ground_truth_training,
                                         model_info['preproc'],
                                         model_info['predict_amplitude'],
                                         ix_shift_input=ix_shift_input,
                                         use_above_alphasnr=use_above_alphasnr)

    if arch.split('_')[0] == 'fir':

        if arch.split('_')[1] == 'gen':
            f_alpha_approx = 10
            fir_def_weights = [int(meta['fs'] * (112/512)), 8, 4]
            model_info['model'] = met_fir.firwin_to_model(fir_def_weights, meta['fs'])
            model_info['met_handle'] = met_fir

        elif arch.split('_')[1] == 'cus':
            dec = int(arch.split('_')[2])
            if dec > 25:
                fir_def_weights = [int(dec), 8, 5]
            else:
                fir_def_weights = [int(meta['fs'] / dec), 8, 4]
            model_info['model'] = met_fir.firwin_to_model(fir_def_weights, meta['fs'])
            model_info['met_handle'] = met_fir

        elif arch.split('_')[1] == 'vsh':
            f_alpha_approx = 10
            fir_def_weights = [int(meta['fs'] / 8), 8, 4]
            model_info['model'] = met_fir.firwin_to_model(fir_def_weights, meta['fs'])
            model_info['met_handle'] = met_fir

        elif arch.split('_')[1] == 'apa':
            f_alpha_approx = 10
            fir_def_weights = [int(meta['fs'] / 4), 8, 4]
            model_info['model'] = met_fir.firwin_to_model(fir_def_weights, meta['fs'])
            model_info['met_handle'] = met_fir

        elif arch.split('_')[1] == 'mph':
            f_alpha_approx = 10
            fir_def_weights = [int(2 * (meta['fs'] / (0.5 * f_alpha_approx)) + 1), 8, 5]
            fir_def_weights = [int(meta['fs'] / 2), 8, 4]
            model_info['model'] = met_fir.firwin_to_model(fir_def_weights, meta['fs'])
            from scipy.signal import minimum_phase
            from scipy.signal import group_delay
            model_info['model']['b'] = minimum_phase(model_info['model']['b'])
            # I think something is wrong with this gd calculation, but maybe it just fails for very short filters
            gd_vec = group_delay((model_info['model']['b'], 1))
            f_vec = (gd_vec[0] * meta['fs']) / (2 * np.pi)
            gd_vec = gd_vec[1]
            gd = - gd_vec[np.argmin(np.abs(f_vec - f_alpha_approx))]

            model_info['model']['gd'] = gd
            model_info['met_handle'] = met_fir

        elif arch.split('_')[1] == 'opt':
            fir_def_weights = [int(2 * (meta['fs'] / (0.5 * meta['f_alpha'])) + 1), meta['f_alpha'] - 1.5, 3]
            model_info['model'] = met_fir.train(x_ev_train, h_ev_train, meta['fs'], fir_def_weights, f_arch=f_arch,
                                                overwrite=overwrite_met, ix_shift_input=ix_shift_input, opt=opt_met)
            model_info['test_function'] = met_fir.test

        elif arch.split('_')[1] == 'alp':
            # fir_def_weights = [int(meta['fs'] * (512 / 512)), meta['f_alpha'] - 1.5, 3]
            fir_def_weights = [int(meta['fs'] * (112/512)), meta['f_alpha'] - 1.5, 3]
            model_info['model'] = met_fir.firwin_to_model(fir_def_weights, meta['fs'])
            model_info['met_handle'] = met_fir
        elif arch.split('_')[1] == 'app':  # like alp but longer
            fir_def_weights = [int(meta['fs'] * (180 / 512)), meta['f_alpha'] - 1.5, 3]
            model_info['model'] = met_fir.firwin_to_model(fir_def_weights, meta['fs'])
            model_info['met_handle'] = met_fir
        elif arch.split('_')[1] == 'vlo':  # like gen but longer
            fir_def_weights = [int(meta['fs'] * (180 / 512)), 8, 4]
            model_info['model'] = met_fir.firwin_to_model(fir_def_weights, meta['fs'])
            model_info['met_handle'] = met_fir

        else:
            raise Exception("arch mispecified: {}".format(arch))
        model_info['model']['f_alpha'] = meta['f_alpha']
        model_info['model']['compensation'] = opt_met.compensation

    elif arch.split('_')[0] == 'net':
        model_info['model'] = met_net.train(x_ev_train, h_ev_train, meta['fs'], arch=arch, f_arch=f_arch, resume=False,
                                            overwrite=overwrite_met, opt=opt_met)
        model_info['met_handle'] = met_net
    elif arch.split('_')[0] == 'reg':
        model_info['model'] = met_reg.train(x_ev_train, h_ev_train, meta['fs'], f_arch=f_arch,
                                            overwrite=overwrite_met, opt=opt_met)
        model_info['met_handle'] = met_reg
    elif arch.split('_')[0] == 'ukf':
        raise Exception('For testing only - not fully implemented.')
        model_info['model'] = {'temp': 0, 'fs': meta['fs']}
        model_info['met_handle'] = met_ukf
    else:
        raise Exception("arch mispecified: {}".format(arch))
    model_info['fs'] = meta['fs']

    return model_info


def core_train(datadir, dataset, sub, ix_run, arch, overwrite_dat=False, overwrite_met=False, use_synth=None,
               predict_ev=None, save_in='', str_model='', opt_met=None, ix_shift_input=0, predict_amplitude=False,
               omit_last_run=False, eeg_setting=None, use_ground_truth_training=False,
               use_above_alphasnr=None):
    """
    Loads data, and passes this to core_arch_select, which returns a model that is based on that data.

    :param datadir:
    :param dataset:
    :param sub:
    :param ix_run:
    :param arch:
    :param overwrite_dat:
    :param overwrite_met:
    :param use_synth:
    :param predict_ev:
    :param save_in:
    :param str_model:
    :param opt_met:
    :param ix_shift_input:
    :param predict_amplitude:
    :param omit_last_run:
    :param eeg_setting:
    :param use_ground_truth_training:
    :return:
    """
    assert isinstance(eeg_setting, str), 'eeg_setting should be string'

    print('Running {}, run {} with method {}.'.format(sub, ix_run, arch))
    if predict_ev is None:
        predict_ev = dict()
        predict_ev['scaling_type'] = 'alpha_filter'
        predict_ev['scaling'] = 2

    if isinstance(sub, list):
        assert isinstance(save_in, list) and len(save_in) == 1, 'save_in should be a list of length 1'
        assert isinstance(str_model, str) and len(str_model) > 0, 'es must be set'
        sub_ = save_in[0]
        pro, meta = data_loader.get_alpha_stitcher(datadir, dataset, sub, omit_last_run=omit_last_run,
                                                   overwrite=overwrite_dat, use_synth=use_synth,
                                                   eeg_setting=eeg_setting)
        f_arch = data_loader.preproc_parser(datadir, dataset, sub_, None, eeg_setting=eeg_setting, use_synth=use_synth,
                                            es='{}.dat'.format(str_model))
    else:
        pro, meta = data_loader.get_alpha(datadir, dataset, sub, ix_run,
                                          overwrite=overwrite_dat, use_synth=use_synth, eeg_setting=eeg_setting)
        f_arch = data_loader.preproc_parser(datadir, dataset, sub, ix_run, eeg_setting=eeg_setting, use_synth=use_synth,
                                            es='{}.dat'.format(str_model))

    if pro is None:
        print('{} {} does not exist, skipping.'.format(sub, ix_run))
        return None

    model_info = core_arch_select(pro, meta, predict_ev, arch, overwrite_met, f_arch,
                                  use_above_alphasnr=use_above_alphasnr,
                                  ix_shift_input=ix_shift_input, predict_amplitude=predict_amplitude,
                                  opt_met=opt_met[arch], use_ground_truth_training=use_ground_truth_training)

    return model_info


def core_test(model_info, datadir, dataset, sub, ix_run, arch,
              overwrite_dat=False, use_synth_test=None, overwrite_res=True,
              use_ground_truth=False, plot=False,
              predict_ev=None, eeg_setting=None):
    assert isinstance(eeg_setting, str), 'eeg_setting should be string'
    if predict_ev is None:
        predict_ev = dict()
        predict_ev['scaling_type'] = 'alpha_filter'
        predict_ev['scaling'] = 1
        if use_ground_truth:
            assert use_synth_test is not None, """If you are checking against ground truth
            then you have to evaluate against a synth method"""

    if ix_run is None:
        pro, meta = data_loader.get_alpha_stitcher(datadir, dataset, sub,
                                                   overwrite=overwrite_dat, use_synth=use_synth_test,
                                                   eeg_setting=eeg_setting)
    else:
        pro, meta = data_loader.get_alpha(datadir, dataset, sub, ix_run,
                                          overwrite=overwrite_dat, use_synth=use_synth_test, eeg_setting=eeg_setting)

    if pro is None:
        print('{} {} does not exist, skipping.'.format(sub, ix_run))
        return None

    x_ev_test, h_ev_test, p_ratio = core_shared(pro, meta, predict_ev, use_ground_truth,
                                       model_info['preproc'],
                                       model_info['predict_amplitude'],
                                       ix_shift_input=model_info['ix_shift_input'])

    # this is a function handle in a dictionary!
    phi_test_error, _, amp_test_error, phi_pred = model_info['met_handle'].test(x_ev_test, h_ev_test, model_info['model'],
                                                                      ix_shift_input=model_info['ix_shift_input'])
    mace = np.mean(np.abs(phi_test_error))

    result = dict()
    result[arch] = dict()
    result[arch]['mae'] = mace
    result[arch]['phi_test_error'] = phi_test_error
    result[arch]['phase'] = phi_pred
    result[arch]['amp'] = None
    result[arch]['amp_test_error'] = amp_test_error
    result[arch]['p_ratio'] = p_ratio

    if plot:
        phi_pred_c, t = model_info['met_handle'].predict_samples(x_ev_test, h_ev_test, model_info['model'], model_info['fs'],
                                                 ix_shift_input=model_info['ix_shift_input'])
        import matplotlib.pyplot as plt
        ix = 20  # arbitrary choice - always print 20th epoch
        plt.plot(t[ix], phi_pred_c[ix])

    return result
