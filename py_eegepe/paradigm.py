#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from . import data_loader
from . import met
from . import data_specific
import hashlib
import random
import pickle

""" Setup to run 'experiments' on data """


def set_seed(ps):
    """
    Sets the seed based on a string (e.g. path)

    :type ps: str
    :param ps: any string to be hashed
    """

    hash_ = gen_hash(ps)
    random.seed(hash_)
    np.random.seed(hash_)  # not tested


def gen_hash(ps):
    """
    Gets the hash of a strin (e.g. path)

    :type ps: str
    :param ps: any string to be hashed
    """

    str_hash = str(ps)
    hash_object = hashlib.md5(str_hash.encode('utf-8'))
    hash = int(hash_object.hexdigest(), 16) % 2 ** 32
    return hash


def get_str_model(es, opt_train, item_arch, paradigm_type):
    """
    Take some input parameters and generate a string for file name.
    This is actually used in the model file name and the results file name.
    Note that if the information is part of opt_train, but relevant to the data that should get loaded
    then it's specified in data_loader.preproc_parser instead (e.g. synth)

    :param es: extra string
    :param opt_train:
    :param item_arch:
    :param paradigm_type: WR/AS/WS
    :return:
    """
    es = es + '_' + item_arch
    es = es + '_' + opt_train['opt_met'][item_arch].preproc  # subsmooth or not
    if opt_train['predict_amplitude']:
        es = es + '_amp'
    try:
        es = es + '_shift_m{:04d}'.format(-opt_train['ix_shift_input'])
    except:
        print(1)
    # es = es + '_shift_m{:04d}'.format(-opt_train['ix_shift_input'])
    if paradigm_type:
        es = es + '_' + paradigm_type
    if opt_train['use_ground_truth_training']:
        es = es + '_gttrained'
    if opt_train['use_above_alphasnr']:
        es = es + f"_above_alphasnr{opt_train['use_above_alphasnr']}_"

    return es


def generate_hashable_dict(sub, arch, es, opt_train, opt_test, **kwargs):
    """Generate a dictionary of items to be hashed"""
    hashable_dict = {'sub': sub, 'arch': arch, 'es': es}
    hashable_dict.update(kwargs)
    hashable_dict.update(opt_test)
    hashable_dict.update(opt_train)

    # remove overwrite strings and opt_met stuff
    hashable_dict = {k: v for k, v in hashable_dict.items() if 'opt_met' not in k}
    hashable_dict = {k: v for k, v in hashable_dict.items() if 'overwrite' not in k}
    hashable_dict = {k: v for k, v in hashable_dict.items() if 'plot' not in k}

    hashable_dict = {k: v for k, v in hashable_dict.items() if v is not None}

    hashable_dict = {k: v for k, v in sorted(hashable_dict.items())}

    return hashable_dict


def load_result(f_res, ip_hash):
    assert isinstance(ip_hash, int), 'ip_hash should be an int'
    result_ = None
    generate_result = True

    if f_res.exists():
        with open(str(f_res), 'rb') as handle:
            r = pickle.load(handle)
        if r['ip_hash'] == ip_hash:
            result_ = r['result_']
            generate_result = False

    return result_, ip_hash, generate_result


def save_result(f_res, result_, ip_hash):
    assert isinstance(ip_hash, int), 'ip_hash should be an int'
    r = dict()
    r['result_'] = result_
    r['ip_hash'] = ip_hash
    with open(str(f_res), 'wb') as handle:
        pickle.dump(r, handle)


def dictionary_to_hash(hashable_dict):
    ip_string = dictionary_to_string(hashable_dict)
    return gen_hash(ip_string)


def dictionary_to_string(hashable_dict):
    ip_string = ''
    for k, v in hashable_dict.items():
        ip_string = ip_string + k
        if not isinstance(v, list):
            v = [v]
        for i in v:
            ip_string = ip_string + str(i) + '_'
    return ip_string


def main_multiple_subject(datadir, dataset, sub_list, arch, opt_train=None, opt_test=None, increment_ix_run=0,
                          gen_fooof=False, max_n_run=20, append_model=False, es=''):
    """
    Train on a single run, then evaluate on that same run, or some other run controlled by increment_ix_run.
    There is no real train/test issue, because in training we are using data away from the event time.
    And at test time, we are trying to get the phase at the unseen event time itself.
    """

    if opt_train is None: opt_train, _ = opt_init(arch)
    if opt_test is None: _, opt_test = opt_init(arch)

    if not isinstance(sub_list, list):
        sub_list = [sub_list]

    sub_result = dict()
    for sub in sub_list:
        print('{}'.format(sub))
        run_result = dict()
        for ix_run in range(max_n_run):
            path_hash = data_loader.preproc_parser(datadir, dataset, sub, ix_run)
            if path_hash is not None:
                set_seed(path_hash)
                if gen_fooof:
                    data_loader.get_alpha(datadir, dataset, sub, ix_run,
                                          eeg_setting=opt_train['eeg_setting'], overwrite=opt_train['overwrite_dat'])
                else:
                    print('Running {}, run {}'.format(sub, ix_run))
                    result = main_multiple_met(datadir, dataset, sub, ix_run, arch, opt_train, opt_test, es=es,
                                               increment_ix_run=increment_ix_run, append_model=append_model)

                    run_result[ix_run] = result
        sub_result[sub] = run_result
    return sub_result


def main_within_subject_full(datadir, dataset, subject, arch, opt_train=None, opt_test=None, es=''):
    """
    Train on all runs from an individual, then evaluate on these same runs.
    There is no real train/test issue, because in training we are using data away from the event time.
    And at test time, we are trying to get the phase at the unseen event time itself.
    """

    if opt_train is None: opt_train, _ = opt_init(arch)
    if opt_test is None: _, opt_test = opt_init(arch)

    # because this takes a while to run, make it so that you can subselect the test list as test_only
    if isinstance(subject, str):
        subject = [subject]

    if isinstance(arch, list):
        if len(arch) == 1:
            arch = arch[0]
        else:
            raise Exception('for this function we need arch as a single entry')

    #
    # not sure if sub_test is allowed to be more than subject name - if so, should make same change in across subject
    str_model = get_str_model(es, opt_train, arch, 'ws')
    sub_test = [subject[0]]

    hashable_dict = generate_hashable_dict(sub_test, arch, es, opt_train, opt_test)
    ip_hash = dictionary_to_hash(hashable_dict)

    f_res = data_loader.preproc_parser(datadir, dataset, sub_test[0], None, eeg_setting=opt_train['eeg_setting'], use_synth=opt_train['use_synth'],
                                       es=f"{str_model}_{ip_hash}_result.dat")

    result, ip_hash, generate_result = load_result(f_res, ip_hash)
    if generate_result or opt_test['overwrite_res']:
        result = dict()
        # predict_ev defaults to validation values in core_train and core_test,
        # to actually predict phase you want to change this

        assert opt_train['eeg_setting'] == opt_test[
            'eeg_setting'], 'eeg_setting should probably be the same for train and test'

        model_info = met.core.core_train(datadir, dataset, subject, None, arch, omit_last_run=True,
                                         save_in=sub_test, str_model=str_model, **opt_train)
        d = datadir / dataset / data_specific.specifier(dataset, subject[0], None)[0]
        n_run = len([x for x in d.iterdir()])
        if opt_test['plot']:  # ugly to have this here...
            met.core.core_test(model_info, datadir, dataset, subject[0], 1, arch, **opt_test)
            return model_info
        else:
            for ix_run in range(1, n_run + 1):
                result_ = met.core.core_test(model_info, datadir, dataset, subject[0], ix_run, arch, **opt_test)
                result[ix_run] = result_

            save_result(f_res, result, ip_hash)

    return result


def main_multiple_met(datadir, dataset, sub, ix_run, arch, opt_train=None, opt_test=None,
                      increment_ix_run=0, es='', append_model=False):
    if opt_train is None: opt_train, _ = opt_init(arch)
    if opt_test is None: _, opt_test = opt_init(arch)

    if isinstance(arch, str): arch = [arch]
    result = dict()
    for item_arch in arch:
        str_model = get_str_model(es, opt_train, item_arch, 'wr')

        hashable_dict = generate_hashable_dict(sub, item_arch, es, opt_train, opt_test,
                                               ix_run=ix_run, increment_ix_run=increment_ix_run)
        ip_hash = dictionary_to_hash(hashable_dict)
        f_res = data_loader.preproc_parser(datadir, dataset, sub, ix_run + increment_ix_run,
                                           eeg_setting=opt_train['eeg_setting'],
                                           use_synth=opt_train['use_synth'],
                                           es=f"{str_model}_{ip_hash}_train_r{ix_run}_result.dat")

        result_, ip_hash, generate_result = load_result(f_res, ip_hash)

        if generate_result or opt_test['overwrite_res']:
            # these following two lines actually do the train/test everything else is file handling

            assert opt_train['eeg_setting'] == opt_test[
                'eeg_setting'], 'eeg_setting should probably be the same for train and test'

            model_info = met.core.core_train(datadir, dataset, sub, ix_run, item_arch, str_model=str_model, **opt_train)
            result_ = met.core.core_test(model_info, datadir, dataset, sub, ix_run + increment_ix_run, item_arch,
                                         **opt_test)
            if append_model:
                result_[list(result_.keys())[0]]['model_info'] = model_info

            if not opt_test['plot']:  # ugly, but don't save if we are making plots
                save_result(f_res, result_, ip_hash)

        if result_ is not None:
            result.update(result_)

    return result


def main_across_subjects(datadir, dataset, sub_list, arch,
                         opt_train=None, opt_test=None, test_only=None, append_model=False, es=''):
    """For convenience """

    if opt_train is None: opt_train, _ = opt_init(arch)
    if opt_test is None: _, opt_test = opt_init(arch)

    # because this takes a while to run, make it so that you can subselect the test list as test_only
    if test_only is None:
        test_only = sub_list
    elif isinstance(test_only, str):
        test_only = [test_only]

    if isinstance(arch, list):
        if len(arch) == 1:
            arch = arch[0]
        else:
            raise Exception('for this function we need arch as a single entry')

    if not isinstance(sub_list, list):
        sub_list = [sub_list]

    str_model = get_str_model(es, opt_train, arch, 'as')

    #
    sub_result = dict()
    for ix in range(len(test_only)):
        sub_test = [test_only[ix]]
        sub_train = list(set(sub_list) - set([test_only[ix]]))
        sub_train = sorted(sub_train)

        hashable_dict = generate_hashable_dict(sub_test, arch, es, opt_train, opt_test,
                                               sub_train=sub_train)
        ip_hash = dictionary_to_hash(hashable_dict)
        f_res = data_loader.preproc_parser(datadir, dataset, sub_test[0], None,
                                           eeg_setting=opt_train['eeg_setting'],
                                           use_synth=opt_train['use_synth'],
                                           es=f"{str_model}_{ip_hash}_result.dat")

        result_, ip_hash, generate_result = load_result(f_res, ip_hash)

        if generate_result or opt_test['overwrite_res']:

            # predict_ev defaults to validation values in core_train and core_test,
            # to actually predict phase you want to change this

            assert opt_train['eeg_setting'] == opt_test[
                'eeg_setting'], 'eeg_setting should probably be the same for train and test'

            model_info = met.core.core_train(datadir, dataset, sub_train, None, arch, save_in=sub_test,
                                             str_model=str_model, **opt_train)

            result_ = met.core.core_test(model_info, datadir, dataset, sub_test, None, arch, **opt_test)
            if append_model:
                result_[list(result_.keys())[0]]['model_info'] = model_info

            if not opt_test['plot']:  # ugly, but don't save if we are making plots
                save_result(f_res, result_, ip_hash)

        sub_result[test_only[ix]] = {0: result_}
        # sub_result[test_only[ix]] = result_

    return sub_result


def opt_init(arch, overwrite_dat=False, overwrite_met=False, overwrite_res=True, use_above_alphasnr=None,
             use_synth=None, epochs=None, es_patience=None, ix_shift_input=0,
             eeg_setting='occipital_alpha_hjorth', predict_amplitude=False,
             use_ground_truth_training=False):
    """
    Remember that if you add something here, it may also need to be explicitly specified
    for the model loading or data loading stage
    i.e. either in paradigm.get_str_model or data_loader.preproc_parser
    :param arch:
    :param overwrite_dat:
    :param overwrite_met:
    :param overwrite_res:
    :param use_above_alphasnr:
    :param use_synth:
    :param epochs:
    :param es_patience:
    :param ix_shift_input:
    :param eeg_setting:
    :param predict_amplitude:
    :param use_ground_truth_training:
    :return:
    """
    if epochs is not None:
        print('Overriding max epochs for all methods.')

    # assert eeg_setting == 'occipital_alpha_hjorth', 'The filename for data set differently is incorrect.'

    opt_train = dict()
    opt_train['use_ground_truth_training'] = use_ground_truth_training
    opt_train['use_synth'] = use_synth  # note that this is either None or a dictionary now
    opt_train['overwrite_dat'] = overwrite_dat
    opt_train['overwrite_met'] = overwrite_met
    opt_train['ix_shift_input'] = ix_shift_input
    opt_train['predict_amplitude'] = predict_amplitude
    opt_train['eeg_setting'] = eeg_setting
    opt_train['use_above_alphasnr'] = use_above_alphasnr
    opt_train['opt_met'] = dict()

    if arch is not None:
        if isinstance(arch, str):
            arch = [arch]

        for arch_ in arch:
            if arch_.split('_')[0] == 'fir':
                opt_train['opt_met'][arch_] = met.met_fir.defaults(None)
            elif arch_.split('_')[0] == 'net':
                opt_train['opt_met'][arch_] = met.met_net.defaults(None)
            elif arch_.split('_')[0] == 'con':
                opt_train['opt_met'][arch_] = met.met_net.defaults(None)
            elif arch_.split('_')[0] == 'reg':
                opt_train['opt_met'][arch_] = met.met_reg.defaults(None)
            elif arch_.split('_')[0] == 'ukf':
                opt_train['opt_met'][arch_] = met.met_ukf.defaults(None)
            else:
                raise Exception('')
            if epochs is not None:
                opt_train['opt_met'][arch_].epochs = epochs
            if es_patience is not None:
                opt_train['opt_met'][arch_].es_patience = es_patience

    print('Setting max epochs:{}, es_patience:{}.'.format(epochs, es_patience))
    opt_test = dict()
    opt_test['use_synth_test'] = use_synth
    opt_test['overwrite_res'] = overwrite_res
    opt_test['use_ground_truth'] = False
    opt_test['plot'] = False
    opt_test['eeg_setting'] = opt_train['eeg_setting']

    return opt_train, opt_test


def opt_quick_config(arch, ix_config, script_type='', overwrite_res=False, overwrite_dat=False,
                     predict_amplitude=False):
    """Quick configuration options for running experiments"""

    if not ix_config:
        # has to be updated manually to match final ix_config
        n_config = 19
        return n_config

    opt_train, opt_test = opt_init(arch, overwrite_dat=overwrite_dat, overwrite_res=overwrite_res,
                                   predict_amplitude=predict_amplitude)

    opt_train['ix_shift_input'] = 0  # if this is -, it shifts our training data to be that amount to the left of the
    # test event markers

    for met in opt_train['opt_met']:
        if met.split('_')[0] == 'net':
            opt_train['opt_met'][met].preproc = 'diff'
        else:
            opt_train['opt_met'][met].preproc = 'subsmooth'

    use_subsmooth = False
    use_hpnc_for_fir = False
    if ix_config == 1:
        ds = '''Train on real data, evaluate on real data'''
        opt_train['use_synth'] = None
        opt_test['use_synth_test'] = opt_train['use_synth']
    elif ix_config == 2:
        ds = '''Kuramoto trained evaluated on Kuramoto test (in the same way as data)'''
        opt_train['use_synth'] = dict({'name': 'kuramoto', 'N': 16, 'A_dist': 'constant'})
        opt_test['use_ground_truth'] = False
        opt_test['use_synth_test'] = opt_train['use_synth']
    elif ix_config == 3:
        ds = '''Kuramoto trained, evaluated on Kuramoto ground truth signal'''
        opt_train['use_synth'] = dict({'name': 'kuramoto', 'N': 16, 'A_dist': 'constant'})
        opt_test['use_ground_truth'] = True
        opt_test['use_synth_test'] = opt_train['use_synth']
    elif ix_config == 4:
        ds = '''Kuramoto GT trained, evaluated on GT Kuramoto'''
        opt_train['use_synth'] = dict({'name': 'kuramoto', 'N': 16, 'A_dist': 'constant'})
        opt_train['use_ground_truth_training'] = True
        opt_test['use_ground_truth'] = True
        opt_test['use_synth_test'] = opt_train['use_synth']
    elif ix_config == 5:
        ds = '''Kuramoto GT trained, evaluated on Kuramoto'''
        opt_train['use_synth'] = dict({'name': 'kuramoto', 'N': 16, 'A_dist': 'constant'})
        opt_train['use_ground_truth_training'] = True
        opt_test['use_ground_truth'] = False
        opt_test['use_synth_test'] = opt_train['use_synth']
    elif ix_config == 6:
        ds = '''Kuramoto HIGH-AMPLITUDE trained, evaluated on GT Kuramoto'''
        opt_train['use_synth'] = dict({'name': 'kuramoto', 'N': 16, 'A_dist': 'constant'})
        opt_train['use_ground_truth_training'] = False
        opt_train['use_above_alphasnr'] = 90  # as a percentile
        opt_test['use_ground_truth'] = True
        opt_test['use_synth_test'] = opt_train['use_synth']
    elif ix_config == 7:
        ds = '''Kuramoto trained, evaluated on real data'''
        opt_train['use_synth'] = dict({'name': 'kuramoto', 'N': 16, 'A_dist': 'constant'})
        opt_test['use_ground_truth'] = False
        opt_test['use_synth_test'] = None
    elif ix_config == 8:
        ds = '''Random frequency Kuramoto trained, evaluated on real data'''
        opt_train['use_synth'] = dict({'name': 'kuramoto_randf', 'N': 16, 'A_dist': 'constant'})
        opt_test['use_synth_test'] = None
    elif ix_config == 9:
        ds = '''Train on real data, evaluate on real data - but turn off diff!'''
        opt_train['use_synth'] = None
        opt_test['use_synth_test'] = opt_train['use_synth']
        use_subsmooth = True
        for met in opt_train['opt_met']:
            opt_train['opt_met'][met].preproc = 'subsmooth'
    elif ix_config == 10:
        ds = '''N = 8, Kuramoto trained evaluated on Kuramoto test (in the same way as data)'''
        opt_train['use_synth'] = dict({'name': 'kuramoto', 'N': 8, 'A_dist': 'constant'})
        opt_test['use_ground_truth'] = False
        opt_test['use_synth_test'] = opt_train['use_synth']
    elif ix_config == 11:
        ds = '''N = 8, Kuramoto trained, evaluated on Kuramoto ground truth signal'''
        opt_train['use_synth'] = dict({'name': 'kuramoto', 'N': 8, 'A_dist': 'constant'})
        opt_test['use_ground_truth'] = True
        opt_test['use_synth_test'] = opt_train['use_synth']
    elif ix_config == 12:
        ds = '''N = 32, Kuramoto trained evaluated on Kuramoto test (in the same way as data)'''
        opt_train['use_synth'] = dict({'name': 'kuramoto', 'N': 32, 'A_dist': 'constant'})
        opt_test['use_ground_truth'] = False
        opt_test['use_synth_test'] = opt_train['use_synth']
    elif ix_config == 13:
        ds = '''N = 32, Kuramoto trained, evaluated on Kuramoto ground truth signal'''
        opt_train['use_synth'] = dict({'name': 'kuramoto', 'N': 32, 'A_dist': 'constant'})
        opt_test['use_ground_truth'] = True
        opt_test['use_synth_test'] = opt_train['use_synth']
    elif ix_config == 14:
        ds = '''Train on real data, evaluate on real data [Use HP for FIR!!!]'''
        use_hpnc_for_fir = True  # just used to change the es (for the file name - below)
        opt_train['use_synth'] = None
        opt_test['use_synth_test'] = opt_train['use_synth']
        for met in opt_train['opt_met']:
            if met.split('_')[0] == 'net':
                pass  # should be OK already
            else:
                opt_train['opt_met'][met].preproc = 'hpnc'
    elif ix_config == 15:
        ds = '''A_dist = beta, Kuramoto trained evaluated on Kuramoto test (in the same way as data)'''
        opt_train['use_synth'] = dict({'name': 'kuramoto', 'N': 16, 'A_dist': 'beta_a2.0_b5.0'})
        opt_test['use_ground_truth'] = False
        opt_test['use_synth_test'] = opt_train['use_synth']
    elif ix_config == 16:
        ds = '''A_dist = beta, Kuramoto trained, evaluated on Kuramoto ground truth signal'''
        opt_train['use_synth'] = dict({'name': 'kuramoto', 'N': 16, 'A_dist': 'beta_a2.0_b5.0'})
        opt_test['use_ground_truth'] = True
        opt_test['use_synth_test'] = opt_train['use_synth']
    elif ix_config == 17:
        ds = '''A_dist = U, Kuramoto trained evaluated on Kuramoto test (in the same way as data)'''
        opt_train['use_synth'] = dict({'name': 'kuramoto', 'N': 16, 'A_dist': 'uniform'})
        opt_test['use_ground_truth'] = False
        opt_test['use_synth_test'] = opt_train['use_synth']
    elif ix_config == 18:
        ds = '''A_dist = U, Kuramoto trained, evaluated on Kuramoto ground truth signal'''
        opt_train['use_synth'] = dict({'name': 'kuramoto', 'N': 16, 'A_dist': 'uniform'})
        opt_test['use_ground_truth'] = True
        opt_test['use_synth_test'] = opt_train['use_synth']
    elif ix_config == 19:
        # TODO: compensation should make it into the hash, but not the filename - need to check repurcusions
        ds = '''(Use different compensation for FIR) Train on real data, evaluate on real data'''
        opt_train['use_synth'] = None
        opt_test['use_synth_test'] = opt_train['use_synth']
        for met in opt_train['opt_met']:
            if met.split('_')[0] == 'fir':
                opt_train['opt_met'][met].compensation = 'f_alpha'
    # elif ix_config == 20:
    #     ds = '''Kuramoto trained, evaluated on Kuramoto ground truth signal - BUT DELAYED!\n
    #     This is untested - use with care.'''
    #     opt_train['use_synth'] = None
    #     opt_train['ix_shift_input'] = int(-25)  # assuming a delay of 50ms, and a frequency of 500Hz
    #     opt_test['use_synth_test'] = opt_train['use_synth']
    # elif ix_config == 21:
    #     ds = '''Kuramoto trained evaluated on Kuramoto test (in the same way as data) - BUT DELAYED!'''
    #     opt_train['use_synth'] = dict({'name': 'kuramoto', 'N': 16, 'A_dist': 'constant'})
    #     opt_train['ix_shift_input'] = int(-25)  # assuming a delay of 50ms, and a frequency of 500Hz
    #     opt_test['use_ground_truth'] = False
    #     opt_test['use_synth_test'] = opt_train['use_synth']
    # elif ix_config == 22:
    #     ds = '''Kuramoto trained, evaluated on Kuramoto ground truth signal - BUT DELAYED!\n
    #     This is untested - use with care.'''
    #     opt_train['use_synth'] = dict({'name': 'kuramoto', 'N': 16, 'A_dist': 'constant'})
    #     opt_train['ix_shift_input'] = int(-25)  # assuming a delay of 50ms, and a frequency of 500Hz
    #     opt_test['use_ground_truth'] = True
    #     opt_test['use_synth_test'] = opt_train['use_synth']

    else:
        raise Exception('ix_config value not defined')

    es = script_type
    if opt_train['use_synth']:
        es = f"{es}_{opt_train['use_synth']['name']}"
        if not (opt_train['use_synth']['N'] == 16):
            es = f"{es}_N{opt_train['use_synth']['N']}"
        if not (opt_train['use_synth']['A_dist'] == 'constant'):
            es = f"{es}_Ad{opt_train['use_synth']['A_dist']}"
    if opt_test['use_synth_test']:
        es = f"{es}_test{opt_test['use_synth_test']['name']}"
    if opt_test['use_ground_truth']:
        es = f"{es}_gt"
    if opt_train['use_ground_truth_training']:
        es = f"{es}_traingt"
    if opt_train['use_above_alphasnr']:
        es = f"{es}_above_alphasnr{opt_train['use_above_alphasnr']}"
    if use_subsmooth:
        es = f"{es}_subsmooth"
    elif use_hpnc_for_fir:
        es = f"{es}_hpncforfir"
    if not (opt_train['ix_shift_input'] == 0):
        str_mod_local = f"shift{opt_train['ix_shift_input']}".replace('-', 'm')
        es = f"{es}_{str_mod_local}"

    print(ds)
    print(es)
    return opt_train, opt_test, es, ds


def get_standard_arch(fig_type):
    arch = []
    if fig_type == 'scan_fir_length':
        for ix in np.arange(64 + 32, 128 + 64, 8):
            arch.append('fir_cus_{:03}'.format(ix))
    elif fig_type == 'scan_fir_length_longer':
        for ix in np.arange(64 + 32, 128 + 128, 8):
            arch.append('fir_cus_{:03}'.format(ix))
    elif fig_type == 'main_set':
        arch.append('fir_gen_000')
        arch.append('fir_alp_000')
        arch.append('net_sgd_000')
        arch.append('net_sgd_001')
        arch.append('net_cug_000')
    elif fig_type == 'main_set_reduced':
        arch.append('fir_alp_000')
        arch.append('net_sgd_000')
        arch.append('net_sgd_001')
    elif fig_type is None:
        # FIR standard
        arch.append('fir_gen_000')
        arch.append('fir_vsh_000')
        arch.append('fir_alp_000')
        arch.append('reg_sta_000')
        arch.append('net_cug_000')
        arch.append('net_cug_001')
        arch.append('net_cug_002')
        arch.append('net_cug_003')
        arch.append('net_sgd_000')
        arch.append('net_sgd_001')
        arch.append('net_sgd_002')
        arch.append('net_sgd_003')
    else:
        raise Exception('Bad fig type spec')
    return arch


def get_arch_nice():
    arch_nice = {'fir_gen_000': 'C-FIR (0.22s)',
                 'fir_opt_000': 'Opt. symmetrical C-FIR',
                 'fir_alp_000': 'IAF C-FIR (0.22s)',
                 'fir_cus_112': 'C-FIR (0.22s)',
                 'fir_vlo_000': 'C-FIR (0.35s)',
                 'fir_app_000': 'IAF C-FIR (0.35s)',
                 'fir_vsh_000': 'C-FIR (0.125s)',
                 'reg_sta_000': 'Opt. reg.',
                 'net_sgd_000': 'OF',
                 'net_sgd_001': 'MLOF',
                 'net_cug_000': 'GRU',
                 'net_sgd_002': 'net_sgd_002',
                 'net_sgd_003': 'net_sgd_003',
                 'net_sgd_004': 'net_sgd_004',
                 'net_sgd_005': 'net_sgd_005',
                 'net_sgd_006': 'net_sgd_006',
                 'net_cug_001': 'net_cug_001',
                 'net_cug_002': 'net_cug_002',
                 }

    # repeat above with 'test_' preamble
    arch_nice.update({f'test_{k}': v for k, v in arch_nice.items()})
    return arch_nice
