#!/usr/bin/env python
# -*- coding: utf-8 -*-

from . import sp
from . import data_specific
from . import inputparser as ip
from . import paradigm
import numpy as np
from scipy import signal
from datetime import datetime
import pickle
try:
    from noisyopt import minimizeCompass, minimizeSPSA
except:
    print('no noisyopt?')
import os
import matplotlib.pyplot as plt
from pathlib import Path
import random


def preproc_parser(datadir, dataset, subject, ix_run, eeg_setting=None, use_synth=None, es=''):
    """
    use_synth has to be specified here because it influences the data that gets loaded.
    That's why it doesn't get specified in the model name in paradigm.py - it gets added automatically
    to the filename because it's part of the data (on the other hand, the ground_truth flag doesn't actually
    influence the data that is loaded - just how it's used).


    :param datadir:
    :param dataset:
    :param subject:
    :param ix_run:
    :param use_synth:
    :param es:
    :return:
    """

    if not(isinstance(datadir, Path)):
        datadir = Path(datadir)

    if not(isinstance(dataset, Path)):
        dataset = Path(dataset)

    if ix_run is not None:
        assert (isinstance(ix_run, int)), "run must be an int"

    if es == '':
        subject_path, run_name = data_specific.specifier(dataset, subject, ix_run)
    else:
        assert isinstance(eeg_setting, str), 'eeg_setting should be available here...'
        subject_path, run_name = data_specific.specifier_proc(dataset, subject, ix_run, es)

        f_temp, f_temp_ext = os.path.splitext(str(run_name))
        run_name = Path(f_temp + '_' + eeg_setting + f_temp_ext)

        if use_synth is not None:
            f_temp, f_temp_ext = os.path.splitext(str(run_name))
            synth_str = '_'.join([str(i) for i in use_synth.values() if i is not None])
            run_name = Path(f_temp + '_' + synth_str + f_temp_ext)

    fdir = datadir / dataset / subject_path
    fname = datadir / dataset / subject_path / run_name

    if not(es == ''):
        fdir.mkdir(parents=True, exist_ok=True)
    else:
        if not (fname.exists()):
            fname = None

    return fname


def sublist(datadir, dataset):
    return data_specific._sublist(datadir, dataset)


def get_alpha(datadir, dataset, sub, ix_run, overwrite=False, use_synth=None,
              eeg_setting=None, do_save=True):
    assert isinstance(eeg_setting, str), 'eeg_setting should be string'

    f_i = preproc_parser(datadir, dataset, sub, ix_run)
    f_fif = preproc_parser(datadir, dataset, sub, ix_run, eeg_setting=eeg_setting, use_synth=use_synth, es='_alpha.dat')
    f_meta = preproc_parser(datadir, dataset, sub, ix_run, eeg_setting=eeg_setting, use_synth=use_synth, es='_alpha_meta.dat')
    f_fooof = preproc_parser(datadir, dataset, sub, ix_run, eeg_setting=eeg_setting, es='_fooof.dat')
    if f_i is None:
        return None, None

    # assert eeg_setting == 'occipital_alpha_hjorth', 'The f_fif and others do not specifiy eeg setting in the filename'
    assert not isinstance(sub, list), 'sub should not be a list for get_alpha'

    if not(f_fif.exists()) or overwrite:
        s, t, events, fs = data_specific.preprocessor(dataset, f_i, eeg_setting=eeg_setting)
        print('------ Session time (minutes): {:0.2f} ------ '.format((np.shape(s)[1] / fs) / 60))
        nyq = fs / 2

        # not strictly possible, but also not a big deal
        s = s - np.mean(s)

        # maybe I Can put this later so I don't have to do all the annoying [0]
        s = np.array(s).reshape(1, -1)
        t = np.array(t).reshape(1, -1)

        # So that datasets are a bit comparable, make avg power in 30 to 50Hz to 1
        nperseg = 4096
        f, s_spec = signal.welch(s, fs, nperseg=nperseg)
        # TODO: if you remove this line, also address the other TODO items (in paradigm.py in particular!)
        s = s / np.sqrt(np.mean(s_spec[0, (f > 30) & (f < 50)]))
        f, s_spec = signal.welch(s, fs, nperseg=nperseg)
        s_spec = s_spec[0]

        fooof_results = sp.run_fooof(f, s_spec, f_fooof)
        f_alpha = sp.get_f_alpha(f, s_spec, fooof_results)

        if '_alpha_' in eeg_setting:
            f_low = 10 - 2
            f_hig = 10 + 2
        elif '_mu_' in eeg_setting:
            f_low = 10 - 2
            f_hig = 10 + 3
        elif '_beta_' in eeg_setting:
            f_low = 16.5
            f_hig = 20.0
        else:
            raise Exception("eeg_setting incorrect")

        # long-ish filter (ideally very long, but limited due to not wanting to spread events around)
        order = int(np.ceil(fs * 1.5) // 2 * 2 + 1)

        alpha_filter = dict()
        alpha_filter['fs'] = fs
        alpha_filter['order'] = order
        alpha_filter['gd'] = np.round((order - 1) / 2)
        alpha_filter['range'] = [f_low, f_hig]
        alpha_filter['a'] = 1
        alpha_filter['b'] = signal.firwin(order, [f_low / nyq, f_hig / nyq], pass_zero=False)

        if use_synth is not None:
            use_synth_local = '_'.join([str(i) for i in use_synth.values() if i is not None])
            if use_synth['name'] == 'kuramoto_randf':
                use_synth_local = use_synth_local.replace('kuramoto_randf', 'kuramoto')

            f_synth_params = preproc_parser(datadir, dataset, sub, ix_run, eeg_setting=eeg_setting, es='_synth_{}.dat'.format(use_synth_local))
            f_name = '{}_r{}'.format(sub, ix_run)
            if f_synth_params.exists():
                with open(str(f_synth_params), 'rb') as handle:
                    d = pickle.load(handle)
                    d['opt'] = None
            else:
                d = dict()
                d['K'] = 8  # np.random.randint(0, 15)
                d['gma'] = np.pi
                d['N'] = 16
                if use_synth['N']:
                    d['N'] = use_synth['N']
                d['A_dist'] = 'constant'
                if use_synth['A_dist']:
                    d['A_dist'] = use_synth['A_dist']
                d['noise'] = 1
                d['fdist'] = 'cauchy'
                d['A_alpha'] = 75
                d['opt'] = ['A_alpha', 'K', 'noise', 'gma']
                d['n_its_rand'] = 500
                d['figure_dir'] = f_synth_params.parent / use_synth_local

        if use_synth is None:
            s_alpha_gt = np.nan * np.zeros(np.shape(s))  # placeholder for synth data

        elif use_synth['name'] == 'synth':
            s, s_alpha_gt, synth_noise, synth_lf = fit_synth(s, fooof_results, f_alpha, fs, nperseg)

        elif use_synth['name'] == 'kuramoto':
            s, s_alpha_gt, synth_noise, wt, v = simulate_eeg(s, fooof_results, f_alpha, fs, nperseg, f_name, **d)
            if do_save:
                with open(str(f_synth_params), 'wb') as handle:
                    pickle.dump(v, handle)

        elif use_synth['name'] == 'kuramoto_randf':
            assert d['opt'] is None, 'You should have pre-fitted Kuramoto model'
            f_alpha_rand = paradigm.gen_hash(sub)/np.power(2, 32)
            f_alpha_rand = 8 + f_alpha_rand * 4.0
            s, s_alpha_gt, synth_noise, wt, v = simulate_eeg(s, fooof_results, f_alpha_rand, fs, nperseg, f_name, **d)

        else:
            raise Exception('Synth method badly specified!')

        s_alpha_acausal = signal.filtfilt(alpha_filter['b'], alpha_filter['a'], s)

        s = s.squeeze()
        t = t.squeeze()
        s_alpha_acausal = s_alpha_acausal.squeeze()
        s_alpha_gt = s_alpha_gt.squeeze()

        custom_raw = {'s': s, 't': t, 's_alpha_acausal': s_alpha_acausal, 's_alpha_gt': s_alpha_gt}
        meta = {'events': events, 'alpha_filter': alpha_filter, 'fs': fs, 'f_alpha': f_alpha,
                'fooof': fooof_results._asdict()}

        if do_save:
            with open(str(f_fif), 'wb') as handle:
                pickle.dump(custom_raw, handle)
            with open(str(f_meta), 'wb') as handle:
                pickle.dump(meta, handle)

        print('Saved processed {} run {}.'.format(sub, ix_run))
    else:
        with open(str(f_fif), 'rb') as handle:
            custom_raw = pickle.load(handle)

        with open(str(f_meta), 'rb') as handle:
            meta = pickle.load(handle)

    return custom_raw, meta


def get_alpha_stitcher(datadir, dataset, sub, overwrite=False, use_synth=None,
                       omit_last_run=False, eeg_setting=None):
    ''' dd '''
    assert isinstance(sub, list), 'for stitcher sub must be a list (even of len 1)'

    initialise_pro = True
    for ix_sub in range(len(sub)):
        sub_ = sub[ix_sub]
        d = datadir / dataset / data_specific.specifier(dataset, sub_, None)[0]
        n_run = len([x for x in d.iterdir()])
        if omit_last_run:
            n_run = n_run - 1
        for ix_run in range(1, n_run + 1):  # run indexing starts at 1
            pro_, meta_ = get_alpha(datadir, dataset, sub_, ix_run,
                                    overwrite=overwrite, use_synth=use_synth, eeg_setting=eeg_setting)

            if initialise_pro and pro_ is not None:
                pro = pro_
                meta = meta_

                meta.pop('fooof', None)
                meta['vec_f_alpha'] = np.array(meta['f_alpha'])

                initialise_pro = False
            elif pro_ is not None:
                meta['vec_f_alpha'] = np.append(meta['f_alpha'], meta_['f_alpha'])
                meta['f_alpha'] = np.mean(meta['vec_f_alpha'])
                augmented_events = meta_['events']
                n_aug = len(pro['t'])
                augmented_events[:, 0] = augmented_events[:, 0] + n_aug
                meta['events'] = np.concatenate((meta['events'], augmented_events), axis=0)
                for key in pro:
                    if key == 's':
                        match_end = - pro_[key][-1] + pro[key][-1]
                    else:
                        match_end = 0
                    pro[key] = np.append(pro[key], pro_[key] + match_end)

    return pro, meta


def gen_synth(noise, fs, p_o, p_s):

    t = np.arange(0, (1 / fs)*len(noise), 1 / fs)
    y_alpha = np.zeros(np.shape(t))
    for ix in range(len(p_s)):
        # n = np.random.randn(np.shape(noise)[0])
        # fac = int(np.round(p_s[ix]['fac']))
        # f_var = p_s[ix]['fsc'] * signal.lfilter(np.ones(fac) / fac, 1, n)
        f_var = p_s[ix]['fsc']*(t-np.mean(t))*(1/500)
        f = p_s[ix]['fof'] + f_var
        y_alpha += p_s[ix]['a'] * np.cos(2 * np.pi * f * t)

    y_lf = p_o['offset'] + p_o['trend'] * t

    y = noise.copy()
    y += y_lf
    y += y_alpha

    return y, y_alpha, y_lf


def fit_synth(s, fooof_results, f_alpha, fs, nperseg, fig_gen=True):

    s_squeeze = s.copy()
    s_squeeze = s_squeeze[0]

    def minf_translate(x):
        p_s = [{'a': x[2], 'fsc': x[3], 'fof': x[4]}]
        p_o = {'offset': x[0], 'trend': x[1]}
        return p_o, p_s

    def minf(x):
        # synth_noise, fs, nperseg are not passed cleanly
        p_o, p_s = minf_translate(x)
        y, _, _ = gen_synth(sig_noise, fs, p_o, p_s)

        freqs, spectrum_s = signal.welch(s_squeeze[fs:-fs], fs, nperseg=nperseg)
        freqs, spectrum_y = signal.welch(y[fs:-fs], fs, nperseg=nperseg)

        cost = np.mean(np.abs(spectrum_s[1:] - spectrum_y[1:]))
        return cost

    sig_noise = sp.gen_arb_noise_from_foof(fooof_results, len(s_squeeze), fs)

    opt_f = dict()
    x0_ = [0, 0, 10, 0.1, f_alpha]
    l_bound = [-1000, -1000, 0, 1e-3, f_alpha - 2]
    u_bound = [+1000, +1000, 25., 4., f_alpha + 2]
    opt_f['deltatol'] = 1e-6
    opt_f['bounds'] = np.array([l_bound, u_bound]).transpose()

    min_s = minimizeCompass(minf, x0=x0_, paired=False, errorcontrol=False, **opt_f)
    p_o, p_s = minf_translate(min_s['x'])
    sig, sig_alpha, synth_lf = gen_synth(sig_noise, fs, p_o, p_s)

    if fig_gen:
        f_synth, synth_opt_spec = signal.welch(sig, fs, nperseg=nperseg)
        f_noise, noise_spec = sp.welch(sig_noise, fs, nperseg)
        f_, s_spec = signal.welch(s_squeeze, fs, nperseg=nperseg)

        xlim = [2, 35]
        plt.figure(6)
        plt.clf()
        plt.subplot(111)
        plt.plot(f_, s_spec, 'r-')
        plt.plot(f_noise, noise_spec, 'm--')
        plt.plot(f_synth, synth_opt_spec, 'b--')
        plt.xlim([0.1, xlim[-1]])
        plt.ylim([0, 2.1 * np.max(s_spec[f_ > 5])])
    #     should save the figure

    sig = np.array(sig).reshape(1, -1)  # ugh
    sig_alpha = np.array(sig_alpha).reshape(1, -1)  # ugh
    synth_lf = np.array(synth_lf).reshape(1, -1)  # ugh
    sig_noise = np.array(sig_noise).reshape(1, -1)  # ugh

    return sig, sig_alpha, sig_noise, synth_lf


def simulate_alpha_phase(t, dt, f, K, gma, N, s, fdist):
    # K = 4 # coupling strength
    # s = 0 # noise scaling
    w0 = 2 * np.pi * f
    N = int(np.round(N))
    Pi = np.zeros((N, np.size(t)))
    Pi[:, 0] = 2 * np.pi * np.random.rand(N)
    dP = np.zeros((N, np.size(t)))

    if fdist == 'cauchy':
        cauchy_icdf = lambda p, w0, g: w0 + g + np.tan(np.pi * (p - 0.5))
        Wi = cauchy_icdf(np.random.rand(N, 1), w0, gma)
    elif fdist == 'gaussian':
        Wi = w0 + gma * np.random.rand(N, 1)
    else:
        raise Exception('Undefined fdist')

    for ix_t in range(1, np.size(t)):
        pip = Pi[:, ix_t-1]
        dp = kuramoto(pip, K, N, Wi, s, dt)
        dP[:, ix_t] = dp
        Pi[:, ix_t] = pip + dt * dp

    # % plot_state_space(Pi, dP);
    return Pi


def kuramoto(x, K, N, Wi, s, dt):
    met = 'euler'
    # it is divided by dt because this whole thing is then multiplied by dt!
    xs = s * np.random.randn(N, 1) / np.sqrt(dt)

    if met == 'rk4':
        #         % original runge-kutta-4 kuramoto from:
        # % https://appmath.wordpress.com/2017/07/23/kuramoto-model-numerical-code-matlab/
        # % not sure if I am handling the noise 100% correctly though
        # % ok, but the way to handle the noise would be to just stick it in the outer loop? (i.e. in simulate_alpha_phase
        # Pi[] = pip + dt * dp + c * np.sqrt(dt) # then we also don't need the weird division by sqrt(dt)
        dp1 = kuramoto_core(x, K, N, Wi, xs)
        dp2 = kuramoto_core(0.5 * dt * dp1, K, N, Wi, xs)
        dp3 = kuramoto_core(x + 0.5 * dt * dp2, K, N, Wi, xs)
        dp4 = kuramoto_core(x + dt * dp3, K, N, Wi, xs)
        dp = (1/6) * (dp1 + dp2 + dp3 + dp4)
    elif met == 'euler':
        dp1 = kuramoto_core(x, K, N, Wi, xs)
        dp = dp1
    else:
        raise Exception('bad kuramoto method?')

    return dp


def kuramoto_core(x, K, N, w, xs):
    # x is 64 x 1
    X = x[:, None] - x[None, :]  # (broadcasting) trying to achieve x - x' in matlab
    dp = w + xs + (K/N)*np.sum(np.sin(X))
    return dp.reshape(-1)


def simulate_eeg(s, fooof_results, f_alpha, fs, nperseg, f_name, *args, **kwargs):
    d = dict()
    d['K'] = 8  # np.random.randint(0, 15)
    d['gma'] = np.pi
    d['N'] = 16
    d['noise'] = 1
    d['fdist'] = 'cauchy'
    d['A_dist'] = 'constant'
    d['A_alpha'] = 75
    d['n_its_rand'] = 500
    d['figure_dir'] = 'kuramoto'
    d['fig_gen'] = True
    d['testing'] = False
    d['opt'] = ['A_alpha', 'K', 'noise', 'gma']
    v, d = ip.inputParserCustom(d, kwargs)  # this function also accepts args (but args is empty here)
    # assert v['A_dist'] == 'constant', 'Have not written alternatives'
    if v['A_dist'] == 'constant':
        A_dist = np.ones((v['N'], 1))
    elif v['A_dist'] == 'uniform':
        A_dist = np.random.rand(v['N'], 1)
        A_dist = A_dist / np.mean(A_dist)  # rescale so that the mean = 1
    elif 'beta' in v['A_dist']:
        x = v['A_dist']
        from scipy.stats import beta
        assert x.split('_')[1][0] == 'a', 'bad beta format?'
        assert x.split('_')[2][0] == 'b', 'bad beta format?'
        a = float(x.split('_')[1][1:])
        b = float(x.split('_')[2][1:])
        A_dist = beta.rvs(a, b, size=(v['N'], 1))
        A_dist = A_dist / np.mean(A_dist)  # rescale so that the mean = 1
    else:
        raise Exception('Bad A_dist')

    def minf_translate(x):
        p = dict()
        print(repr(x))
        for _, pli in enumerate(param_list):
            idx = np.array([x == pli for x in v['opt']])
            p[pli] = (x[idx][0]*scaling[idx][0] if pli in v['opt'] else v[pli])
            if not isinstance(x, np.ndarray):
                raise Exception('x should be ndarray')
        return p

    def minf_bounds(v_opt, single=False):
        default = dict()
        default['K'] = [0, np.random.rand(), 1, 20.0]
        default['gma'] = [0, np.random.rand(), 1, np.pi * 2]
        default['N'] = [0, np.random.rand(), 1, 64.0]
        default['noise'] = [0, np.random.rand(), 1, np.pi * 2]
        default['fdist'] = [None, None, None, None]
        default['A_alpha'] = [0.15, 0.15 + np.random.rand() * 0.6, 1, 100.0]

        if single:
            # then just return a sample
            return default[v_opt][1] * default[v_opt][3]
        else:
            # lower, start, upper, scaling
            l, s, u, g = [], [], [], []
            for _, pli in enumerate(v_opt):
                l.append(default[pli][0])
                s.append(default[pli][1])
                u.append(default[pli][2])
                g.append(default[pli][3])

            return np.array(l), np.array(s), np.array(u), np.array(g)

    def minf(x, A_dist, seed=None):
        """

        :param x:
        :param seed:
        :return:
        """
        if seed is not None:
            # guessing this is how this works...
            random.seed(seed)
            np.random.seed(seed)
        p = minf_translate(x)

        Pi = simulate_alpha_phase(t, dt, f_alpha, p['K'], p['gma'], p['N'], p['noise'], p['fdist'])
        sig_alpha = p['A_alpha'] * A_dist * np.cos(Pi)

        sig = np.mean(sig_alpha, axis=0) + sig_noise

        _, s_spec_ = signal.welch(s_squeeze, fs, nperseg=nperseg)
        f_synth_, sig_spec = signal.welch(sig, fs, nperseg=nperseg)

        ix_f = np.logical_and(f_synth_ > 2.5, f_synth_ < 30)
        cost = np.round(np.sqrt(np.mean(np.square(s_spec_[ix_f] - sig_spec[ix_f]))), 3)
        print(cost)
        return cost

    def minimize_random_local(v_opt, A_dist, n_its=100):
        cost_best = np.Inf
        x_best = None
        for ix in range(n_its):
            _, x, _, _ = minf_bounds(v_opt)
            cost = minf(x, A_dist)
            if cost < cost_best:
                cost_best = cost
                x_best = x

        return x_best

    n_its = 1000 if v['testing'] else 1
    for ix in range(n_its):

        paradigm.set_seed(paradigm.gen_hash('{}r{}'.format(f_name, ix)))

        if v['testing']:
            for o in v['opt']:
                v[o] = minf_bounds(o, single=True)
            v['opt'] = None

        dt = 1/fs
        t = np.arange(0, np.size(s) * dt, dt)
        s_squeeze = s.copy()
        s_squeeze = s_squeeze[0]

        sig_noise = sp.gen_arb_noise_from_foof(fooof_results, len(s_squeeze), fs)

        if v['opt'] is not None:
            param_list = ['K', 'gma', 'N', 'noise', 'A_alpha', 'fdist']
            l_bound, _x0, u_bound, scaling = minf_bounds(v['opt'])
            x0 = minimize_random_local(v['opt'], A_dist, n_its=v['n_its_rand'])
            deltatol = 1e-4
            deltainit = 0.2  # since we do a (random) previous min step
            bounds = np.array([l_bound, u_bound]).transpose()
            paired = False
            # min_s = minimizeCompass(minf, x0, paired=paired, errorcontrol=paired,
            #                         bounds=bounds, deltainit=deltainit, deltatol=deltatol)
            # # min_s = minimizeSPSA(minf, x0, paired=paired, a=deltainit, c=deltainit, bounds=bounds)
            # min_s['x']
            p = minf_translate(x0)
            for pli in p:
                v[pli] = p[pli]

        Pi = simulate_alpha_phase(t, dt, f_alpha, v['K'], v['gma'], v['N'], v['noise'], v['fdist'])
        wt = sp.circ_mean(Pi, axis=0)
        sig_alpha = v['A_alpha'] * A_dist * np.cos(Pi)
        sig = np.mean(sig_alpha, axis=0) + sig_noise

        if v['fig_gen']:
            f_synth, sig_spec = signal.welch(sig, fs, nperseg=nperseg)
            # f_noise, noise_spec = sp.welch(sig, fs, nperseg)  #
            f_, s_spec = signal.welch(s_squeeze, fs, nperseg=nperseg)  # original
            ix_f = np.logical_and(f_synth > 2.5, f_synth < 30)
            e = np.round(np.sqrt(np.mean(np.square(s_spec[ix_f] - sig_spec[ix_f]))), 3)
            print(e)

            xlim = [2, 35]
            plt.clf()
            plt.plot(f_, s_spec, 'k-', label='Data')  # original
            # plt.plot(f_noise, noise_spec, 'm--')
            plt.plot(f_synth, sig_spec, 'r--', label='Simulated')
            plt.xlim([0.1, xlim[-1]])
            plt.ylim([0, 2.1 * np.max(s_spec[f_ > 5])])
            # plt.legend(loc='upper right')
            strtitle = '{}_{}_{}_A{:.2f}_noise{:.1f}_K{:.2f}_gma{:.1f}_N{}'.format(f_name, e, v['fdist'], v['A_alpha'],
                                                                                   v['noise'], v['K'], v['gma'], v['N'])
            if v['opt'] is not None:
                mydir = v['figure_dir'] / 'opt'
            else:
                mydir = v['figure_dir'] / 'set'

            plt.title(strtitle)
            Path(mydir).mkdir(parents=True, exist_ok=True)
            plt.savefig('{}/{}.png'.format(mydir, strtitle))

            # enable to print signal stretches examples:
            # plot_signal_stretch(t, sig, sig_alpha, f_name)

    sig = np.array(sig).reshape(1, -1)  # ugh
    sig_alpha = np.array(sig_alpha).mean(axis=0).reshape(1, -1)  # ugh
    sig_noise = np.array(sig_noise).reshape(1, -1)  # ugh

    import pycircstat
    wt_ = wt.reshape(1, -1)
    s_ = np.angle(signal.hilbert(sig_alpha))
    cc_ = pycircstat.corrcc(s_, wt_)
    print(f'cc between mean phase and hb phase: {cc_}')

    return sig, sig_alpha, sig_noise, wt, v


def plot_signal_stretch(t, sig, sig_alpha, f_name):
    """
    This is to print an example of a time series
    you can run it by just commenting in the call above -
    :param t:
    :param sig:
    :param sig_alpha:
    :param f_name:
    :return:
    """

    plt.clf
    plt.close('all')
    t0 = 27
    t_len = 1
    y_sec = sig[np.all(np.stack((t < t0+t_len, t > t0), axis=0), axis=0)]
    y0 = np.mean(y_sec)
    y_sec = y_sec - y0
    sig_local = sig - y0

    ylim = np.ceil(1.1 * np.array([-1, 1]) * np.max(np.abs(y_sec)))
    f_width = (18 / 4) * 2.5 * (1 / 2.54)
    f_height = 4 * (1 / 2.54)
    dpi = 150
    dpi_print = 300
    img_fmt = 'svg'
    fig, ax = plt.subplots(1, 1, figsize=(f_width, f_height), dpi=dpi)
    ax.plot(t - t0, sig_local)
    ax.plot(t - t0, np.mean(sig_alpha, axis=0))
    ax.set_xlim(0, t_len)
    ax.set_ylim(ylim)

    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (AU)')
    fig.tight_layout()
    import seaborn as sns
    sns.despine()
    f_name = f"paper{''}_kuramoto_signal{f_name}.{img_fmt}"
    fig.savefig(f_name, format=img_fmt, dpi=dpi_print)
