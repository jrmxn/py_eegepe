from . import num_subplots
import numpy as np
from numpy import pi
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def phase_errors(result, plot_type=None):
    if plot_type is None:
        if len(result)>3:
            fig = phase_errors_subplots(result)
        else:
            fig = phase_errors_stacked(result)
    elif plot_type is 'stacked':
        fig = phase_errors_stacked(result)
    elif plot_type is 'subplots':
        fig = phase_errors_subplots(result)

    return fig


def phase_errors_stacked(result):
    fig = plt.figure(2, figsize=(12, 4), dpi=80, facecolor='w', edgecolor='k')
    plt.clf()
    ax1 = fig.add_subplot(121)

    x_phi = np.arange(-pi, pi, pi / 16)
    title = 'MAE, '
    for ix, key in enumerate(result.keys()):
        fc = list(plt.get_cmap('Dark2')(ix))
        fc[-1] = 0.25 # transparent
        ax1.hist(result[key]['phi_test_error'], bins=x_phi, lw=2, fc=fc, label=key)
        title = title + '{}: {:0.2f}, '.format(key, result[key]['mae'])
        if ix>0 and ix % 2 == 0:
            title = title + '\n'
    ax1.set_xlim([-pi, pi])
    ax1.set_title(title)
    # plt.text(0.5, 0.95, title, horizontalalignment='center', fontsize=12, transform=ax1.transAxes)
    ax1.legend(prop={'size': 10})

    return fig


def phase_errors_subplots(result):

    plt.clf()
    p, _ = num_subplots.num_subplots(len(result))
    x_phi = np.arange(-pi, pi, pi / 12)
    fig = plt.figure(figsize=(1 + p[1]*4, 1 + p[0]*3), dpi=80)
    for ix, key in enumerate(result.keys()):
        ax1 = fig.add_subplot(p[0], p[1], ix + 1)
        fc = list(plt.get_cmap('Dark2')(ix))
        fc[-1] = 0.5 # transparent
        ax1.hist(result[key]['phi_test_error'], bins=x_phi, lw=2, fc=fc, label=key)
        title = 'MAE: {}: {:0.2f} '.format(key, result[key]['mae'])
        ax1.set_title(title)
        ax1.set_xlim([-pi, pi])
        ax1.legend(prop={'size': 10})

    return fig


def get_mace(result):
    # p, _ = num_subplots.num_subplots(len(result))
    # x_phi = np.arange(-pi, pi, pi / 12)
    list_subject = list(result.keys())
    list_met = list(result[list_subject[0]][list(result[list_subject[0]].keys())[0]].keys())

    mace = dict()
    for met in list_met:
        mace[met] = dict(zip(list_subject, [[] for ix in range(len(list_subject))]))

    for subject in list_subject:
        for ix_run in result[subject].keys():
            for met in list_met:
                if not (result[subject][ix_run] == {}):
                    phi_test_error = result[subject][ix_run][met]['phi_test_error']
                    mace_ = np.mean(np.abs(phi_test_error))
                    mace[met][subject].append(mace_)

    mace = pd.DataFrame(mace)
    return mace


def get_mace_run_averaged(result):
    # p, _ = num_subplots.num_subplots(len(result))
    # x_phi = np.arange(-pi, pi, pi / 12)
    list_subject = list(result.keys())
    list_met = list(result[list_subject[0]][list(result[list_subject[0]].keys())[0]].keys())

    mace = dict()
    phi_test_error_mat = dict()
    for met in list_met:
        mace[met] = dict(zip(list_subject, [[] for ix in range(len(list_subject))]))
        phi_test_error_mat[met] = dict(zip(list_subject, [[] for ix in range(len(list_subject))]))

    for subject in list_subject:
        for ix_run in result[subject].keys():
            for met in list_met:
                if not (result[subject][ix_run] == {}):
                    phi_test_error = result[subject][ix_run][met]['phi_test_error']
                    phi_test_error_mat[met][subject] = np.append(phi_test_error_mat[met][subject], phi_test_error)
                    # phi_test_error_mat[met][subject].append(phi_test_error)

    for subject in list_subject:
            for met in list_met:
                mace[met][subject] = np.mean(np.abs(phi_test_error_mat[met][subject]))

    mace = pd.DataFrame(mace)
    return mace


def phase_errors_subjects(result, do_plot=True, fig_opt=None):

    mace = get_mace(result)
    list_met = mace.columns
    mace_mean = mace.applymap(np.mean)
    mace_std = mace.applymap(np.std)
    mace_n = mace.applymap(len)
    mace_sem = mace_std / np.sqrt(mace_n)

    if do_plot:
        if fig_opt is None:
            fig_opt = dict()
            fig_opt['figsize'] = (5, 5)
            fig_opt['dpi'] = 80

        fig = plt.figure(**fig_opt)
        ax = fig.add_subplot(111)
        if len(mace_mean.columns)==1:
            mace_mean.hist()
        else:
            ax = mace_mean.plot(ax=ax, x=list_met[0], y=list(set(list_met) - set([list_met[0]])), style='o')
    else:
        fig = None

    return fig

