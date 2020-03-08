#!/usr/bin/env python
# -*- coding: utf-8 -*-
from pathlib import Path
import os.path
import sys
import importlib

module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path: sys.path.append(module_path)
import py_eegepe
import numpy as np

importlib.reload(py_eegepe)
import py_eegepe.paradigm as paradigm
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns

"""as: across subject!"""
script_type = 'as'

#%%
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
sns.set_style("ticks")
sns.set_context("paper", rc={"lines.linewidth": 1.0})
plt.close('all')
s = 7
params = {
    'axes.labelsize': s+1,
    'axes.titlesize': s,
    'legend.fontsize': s,
    'xtick.labelsize': s,
    'ytick.labelsize': s,
    'text.usetex': False,
}
dpi = 150
dpi_print = 300
rcParams.update(params)

#%%
gl = py_eegepe.data_specific.git_label()
print(str(gl.label))
print(str(gl.tag))

datadir = Path('/home/mcintosh/Cloud/DataPort/')
dataset = Path('2017-04-21_LIINC_go_nogo')
figures = Path() / 'figures'
img_fmt = 'svg'

arch_nice = paradigm.get_arch_nice()
sub_list = paradigm.data_loader.sublist(datadir, dataset)

arch = paradigm.get_standard_arch('main_set')
opt_train, opt_test = paradigm.opt_init(arch, overwrite_dat=False)
paradigm.main_multiple_subject(datadir, dataset, sub_list, arch, opt_train=opt_train, opt_test=opt_test, gen_fooof=True)

# %% Settings
do_plot = True
es1 = 'as_kuramoto_testkuramoto'
es2 = 'as_kuramoto_testkuramoto_gt'
es3 = 'as_kuramoto_testkuramoto_gt_traingt'
es4 = 'as_kuramoto_testkuramoto_gt_above_alphasnr90'
g_as_kuramoto_testkuramoto_gt = []
g_as_kuramoto_testkuramoto_gt_traingt = []
g_as_kuramoto_testkuramoto_gt_above_alphasnr = []
n_config = paradigm.opt_quick_config(None, None)
for ix_config in range(1, n_config + 1):
    if ix_config == 1:
        fig_type = 'main_set'
    else:
        fig_type = 'main_set_reduced'
    arch = paradigm.get_standard_arch(fig_type)

    opt_train, opt_test, es, ds = paradigm.opt_quick_config(arch, ix_config, script_type=script_type)
    # %%
    arch_select = ['test_' + arch_ for arch_ in arch]

    #%%
    colix = np.arange(0, len(arch_select))
    cmap = sns.color_palette("colorblind", len(arch_select))
    cmap = [cmap[ix] for ix in colix]
    d_cmap = {
        'fir_gen_000': (0.00392156862745098, 0.45098039215686275, 0.6980392156862745),
        'fir_alp_000': (0.8705882352941177, 0.5607843137254902, 0.0196078431372549),
        'net_sgd_000': (0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
        'net_sgd_001': (0.8352941176470589, 0.3686274509803922, 0.0),
        'net_cug_000': (0.8, 0.47058823529411764, 0.7372549019607844)
              }
    for ix, arch_ in enumerate(arch):
        cmap[ix] = d_cmap[arch_]
    #%%
    # note that I made it so last run is omitted from training! so you could just focus on that one...
    df = pd.DataFrame(index=sub_list, columns=arch)
    df = df.fillna(0)  # with 0s rather than NaNs
    df_p = df.copy()
    for arch_ in arch:
        for sub in sub_list:
            print(sub)
            x = paradigm.main_across_subjects(datadir, dataset, sub_list, arch_,
                                              opt_train=opt_train, opt_test=opt_test, test_only=sub)
            mae = x[sub][0][arch_]['mae']
            # print('{} {}: {}'.format(sub, arch_, mae))
            # x = x[list(x.keys())[-1]][arch_]['mae']
            df.loc[sub, arch_] = mae
            df_p.loc[sub, arch_] = np.median(x[sub][0][arch_]['p_ratio'])

            if arch_ == 'net_sgd_000':
                if es == es2:
                    g_as_kuramoto_testkuramoto_gt.append(x[sub][0][arch_]['phase'])
                elif es == es3:
                    g_as_kuramoto_testkuramoto_gt_traingt.append(x[sub][0][arch_]['phase'])
                elif es == es4:
                    g_as_kuramoto_testkuramoto_gt_above_alphasnr.append(x[sub][0][arch_]['phase'])

    if do_plot:
        plt.close('all')

        df_mean_summary = df
        arch_local = df_mean_summary.columns.to_list()
        # labels = [arch_nice[l] for l in arch_local]
        labels_nice = [arch_nice[l] for l in df.columns.to_list()]
        if len(labels_nice) == 5:
            f_width = (18 / 4) * (4 / 3) * (1 / 2.54)
        else:
            f_width = 4.5 * (1 / 2.54)
        f_height = 8 * (1 / 2.54)
        plt.close('all')
        fig, ax = plt.subplots(1, 1, figsize=(f_width, f_height), dpi=dpi)
        ax.set_ylim([0.29, 1.76])
        py_eegepe.summary.results_plot.box_whisker(ax, df_mean_summary, cmap, labels=labels_nice)
        fig.tight_layout()
        # if ix_config:
        #     str_label = '(c)'
        # fig.text(0.02, 0.965, str_label, fontsize=14, weight='bold')
        sns.despine()

        f_name = f"paper_{es}_fig_{1}e.{img_fmt}"
        plt.savefig(figures / f_name, format=img_fmt, dpi=dpi_print)

        # plt.show()

        if fig_type == 'main_set':
            df_squash = pd.DataFrame()
            df_squash['fir'] = df_mean_summary[['fir_gen_000', 'fir_alp_000']].mean(axis=1)
            df_squash['learning'] = df_mean_summary[['net_sgd_000', 'net_sgd_001', 'net_cug_000']].mean(axis=1)
            p_learning, _ = py_eegepe.summary.results_plot.get_paired_p_values(df_squash)
            if p_learning[0] > 1e-3:
                p_learning_str = f'{p_learning[0]:0.3f}'
            else:
                p_learning_str = f'{p_learning[0]:0.1e}'
                p_learning_str = p_learning_str.replace('e', ' × 10')
            print(f'p = {p_learning_str}, averaged FIR methods, compared with averaged learning methods, Wilcoxon signed-rank test')

        print(ds)
        print('-------------')
        print('-------------')

    if do_plot:
        if es == es1:
            df1 = df
            df_p1 = df_p
        if es == es2:
            df2 = df
            df_p2 = df_p
            plt.close('all')
            fig_snr, ax_snr = plt.subplots(1, 2, figsize=(f_width * 2, f_height), dpi=dpi)

            # TODO: this all looks very bug prone because of assumptions about matching dfx and df_px
            assert (df_p1 == df_p2).loc[:, 'net_sgd_000'].all(), 'df_p1 does not match df_p2!'
            for ix in range(2):
                if ix == 0:
                    c_lines = np.array([222, 200, 5])/255
                    df_px = df_p1
                    dfx = df1
                else:
                    c_lines = np.array([222, 50, 0])/255
                    df_px = df_p2
                    dfx = df2

                ax_snr[0].plot(df_px.loc[:, 'net_sgd_000'], dfx.loc[:, 'net_sgd_000'], 'o', color=c_lines, markersize=5,
                            markerfacecolor=c_lines, markeredgewidth=0.75, markeredgecolor=[1, 1, 1], linewidth=0.75, zorder=3, alpha=0.95)

            c_lines = np.ones(3) * 0.25
            ax_snr[1].plot(df_px.loc[:, 'net_sgd_000'], (df2 - df1).loc[:, 'net_sgd_000'], 'o', color=c_lines, markersize=5,
                        markerfacecolor=c_lines, markeredgewidth=0.75, markeredgecolor=[1, 1, 1], linewidth=0.75,
                        zorder=3, alpha=0.95)

            fig_snr.tight_layout()
            sns.despine()

            ax_snr[0].set_ylim([0, None])
            ax_snr[0].set_xlim([0, 6.0])
            ax_snr[0].set_xticks(np.arange(0, 6))
            ax_snr[0].set_xlabel('SNR')
            ax_snr[0].set_ylabel(r'MACE (rad.)')

            ax_snr[1].set_ylim([0, None])
            ax_snr[1].set_xlim([0, 6.0])
            ax_snr[1].set_xticks(np.arange(0, 6))
            ax_snr[1].set_xlabel('SNR')
            ax_snr[1].set_ylabel(r'$\Delta$MACE (rad.)')
            fig_snr.subplots_adjust(top=1.0, bottom=0.325, left=0.1, right=0.9, wspace=0.4, hspace=0)


            f_name = f"paper_{es}_fig_{1}_snr.{img_fmt}"
            fig_snr.savefig(figures / f_name, format=img_fmt, dpi=dpi_print)

            # now spit out the comparison betweed GT evaluation and non-causal filtering:
            print('---\n---\n---\n---')
            for met in df.columns:
            # met = 'net_sgd_000'
                print(f"Comparing non-causal vs ground-truth evaluation for {met}:")
                df_eval = pd.concat([df1.loc[:, met], df2.loc[:, met]], axis=1)
                df_eval.columns = ['nc', 'gt']
                p_eval_comparison, _ = py_eegepe.summary.results_plot.get_paired_p_values(df_eval)
            print('---\n---\n---\n---')


import pycircstat
g = g_as_kuramoto_testkuramoto_gt
# g = g_as_kuramoto_testkuramoto_gt_traingt
# g = g_as_kuramoto_testkuramoto_gt_above_alphasnr
gg = np.array([])
for ix_g in range(len(g)):
    g_local = g[ix_g]
    # g_local = np.random.rand(np.shape(g_local)[0]) * np.pi * 2
    print(pycircstat.rayleigh(g_local)[0])
    gg = np.append(gg, g_local)
    plt.close('all'); plt.hist(g_local); plt.show()
plt.close('all');plt.hist(gg);plt.show()
print(pycircstat.rayleigh(gg))
print(1)
