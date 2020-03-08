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

"""wr: within run!"""
script_type = 'wr'
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
# ix_fig = 1
# arch = paradigm.get_standard_arch(ix_fig)

sub_list = paradigm.data_loader.sublist(datadir, dataset)

arch = paradigm.get_standard_arch('main_set')
opt_train, opt_test = paradigm.opt_init(arch, overwrite_dat=False)
paradigm.main_multiple_subject(datadir, dataset, sub_list, arch, opt_train=opt_train, opt_test=opt_test, gen_fooof=True)

# %% Settings
do_plot = True
n_config = paradigm.opt_quick_config(None, None)
for ix_config in range(0, n_config + 1):
    ix_config_local = ix_config
    if ix_config == 0:
        fig_type = 'scan_fir_length'
        ix_config_local = 1
    elif ix_config == 1:
        fig_type = 'main_set'
    else:
        fig_type = 'main_set_reduced'
    arch = paradigm.get_standard_arch(fig_type)
    opt_train, opt_test, es, ds = paradigm.opt_quick_config(arch, ix_config_local, script_type=script_type)
    print(es)

    # %% within run
    sub_result = paradigm.main_multiple_subject(datadir, dataset, sub_list, arch, opt_train, opt_test)
    mace = py_eegepe.summary.phase_plot.get_mace_run_averaged(sub_result)

    # %% predict next run
    increment_ix_run = 1
    sub_result_test = paradigm.main_multiple_subject(datadir, dataset, sub_list, arch, opt_train, opt_test,
                                                     increment_ix_run=increment_ix_run)
    if do_plot:
        mace_test = py_eegepe.summary.phase_plot.get_mace_run_averaged(sub_result_test)
        mace_test.columns = ['test_' + c for c in list(mace_test.columns)]
        df_mean = pd.concat([mace, mace_test], axis=1)

        arch_select = ['test_' + arch_ for arch_ in arch]

        # use_result = sub_result_test
        ##

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
        # %%
        for ix_arch, arch_ in enumerate(arch_select):
            if arch_ in arch_nice:
                arch_nice_ = arch_nice[arch_]
            else:
                arch_nice_ = 'X_' + arch_
            df = df_mean[arch_]
            n = df.shape[0]
            mea_df_mean = np.mean(df)
            sem_df_mean = np.std(df)/np.sqrt(n)
            s = '{}: {:0.2f}+-{:0.2f} SEM'.format(arch_nice_, mea_df_mean, sem_df_mean)

        # %%
        if fig_type == 'scan_fir_length':
            f_width = (18 / 4) * (4 / 3) * (1 / 2.54)
            f_height = 4 * (1 / 2.54)
            fig, ax = plt.subplots(1, 1, figsize=(f_width, f_height), dpi=dpi)
            y = df_mean.T
            y = y.iloc[y.index.str.contains("test"), :]
            y_mea = y.mean(axis=1)
            yerr = 2 * y.sem(axis=1)
            from scipy.interpolate import pchip, Akima1DInterpolator

            x = np.arange(1, len(y_mea.index) + 1)
            x = [int(y_mea.index[ix].split('_')[-1])/512.0 for ix in range(len(y_mea))]

            x_pchip = np.arange(x[0], x[-1], 1e-3)
            h_pchip = Akima1DInterpolator(x, np.array(y_mea))
            h_pchip_ups = Akima1DInterpolator(x, np.array(y_mea) + yerr)
            h_pchip_dow = Akima1DInterpolator(x, np.array(y_mea) - yerr)

            y_ups = h_pchip_ups(x_pchip)
            y_dow = h_pchip_dow(x_pchip)
            y_ = h_pchip(x_pchip)

            # for ix_x, x_ in enumerate(x):
            #     ax.plot(x_ * np.ones(2), x_mea[ix_x] + np.array([-1, 1])*yerr[ix_x], color=cmap[0])
            #     ax.plot(x_ + np.array([-0.1, +0.1]), (x_mea[ix_x] + yerr[ix_x]) * np.ones(2), color=cmap[0])
            #     ax.plot(x_ + np.array([-0.1, +0.1]), (x_mea[ix_x] - yerr[ix_x]) * np.ones(2), color=cmap[0])
            plt.fill_between(x_pchip, y_dow, y_ups,
                             facecolor=cmap[0],  # The fill color
                             color=cmap[0],  # The outline color
                             alpha=0.2)
            plt.plot(x_pchip, y_, color=cmap[0])

            # plt.xticks(x[0::2], labels=labels[::2])
            plt.xticks(np.arange(0.175, x[-1]+0.025, 0.05))
            plt.ylabel('MACE')
            plt.xlabel('Filter length (s)')
            plt.tight_layout()
            sns.despine()

            f_name = f"paper_{es}_fig_{0}d.{img_fmt}"
            # plt.show()
            plt.savefig(figures / f_name, format=img_fmt, dpi=dpi_print)

            if ix_config == 1:
                break
        else:
            # this is the main plotting function
            df_mean_summary = df_mean.loc[:, arch_select]
            # labels = [arch_nice[l] for l in df_mean_summary.columns.to_list()]
            labels_nice = [arch_nice[l] for l in df_mean_summary.columns.to_list()]
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
            #     str_label = '(a)'
            # fig.text(0.02, 0.965, str_label, fontsize=14, weight='bold')
            sns.despine()

            if fig_type == 'main_set':

                # hardcoded!
                df_squash = pd.DataFrame()
                df_squash['fir'] = df_mean_summary[['test_fir_gen_000', 'test_fir_alp_000']].mean(axis=1)
                df_squash['learning'] = df_mean_summary[['test_net_sgd_000', 'test_net_sgd_001', 'test_net_cug_000']].mean(axis=1)
                p_learning, _ = py_eegepe.summary.results_plot.get_paired_p_values(df_squash)
                if p_learning[0] > 1e-3:
                    p_learning_str = f'{p_learning[0]:0.3f}'
                else:
                    p_learning_str = f'{p_learning[0]:0.1e}'
                    p_learning_str = p_learning_str.replace('e', ' × 10')
                print(f'p = {p_learning_str}, averaged FIR methods, compared with averaged learning methods, Wilcoxon signed-rank test')

                f_name = f"paper_{es}_fig_{1}e.{img_fmt}"
                plt.savefig(figures / f_name, format=img_fmt, dpi=dpi_print)
