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

from py_eegepe import data_loader

"""ws: within subject!"""
script_type = 'ws'
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
figures = Path('/hdd/Local/gitprojects/py_eegepe/manuscript/figures/psd')
img_fmt = 'svg'

arch_nice = paradigm.get_arch_nice()
ix_fig = 1
arch = paradigm.get_standard_arch(ix_fig)

sub_list = paradigm.data_loader.sublist(datadir, dataset)

opt_train, opt_test = paradigm.opt_init(arch, overwrite_dat=False)
sub_result = paradigm.main_multiple_subject(datadir, dataset, sub_list, arch, opt_train=opt_train, opt_test=opt_test, gen_fooof=True)
#%%
colix = np.arange(0, len(arch))
cmap = sns.color_palette("colorblind", len(arch))
cmap = [cmap[ix] for ix in colix]

# %%

# sub_list = [sub_list[0]]
f_width = (18/4) * (4/3) * (1/2.54)
f_height = 4 * (1/2.54)
do_save = True
for sub in sub_list:
    fig, ax = plt.subplots(1, 1, figsize=(f_width, f_height), dpi=dpi)
    # The actual Kuramoto plot routing is in the kuramoto fitting function in data_loader
    # if you go into get_alpha and uncomment:
    # plot_signal_stretch(t, sig, sig_alpha, f_name)
    # and here uncomment:
    # do_save = False
    # and note that figure dimensions are controlled in plot_signal_stretch
    # you can also generate time series examples
    use_synth = dict({type: 'kuramoto', 'N': 16, 'A_dist': 'constant'})
    pro_, meta_ = data_loader.get_alpha(datadir, dataset, sub, 1, overwrite=True, use_synth=use_synth,
                            eeg_setting='occipital_alpha_hjorth', do_save=False)
    fig = plt.gcf()
    ax = fig.axes[0]
    # plt.ylabel('MACE')
    # plt.ylim([0.39, 1.31])
    ax.set_ylim(0, 50)
    ax.set_xlabel('Frequency (Hz)')
    # ax.set_ylabel('PSD (μV²/Hz)')
    ax.set_ylabel('PSD (AU)')
    ax.title.set_text('')
    fig.tight_layout()
    sns.despine()
    f_name = f"paper{''}_kuramoto_{sub}.{img_fmt}"
    if do_save:
        fig.savefig(figures / f_name, format=img_fmt, dpi=dpi_print)
