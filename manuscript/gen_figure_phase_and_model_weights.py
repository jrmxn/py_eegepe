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
figures = Path('/hdd/Local/gitprojects/py_eegepe/manuscript/figures/time_series')
img_fmt = 'svg'

arch_nice = paradigm.get_arch_nice()
ix_fig = 1
arch = paradigm.get_standard_arch(ix_fig)
arch = ['net_sgd_000']
sub_list = paradigm.data_loader.sublist(datadir, dataset)

opt_train, opt_test = paradigm.opt_init(arch, overwrite_dat=False)
sub_result = paradigm.main_multiple_subject(datadir, dataset, sub_list, arch, opt_train=opt_train, opt_test=opt_test, gen_fooof=True)
#%%
colix = np.arange(0, len(arch))
cmap = sns.color_palette("colorblind", len(arch))
cmap = [cmap[ix] for ix in colix]

# %%
ix_config = 1
opt_train, opt_test, es, ds = paradigm.opt_quick_config(arch, ix_config, script_type=script_type)
opt_test['plot'] = True
opt_test['overwrite_res'] = True  # doesn't actually overwrite because plot = True
# %%
# This is training fully within subject
df = pd.DataFrame(index=sub_list, columns=arch)
df = df.fillna(0)  # with 0s rather than NaNs
weights = []
cmap = [(0.00784313725490196, 0.6196078431372549, 0.45098039215686275),
(0.00784313725490196, 0.3, 0.15098039215686275)]
f_width = 6
f_height = 3
f_width = (18/4) * (4/3) * (1/2.54)
f_height = 4 * (1/2.54)
for sub in sub_list:

    fig, ax = plt.subplots(1, 1, figsize=(f_width, f_height), dpi=dpi)
    for arch_ in arch:
        model_info = paradigm.main_within_subject_full(datadir, dataset, sub, arch_,
                                              opt_train=opt_train, opt_test=opt_test, es='')
        if arch_ == 'net_sgd_000':
            net_sgd_000 = model_info

    fig = plt.gcf()
    ax = fig.axes[0]
    ax.set_ylabel('Phase (rad.)')
    ax.set_xlim(-0.5, 0)
    ax.set_ylim(-np.pi - 0.1, np.pi + 0.1)
    ax.set_xlabel('Time (s)')
    line = ax.get_children()[0]
    line.set_linewidth(1.0)
    line.set_color(cmap[0])
    # ax.title.set_text('')
    fig.tight_layout()
    f_name = f"paper_{sub}.{img_fmt}"
    sns.despine()
    fig.savefig(figures / f_name, format=img_fmt, dpi=dpi_print)

    w = net_sgd_000['model'].get_weights()
    weights.append(np.flipud(w[0]))


fs = model_info['fs']
dt = 1/fs
plt.close('all')
t_f = np.arange(0, dt*np.shape(weights[0])[0], dt)
fig, ax = plt.subplots(1, 1, figsize=(f_width, f_height), dpi=dpi)

for w in weights:
    ax.plot(t_f, w[:, 0], color=cmap[0], linewidth=0.5)
    ax.plot(t_f, w[:, 1], color=cmap[1], linewidth=0.5)
ax.set_ylabel('Filter magnitude (A.U.)')
ax.set_xlabel('Time (s)')
fig.tight_layout()
sns.despine()
f_name = f"paper_filter.{img_fmt}"
fig.savefig(figures / f_name, format=img_fmt, dpi=dpi_print)
# plt.show()
print(1)
