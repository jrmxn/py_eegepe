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
figures = Path() / 'figures'
img_fmt = 'svg'

arch_nice = paradigm.get_arch_nice()

sub_list = paradigm.data_loader.sublist(datadir, dataset)

arch = paradigm.get_standard_arch('main_set')
opt_train, opt_test = paradigm.opt_init(arch, overwrite_dat=False)
paradigm.main_multiple_subject(datadir, dataset, sub_list, arch,
                                            opt_train=opt_train, opt_test=opt_test, gen_fooof=True)
#%%
colix = np.arange(0, len(arch))
cmap = sns.color_palette("colorblind", len(arch))
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
# %% Settings
do_plot = True
n_config = paradigm.opt_quick_config(arch, None)
for ix_config in range(1, n_config + 1):
    if ix_config == 1:
        fig_type = 'main_set'
    else:
        fig_type = 'main_set_reduced'
    arch = paradigm.get_standard_arch(fig_type)
    opt_train, opt_test, es, ds = paradigm.opt_quick_config(arch, ix_config, script_type=script_type)
    # opt_train['overwrite_met'] = True
    # %%
    # This is training fully within subject
    df = pd.DataFrame(index=sub_list, columns=arch)
    df = df.fillna(0)  # with 0s rather than NaNs
    for arch_ in arch:
        for sub in sub_list:
            print(sub)
            # if ('TTL' in sub) or ('DX' in sub):
            #     opt_test['overwrite_res'] = True
            # else:
            #     opt_test['overwrite_res'] = False
            x = paradigm.main_within_subject_full(datadir, dataset, sub, arch_,
                                                  opt_train=opt_train, opt_test=opt_test, es='')
            vec_run = list(x.keys())
            mae_last_run = x[vec_run[-1]][arch_]['mae']
            mae_average = np.mean([x[ix_run][arch_]['mae'] for ix_run in vec_run if x[ix_run] is not None])

            # if we switch this from mae_average to mae_last_run then... we get results as if held out on last run
            # since we are always holding out on the last run (I think)
            df.loc[sub, arch_] = mae_average

    if do_plot:
        labels_nice = [arch_nice[l] for l in df.columns.to_list()]
        if len(labels_nice) == 5:
            f_width = (18 / 4) * (4 / 3) * (1 / 2.54)
        else:
            f_width = 4.5 * (1 / 2.54)
        f_height = 8 * (1 / 2.54)
        plt.close('all')
        fig, ax = plt.subplots(1, 1, figsize=(f_width, f_height), dpi=dpi)
        py_eegepe.summary.results_plot.box_whisker(ax, df, cmap, labels=labels_nice)
        fig.tight_layout()
        # if ix_config:
        #     str_label = '(b)'
        # fig.text(0.02, 0.965, str_label, fontsize=8, weight='bold')
        sns.despine()

        f_name = f"paper_{es}_fig_{1}e.{img_fmt}"
        plt.savefig(figures / f_name, format=img_fmt, dpi=dpi_print)
        print(1)
        # plt.show()
