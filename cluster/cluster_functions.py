# -*- coding: utf-8 -*-
#!/usr/bin/env python3

from pathlib import Path
import os.path
import sys
import importlib
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path: sys.path.append(module_path)
import py_eegepe
importlib.reload(py_eegepe)
import py_eegepe.paradigm as paradigm
import py_eegepe.data_loader as data_loader
gl = py_eegepe.data_specific.git_label()
import multiprocessing
import tensorflow as tf
import socket
import argparse

n_cpu = multiprocessing.cpu_count()
n_pcore = int(n_cpu/2)
config = tf.ConfigProto()
config.intra_op_parallelism_threads = n_pcore
config.inter_op_parallelism_threads = n_pcore
tf.Session(config=config)
if socket.gethostname() == 'OJDEF-Desktop':
    DATADIR = Path('/home/mcintosh/Cloud/DataPort/')
else:
    DATADIR = Path('/rigel/dsi/users/jrm2263/Local/Data/')


print(str(gl.label))
print(str(gl.tag))
print('Physical cores: {}'.format(n_pcore))


def run_main_multiple_subject(ix_sub=0, arch=None, dataset=None, ix_config=None, datadir=None):

    if datadir is None or datadir is '':
        datadir = DATADIR
    if dataset is None or dataset is '':
        dataset = '170421_EEG_linbi'

    sub_list = data_loader.sublist(datadir, dataset)
    sub_list = sub_list[ix_sub]
    # is the following line right?
    # if not isinstance(sub_list, list): x = [sub_list]
    # opt_train, opt_test = paradigm.opt_init(overwrite_dat=False, arch=arch, epochs=epochs,
    #                                         predict_amplitude=predict_amplitude, es_patience=es_patience)
    print(opt_train)
    opt_train, opt_test, es, ds = paradigm.opt_quick_config(arch, ix_config, script_type='as')

    sub_result = paradigm.main_multiple_subject(datadir, dataset, sub_list, arch, opt_train, opt_test)
    print('Done')


def run_main_across_subjects(ix_sub=0, arch=None, dataset=None, ix_config=None, datadir=None):

    if datadir is None or datadir is '':
        datadir = DATADIR
    if dataset is None or dataset is '':
        dataset = '170421_EEG_linbi'

    sub_list = data_loader.sublist(datadir, dataset)
    test_only = sub_list[ix_sub]
    opt_train, opt_test, es, ds = paradigm.opt_quick_config(arch, ix_config, script_type='as')

    sub_result = paradigm.main_across_subjects(datadir, dataset, sub_list, arch,
                                               opt_train=opt_train, opt_test=opt_test, test_only=test_only)
    print('Done')


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    FUNCTION_MAP = {'run_main_multiple_subject': run_main_multiple_subject,
                    'run_main_across_subjects': run_main_across_subjects}
    parser = argparse.ArgumentParser(description='something')
    parser.add_argument('--function', choices=FUNCTION_MAP.keys(), default='run_main_multiple_subject')
    parser.add_argument('--ix_sub', type=int, default=0, help='')
    parser.add_argument('--arch', nargs='+', default=['fir_opt_000', 'net_gru_000'])
    parser.add_argument('--dataset', type=str, default='170421_EEG_linbi')
    parser.add_argument('--datadir', type=str, default='')
    parser.add_argument('--ix_config', type=int, default=0)
    # parser.add_argument('--predict_amplitude', type=str2bool, default=False)

    args = parser.parse_args()
    method = FUNCTION_MAP[args.function]

    method(ix_sub=args.ix_sub, arch=args.arch,
           dataset=args.dataset, ix_config=args.ix_config, datadir=args.datadir)
