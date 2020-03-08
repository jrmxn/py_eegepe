import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


def slice_matrix(sig_ev, time_ev):
    # sig_ev, time_ev = sp.slice_signal_matrix(event_ix, sig, time, fs, \
    # t_from=t_from, t_to=t_to, remove_baseline=remove_baseline, ix_back=ix_back, t_baseline=t_baseline)

    n = np.sqrt(np.shape(sig_ev)[0])
    mea_sig_ev = np.mean(sig_ev, axis=0)
    sem_sig_ev = np.std(sig_ev, axis=0)/np.sqrt(n)
    # n.b. not currently dealing with nan in sig_ev for n
    # fig = plt.figure()
    h_fill = plt.fill_between(time_ev, mea_sig_ev - sem_sig_ev, mea_sig_ev + sem_sig_ev, alpha = 0.2)
    h_line = plt.plot(time_ev, mea_sig_ev)

    res = dict()
    res['h_fill'] = h_fill
    res['h_plot'] = h_line
    res['mea_sig_ev'] = mea_sig_ev
    res['sem_sig_ev'] = sem_sig_ev
    res['sig_ev'] = sig_ev
    res['time_ev'] = time_ev
    return res


