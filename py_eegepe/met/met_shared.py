import numpy as np
from scipy import signal


def met_preproc(s, fs, met_try='diff', o=None):
    if o is None:
        o = (2 * fs) + 1

    if met_try == 'diff':
        s = np.diff(np.concatenate((np.array([s[0]]), s)))
    elif met_try == 'hpnc':
        # note that this is acausal! (on purpose, to answer reviewer)
        o = int(2 * (5 * fs) + 1)
        hp_filter = {'a': 1, 'b': signal.firwin(o, [1/(fs/2)], pass_zero=False)}
        #X s = signal.lfilter(hp_filter['b'], hp_filter['a'], s)
        s = signal.filtfilt(hp_filter['b'], hp_filter['a'], s)

    elif met_try == 'subsmooth':
        s = s - signal.lfilter(np.ones(o)/o, 1, s)
    else:
        raise Exception("met_try in sig_preproc mispecified")

    s = s/np.std(s)

    return s


def slice_signal(event_ix, sig, time, ix_back=0, ix_forward=0):
    ix_cut_before_event = ix_back
    ix_cut_after_previous_event = ix_forward
    # try to include first event
    ix_start = event_ix[0] - np.round(np.mean(np.diff(event_ix)))
    if ix_start < 0: ix_start = 0
    ev_cut = np.zeros((len(event_ix), 2))
    ev_cut[0][0] = ix_start
    ev_cut[0][1] = event_ix[0]
    for ix_a in range(1, len(event_ix)):  # 1 indexing is on purpose.
        ix_from = event_ix[ix_a - 1] + ix_cut_after_previous_event
        ix_to = event_ix[ix_a] - ix_cut_before_event
        assert ix_from < ix_to, 'ix_back or ix_forward is too big. ix_from: {}, ix_to: {}'.format(ix_from, ix_to)
        ev_cut[ix_a][0] = np.round(ix_from)
        ev_cut[ix_a][1] = np.round(ix_to)

    sig_ev = list()
    time_ev = list()
    for ix_ev_cut in range(0, len(ev_cut)):
        ix_s = int(ev_cut[ix_ev_cut][0])
        ix_e = int(ev_cut[ix_ev_cut][1])
        sig_ev.append(sig[ix_s:ix_e])
        time_ev.append(time[ix_s:ix_e])

    return sig_ev, time_ev


def slice_signal_matrix(event_ix, sig, time, fs, t_from=0, t_to=0, remove_baseline=False, ix_back=0, t_baseline=0):
    """ t_from should be negative if you want to go before event"""
    ix_from = np.round(t_from*fs).astype(int)
    ix_to = np.round(t_to*fs).astype(int)
    sec = np.arange(ix_from, ix_to)  # sec = [-50:50];
    ix_mat = np.tile(np.array(event_ix - ix_back).reshape(-1, 1), (1, len(sec)))  # ix_mat = ix;%repmat not needed
    sec_mat = np.tile(sec, (len(event_ix), 1))  # sec_mat = sec;%repmat not needed
    ixsec_mat = ix_mat + sec_mat
    sig_ev = sig[ixsec_mat]
    time_ev = np.linspace(t_from, t_to, len(sec))

    if remove_baseline:
        sig_baseline = np.mean(sig_ev[:, time_ev < t_baseline], axis=1)
        sig_baseline = np.tile(sig_baseline.reshape(-1, 1), (1, len(sec)))
        sig_ev = sig_ev - sig_baseline

    return sig_ev, time_ev


def split_train_val(x_ev, h_ev, es_validation):
    vec_ix = np.random.permutation(len(x_ev))
    vec_ix_cutoff = int(np.round(len(x_ev) * es_validation))
    vec_ix_slice_train = vec_ix[vec_ix_cutoff:]
    x_ev_train = [x_ev[ix] for ix in vec_ix_slice_train]
    h_ev_train = [h_ev[ix] for ix in vec_ix_slice_train]
    vec_ix_slice_val = vec_ix[:vec_ix_cutoff]
    x_ev_val = [x_ev[ix] for ix in vec_ix_slice_val]
    h_ev_val = [h_ev[ix] for ix in vec_ix_slice_val]
    return x_ev_train, h_ev_train, x_ev_val, h_ev_val


class TrainDefault:
    def __init__(self):
        self._epochs = 75
        self._es_min_delta = 0
        self._es_patience = 12
        self._early_stopping = True
        self._validation = None

    @property
    def epochs(self):
        return self._epochs

    @property
    def es_min_delta(self):
        return self._es_min_delta

    @property
    def es_patience(self):
        return self._es_patience

    @property
    def early_stopping(self):
        return self._early_stopping

    @property
    def validation(self):
        validation = self._validation
        if validation is not None:
            if validation >= 1: raise Exception('es_validation cannot be greater than 1')
            if validation == 0: validation = None

        return validation

    @epochs.setter
    def epochs(self, value):
        self._epochs = value

    @es_min_delta.setter
    def es_min_delta(self, value):
        self._es_min_delta = value

    @es_patience.setter
    def es_patience(self, value):
        self._es_patience = value

    @validation.setter
    def validation(self, value):
        self._validation = value

    @early_stopping.setter
    def early_stopping(self, value):
        self._early_stopping = value
