from .met_shared import split_train_val as split_train_val
from .met_shared import TrainDefault as TrainDefault
import numpy as np
from scipy import signal
from pathlib import Path
import datetime
import pickle
from pykalman import AdditiveUnscentedKalmanFilter, UnscentedKalmanFilter

try:
    from noisyopt import minimizeCompass
except:
    print('no noisyopt?')
# from scipy.optimize import minimize
# from scipy.optimize import show_options


def defaults(opt):
    if opt is None:
        opt = TrainDefault()
        opt.epochs = 1000
        opt.validation = 0.2
        opt.preproc = None
    elif isinstance(opt, dict):
        assert False, 'need to add this option'
    return opt


def train(x_ev, h_ev, fs, x0, f_arch=None, overwrite=False, ix_shift_input=0,
          opt=None):
    """
    fff
    """
    x0_rand = lambda: [np.random.randint(51, 251), 9.5 + 2 * np.random.randn(1)[0], np.abs(2 * np.random.randn(1)[0])]
    if x0 is None: x0 = x0_rand()
    if f_arch is None: f_arch = Path('fir_{}{}.dat'.format('arch', datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
    if isinstance(f_arch, str): f_arch = Path(f_arch)
    opt_local = defaults(opt)

    if opt_local.validation is None:
        x_ev_train = x_ev
        h_ev_train = h_ev
    else:
        x_ev_train, h_ev_train, x_ev_val, h_ev_val = split_train_val(x_ev, h_ev, opt_local.validation)

    if f_arch.exists() and not overwrite:
        with open(str(f_arch), 'rb') as handle:
            ukf_settings = pickle.load(handle)
        model = ukf_settings['model']

    if not(f_arch.exists()) or overwrite:

        _, phi_mae_full = test(x_ev_train, h_ev_train, model, ix_shift_input=ix_shift_input)
        # TODO: not updated from FIR definition from here down

        # model = firwin_to_model(min_res['x'], fs)

        # ukf_settings = {'model': model, 'opt_f': opt_f, 'weights_filter': min_res, 'history':history}

        with open(str(f_arch), 'wb') as handle:
            pickle.dump(ukf_settings, handle)

    return model


def test(x_ev_test, h_ev_test, model, ix_shift_input=0, edge_fudge=None, trial_wise_compensation=False):
    # assert ix_shift_input==0, "ix_shift_input==0 is fully implemented, but untested"

    fs = model['fs']
    if edge_fudge is None:
        # this is because of the hilbert edges
        edge_fudge = int(np.round(fs*0.01))
    phi_pred, phi_ev_pred = predict(x_ev_test, model, ix_shift_input=ix_shift_input, edge_fudge=edge_fudge, trial_wise_compensation=trial_wise_compensation)

    phi_ev_test = [np.angle(h_ev_test_) for h_ev_test_ in h_ev_test]


    phi_error_full = [(phi_ev_test[ix][edge_fudge:] - phi_ev_pred[ix][edge_fudge:]) for ix in range(len(phi_ev_pred))]
    phi_error_full = [np.mean(np.abs(np.angle(np.exp(1j * phi_err_full_)))) for phi_err_full_ in phi_error_full]
    #  phi_mae_full is used by optimisation I think
    phi_mae_full = np.mean(phi_error_full)

    # extract only the event related bits
    phi = np.array([phi_ev_test_[-1] for phi_ev_test_ in phi_ev_test])
    phi_err = np.angle(np.exp(1j * (phi - phi_pred)))

    amp_err = np.nan

    return phi_err, phi_mae_full, amp_err


def predict(x_ev_test, model, ix_shift_input=0, edge_fudge=None, trial_wise_compensation=False):
    fs = model['fs']
    # %%
    # TODO: this does something but probably needs A LOT of tuning/tweaking - may also just be wrong
    dt = 1/fs
    def f(state, noise):
        xp = state[0]
        xv = state[1]
        xw = state[2]
        xg = state[3]
        xo = state[4]
        dt2 = np.power(dt, 2)
        xw2 = np.power(xw, 2)
        A = np.array([
            [1-dt2*xw + xg, dt, 0, 0, 0],
            [-dt*xw2, 1 + xg, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 1],
        ])
        return np.matmul(A, state) + noise

    def g(state, noise):
        xp = state[0]
        xv = state[1]
        xw = state[2]
        xg = state[3]
        xo = state[4]
        return xp + xo + noise

    initial_state_mean = np.array([1, 1, 2*np.pi*10, 0, 0])
    R = np.array([[1]])
    Q = np.sqrt(dt)*np.eye(5) * np.array([0.1, 0.1, 0.1, 1e-4, 1]).reshape(-1, 1)
    # sigma = np.diag(Q)

    # %%
    ukf = UnscentedKalmanFilter(f, g,
                                transition_covariance=Q, observation_covariance=R,
                                initial_state_mean=initial_state_mean)
    x_ev_test_ = x_ev_test[0]
    (filtered_state_means, filtered_state_covariances) = ukf.filter(x_ev_test_)
    import matplotlib.pyplot as plt
    plt.plot(filtered_state_means[:, 0])
    plt.plot(filtered_state_means[:, 1])
    plt.xlim([1000, 1400])
    plt.ylim([-0.25, 0.25])
    plt.show()
    # %%

    if edge_fudge is None:
        # this is because of the hilbert edges
        edge_fudge = int(np.floor(fs*0.01))

    y_ev_pred = [signal.lfilter(model['b'], model['a'], x_ev_test_) for x_ev_test_ in x_ev_test]

    phi_ev_pred = [np.angle(signal.hilbert(y_ev_pred_)) for y_ev_pred_ in y_ev_pred]

    # now we need to extend phi_ev_pred
    f_alpha_est = np.array(
        [np.median(np.diff(np.unwrap(phi_ev_pred_))) / (2 * np.pi * (1 / fs)) for phi_ev_pred_ in phi_ev_pred])
    if trial_wise_compensation:
        phi_compensate = ((gd + edge_fudge - ix_shift_input) / fs) * 2 * np.pi * f_alpha_est
        phi_ev_pred = [phi_ev_pred[ix] + phi_compensate[ix] for ix in range(len(phi_ev_pred))]
    else:
        # the usual case because trial wise compensation is noisy
        f_alpha_est = np.median(f_alpha_est)
        # we add edge_fudge because that we want to propagate our estimate forward a little bit extra
        # (since we are then going to roll the edge fudge length off the end of the signal)
        # we subtract ix_shift_input because if it is, e.g. -10 that means our input was shifted to the left by 10, but
        # we still want to predict at time 0. So we need to propagate the phase forward by an extra 10 (i.e. - -10)
        phi_compensate = ((gd + edge_fudge - ix_shift_input) / fs) * 2 * np.pi * f_alpha_est
        phi_ev_pred = [phi_ev_pred_ + phi_compensate for phi_ev_pred_ in  phi_ev_pred]


    phi_ev_pred = [np.roll(phi_ev_pred_, edge_fudge) for phi_ev_pred_ in phi_ev_pred]
    # TODO: should set the edge fudge length at the start of phi_ev_pred to nan and make sure that is dealt with upstream

    phi_pred = np.array([phi_ev_pred_[-1] for phi_ev_pred_ in phi_ev_pred])

    return phi_pred, phi_ev_pred
