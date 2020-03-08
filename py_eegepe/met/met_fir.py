from .met_shared import split_train_val as split_train_val
from .met_shared import TrainDefault as TrainDefault
import numpy as np
from scipy import signal
from pathlib import Path
import datetime
import pickle
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
        opt.es_min_delta = 1e-5
        opt.es_patience = 12
        opt.early_stopping = True
        opt.validation = 0.2
        opt.preproc = 'subsmooth'
        # this gets used in core.py, and attached to the model spec to be used in predict
        opt.compensation = 'f_alpha_from_phase'  # 'f_alpha','trial_wise'
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
            fir_settings = pickle.load(handle)
        model = fir_settings['model']

    if not(f_arch.exists()) or overwrite:
        def obj(x):
            model = firwin_to_model(x, fs)
            _, phi_mae_full = test(x_ev_train, h_ev_train, model, ix_shift_input=ix_shift_input)
            # print(phi_mae_full)
            return phi_mae_full

        f_alpha_approx = 10
        l_bound = [int(5), 4., 0.1]
        u_bound = [int(2 * (fs / (0.2 * f_alpha_approx)) + 1), 15., 10.]
        opt_f = dict()
        opt_f['scaling'] = [1., 10., 10.]
        opt_f['deltainit'] = 50
        opt_f['deltatol'] = 5e-4
        opt_f['redfactor'] = 1.5
        opt_f['bounds'] = np.array([l_bound, u_bound]).transpose()

        f_eval = np.Inf
        x0_iter = x0
        print('Starting FIR opt...')
        history = {'min_res': [], 'f_eval_iter': [], 'f_eval': [], 'x0': [], 'f_eval_val': [], 'improvement': []}
        for ix in range(opt_local.epochs + 1):
            min_res_iter = minimizeCompass(obj, x0=x0_iter, paired=False, errorcontrol=False, **opt_f)
            f_eval_iter = min_res_iter['fun']
            history['min_res'].append(min_res_iter)
            history['f_eval_iter'].append(f_eval_iter)
            history['f_eval'].append(f_eval)
            history['x0'].append(x0_iter)

            if opt_local.validation is not None:
                model_val = firwin_to_model(min_res_iter['x'], fs)
                _, phi_mae_full_val = test(x_ev_val, h_ev_val, model_val, ix_shift_input=ix_shift_input)
                history['f_eval_val'].append(phi_mae_full_val)
                # raise Exception("validation is not used in met_fir training currently")

                if ix > 0:
                    history['improvement'].append(history['f_eval_val'][-1] < (np.min(history['f_eval_val'][:-1]) - opt_local.es_min_delta))

                if ix >= opt_local.es_patience and opt_local.early_stopping:
                    not_improvement = np.invert(history['improvement'][-opt_local.es_patience:])
                    if np.all(not_improvement):
                        print('Early stopping.')
                        break

            if f_eval_iter < f_eval:
                f_eval = f_eval_iter
                min_res = min_res_iter
                x0 = x0_iter

            if (ix % 5 == 0) or (ix == opt_local.epochs):
                print('Iteration {} complete, MACE: {:0.4f}, best MACE: {:0.4f}'.format(ix, f_eval_iter, f_eval))
            x0_iter = x0_rand()

        model = firwin_to_model(min_res['x'], fs)

        fir_settings = {'model': model, 'opt_f': opt_f, 'weights_filter': min_res, 'history':history}

        with open(str(f_arch), 'wb') as handle:
            pickle.dump(fir_settings, handle)

    return model


def test(x_ev_test, h_ev_test, model, ix_shift_input=0, edge_fudge=None):
    # assert ix_shift_input==0, "ix_shift_input==0 is fully implemented, but untested"

    fs = model['fs']
    if edge_fudge is None:
        # this is because of the hilbert edges
        edge_fudge = int(np.round(fs*0.01))
    phi_pred, phi_ev_pred = predict(x_ev_test, model, ix_shift_input=ix_shift_input, edge_fudge=edge_fudge)

    phi_ev_test = [np.angle(h_ev_test_) for h_ev_test_ in h_ev_test]

    phi_error_full = [(phi_ev_test[ix][edge_fudge:] - phi_ev_pred[ix][edge_fudge:]) for ix in range(len(phi_ev_pred))]
    phi_error_full = [np.mean(np.abs(np.angle(np.exp(1j * phi_err_full_)))) for phi_err_full_ in phi_error_full]
    #  phi_mae_full is used by optimisation I think
    phi_mae_full = np.mean(phi_error_full)

    # extract only the event related bits
    phi = np.array([phi_ev_test_[-1] for phi_ev_test_ in phi_ev_test])
    phi_err = np.angle(np.exp(1j * (phi - phi_pred)))

    amp_err = np.nan

    return phi_err, phi_mae_full, amp_err, phi_pred


def predict(x_ev_test, model, ix_shift_input=0, edge_fudge=None):
    fs = model['fs']
    gd = model['gd']
    compensation = model['compensation']
    if edge_fudge is None:
        # this is because of the hilbert edges
        edge_fudge = int(np.floor(fs*0.01))

    y_ev_pred = [signal.lfilter(model['b'], model['a'], x_ev_test_) for x_ev_test_ in x_ev_test]

    phi_ev_pred = [np.angle(signal.hilbert(y_ev_pred_)) for y_ev_pred_ in y_ev_pred]

    # now we need to extend phi_ev_pred
    f_alpha_est = np.array(
        [np.median(np.diff(np.unwrap(phi_ev_pred_))) / (2 * np.pi * (1 / fs)) for phi_ev_pred_ in phi_ev_pred])
    if compensation == 'trial_wise':
        phi_compensate = ((gd + edge_fudge - ix_shift_input) / fs) * 2 * np.pi * f_alpha_est
        phi_ev_pred = [phi_ev_pred[ix] + phi_compensate[ix] for ix in range(len(phi_ev_pred))]
    elif compensation == 'f_alpha_from_phase':
        # the usual case because trial wise compensation is noisy
        f_alpha_est = np.median(f_alpha_est)
        # we add edge_fudge because that we want to propagate our estimate forward a little bit extra
        # (since we are then going to roll the edge fudge length off the end of the signal)
        # we subtract ix_shift_input because if it is, e.g. -10 that means our input was shifted to the left by 10, but
        # we still want to predict at time 0. So we need to propagate the phase forward by an extra 10 (i.e. - -10)
        phi_compensate = ((gd + edge_fudge - ix_shift_input) / fs) * 2 * np.pi * f_alpha_est
        phi_ev_pred = [phi_ev_pred_ + phi_compensate for phi_ev_pred_ in phi_ev_pred]
    elif compensation == 'f_alpha':
        # the usual case because trial wise compensation is noisy
        f_alpha_est = model['f_alpha']
        # same comments are for f_alpha_from_phase
        phi_compensate = ((gd + edge_fudge - ix_shift_input) / fs) * 2 * np.pi * f_alpha_est
        phi_ev_pred = [phi_ev_pred_ + phi_compensate for phi_ev_pred_ in phi_ev_pred]
    else:
        raise Exception('compensation not specified')

    phi_ev_pred = [np.roll(phi_ev_pred_, edge_fudge) for phi_ev_pred_ in phi_ev_pred]
    # TODO: should set the edge fudge length at the start of phi_ev_pred to nan and make sure that is dealt with upstream

    phi_pred = np.array([phi_ev_pred_[-1] for phi_ev_pred_ in phi_ev_pred])

    return phi_pred, phi_ev_pred


def firwin_to_model(x, fs):
    nyq = fs / 2
    order = int(np.round([x[0]]))
    f_low = x[1]
    f_range = x[2]
    b = signal.firwin(order, [f_low / nyq, (f_low + f_range) / nyq], pass_zero=False)
    a = 1
    gd = int(np.round((x[0] - 1) / 2))
    model = {'b': b, 'a': a, 'fs': fs, 'gd': gd, 'order': order, 'f_low': f_low, 'f_range': f_range}
    return model


def iirwin_to_model(x, fs):
    # not tested at all
    # warnings.filterwarnings("ignore") # would need to import watnings (also this is for all I think)
    nyq = fs / 2
    order = int(np.round([x[0]]))
    f_low = x[1]
    f_range = x[2]
    b, a = signal.iirfilter(order, [f_low / nyq, (f_low + f_range) / nyq], btype='band', analog = False, ftype = 'butter')
    w, gd = signal.group_delay((b, a))
    gd = gd[np.argmin(np.abs(nyq*(w/np.pi)-(f_low + f_range/2)))]
    gd = int(np.round(gd))
    model = {'b': b, 'a': a, 'fs': fs, 'gd': gd, 'order': order, 'f_low': f_low, 'f_range': f_range}
    return model


def b_to_model(x, order, fs):
    nyq = fs / 2
    f_low = x[1]
    f_range = x[2]
    gd = int(np.round((order - 1) / 2))
    b = x
    model = {'b': b, 'a': 1, 'fs':fs, 'gd':gd, 'order':order, 'f_low': f_low, 'f_range':f_range}
    return model


def train_direct(x_ev_train, h_ev_train, fs, x0, f_arch=None, overwrite=False, epochs=5):
    """
    fff
    """
    assert False, "have not updated to use ix_shift_input"
    x0_rand = lambda: [125, 9.5 + 2 * np.random.randn(1)[0], np.abs(2 * np.random.randn(1)[0])]
    if x0 is None:
        x0 = x0_rand()

    model_base = firwin_to_model(x0, fs)

    if isinstance(f_arch, str):
        f_arch = Path(f_arch)

    if f_arch is None:
        f_arch = Path('fir_{}{}.dat'.format('arch', datetime.datetime.now().strftime("%Y%m%d%H%M%S")))

    if f_arch.exists() and not overwrite:
        with open(str(f_arch), 'rb') as handle:
            fir_settings = pickle.load(handle)

        model = fir_settings['model']

    if not(f_arch.exists()) or overwrite:
        def obj(x):
            model = b_to_model(x, model_base['order'], fs)
            _, phi_mae_full = test(x_ev_train, h_ev_train, model)
            return phi_mae_full

        opt_f = dict()
        opt_f['deltatol'] = 5e-4
        opt_f['redfactor'] = 1.5
        opt_f['deltainit'] = np.median(np.abs(model_base['b']))

        f_eval = np.Inf
        x0_iter = model_base['b']
        print('Starting FIR opt...')
        for ix in range(epochs + 1):
            min_res_ = minimizeCompass(obj, x0=x0_iter, paired=False, errorcontrol=False, **opt_f)

            f_eval_ = min_res_['fun']
            if f_eval_ < f_eval:
                f_eval = f_eval_
                min_res = min_res_
                x0 = x0_iter
            x0_iter = x0_rand()
            if (ix % 2 == 0) or (ix == epochs):
                print('Iteration {} complete, MACE: {:0.3f}, best MACE: {:0.3f}'.format(ix, f_eval_, f_eval))

        model = b_to_model(min_res['x'], fs)

        fir_settings = {'model': model, 'opt_f': opt_f, 'weights_filter': min_res}

        with open(str(f_arch), 'wb') as handle:
            pickle.dump(fir_settings, handle)

    return model
