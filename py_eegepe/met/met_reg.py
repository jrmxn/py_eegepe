# from .met_net_arch import get_arch as get_arch
# from .met_shared import split_train_val as split_train_val
from .met_shared import TrainDefault as TrainDefault
from sklearn import linear_model

import numpy as np
from pathlib import Path
import datetime
import pickle


def defaults(opt):
    if opt is None:
        opt = TrainDefault()
        # opt.epochs = 1000
        # opt.es_min_delta = 1e-5
        # opt.es_patience = 5
        # opt.early_stopping = True
        # opt.validation = 0.2
        opt.n_sec = 250
        opt.preproc = 'diff'
    elif isinstance(opt, dict):
        assert False, 'need to add this option?'
    return opt


def train(x_ev, h_ev, fs, f_arch=None, resume=False, overwrite=False,
          opt=None):
    """
    fff
    """
    # frame = inspect.currentframe() # this is how you might go about saving inputs...
    # args, _, _, values = inspect.getargvalues(frame)
    if f_arch is None: f_arch = Path('net_{}{}.dat'.format('arch', datetime.datetime.now().strftime("%Y%m%d%H%M%S")))
    if isinstance(f_arch, str): f_arch = Path(f_arch)
    opt_local = defaults(opt)

    # assert opt_local.validation or not opt_local.early_stopping, "Can't early stop without validation."

    x_ev_train = x_ev
    h_ev_train = h_ev

    x_ev_train_shaped = list_to_network_list(x_ev_train)
    h_ev_train_shaped = list_to_network_list(h_ev_train)
    y_ev_train_shaped = hilbert_to_net_output(h_ev_train_shaped)

    model = linear_model.LinearRegression()
    model.fit_intercept = False
    model.set_params()

    # if model.input_shape[1] is not None:
        # This is if instead of using gru/lstm we want to use an image like input
    n_sec = opt_local.n_sec
    x_ev_train_shaped, y_ev_train_shaped = reshape_image_based(x_ev_train_shaped, y_ev_train_shaped, n_sec)

    if isinstance(f_arch, str):
        f_arch = Path(f_arch)

    if f_arch is None:
        f_arch = Path('reg_{}{}.dat'.format('arch', datetime.datetime.now().strftime("%Y%m%d%H%M%S")))

    if f_arch.exists() and not overwrite:
        # print('Loading {}'.format(f_arch))
        with open(str(f_arch), 'rb') as handle:
            reg_settings = pickle.load(handle)

        model.coef_ = reg_settings['weights']
        model.intercept_ = 0.0
        model.set_params()  # no idea if this does anything

    if not(f_arch.exists()) or resume or overwrite:
        print('Generating {}'.format(f_arch))

        # x_ev_val_shaped = list_to_network_list(x_ev_val)
        # h_ev_val_shaped = list_to_network_list(h_ev_val)
        # y_ev_val_shaped = hilbert_to_net_output(h_ev_val_shaped)


        # x_ev_val_shaped, y_ev_val_shaped = reshape_image_based(x_ev_val_shaped, y_ev_val_shaped, n_sec)

        # steps_per_epoch_val = int(np.round(len(x_ev_val_shaped)))
        # if opt_local.early_stopping:
        #     # 'an absolute change of less than min_delta, will count as no improvement'
        #     callbacks_ = [callbacks.EarlyStopping(monitor='val_loss', min_delta=opt_local.es_min_delta, patience=opt_local.es_patience, verbose=0, mode='min')]
        # else:
        #     callbacks_ = None

        x_ev_train_shaped = np.concatenate(x_ev_train_shaped, axis=0)
        y_ev_train_shaped = np.concatenate(y_ev_train_shaped, axis=0)
        model.fit(x_ev_train_shaped, y_ev_train_shaped)
        # m = model.fit_generator(train_generator(), steps_per_epoch=steps_per_epoch_train,
        #                     epochs=opt_local.epochs, verbose=2,
        #                     validation_data=val_generator(),
        #                     validation_steps=steps_per_epoch_val,
        #                     callbacks=callbacks_)

        # net_settings['finished']
        # if np.max(m.epoch) + 1 >= opt_local.epochs:
        #     finished = False

        weights = model.coef_

        # if resume:
        #     epochs = net_settings['epochs'] + opt_local.epochs  # don't like the structure here
        # else:
        #     epochs = opt_local.epochs

        net_settings = {'weights': weights}

        with open(str(f_arch), 'wb') as handle:
            pickle.dump(net_settings, handle)

    return model


def test(x_ev, h_ev, model, ix_shift_input=0):
    """
    d
    """
    phi_pred, amp_pred = predict(x_ev, model)

    h_ev_test_shaped = list_to_network_list(h_ev)
    y_ev_test_shaped = hilbert_to_net_output(h_ev_test_shaped)

    phi = []
    amp = []
    for ix in range(len(x_ev)):
        y = y_ev_test_shaped[ix][0]
        phi_ = np.arctan2(y[-1, 1], y[-1, 0])
        amp_ = np.sqrt(np.power(y[-1, 1], 2) + np.power(y[-1, 0], 2))
        phi.append(phi_)
        amp.append(amp_)
    phi = np.array(phi)
    amp = np.array(amp)

    phi_err = np.angle(np.exp(1j * (phi - phi_pred)))
    amp_err = np.sqrt(np.power(amp, 2) + np.power(amp_pred, 2))

    return phi_err, None, amp_err


def predict(x_ev, model):
    """
    d
    """

    n_sec = np.shape(model.coef_)[1]
    x_ev_test_shaped = list_to_network_list(x_ev)
    x_ev_test_shaped, _ = reshape_image_based(x_ev_test_shaped, None, n_sec, last_only=False)

    phi_pred = []
    amp_pred = []
    for ix in range(len(x_ev)):
        x = x_ev_test_shaped[ix]
        y_pred = model.predict(x)
        phi_pred_ = np.arctan2(y_pred[-1, 1], y_pred[-1, 0])
        amp_pred_ = np.sqrt(np.power(y_pred[-1, 1], 2) + np.power(y_pred[-1, 0], 2))
        phi_pred.append(phi_pred_)
        amp_pred.append(amp_pred_)
    phi_pred = np.array(phi_pred)
    amp_pred = np.array(amp_pred)

    phi_pred = np.array(phi_pred)
    amp_pred = np.array(amp_pred)

    return phi_pred, amp_pred


def list_to_network_list(signal_in):
    """Reshape to be the right way round to train/test on for keras"""
    signal_out = [signal_in[ix].reshape(1, -1, 1) for ix in range(len(signal_in))]
    return signal_out


def hilbert_to_net_output(sig_in):
    sig_out = [np.concatenate((np.real(sig), np.imag(sig)), axis=2) for sig in sig_in]
    return sig_out


def reshape_image_based(x_ev_shaped_, y_ev_shaped_, n_sec, last_only=False):
    for ix in range(len(x_ev_shaped_)):
        x = x_ev_shaped_[ix][0, :, 0]

        sec = np.arange(-n_sec + 1, 1)
        ev = np.arange(0, len(x))
        ix_mat = np.tile(ev.reshape(-1, 1), (1, n_sec))  # ix_mat = ix;%repmat not needed
        sec_mat = np.tile(sec, (len(x), 1))  # sec_mat = sec;%repmat not needed
        ixsec_mat = ix_mat + sec_mat
        rm_mat = np.any(np.stack((ixsec_mat < 0, ixsec_mat >= len(x)), 2), 2)
        ixsec_mat[rm_mat] = 0
        x_mat = x[ixsec_mat]

        # remove rows where indeces don't match -
        rm_rows = np.where(np.any(x_mat == x[0], 1))
        x_mat = np.delete(x_mat, rm_rows, 0)

        if last_only:
            x_ev_shaped_[ix] = x_mat[-1, :].reshape(1, -1)
        else:
            x_ev_shaped_[ix] = x_mat

        if y_ev_shaped_ is not None:
            y = np.squeeze(y_ev_shaped_[ix])
            y = np.delete(y, rm_rows, 0)
            if last_only:
                y_ev_shaped_[ix] = y[-1, :].reshape(1, -1)
            else:
                y_ev_shaped_[ix] = y

    return x_ev_shaped_, y_ev_shaped_
