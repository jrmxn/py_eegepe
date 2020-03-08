from .met_net_arch import get_arch as get_arch
from .met_shared import split_train_val as split_train_val
from .met_shared import TrainDefault as TrainDefault

import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
from tensorflow.python.keras import callbacks
from os import sep


def defaults(opt):
    """

    :param opt:
    :return:
    """
    if opt is None:
        opt = TrainDefault()
        opt.epochs = 5000
        opt.es_min_delta = 1e-5
        opt.es_patience = 12
        opt.early_stopping = True
        opt.validation = 0.2
        opt.preproc = 'diff'
        opt.tensorboard = False
    elif isinstance(opt, dict):
        assert False, 'need to add this option?'
    return opt


def train(x_ev, h_ev, fs, arch='net_gru_000', f_arch=None, resume=False, overwrite=False,
          opt=None):
    """

    :param x_ev:
    :param h_ev:
    :param fs:
    :param arch:
    :param f_arch:
    :param resume:
    :param overwrite:
    :param opt:
    :return:
    """
    # frame = inspect.currentframe() # this is how you might go about saving inputs...
    # args, _, _, values = inspect.getargvalues(frame)
    if f_arch is None: f_arch = Path('net_{}{}.dat'.format(arch, datetime.now().strftime("%Y%m%d%H%M%S")))
    if isinstance(f_arch, str): f_arch = Path(f_arch)
    opt_local = defaults(opt)

    assert opt_local.validation or not opt_local.early_stopping, "Can't early stop without validation."

    if opt_local.validation is None:
        x_ev_train = x_ev
        h_ev_train = h_ev
    else:
        x_ev_train, h_ev_train, x_ev_val, h_ev_val = split_train_val(x_ev, h_ev, opt_local.validation)

    x_ev_train_shaped = list_to_network_list(x_ev_train)
    h_ev_train_shaped = list_to_network_list(h_ev_train)
    y_ev_train_shaped = hilbert_to_net_output(h_ev_train_shaped)

    model = get_arch(arch)

    if model.input_shape[1] is not None:
        # This is if instead of using gru/lstm we want to use an image like input
        n_sec = model.input_shape[1]
        x_ev_train_shaped, y_ev_train_shaped = reshape_image_based(x_ev_train_shaped, y_ev_train_shaped, n_sec)

    if isinstance(f_arch, str):
        f_arch = Path(f_arch)

    if f_arch is None:
        f_arch = Path('net_{}{}.dat'.format(arch, datetime.now().strftime("%Y%m%d%H%M%S")))

    def train_generator():
        ix = 0
        # added this so if our dataset spans subjects we shuffle the sets
        vec_ix = np.random.permutation(len(x_ev_train_shaped))
        while True:
            xx = x_ev_train_shaped[vec_ix[ix]]
            yy = y_ev_train_shaped[vec_ix[ix]]
            ix += 1
            ix = ix % len(x_ev_train_shaped)
            yield xx, yy

    if f_arch.exists() and not overwrite:
        # print('Loading {}'.format(f_arch))
        with open(str(f_arch), 'rb') as handle:
            net_settings = pickle.load(handle)

        weights = net_settings['weights']

        if 'finished' in net_settings:
            resume_overwrite = not(net_settings['finished'])
        else:
            resume_overwrite = False
        resume = resume or resume_overwrite

        # you have to evaluate it before setting for some reason (maybe an older bug):
        g = model.predict(np.zeros(np.shape(x_ev_train_shaped[0])))
        model.set_weights(weights=weights)

    finished = True
    if not(f_arch.exists()) or resume or overwrite:
        print('Generating {}'.format(f_arch))
        if opt_local.epochs < 5:
            print('Only {} epochs selected... testing?'.format(opt_local.epochs))
        steps_per_epoch_train = int(np.round(len(x_ev_train)))

        if opt.tensorboard:
            logdir = Path(sep.join(f_arch.parts[:-3])) / 'logs' / 'scalars' / f_arch.parts[-1].replace('.dat', '')
            tensorboard_callback = callbacks.TensorBoard(log_dir=str(logdir))

        if opt_local.validation is None:
            model.fit_generator(train_generator(), steps_per_epoch=steps_per_epoch_train, epochs=opt_local.epochs, verbose=2)
        else:
            x_ev_val_shaped = list_to_network_list(x_ev_val)
            h_ev_val_shaped = list_to_network_list(h_ev_val)
            y_ev_val_shaped = hilbert_to_net_output(h_ev_val_shaped)

            if model.input_shape[1] is not None:
                x_ev_val_shaped, y_ev_val_shaped = reshape_image_based(x_ev_val_shaped, y_ev_val_shaped, n_sec)

            def val_generator():
                ix = 0
                vec_ix = np.random.permutation(len(x_ev_val_shaped))
                while True:
                    xx = x_ev_val_shaped[vec_ix[ix]]
                    yy = y_ev_val_shaped[vec_ix[ix]]
                    ix += 1
                    ix = ix % len(x_ev_val_shaped)
                    yield xx, yy

            steps_per_epoch_val = int(np.round(len(x_ev_val_shaped)))
            if opt_local.early_stopping:
                # 'an absolute change of less than min_delta, will count as no improvement'
                callbacks_ = [
                    callbacks.EarlyStopping(monitor='val_loss', min_delta=opt_local.es_min_delta,
                                            patience=opt_local.es_patience, verbose=0, mode='min',
                                            restore_best_weights=True)]

                callbacks_.append(tensorboard_callback) if opt.tensorboard else None
            else:
                callbacks_ = None
                callbacks_.append(tensorboard_callback) if opt.tensorboard else None

            m = model.fit_generator(train_generator(), steps_per_epoch=steps_per_epoch_train,
                                    epochs=opt_local.epochs, verbose=2,
                                    validation_data=val_generator(),
                                    validation_steps=steps_per_epoch_val,
                                    callbacks=callbacks_)

            # net_settings['finished']
            if np.max(m.epoch) + 1 >= opt_local.epochs:
                finished = False

        weights = model.get_weights()

        if resume:
            epochs = net_settings['epochs'] + opt_local.epochs  # don't like the structure here
        else:
            epochs = opt_local.epochs

        net_settings = {'weights': weights, 'arch': arch, 'epochs': epochs, 'finished': finished}

        with open(str(f_arch), 'wb') as handle:
            pickle.dump(net_settings, handle)

    return model


def test(x_ev, h_ev, model, ix_shift_input=0):
    """

    :param x_ev:
    :param h_ev:
    :param model:
    :param ix_shift_input:
    :return:
    """
    # assert ix_shift_input == 0, "ix_shift_input==0 should be implemented already, but looking at code here not 100% sure"

    # if not(model.input_shape == (None, None, 1)):
    #     # This is if instead of using gru/lstm we want to use an image like input
    # think not needed -
    #     assert False, "have not tested this - weird that it takes x_ev instead of x_ev_shaped"
    #     n_sec = model.input_shape[1]
    #     x_ev, _ = reshape_image_based(x_ev, None, n_sec, last_only=False)

    phi_pred, amp_pred = predict(x_ev, model)

    h_ev_test_shaped = list_to_network_list(h_ev)
    y_ev_test_shaped = hilbert_to_net_output(h_ev_test_shaped)

    phi = []
    amp = []
    for ix in range(len(x_ev)):
        y = y_ev_test_shaped[ix]
        phi_ = np.arctan2(y[0][-1][1], y[0][-1][0])
        amp_ = np.sqrt(np.power(y[0][-1][1], 2) + np.power(y[0][-1][0], 2))
        phi.append(phi_)
        amp.append(amp_)
    phi = np.array(phi)
    amp = np.array(amp)

    phi_err = np.angle(np.exp(1j * (phi - phi_pred)))
    amp_err = np.sqrt(np.power(amp, 2) + np.power(amp_pred, 2))

    return phi_err, None, amp_err, phi_pred


def predict_samples(x_ev, h_ev, model, fs, ix_shift_input=0):
    """

    :param x_ev:
    :param h_ev:
    :param model:
    :param ix_shift_input:
    :return:
    """
    n_sec = None
    x_ev_test_shaped = list_to_network_list(x_ev)
    if model.input_shape[1] is not None:
        # This is if instead of using gru/lstm we want to use an image like input
        n_sec = model.input_shape[1]
        x_ev_test_shaped, _ = reshape_image_based(x_ev_test_shaped, None, n_sec, last_only=False)

    phi_pred = []
    t = []
    for ix in range(len(x_ev)):
        x = x_ev_test_shaped[ix]
        y_pred = model.predict(x)
        if n_sec is None:
            #     then we are doing a time series thing
            phi_pred_ = np.arctan2(y_pred[0, :, 1], y_pred[0, :, 0])
        else:
            phi_pred_ = np.arctan2(y_pred[:, 1], y_pred[:, 0])

        t_ = np.linspace(0, (1/fs)*(len(phi_pred_)-1), len(phi_pred_))
        t.append(t_ - t_[-1])
        phi_pred.append(phi_pred_)
    return phi_pred, t


def predict(x_ev, model):
    """

    :param x_ev:
    :param model:
    :return:
    """

    n_sec = None
    x_ev_test_shaped = list_to_network_list(x_ev)
    if model.input_shape[1] is not None:
        # This is if instead of using gru/lstm we want to use an image like input
        n_sec = model.input_shape[1]
        x_ev_test_shaped, _ = reshape_image_based(x_ev_test_shaped, None, n_sec, last_only=False)

    phi_pred = []
    amp_pred = []
    for ix in range(len(x_ev)):
        x = x_ev_test_shaped[ix]
        y_pred = model.predict(x)
        if n_sec is None:
            #     then we are doing a time series thing
            phi_pred_ = np.arctan2(y_pred[0][-1][1], y_pred[0][-1][0])
            amp_pred_ = np.sqrt(np.power(y_pred[0][-1][1], 2) + np.power(y_pred[0][-1][0], 2))
        else:
            phi_pred_ = np.arctan2(y_pred[-1][1], y_pred[-1][0])
            amp_pred_ = np.sqrt(np.power(y_pred[-1][1], 2) + np.power(y_pred[-1][0], 2))
        phi_pred.append(phi_pred_)
        amp_pred.append(amp_pred_)
    phi_pred = np.array(phi_pred)
    amp_pred = np.array(amp_pred)

    return phi_pred, amp_pred


def list_to_network_list(signal_in):
    """
    Reshape to be the right way round to train/test on for keras
    :param signal_in:
    :return:
    """
    signal_out = [signal_in[ix].reshape(1, -1, 1) for ix in range(len(signal_in))]
    return signal_out


def hilbert_to_net_output(sig_in):
    """

    :param sig_in:
    :return:
    """
    sig_out = [np.concatenate((np.real(sig), np.imag(sig)), axis=2) for sig in sig_in]
    return sig_out


def reshape_image_based(x_ev_shaped_, y_ev_shaped_, n_sec, last_only=False):
    """

    :param x_ev_shaped_:
    :param y_ev_shaped_:
    :param n_sec:
    :param last_only:
    :return:
    """
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
