import numpy as np
from scipy import signal
import pickle
import matplotlib.pyplot as plt


def nextpow2(x):
    return 1 if x == 0 else 2**(x - 1).bit_length()


def gen_arb_noise_from_welch(fx, fs, one_sided=True):
    """ Takes the output of signal.welch, as well as the original signal length to generate noise"""
    fx = np.power(fx * fs * len(fx), 0.5)
    noise = gen_arb_noise(fx, one_sided=one_sided)
    return noise


def pinknoise(N, f_pow=1):
    """ - """
    if N % 2 == 0:
        M = N + 1
    else:
        M = N
    f = np.arange(0, np.ceil(M/2))
    p = np.power(f, -f_pow)
    y = gen_arb_noise(p, one_sided=True)
    if not M == N:
        y = y[:-1]
    return y


def welch(s, fs, nperseg):
    sec = np.arange(0, nperseg)  # sec = [-50:50];
    event_ix = np.arange(0, len(s) - len(sec), nperseg)
    ix_mat = np.tile(np.array(event_ix).reshape(-1, 1), (1, len(sec)))  # ix_mat = ix;%repmat not needed
    sec_mat = np.tile(sec, (len(event_ix), 1))  # sec_mat = sec;%repmat not needed
    ixsec_mat = ix_mat + sec_mat  # ixsec_mat = ix_mat + sec_mat;
    s0 = s[ixsec_mat.transpose()]  # how to do this without looping?

    # can do some rejection here

    h = np.hamming(nperseg)
    h = h / np.linalg.norm(h)
    s0_h = np.multiply(s0, np.tile(h.reshape(-1, 1), (1, np.shape(s0)[1])))
    n = nextpow2(nperseg)
    f_spec = np.arange(0, n) * (fs / n)
    a_spec = np.fft.fft(s0_h, n, axis=0)
    s_spec = np.power(np.abs(a_spec), 2)
    s_spec = s_spec / (fs / 2)
    s_spec = s_spec[0:int(n / 2 + 1), :]
    f_spec = f_spec[0:int(n / 2 + 1)]
    s_spec = np.mean(s_spec, axis=1)

    return f_spec, s_spec


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    return result[int(result.size / 2):]


def get_f_alpha(f, s_spec, fooof_results):
    if f[0] == 0:
        f = f[1:]
        s_spec = s_spec[1:]

    log_oof = fooof_results.background_params[0] - np.log10(np.power(f, fooof_results.background_params[1]))
    oof = np.power(10, log_oof)

    oof_bands = s_spec - oof
    oof_alpha = oof_bands.copy()
    case_alpha = np.any((f < 8., f > 13.), axis=0)
    oof_alpha[case_alpha] = np.NaN
    ix_f_alpha = np.nanargmax(oof_alpha)
    f_alpha = f[ix_f_alpha]

    return f_alpha


def run_fooof(f, s_spec, f_fooof, overwrite=False, fooof_figure=True):
    if not (f_fooof.exists()) or overwrite:
        from fooof import FOOOF
        # fig_fooof = plt.figure(100)
        # This breaks tensorflow... (but works)
        peak_width_limits = [0.25, 8.0]
        freq_range = [2.5, 35]
        fm = FOOOF(peak_width_limits=peak_width_limits)
        fm.report(f, s_spec, freq_range)
        fooof_results = fm.get_results()

        with open(str(f_fooof), 'wb') as handle:
            pickle.dump(fooof_results, handle)

        # should save the figure
        if not fooof_figure:
            plt.close()
        else:
            plt.savefig(str(f_fooof).replace('.dat', '.png'))

        print('Generate FOOOF result.')
    else:
        with open(str(f_fooof), 'rb') as handle:
            fooof_results = pickle.load(handle)

    return fooof_results


def gen_arb_noise(fx, one_sided=False):
    """Takes the output of an FFT and generates noise with the same spectrum"""

    fx = fx + 0 * 1j
    if one_sided:
        fx = np.concatenate((fx[1:], np.flip(fx, 0)))

    N = len(fx)

    Np = int(np.floor((N - 1) / 2))

    phases = 2 * np.pi * np.random.rand(Np)
    c_phases = np.cos(phases) + 1j * np.sin(phases)
    fx[1:Np + 1] = fx[1:Np + 1] * c_phases
    fx[-1:-1 - Np:-1] = np.conjugate(fx[1:Np + 1])

    noise = np.real(np.fft.ifft(fx))

    return noise


def bode(b, fs):
    w, h = signal.freqz(b)
    fig = plt.gcf()
    plt.title('Digital filter frequency response')
    ax1 = fig.add_subplot(111)
    f = w * (1 / np.pi) * (fs / 2)

    plt.plot(f, 20 * np.log10(abs(h)), 'b')
    plt.ylabel('Amplitude [dB]', color='b')
    plt.xlabel('Frequency [Hz]')

    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    plt.plot(f, angles, 'g')
    plt.ylabel('Angle (radians)', color='g')
    plt.grid()
    plt.axis('tight')
    plt.show()


def circ_mean(alpha, w=None, axis=0):
    if w is None:
        w = np.ones(np.shape(alpha))

    r = np.sum(w * np.exp(1j * alpha), axis=axis)

    mu = np.angle(r)

    ul = None
    ll = None

    return mu


def gen_arb_noise_from_foof(fooof_results, N, fs):
    # fs_high_res = int(np.ceil(len(s_squeeze) / 2))
    # f_high_res = np.linspace(0, fs / 2, fs_high_res)

    if N % 2 == 0:
        M = N + 1
    else:
        M = N
    MO2 = np.ceil(M / 2)

    f_high_res = np.arange(0, MO2) * (fs / (2 * MO2))

    log_oof_high_res = fooof_results.background_params[0] - np.log10(np.power(f_high_res, fooof_results.background_params[1]))
    oof_high_res = np.power(10, log_oof_high_res)
    synth_noise = gen_arb_noise_from_welch(oof_high_res, fs)
    if not M == N:
        synth_noise = synth_noise[:-1]

    assert abs(len(synth_noise) - N) < 1, 'Synth noise length is wrong'
    return synth_noise

    # this is ugly but not very important (sometimes synth_noise is 1 too short)
    # assert abs(len(synth_noise) - len(s_squeeze)) < 3, 'Synth noise length is very wrong'
    # if len(synth_noise)>len(s_squeeze):
    #     synth_noise = synth_noise[:len(s_squeeze)]
    # while len(synth_noise)<len(s_squeeze):
    #     synth_noise = np.append(synth_noise, synth_noise[-1])
    # assert abs(len(synth_noise) - len(s_squeeze)) < 1, 'Synth noise length is not fixed'