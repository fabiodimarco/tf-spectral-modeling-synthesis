import tensorflow as tf
import numpy as np


def logb(x, base=2.0):
    return tf.math.log(x) / tf.math.log(base)


def next_power_of_2(x):
    x = tf.cast(x, tf.float32)
    return 2.0 ** tf.math.ceil(logb(x, 2.0))


def db_to_lin(x):
    return tf.math.exp(tf.math.log(10.0) / 20.0 * x)


def lin_to_db(x):
    eps = 1e-9  # ≈ 180 dB, db_to_lin(-180.0)
    safe_x = tf.where(x <= eps, eps, x)
    return 20.0 * logb(safe_x, 10.0)


def midi_to_hz(notes):
    return 440.0 * (2.0 ** ((notes - 69.0) / 12.0))


def hz_to_midi(freq):
    notes = 12.0 * (logb(freq, 2.0) - logb(440.0, 2.0)) + 69.0
    # Map 0 Hz to MIDI 0 (Replace -inf MIDI with 0.)
    notes = tf.where(tf.less_equal(freq, 0.0), 0.0, notes)

    return notes


def slice_axis(input, axis, start=None, stop=None, step=None):
    nd = tf.rank(input)
    axis = (nd + axis) if axis < 0 else axis

    if start is None:
        start = 0
    elif start < 0:
        start = tf.shape(input)[axis] + start

    if stop is None:
        stop = tf.shape(input)[axis]
    elif stop < 0:
        stop = tf.shape(input)[axis] + stop

    if step is None:
        step = 1

    begin = tf.zeros(nd, dtype=tf.int32)
    end = tf.shape(input)
    strides = tf.ones(nd, dtype=tf.int32)

    begin = tf.tensor_scatter_nd_update(begin, [[axis]], [start])
    end = tf.tensor_scatter_nd_update(end, [[axis]], [stop])
    strides = tf.tensor_scatter_nd_update(strides, [[axis]], [step])
    slice = tf.strided_slice(input, begin, end, strides)

    return slice


def phase_diff(p0, p1, mod=2.0*np.pi):
    dd = p0 - p1
    ddmod = tf.math.floormod(dd + 0.5 * mod, mod) - 0.5 * mod
    ddmod = tf.where((ddmod == -0.5 * mod) & (dd > 0.0), 0.5 * mod, ddmod)

    return ddmod


def phase_unwrap(p, discont=np.pi, axis=-1):
    p0 = slice_axis(p, axis, start=1)  # [1:]
    p1 = slice_axis(p, axis, stop=-1)  # [:-1]
    p2 = slice_axis(p, axis, stop=1)  # [:1]

    dd = p0 - p1
    ddmod = tf.math.floormod(dd + np.pi, 2 * np.pi) - np.pi
    ddmod = tf.where((ddmod == -np.pi) & (dd > 0), np.pi, ddmod)
    ph_correct = ddmod - dd
    ph_correct = tf.where(tf.math.abs(dd) < discont, 0.0, ph_correct)

    up = p0 + tf.math.cumsum(ph_correct, axis=axis)
    up = tf.concat([p2, up], axis=axis)

    return up


def savitzky_golay(x, window_size, order, axis=-1, mode='mirror', cval=0.0):
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number.")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order.")

    order_range = range(order + 1)
    half_window = (window_size - 1) // 2

    # compute slices
    x0 = slice_axis(x, axis, stop=1)
    x1 = slice_axis(x, axis, start=-1)
    x2 = slice_axis(x, axis, start=half_window, stop=0, step=-1)
    x3 = slice_axis(x, axis, start=-1, stop=-half_window - 1, step=-1)

    # compute input permutation and inverse permutation
    nd = tf.rank(x)
    perm = tf.range(nd)
    perm = tf.tensor_scatter_nd_update(
        perm, [[axis], [nd - 1]], [perm[-1], perm[axis]])
    inv_perm = tf.math.invert_permutation(perm)

    # precompute coefficients
    b = tf.constant([[k ** i for i in order_range]
                     for k in range(-half_window, half_window + 1)],
                    dtype=tf.float32)
    m = tf.linalg.pinv(b)
    m = m[0, :]
    m = m[:, tf.newaxis, tf.newaxis]

    # mode       |   Ext   |         Input          |   Ext
    # -----------+---------+------------------------+---------
    # 'mirror'   | 4  3  2 | 1  2  3  4  5  6  7  8 | 7  6  5
    # 'extent'   |-2 -1  0 | 1  2  3  4  5  6  7  8 | 9 10 11
    # 'nearest'  | 1  1  1 | 1  2  3  4  5  6  7  8 | 8  8  8
    # 'constant' | 0  0  0 | 1  2  3  4  5  6  7  8 | 0  0  0

    if mode == 'mirror':
        firstvals = x2
        lastvals = x3
    elif mode == 'extent':
        firstvals = 2 * x0 - x2
        lastvals = 2 * x1 - x3
    elif mode == 'nearest':
        firstvals = tf.ones_like(x2) * x0
        lastvals = tf.ones_like(x3) * x1
    elif mode == 'constant':
        firstvals = tf.ones_like(x2) * cval
        lastvals = tf.ones_like(x3) * cval
    else:
        raise TypeError("selected mode is invalid.")

    y = tf.concat([firstvals, x, lastvals], axis=axis)
    y = tf.transpose(y, perm=perm)
    y = tf.expand_dims(y, axis=-1)
    y = tf.nn.conv1d(y, m, stride=1, padding='VALID')
    y = tf.squeeze(y, axis=-1)
    y = tf.transpose(y, perm=inv_perm)

    return y


@tf.function
def mod_cumsum(x, mod, axis=-1):
    size = tf.shape(x)[axis]

    y = tf.TensorArray(tf.float32, size=size, dynamic_size=False)

    x = x % mod
    s = tf.gather(x, 0, axis=axis) * 0.0
    for i in tf.range(size):
        v = tf.gather(x, i, axis=axis)
        s = (s + v) % mod
        y = y.write(i, s)

    y = y.stack()

    # compute input permutation and inverse permutation
    nd = tf.rank(x)
    perm = tf.range(nd)
    perm = tf.tensor_scatter_nd_update(
        perm, [[axis], [0]], [perm[0], perm[axis]])

    y = tf.transpose(y, perm=perm)

    return y


def non_zero_mean(x, axis=-1):
    x_sum = tf.math.reduce_sum(x, axis=axis, keepdims=True)
    x_count = tf.math.count_nonzero(
        x > 0.0, axis=axis, keepdims=True, dtype=tf.float32)
    x_mean = tf.where(x_count > 0.0, x_sum / x_count, x_sum)

    return x_mean


def generalized_cos_window(length, name):
    if name == 'rect':
        c = [1.0]
    elif name == 'hann' or name == 'hanning':
        c = [0.5, 0.5]
    elif name == 'hamming':
        c = [25.0 / 46.0, 21.0 / 46.0]
    elif name == 'blackman':
        c = [0.42, 0.5, 0.08]
    elif name == 'blackman_harris':
        c = [0.35875, 0.48829, 0.14128, 0.01168]
    elif name == 'blackman_nuttall':
        c = [0.3635819, 0.4891775, 0.1365995, 0.0106411]
    elif name == 'flattop':
        c = [0.21557895, 0.41663158, 0.277263158, 0.083578947, 0.006947368]
    else:
        c = [0.0]

    c = tf.constant(c, dtype=tf.float64)
    t = tf.range(0, length, dtype=tf.float64)
    t = t / (tf.cast(length, dtype=tf.float64) - 1.0) - 0.5
    f = tf.range(0, len(c), dtype=tf.float64)
    c_diff = c * f

    c = c[tf.newaxis, :]
    c_diff = c_diff[tf.newaxis, :]
    t = t[tf.newaxis, :]
    f = f[:, tf.newaxis]

    omega = 2.0 * np.pi * f * t
    window = tf.matmul(c, tf.math.cos(omega))
    window_diff = tf.matmul(c_diff, -tf.math.sin(omega))

    window = tf.cast(tf.squeeze(window), dtype=tf.float32)
    window_diff = tf.cast(tf.squeeze(window_diff), dtype=tf.float32)

    return window, window_diff


def midi_to_f0_estimate(note_number, samples, frame_step):
    frames = samples / frame_step
    frames = tf.cast(tf.math.ceil(frames), dtype=tf.int32)

    f0_estimate = midi_to_hz(note_number)

    f0_estimate = tf.reshape(f0_estimate, shape=(1, 1, 1))
    f0_estimate = tf.broadcast_to(f0_estimate, shape=(1, frames, 1))

    return f0_estimate


def get_harmonic_frequencies(f0, harmonics):
    harmonic_numbers = tf.range(1, harmonics + 1, dtype=tf.float32)
    harmonic_numbers = harmonic_numbers[tf.newaxis, tf.newaxis, :]

    h_freq = f0 * harmonic_numbers

    return h_freq


def get_number_harmonics(f0, sample_rate):
    period = sample_rate / f0
    harmonics = tf.cast(0.5 * period, dtype=tf.int32)

    return harmonics


def compute_stft(signals, frame_length, frame_step, window,
                 fft_length=None, normalize_window=False):
    if fft_length is None:
        fft_length = next_power_of_2(tf.cast(frame_length, dtype=tf.float32))

    fft_length = tf.cast(fft_length, dtype=tf.int32)

    # zero padding x to center first window at sample 0 and analyze last sample
    pad0 = tf.cast(frame_length // 2, dtype=tf.int32)
    pad1 = tf.cast(frame_length - pad0 - 1, dtype=tf.int32)

    # pad0 = tf.cast(0, dtype=tf.int32)
    # pad1 = tf.cast(frame_length - pad0 - 1, dtype=tf.int32)

    pad_signals = tf.pad(signals, paddings=[[0, 0], [pad0, pad1]])

    if normalize_window:
        window = 2.0 * window / tf.math.reduce_sum(window)

    # frame signals
    framed_signals = tf.signal.frame(pad_signals, frame_length, frame_step)
    framed_signals = framed_signals * window
    framed_signals = tf.pad(
        framed_signals,
        paddings=[[0, 0], [0, 0], [0, fft_length - frame_length]])

    # zero-phase window
    framed_signals_split = tf.split(
        framed_signals, [pad0, fft_length - pad0], axis=-1)
    framed_signals_split.reverse()
    framed_signals = tf.concat(framed_signals_split, axis=-1)

    # FFT of the real windowed signals in framed_signals
    return tf.signal.rfft(framed_signals, [fft_length])


def compute_ipc_stft(signals, frame_length, frame_step, window_name='blackman',
                     fft_length=None):
    if fft_length is None:
        fft_length = next_power_of_2(tf.cast(frame_length, dtype=tf.float32))

    # compute normalized window
    window, window_diff = generalized_cos_window(frame_length, window_name)

    # compute STFT and its time-derivative
    stft = compute_stft(
        signals, frame_length, frame_step, window, fft_length)
    stft_diff = compute_stft(
        signals, frame_length, frame_step, window_diff, fft_length)

    # STFT power
    pow_stft = tf.math.square(tf.math.abs(stft))
    # avoiding division by zero
    pow_stft = tf.where(pow_stft > 1e-9, pow_stft, 1e-9)
    pow_stft = tf.cast(pow_stft, dtype=tf.complex64)

    # compute bin-wise instantaneous frequency
    inst_freq = -tf.math.imag(stft_diff * tf.math.conj(stft) / pow_stft)
    inst_freq = (fft_length / frame_length) * inst_freq
    freq_shift = tf.range(0, fft_length // 2 + 1, dtype=tf.float32)
    freq_shift = freq_shift[tf.newaxis, tf.newaxis, :]
    inst_freq = inst_freq + freq_shift

    # compute instantaneous-phase-corrected stft (iPC-STFT)
    norm_omega = (frame_step / fft_length) * inst_freq
    phase = mod_cumsum(norm_omega, mod=1.0, axis=1)
    phase = phase % 1.0
    phase = phase * (2.0 * np.pi)
    phase = tf.cast(phase, dtype=tf.complex64)

    # normalize stft
    win_norm = 2.0 / tf.math.reduce_sum(window)
    win_norm = tf.cast(win_norm, dtype=tf.complex64)
    stft = stft * win_norm

    i = tf.complex(0.0, 1.0)
    ipc_matrix = tf.math.exp(-i * phase)
    ipc_stft = ipc_matrix * stft

    return ipc_stft, stft, ipc_matrix, inst_freq


def normalized_autocorrelation_function(
        signals, frame_step, corr_length, lags, pre_padding=0):
    frame_length = corr_length + lags - 1
    fft_length = next_power_of_2(tf.cast(frame_length, dtype=tf.float32))

    # cast length to int32
    frame_step = tf.cast(frame_step, dtype=tf.int32)
    corr_length = tf.cast(corr_length, dtype=tf.int32)
    lags = tf.cast(lags, dtype=tf.int32)
    frame_length = tf.cast(frame_length, dtype=tf.int32)
    fft_length = tf.cast(fft_length, dtype=tf.int32)

    pad0 = tf.cast(pre_padding, dtype=tf.int32)
    pad1 = tf.cast(frame_length - pad0 - 1, dtype=tf.int32)

    pad_signals = tf.pad(signals, paddings=[[0, 0], [pad0, pad1]])

    framed_signals = tf.signal.frame(
        pad_signals, frame_length, frame_step)

    # compute energy normalized auto-correlation function (nacf) in time domain
    # nacf_time = np.zeros(shape=(1, framed_signals.shape[1], lags))
    # sig0 = framed_signals[:, :, 0:corr_length].numpy()
    # energy0 = np.sum(sig0 * sig0, axis=-1, keepdims=True)
    # for tau in range(lags):
    #     sig1 = framed_signals[:, :, tau:corr_length + tau].numpy()
    #     energy1 = np.sum(sig1 * sig1, axis=-1, keepdims=True)
    #
    #     acf = np.sum(sig0 * sig1, axis=-1, keepdims=True)
    #     energy = energy0 + energy1
    #
    #     nacf_time[:, :, tau:tau+1] = np.where(
    #         energy > 0.0, 2.0 * acf / energy, 0.0)

    # compute auto-correlation function (acf) in frequency domain
    stft0 = tf.signal.rfft(framed_signals, [fft_length])
    stft1 = tf.signal.rfft(framed_signals[:, :, :corr_length], [fft_length])

    stft = tf.math.multiply(stft0, tf.math.conj(stft1))
    acf = tf.signal.irfft(stft)
    acf = acf[:, :, :lags]

    # compute energy normalized auto-correlation (nacf)
    full_energy = acf[:, :, :1]
    sub_signals = framed_signals[:, :, :lags - 1]
    sum_signals = framed_signals[:, :, corr_length:corr_length + lags - 1]
    sub_signals = tf.pad(sub_signals, paddings=[[0, 0], [0, 0], [1, 0]])
    sum_signals = tf.pad(sum_signals, paddings=[[0, 0], [0, 0], [1, 0]])
    sub_energy = tf.math.cumsum(sub_signals * sub_signals, axis=-1)
    sum_energy = tf.math.cumsum(sum_signals * sum_signals, axis=-1)

    energy = 2.0 * full_energy - sub_energy + sum_energy
    nacf = tf.where(energy > 0.0, 2.0 * acf / energy, 0.0)

    return nacf, full_energy


def refine_f0(signals, f0_estimate, sample_rate, frame_step,
              corr_periods=16.0, clarity_threshold=0.9):
    # preprocess f0_estimate to remove zeros
    f0_mean = non_zero_mean(f0_estimate, axis=1)
    f0_estimate = tf.where(f0_estimate > 0.0, f0_estimate, f0_mean)

    min_f0 = tf.math.reduce_min(f0_estimate)
    max_period = sample_rate / min_f0

    # compute array lengths
    corr_length = tf.math.round(corr_periods * max_period)
    # force odd corr_length
    corr_length = corr_length + (1.0 - tf.math.floormod(corr_length, 2))
    lags = tf.math.round(2.0 * max_period)
    # zero padding x to center first window at sample 0
    pre_padding = corr_length // 2

    nacf, energy = normalized_autocorrelation_function(
        signals, frame_step, corr_length, lags, pre_padding)

    # find f0 refinement by choosing the highest nacf peak
    # in range or +/- 1 semitone from f0_estimate
    pos, pos_shift, pos_value = parabolic_interp(nacf)
    p_mask = peak_mask(nacf, 0.0)

    peak_pos = pos * p_mask
    peak_value = pos_value * p_mask

    semitone_up = 2.0 ** (1 / 12.0)
    semitone_down = 2.0 ** (-1 / 12.0)

    period_estimate = sample_rate / f0_estimate
    max_period = period_estimate / semitone_down
    min_period = period_estimate / semitone_up

    r_mask = tf.where(
        tf.math.logical_and(peak_pos > min_period, peak_pos < max_period),
        1.0, 0.0)

    peak_pos = peak_pos * r_mask
    peak_value = peak_value * r_mask

    clarity = tf.math.reduce_max(peak_value, axis=-1, keepdims=True)
    max_mask = tf.where(
        tf.math.logical_and(clarity > clarity_threshold, peak_value == clarity),
        1.0, 0.0)
    peak_pos = peak_pos * max_mask

    period = non_zero_mean(peak_pos, axis=-1)

    period_mean = non_zero_mean(period, axis=1)

    f0_mean = tf.where(period_mean > 0.0, sample_rate / period_mean, f0_mean)
    f0 = tf.where(period > 0.0, sample_rate / period, f0_mean)

    return f0, clarity, energy


def harmonic_analysis_to_f0(h_freq, h_mag, db_threshold=-100.0):
    harmonics = tf.shape(h_freq)[-1]
    harmonic_numbers = tf.range(1, harmonics + 1, dtype=tf.float32)
    harmonic_numbers = harmonic_numbers[tf.newaxis, tf.newaxis, :]

    mean_mag = tf.math.reduce_mean(h_mag, axis=-1)
    f0 = tf.math.reduce_mean(
        h_freq / harmonic_numbers * h_mag, axis=-1) / mean_mag

    lin_threshold = db_to_lin(db_threshold)
    f0 = tf.where(mean_mag > lin_threshold, f0, 0.0)
    f0_mean = non_zero_mean(f0, axis=1)
    f0 = tf.where(f0 > 0.0, f0, f0_mean)

    return f0


def peak_mask(signals, threshold):
    # positions above threshold
    threshold_mask = tf.where(signals[:, :, 1:-1] > threshold, 1, 0)

    # positions higher than the next one
    next_mask = tf.where(signals[:, :, 1:-1] > signals[:, :, 2:], 1, 0)

    # positions higher than the previous one
    prev_mask = tf.where(signals[:, :, 1:-1] > signals[:, :, :-2], 1, 0)

    # positions fulfilling the three criteria
    mask = threshold_mask * next_mask * prev_mask

    mask = tf.cast(mask, dtype=tf.float32)

    return mask


def parabolic_interp(signals):
    val = signals[:, :, 1:-1]
    l_val = signals[:, :, :-2]
    r_val = signals[:, :, 2:]

    delta = l_val - r_val
    den = l_val - 2 * val + r_val

    # center of parabola shift from zero
    pos_shift = tf.where(
        den == 0.0,
        0.0,
        0.5 * delta / den)

    # center of parabola value
    pos_value = val - 0.25 * delta * pos_shift

    # center of parabola position
    pos = tf.range(1, tf.shape(signals)[-1] - 1, dtype=tf.float32)
    pos = pos[tf.newaxis, tf.newaxis, :]
    pos = tf.broadcast_to(pos, shape=tf.shape(pos_shift))

    pos = pos + pos_shift

    return pos, pos_shift, pos_value


def stft_peak_detection(stft, db_threshold):
    mag = lin_to_db(tf.math.abs(stft))
    phase = phase_unwrap(tf.math.angle(stft), axis=-1)

    pos, pos_shift, pos_value = parabolic_interp(mag)
    mask = peak_mask(mag, db_threshold)

    # position of peaks
    peak_pos = pos * mask

    # magnitude of peaks
    peak_mag = pos_value * mask

    # phase of peaks by linear interpolation
    pos_shift = pos_shift * mask

    peak_phase = tf.where(
        pos_shift >= 0.0,
        phase[:, :, 1:-1] * (1.0 - pos_shift) + phase[:, :, 2:] * pos_shift,
        phase[:, :, 1:-1] * (1.0 + pos_shift) - phase[:, :, :-2] * pos_shift)

    peak_phase = peak_phase * mask
    peak_phase = peak_phase % (2.0 * np.pi)

    return peak_pos, peak_mag, peak_phase


def harmonic_detection(h_freq_estimate, peak_freq, peak_mag, peak_phase):
    channels = np.shape(peak_freq)[0]
    frames = np.shape(peak_freq)[1]
    harmonics = np.shape(h_freq_estimate)[2]

    h_freq = np.zeros(shape=(channels, frames, harmonics), dtype=np.float32)
    h_mag = np.zeros(shape=(channels, frames, harmonics), dtype=np.float32)
    h_phase = np.zeros(shape=(channels, frames, harmonics), dtype=np.float32)

    for n in range(channels):
        for m in range(frames):
            hfe = h_freq_estimate[n, m, :]
            freq = peak_freq[n, m, :]
            mag = peak_mag[n, m, :]
            phase = peak_phase[n, m, :]

            indices = np.nonzero(freq)
            freq = freq[indices]
            mag = mag[indices]
            phase = phase[indices]

            n_freq = np.shape(freq)[0]

            j = 0
            for i in range(harmonics):
                if j == n_freq:
                    break

                cur_diff = np.abs(hfe[i] - freq[j])
                diff = cur_diff
                while cur_diff <= diff:
                    j += 1
                    if j == n_freq:
                        break
                    diff = cur_diff
                    cur_diff = np.abs(hfe[i] - freq[j])

                if i == harmonics - 1 or np.abs(
                        hfe[i + 1] - freq[j - 1]) > diff:
                    h_freq[n, m, i] = freq[j - 1]
                    h_mag[n, m, i] = mag[j - 1]
                    h_phase[n, m, i] = phase[j - 1]
                else:
                    j -= 1

    return h_freq, h_mag, h_phase


def compute_freq_correction(h_freq, h_phase, sample_rate, frame_step):
    frame_rate = sample_rate / frame_step
    freq_integral = 0.5 * (h_freq[:, :-1, :] + h_freq[:, 1:, :]) / frame_rate
    norm_phase = h_phase / (2.0 * np.pi)

    p0 = tf.pad(norm_phase[:, :-1, :], ((0, 0), (1, 0), (0, 0)))
    p1 = norm_phase
    freq_integral = tf.pad(freq_integral, ((0, 0), (1, 0), (0, 0)))

    freq_correction = phase_diff(p1, p0 + freq_integral, mod=1.0)

    return freq_correction


def generate_phase(h_freq, sample_rate, frame_step,
                   freq_correction=None, initial_phase=None):
    if freq_correction is None:
        channels = tf.shape(h_freq)[0]
        harmonics = tf.shape(h_freq)[2]
        freq_correction = tf.zeros(shape=(channels, 1, harmonics))

    if initial_phase is None:
        channels = tf.shape(h_freq)[0]
        harmonics = tf.shape(h_freq)[2]
        initial_phase = tf.zeros(shape=(channels, 1, harmonics))

    frame_rate = sample_rate / frame_step
    freq_integral = 0.5 * (h_freq[:, :-1, :] + h_freq[:, 1:, :]) / frame_rate
    freq_integral = tf.pad(freq_integral, ((0, 0), (1, 0), (0, 0)))
    freq_integral = freq_integral + freq_correction
    h_phase = mod_cumsum(freq_integral, mod=1.0, axis=1)
    h_phase = h_phase * (2.0 * np.pi) + initial_phase
    h_phase = h_phase % (2.0 * np.pi)

    return h_phase


def harmonic_synthesis(h_freq, h_mag, h_phase, sample_rate, frame_step,
                       method='pi'):
    # synthesis methods
    # pi: parameter interpolation
    # ola: overlap and add

    # remove components above nyquist frequency
    h_mag = tf.where(
        tf.greater_equal(h_freq, sample_rate / 2.0),
        tf.zeros_like(h_mag), h_mag)

    # the last element is repeated so that the synthesized audio has a size
    # greater than or equal to that of the analysis audio
    h_freq = tf.concat([h_freq, h_freq[:, -1:, :]], axis=1)
    h_mag = tf.concat([h_mag, h_mag[:, -1:, :]], axis=1)
    h_phase = tf.concat([h_phase, h_phase[:, -1:, :]], axis=1)

    h_freq = tf.expand_dims(h_freq, axis=-1)
    h_phase = tf.expand_dims(h_phase, axis=-1)
    h_mag = tf.expand_dims(h_mag, axis=-1)

    audio = None

    if method == 'pi':  # pi: parameter interpolation
        w = h_freq * (2.0 * np.pi)
        m0 = h_mag[:, :-1, :, :]
        m1 = h_mag[:, 1:, :, :]
        w0 = w[:, :-1, :, :]
        w1 = w[:, 1:, :, :]
        p0 = h_phase[:, :-1, :, :]
        p1 = h_phase[:, 1:, :, :]

        # compute M (phase interpolation unwrapping)
        period = frame_step / sample_rate
        x = ((p0 + w0 * period - p1) + (w1 - w0) * (period/2.0)) / (2.0 * np.pi)
        m = tf.math.round(x)

        # compute alpha(M) and beta(M)
        x0 = p1 - p0 - w0 * period + 2.0 * np.pi * m
        x1 = w1 - w0

        period2 = period * period
        period3 = period2 * period

        alpha = (3.0 / period2) * x0 + (-1.0 / period) * x1
        beta = (-2.0 / period3) * x0 + (1.0 / period2) * x1

        # parameters interpolation
        samples = tf.range(0, frame_step, dtype=tf.float32)
        samples = samples[tf.newaxis, tf.newaxis, tf.newaxis, :]

        # cubic phase interpolation
        t = samples / sample_rate
        # freqs = (w0 + (2.0 * alpha + 3.0 * beta * t) * t) / (2.0 * np.pi)
        phases = p0 + (w0 + (alpha + beta * t) * t) * t

        # linear amplitude interpolation
        gamma = samples / frame_step
        mags = (1.0 - gamma) * m0 + gamma * m1

        wavs = tf.cos(phases)
        wavs = mags * wavs
        wavs = tf.reduce_sum(wavs, axis=2)
        audio = tf.reshape(wavs, shape=(tf.shape(wavs)[0], -1))

    elif method == 'ola':  # ola: overlap and add
        # triangular window
        window = tf.range(0, frame_step+1, dtype=tf.float32) / frame_step
        window = tf.concat([window[:-1], window[::-1]], axis=0)
        window = window[tf.newaxis, tf.newaxis, :]

        # time axis
        t = tf.range(-frame_step, frame_step+1, dtype=tf.float32) / sample_rate
        t = t[tf.newaxis, tf.newaxis, tf.newaxis, :]

        phases = 2.0 * np.pi * h_freq * t + h_phase
        wavs = tf.cos(phases)
        wavs = h_mag * wavs
        wavs = tf.reduce_sum(wavs, axis=2)
        wavs = window * wavs
        audio = tf.signal.overlap_and_add(wavs, frame_step)
        audio = audio[:, frame_step:-(frame_step + 1)]

    return audio


def harmonic_analysis(signals, h_freq_estimate, sample_rate, frame_step,
                      frame_length, min_fft_length=None, db_threshold=-180.0):
    sample_rate = tf.cast(sample_rate, dtype=tf.float32)
    frame_step = tf.cast(frame_step, dtype=tf.int32)
    frame_length = tf.cast(frame_length, dtype=tf.float32)

    # force odd length
    frame_length = frame_length + (1.0 - tf.math.floormod(frame_length, 2))

    if min_fft_length is None or min_fft_length < frame_length:
        fft_length = next_power_of_2(frame_length)
    else:
        fft_length = min_fft_length

    frame_length = tf.cast(frame_length, dtype=tf.int32)
    fft_length = tf.cast(fft_length, dtype=tf.int32)

    window, _ = generalized_cos_window(frame_length, 'blackman')
    stft = compute_stft(signals, frame_length, frame_step, window, fft_length,
                        normalize_window=True)

    peak_pos, peak_mag, peak_phase = stft_peak_detection(stft, db_threshold)
    peak_freq = sample_rate * peak_pos / tf.cast(fft_length, dtype=tf.float32)

    # this part is implemented with numpy because I was not able to get
    # a fast implementation in tensorflow
    h_freq, h_mag, h_phase = harmonic_detection(
        h_freq_estimate.numpy(),
        peak_freq.numpy(),
        peak_mag.numpy(),
        peak_phase.numpy())

    h_freq = tf.cast(h_freq, dtype=tf.float32)
    h_mag = tf.cast(h_mag, dtype=tf.float32)
    h_phase = tf.cast(h_phase, dtype=tf.float32)

    h_mag = tf.where(h_freq == 0.0, 0.0, db_to_lin(h_mag))

    return h_freq, h_mag, h_phase


def fill_zeros(h_freq, h_phase, sample_rate, frame_step):
    # replace isolated zeros with linear interpolation
    h_freq_new = tf.pad(h_freq, ((0, 0), (1, 1), (0, 0)))

    cond = (h_freq_new[:, 2:, :] != 0.0) & \
           (h_freq_new[:, :-2, :] != 0.0) & \
           (h_freq_new[:, 1:-1, :] == 0.0)

    h_freq_new = tf.where(cond,
                          0.5 * (h_freq_new[:, 2:, :] + h_freq_new[:, :-2, :]),
                          h_freq_new[:, 1:-1, :])

    # replace freq remaining zeros with mean value
    freq_mean = non_zero_mean(h_freq_new, axis=1)
    h_freq_new = tf.where(h_freq_new == 0.0, freq_mean, h_freq_new)

    # generate phase from frequency integration to replace unidentified zones
    freq_correction = compute_freq_correction(
        h_freq_new, h_phase, sample_rate, frame_step)
    freq_correction = tf.where(h_freq == 0.0, 0.0, freq_correction)

    g_phase = generate_phase(
        h_freq_new, sample_rate, frame_step, freq_correction)
    h_phase_new = tf.where(h_freq == 0.0, g_phase, h_phase)

    return h_freq_new, h_phase_new


def iterative_harmonic_analysis(signals, f0_estimate, sample_rate, frame_step,
                                frame_length_list=None, min_fft_length=None,
                                db_threshold=-180.0,
                                semitone_variation_tol=None):
    max_f0 = tf.math.reduce_max(f0_estimate)
    harmonics = get_number_harmonics(max_f0, sample_rate)
    h_freq_estimate = get_harmonic_frequencies(f0_estimate, harmonics)

    if frame_length_list is None:
        f0_mean = float(tf.math.reduce_mean(f0_estimate))
        period = sample_rate / f0_mean
        frame_length_list = [int(round(8.0 * period))] * 4

    h_freq, h_mag, h_phase = harmonic_analysis(
        signals=signals,
        h_freq_estimate=h_freq_estimate,
        sample_rate=sample_rate,
        frame_step=frame_step,
        frame_length=frame_length_list[0],
        min_fft_length=min_fft_length,
        db_threshold=db_threshold)

    i = tf.complex(0.0, 1.0)

    for frame_length in frame_length_list[1:]:
        harmonic = harmonic_synthesis(h_freq, h_mag, h_phase,
                                      sample_rate, frame_step)

        residual = signals - harmonic[:, :signals.shape[1]]

        f0 = h_freq
        m0 = h_mag
        p0 = h_phase

        f1, m1, p1 = harmonic_analysis(
            signals=residual,
            h_freq_estimate=h_freq_estimate,
            sample_rate=sample_rate,
            frame_step=frame_step,
            frame_length=frame_length,
            min_fft_length=min_fft_length,
            db_threshold=db_threshold)

        # limits the frequency variations with respect to the first iteration
        if semitone_variation_tol is not None:
            up_tol = 2.0 ** (semitone_variation_tol / 12.0)
            down_tol = 2.0 ** (-semitone_variation_tol / 12.0)

            f1 = tf.where((f0 > 0.0) &
                          (f1 > 0.0) &
                          (f1 < h_freq_estimate * up_tol) &
                          (f1 > h_freq_estimate * down_tol),
                          f1, 0.0)

            m1 = tf.where(f1 == 0.0, 0.0, m1)
            p1 = tf.where(f1 == 0.0, 0.0, p1)

        m_sum = m0 + m1
        h_freq = tf.where(m_sum > 0.0,
                          m0 / m_sum * f0 + m1 / m_sum * f1,
                          0.0)

        m0 = tf.cast(m0, dtype=tf.complex64)
        m1 = tf.cast(m1, dtype=tf.complex64)
        p0 = tf.cast(p0, dtype=tf.complex64)
        p1 = tf.cast(p1, dtype=tf.complex64)

        h_complex = m0 * tf.math.exp(i * p0) + m1 * tf.math.exp(i * p1)

        h_mag = tf.math.abs(h_complex)
        h_phase = tf.math.angle(h_complex) % (2.0 * np.pi)

    # remove zeros from frequency and phase tracks
    h_freq, h_phase = fill_zeros(h_freq, h_phase, sample_rate, frame_step)

    return h_freq, h_mag, h_phase

