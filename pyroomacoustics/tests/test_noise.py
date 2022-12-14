import math
import pyroomacoustics as pra
from pathlib import Path
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt
import pytest

tol = 1e-5


@pytest.mark.parametrize(
    "dist,n_fft,from_signal,padding,n_samples,signal_n_samples,fs,ndim,plot,use_cholesky",
    [
        (0.5, 512, False, "none", 16000 * 30, 16000 * 30, 16000, 2, False, False),
        (0.05, 512, False, "none", 16000 * 30, 16000 * 30, 16000, 2, False, False),
        (0.1, 512, False, "none", 16000 * 30, 16000 * 30, 16000, 2, False, False),
        (0.5, 512, False, "none", 16000 * 30, 16000 * 30, 16000, 3, False, False),
        (0.05, 512, False, "none", 16000 * 30, 16000 * 30, 16000, 3, False, False),
        (0.1, 512, False, "none", 16000 * 30, 16000 * 30, 16000, 3, False, False),
        (0.1, 512, False, "none", 16000 * 30, 16000 * 30, 16000, 3, False, True),
        (0.1, 512, True, "none", 16000 * 30, 16000 * 30, 16000, 3, False, False),
        (0.1, 512, True, "none", 16000 * 30, 15999 * 15, 16000, 3, False, False),
        (0.1, 512, True, "repeat", 16000 * 30, 15999 * 15, 16000, 3, False, False),
        (0.1, 512, True, "reflect", 16000 * 30, 15999 * 15, 16000, 3, False, False),
    ],
)
def test_diffuse_noise(
    dist,
    n_fft,
    from_signal,
    padding,
    n_samples,
    signal_n_samples,
    fs,
    ndim,
    plot,
    use_cholesky,
):
    """
    Used for tests to compare the empirical and theoretical coherence

    noise: (n_samples, n_chan)
        The noise sample
    """

    snr = 10.0
    hop = n_fft // 4
    n_mics = 2

    mic_array = np.random.randn(3, n_mics) * dist
    n_mics = mic_array.shape[1]

    signal = np.random.randn(n_mics, n_samples)
    c = pra.constants.get("c")

    if from_signal:
        noise_signal = np.random.randn(signal_n_samples)
    else:
        noise_signal = None

    diffuse_noise_obj = pra.DiffuseNoise(
        snr=snr,
        signal=noise_signal,
        padding=padding,
        n_fft=n_fft,
        hop=hop,
        use_cholesky=use_cholesky,
        smooth_filters=True,
    )
    n_fft_a = math.ceil(np.pi * n_fft)
    hop_a = n_fft_a // 4
    dn_obj_analysis = pra.DiffuseNoise(snr=snr, signal=None, n_fft=n_fft_a, hop=hop_a)

    diffuse_noise = diffuse_noise_obj.generate(signal, mic_array, fs, c)
    filters = diffuse_noise_obj.make_filters(mic_array, fs, c)

    # compute the coherence matrix
    coh = dn_obj_analysis.compute_coherence_theory(mic_array, fs, c)

    # empirical coherence
    coh_data = dn_obj_analysis.compute_coherence_empirical(diffuse_noise)

    # check the SNR
    snr_est = pra.compute_snr(signal, diffuse_noise)

    # compute the performance measure defined in Mirabilii et al.
    mse = np.mean(abs(coh - coh_data) ** 2)
    mse = pra.dB(mse, power=True)
    smoothness = np.mean(abs(filters[1:] - filters[:-1]) ** 2)
    smoothness = pra.dB(smoothness, power=True)
    balance = np.sum(abs(filters)) / (
        filters.shape[0] * filters.shape[1] * np.sqrt(filters.shape[2])
    )
    balance = pra.dB(balance, power=True)

    assert mse < -20
    assert abs(snr - snr_est) < tol

    if plot:
        print(f"{mse=} {smoothness=} {balance=}")

        discrete_freqs = np.arange(n_fft_a // 2 + 1) / n_fft_a * fs

        n_plots = n_mics * (n_mics - 1) // 2
        fig, axes = plt.subplots(1, n_plots, sharey=True, squeeze=False)
        idx = 0
        distance = np.linalg.norm(mic_array[:, :, None] - mic_array[:, None, :], axis=0)
        for r in range(n_mics):
            for c in range(r + 1, n_mics):
                axes[0, idx].plot(discrete_freqs, coh[:, r, c], "r", label="theory")

                meas = coh_data[:, r, c]
                axes[0, idx].plot(discrete_freqs, meas.real, "g", label="measured")
                axes[0, idx].plot(
                    discrete_freqs, meas.imag, "g", label="measured (imag)"
                )

                axes[0, idx].set_title(f"dist={distance[r, c]:.3f}")
                axes[0, idx].set_xlabel("freq. (Hz)")
                if idx == 0:
                    axes[0, idx].set_ylabel("Coherence")

                idx += 1

        plt.show()


@pytest.mark.parametrize(
    "snr,siglen", [(-5, 1000), (0, 1000), (10, 1000), (40, 1000), (100, 1000)]
)
def test_compute_snr(snr, siglen):

    signal = np.random.randn(siglen)
    noise = np.random.randn(siglen)

    # we change the scale of the noise in the mix
    mix1 = pra.mix_signal_noise(signal, noise, snr)
    snr_est_1 = pra.compute_snr(signal, mix1 - signal)

    # here we change the scale of the signal in the mix
    mix2 = pra.mix_signal_noise(signal, noise, snr, scale_noise=False)
    snr_est_2 = pra.compute_snr(mix2 - noise, noise)

    assert abs(snr_est_1 - snr) < tol
    assert abs(snr_est_2 - snr) < tol


@pytest.mark.parametrize(
    "snr,siglen", [(-5, 1000), (0, 1000), (10, 1000), (40, 1000), (100, 1000)]
)
def test_scale_signal(snr, siglen):

    reference = np.random.randn(siglen)
    signal = np.random.randn(siglen)

    signal_scaled = pra.scale_signal(signal, reference, snr)

    snr_est = pra.compute_snr(signal_scaled, reference)

    assert abs(snr - snr_est) < tol


@pytest.mark.parametrize(
    "snr,siglen", [(-5, 1000), (0, 1000), (10, 1000), (40, 1000), (100, 1000)]
)
def test_white_noise(snr, siglen):

    white_noise = pra.WhiteNoise(snr=snr)

    # test generate function
    signal = np.random.randn(siglen)
    noise = white_noise.generate(signal)

    noisy_signal = white_noise.add(signal)

    snr_est = pra.compute_snr(signal, noise)
    snr_est_2 = pra.compute_snr(signal, noisy_signal - signal)

    assert abs(snr - snr_est) < tol
    assert abs(snr - snr_est_2) < tol


if __name__ == "__main__":

    fs = 16000
    c = 343
    n_samples = 60 * fs  # 1 min. data
    n_mics = 3
    n_fft = 512
    hop = 128
    mic_array = np.array([[0.0, 0.0, 0.0], [0.0, 0.4, 0.0]])
    mic_array = np.array([[0.0, 0.0, 0.0], [0.0, 0.05, 0.0], [0.0, 0.1, 0.0]])
    mic_array = np.random.randn(3, 3) * 0.1
    dist = np.linalg.norm(mic_array[:, None, :] - mic_array[None, :, :], axis=-1)

    snr = 10.0
    from_signal = True
    padding = "repeat"
    n_samples = fs * 30
    signal_n_samples = fs * 30
    ndim = 3
    plot = True
    use_evd = True

    test_diffuse_noise(
        snr,
        n_fft,
        hop,
        from_signal,
        padding,
        n_samples,
        signal_n_samples,
        fs,
        ndim,
        plot,
        use_evd,
    )
